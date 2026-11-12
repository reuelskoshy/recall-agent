"""The analysis stage of a run: extract new listings (LLM), then judge every open job (code).

Extraction:
- Only open jobs whose *current* content hash has no extraction yet, newest
  first, at most JOBS_EXTRACT_MAX_PER_RUN per run, JOBS_EXTRACT_CONCURRENCY at
  a time, each under JOBS_EXTRACT_TIMEOUT_S. The prefilter (prefilter.py)
  first drops listings the rules would certainly reject and puts senior
  titles last. One analysis row per (job,
  content hash): an unchanged listing is never sent to the LLM again.
- A listing the model cannot parse, or that times out, counts one attempt;
  after JOBS_EXTRACT_MAX_ATTEMPTS it is left with verdict `error`.
- A *transient* LLM failure (outage, open circuit breaker) is the provider's
  fault, not the listing's: no attempt is counted and the stage stops starting
  new extractions. The rest wait for the next run.

Judging:
- Every open job with an extraction is re-judged on every run (no LLM, a few
  milliseconds each), so a profile edit or a rules change applies everywhere.
  Rows are only written when the verdict or reasons changed.
- Without a profile there is nothing to judge against: verdicts stay empty.

Logs and stored errors hold counts and error types only, never listing or
profile text.
"""

import asyncio
import logging
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.models import CandidateProfile, Job, JobAnalysis, ProfileEntry, ProfileSkill
from app.jobs.extraction import ExtractionError, extract
from app.jobs.prefilter import prefilter, priority
from app.jobs.rules import RULES_VERSION, Listing, ProfileFacts, judge
from app.jobs.schemas import EXTRACTOR_VERSION, ExtractedJob
from app.jobs.scoring import (
    ScoreWeights,
    ScoringProfile,
    cosine,
    job_text,
    profile_text,
    score_job,
)
from app.llm.base import LLMError, LLMProvider
from app.rag.embeddings import EmbeddingProvider, Vector

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Pending:
    job_id: int
    content_hash: str
    title: str
    location_raw: str
    text: str


@dataclass(slots=True)
class AnalysisOutcome:
    counts: Counter[str]
    llm_unavailable: str | None = None
    """Set when a transient LLM failure stopped extraction early (a safe message)."""


async def analyse_jobs(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    llm: LLMProvider | None,
    embedder: EmbeddingProvider | None = None,
) -> AnalysisOutcome:
    """Extract (if an LLM is given), judge, then score. Never raises except on
    cancellation or a database error, which the caller records. Without an
    embedder, scores are computed without the semantic component."""
    counts: Counter[str] = Counter()
    outcome = AnalysisOutcome(counts=counts)
    if llm is not None:
        await _extract_pending(session_factory, settings, llm, outcome)
    await judge_open_jobs(session_factory, counts)
    await score_open_jobs(session_factory, settings, embedder, counts)
    return outcome


# ---- Extraction -------------------------------------------------------------------------


async def _pending_jobs(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    counts: Counter[str] | None = None,
) -> list[_Pending]:
    """The listings to extract this run, newest first, at most the per-run cap.

    With the prefilter on and a profile present, listings the rules would
    certainly reject are left out (counted as prefiltered_*), and senior titles
    go after the rest. Only titles and locations are read until the cut.
    """
    current = and_(JobAnalysis.job_id == Job.id, JobAnalysis.content_hash == Job.content_hash)
    stmt = (
        select(Job.id, Job.content_hash, Job.title, Job.location_raw)
        .outerjoin(JobAnalysis, current)
        .where(
            Job.closed_at.is_(None),
            or_(
                JobAnalysis.id.is_(None),
                and_(
                    JobAnalysis.extraction.is_(None),
                    JobAnalysis.extraction_attempts < settings.jobs_extract_max_attempts,
                ),
                # Extracted by an older prompt that lacks fields the rules now read.
                and_(
                    JobAnalysis.extraction.is_not(None),
                    JobAnalysis.extractor_version != EXTRACTOR_VERSION,
                ),
            ),
        )
        .order_by(Job.first_seen_at.desc(), Job.id.desc())
    )
    limit = settings.jobs_extract_max_per_run
    async with session_factory() as session:
        profile = None
        if settings.jobs_prefilter_enabled:
            profile = await session.get(CandidateProfile, 1)
        if profile is None:
            rows = (await session.execute(stmt.limit(limit))).all()
        else:
            facts = ProfileFacts.from_profile(profile)
            kept = []
            for row in (await session.execute(stmt)).all():
                skip = prefilter(row.title, row.location_raw, facts)
                if skip is None:
                    kept.append(row)
                elif counts is not None:
                    counts[f"prefiltered_{skip.rule}"] += 1
            # Stable sort: newest first within each priority.
            rows = sorted(kept, key=lambda r: priority(r.title))[:limit]
        if not rows:
            return []
        texts = dict(
            (
                await session.execute(
                    select(Job.id, Job.raw_text).where(Job.id.in_([r.id for r in rows]))
                )
            ).all()
        )
    return [_Pending(r.id, r.content_hash, r.title, r.location_raw, texts[r.id]) for r in rows]


async def _extract_pending(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    llm: LLMProvider,
    outcome: AnalysisOutcome,
) -> None:
    counts = outcome.counts
    pending = await _pending_jobs(session_factory, settings, counts)
    counts["to_extract"] += len(pending)
    gate = asyncio.Semaphore(settings.jobs_extract_concurrency)
    stop = asyncio.Event()

    async def one(item: _Pending) -> None:
        async with gate:
            if stop.is_set():
                counts["deferred"] += 1
                return
            try:
                job = await extract(
                    llm,
                    item.title,
                    item.location_raw,
                    item.text,
                    timeout_s=settings.jobs_extract_timeout_s,
                )
            except LLMError as exc:
                if exc.transient:
                    if not stop.is_set():
                        stop.set()
                        outcome.llm_unavailable = "the LLM is unavailable; extraction deferred"
                    counts["deferred"] += 1
                    return
                await _record_failure(session_factory, item, "LLMError")
                counts["extraction_failed"] += 1
                return
            except TimeoutError:
                await _record_failure(session_factory, item, "TimeoutError")
                counts["extraction_failed"] += 1
                return
            except ExtractionError as exc:
                await _record_failure(session_factory, item, f"ExtractionError: {exc}")
                counts["extraction_failed"] += 1
                return
            except Exception as exc:  # a provider bug: this listing's attempt, by type only
                await _record_failure(session_factory, item, type(exc).__name__)
                counts["extraction_failed"] += 1
                return
            await _record_success(session_factory, item, job)
            counts["extracted"] += 1

    await asyncio.gather(*(one(item) for item in pending))


def _analysis_insert(item: _Pending, **values: Any) -> Any:
    return insert(JobAnalysis).values(
        job_id=item.job_id,
        content_hash=item.content_hash,
        extractor_version=EXTRACTOR_VERSION,
        **values,
    )


async def _record_success(
    session_factory: async_sessionmaker[AsyncSession], item: _Pending, job: ExtractedJob
) -> None:
    data = job.model_dump(mode="json")
    stmt = _analysis_insert(item, extraction=data, extraction_attempts=0)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_job_analyses_job_id_content_hash",
        set_={
            "extraction": stmt.excluded.extraction,
            "extractor_version": stmt.excluded.extractor_version,
            "extraction_error": None,
            "verdict": None,  # judged right after, in this run
            "rules_version": None,
        },
    )
    async with session_factory() as session:
        await session.execute(stmt)
        await session.commit()


async def _record_failure(
    session_factory: async_sessionmaker[AsyncSession], item: _Pending, error: str
) -> None:
    """One more failed attempt; the verdict is `error` until an attempt succeeds."""
    stmt = _analysis_insert(item, extraction_error=error, extraction_attempts=1, verdict="error")
    stmt = stmt.on_conflict_do_update(
        constraint="uq_job_analyses_job_id_content_hash",
        set_={
            "extraction_error": error,
            "extraction_attempts": JobAnalysis.extraction_attempts + 1,
            "verdict": "error",
            "evaluated_at": func.now(),
        },
    )
    async with session_factory() as session:
        await session.execute(stmt)
        await session.commit()
    logger.info("Extraction failed for job %d (%s)", item.job_id, error.split(":")[0])


# ---- Judging ----------------------------------------------------------------------------


async def judge_open_jobs(
    session_factory: async_sessionmaker[AsyncSession], counts: Counter[str]
) -> None:
    """Re-judge every open job's current extraction against the profile."""
    async with session_factory() as session:
        profile = await session.get(CandidateProfile, 1)
        if profile is None:
            counts["not_judged_no_profile"] += 1
            return
        facts = ProfileFacts.from_profile(profile)
        rows = (
            await session.execute(
                select(JobAnalysis, Job.title, Job.location_raw, Job.raw_text)
                .join(
                    Job,
                    and_(
                        JobAnalysis.job_id == Job.id,
                        JobAnalysis.content_hash == Job.content_hash,
                    ),
                )
                .where(Job.closed_at.is_(None), JobAnalysis.extraction.is_not(None))
            )
        ).all()
        for analysis, title, location_raw, raw_text in rows:
            try:
                extracted = ExtractedJob.model_validate(analysis.extraction)
            except ValueError:
                counts["unreadable_extraction"] += 1
                continue
            verdict, reasons = judge(extracted, facts, Listing(title, location_raw, raw_text))
            stored = [r.model_dump(mode="json") for r in reasons]
            counts[verdict] += 1
            if (
                analysis.verdict == verdict
                and analysis.reasons == stored
                and analysis.rules_version == RULES_VERSION
            ):
                continue
            await session.execute(
                update(JobAnalysis)
                .where(JobAnalysis.id == analysis.id)
                .values(
                    verdict=verdict,
                    reasons=stored,
                    rules_version=RULES_VERSION,
                    evaluated_at=func.now(),
                )
            )
            counts["rejudged"] += 1
        await session.commit()


# ---- Scoring ----------------------------------------------------------------------------

SCORED_VERDICTS = ("eligible", "flagged")
EMBED_BATCH = 32


async def score_open_jobs(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    embedder: EmbeddingProvider | None,
    counts: Counter[str],
) -> None:
    """Score every open eligible or flagged job; clear the score of the rest.

    Each listing version is embedded once (stored on its analysis row); the
    profile is embedded on every run, since it can change at any time. If
    embedding fails, the jobs are still scored, without the semantic part.
    Rows are only written when the score or its components changed.
    """
    async with session_factory() as session:
        profile = await session.get(CandidateProfile, 1)
        if profile is None:
            return
        skills = list((await session.scalars(select(ProfileSkill.name))).all())
        entry_techs = (await session.scalars(select(func.unnest(ProfileEntry.technologies)))).all()
        scoring_profile = ScoringProfile.from_profile(profile, [*skills, *entry_techs])
        weights = ScoreWeights.from_settings(settings)
        profile_vector = await _embed_profile(session, profile, embedder, skills, counts)

        current = and_(JobAnalysis.job_id == Job.id, JobAnalysis.content_hash == Job.content_hash)
        rows = (
            await session.execute(
                select(
                    JobAnalysis.id,
                    JobAnalysis.extraction,
                    JobAnalysis.embedding,
                    JobAnalysis.score,
                    JobAnalysis.score_components,
                    Job.title,
                    Job.raw_text,
                    func.coalesce(Job.posted_at, Job.first_seen_at).label("seen_at"),
                )
                .join(Job, current)
                .where(Job.closed_at.is_(None), JobAnalysis.verdict.in_(SCORED_VERDICTS))
            )
        ).all()

        parsed: dict[uuid.UUID, ExtractedJob] = {}
        for row in rows:
            try:
                parsed[row.id] = ExtractedJob.model_validate(row.extraction)
            except ValueError:
                counts["unreadable_extraction"] += 1
        vectors: dict[uuid.UUID, Vector] = {
            row.id: list(row.embedding) for row in rows if row.embedding is not None
        }
        missing = [row for row in rows if row.id in parsed and row.id not in vectors]
        if embedder is not None and missing:
            vectors.update(await _embed_jobs(session, embedder, missing, parsed, counts))

        now = datetime.now(UTC)
        for row in rows:
            job = parsed.get(row.id)
            if job is None:
                continue
            vector = vectors.get(row.id)
            similarity = (
                cosine(profile_vector, vector)
                if profile_vector is not None and vector is not None
                else None
            )
            score, components = score_job(
                job, scoring_profile, weights, similarity=similarity, seen_at=row.seen_at, now=now
            )
            counts["scored"] += 1
            if row.score == score and row.score_components == components:
                continue
            await session.execute(
                update(JobAnalysis)
                .where(JobAnalysis.id == row.id)
                .values(score=score, score_components=components)
            )

        # Jobs that are no longer eligible or flagged lose their score.
        cleared = await session.execute(
            update(JobAnalysis)
            .where(
                JobAnalysis.score.is_not(None),
                or_(JobAnalysis.verdict.is_(None), JobAnalysis.verdict.not_in(SCORED_VERDICTS)),
            )
            .values(score=None, score_components=None)
        )
        counts["score_cleared"] += getattr(cleared, "rowcount", 0) or 0
        await session.commit()


async def _embed_profile(
    session: AsyncSession,
    profile: CandidateProfile,
    embedder: EmbeddingProvider | None,
    skills: list[str],
    counts: Counter[str],
) -> Vector | None:
    if embedder is None:
        return None
    text = profile_text(profile.headline, profile.summary, skills)
    try:
        vector = (await embedder.embed_documents([text]))[0]
    except Exception as exc:  # the model failed to load or run: score without it
        logger.warning("Profile embedding failed (%s); scoring without it", type(exc).__name__)
        counts["embedding_failed"] += 1
        return None
    await session.execute(
        update(CandidateProfile).where(CandidateProfile.id == 1).values(embedding=vector)
    )
    return vector


async def _embed_jobs(
    session: AsyncSession,
    embedder: EmbeddingProvider,
    rows: list[Any],
    parsed: dict[uuid.UUID, ExtractedJob],
    counts: Counter[str],
) -> dict[uuid.UUID, Vector]:
    """Embed listings in batches; on a failure, the rest are scored without it."""
    vectors: dict[uuid.UUID, Vector] = {}
    for start in range(0, len(rows), EMBED_BATCH):
        batch = rows[start : start + EMBED_BATCH]
        texts = [job_text(r.title, parsed[r.id], r.raw_text) for r in batch]
        try:
            embedded = await embedder.embed_documents(texts)
        except Exception as exc:
            logger.warning("Job embedding failed (%s); scoring without it", type(exc).__name__)
            counts["embedding_failed"] += len(rows) - start
            break
        for row, vector in zip(batch, embedded, strict=True):
            vectors[row.id] = vector
            await session.execute(
                update(JobAnalysis).where(JobAnalysis.id == row.id).values(embedding=vector)
            )
        counts["embedded"] += len(batch)
    return vectors
