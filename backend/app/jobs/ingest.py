"""One ingestion run: fetch every feed, store listings idempotently, record the outcome.

Failure design (docs/jobs-plan.md 2.4):

- **Per-feed isolation.** Each feed runs under its own deadline and its own
  try/except. A board that times out, 404s or returns garbage becomes one
  entry in `errors`; every other feed still runs.
- **One commit per listing**, so a run cut short keeps everything stored so
  far, and the next run simply continues (the upsert is idempotent).
- **Always a terminal status.** `completed` (no errors), `partial` (some feeds
  failed), `failed` (every feed failed). A run deadline cancels unfinished
  feeds and still records the result. If the process dies, the reaper marks
  the run `interrupted`.
- **One run at a time**: a partial unique index on `status = 'running'` makes a
  second start fail with `RunAlreadyRunningError`, atomically.
- **Private data stays out**: error entries carry the feed key, the exception
  type and our own message. Search queries are identified by number, never
  by text, and unexpected exceptions are logged by type only.
"""

import asyncio
import logging
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import httpx
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.models import Company, IngestionRun, Job
from app.jobs.analyse import analyse_jobs
from app.jobs.dedup import upsert_listing
from app.jobs.robots import RobotsChecker
from app.jobs.sources.ashby import AshbyBoard
from app.jobs.sources.base import Feed, HttpPolicy, Sleep, SourceError
from app.jobs.sources.greenhouse import GreenhouseBoard
from app.jobs.sources.hn import HnWhoIsHiring
from app.jobs.sources.lever import LeverBoard
from app.jobs.sources.web_search import WebSearchFeed
from app.llm.base import LLMProvider
from app.rag.embeddings import EmbeddingProvider
from app.web.fetch import USER_AGENT
from app.web.search import SearchProvider, build_search_provider, web_search_enabled
from app.web.ssrf import SafeTransport

logger = logging.getLogger(__name__)

BOARD_FEEDS: dict[str, type[GreenhouseBoard | LeverBoard | AshbyBoard]] = {
    "greenhouse": GreenhouseBoard,
    "lever": LeverBoard,
    "ashby": AshbyBoard,
}


class RunAlreadyRunningError(Exception):
    """Another ingestion run is in progress."""


@dataclass(slots=True)
class RunSummary:
    """The recorded outcome of one run."""

    run_id: uuid.UUID
    status: str
    counts: dict[str, dict[str, int]]
    errors: list[dict[str, str]] = field(default_factory=list)


async def run_ingestion(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    *,
    trigger: str = "manual",
    feeds: list[Feed] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    robots_transport: httpx.AsyncBaseTransport | None = None,
    search: SearchProvider | None = None,
    llm: LLMProvider | None = None,
    embedder: EmbeddingProvider | None = None,
    sleep: Sleep = asyncio.sleep,
) -> RunSummary:
    """Start a run: fetch every feed, then analyse (extract with `llm`, judge in code),
    and record the outcome. Raises only RunAlreadyRunningError (before anything is
    fetched).

    Both stages share one deadline (JOBS_RUN_TIMEOUT_S). Without `llm`, new
    listings are not extracted, but existing extractions are still re-judged.
    Without `embedder`, scores leave out the semantic component.
    `feeds`, the transports, `search` and `sleep` are injectable for tests;
    by default feeds come from the watchlist and settings.
    """
    run_id = await _start_run(session_factory, trigger)
    deadline = asyncio.get_running_loop().time() + settings.jobs_run_timeout_s
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    errors: list[dict[str, str]] = []
    status = "failed"
    policy = HttpPolicy(max_attempts=settings.jobs_http_max_attempts, backoff_s=1.0, sleep=sleep)
    timeout = httpx.Timeout(settings.jobs_http_timeout_s)
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    try:
        async with (
            httpx.AsyncClient(
                timeout=timeout, headers=headers, transport=transport, follow_redirects=True
            ) as client,
            httpx.AsyncClient(
                timeout=timeout,
                transport=robots_transport or SafeTransport(),
                follow_redirects=True,
                max_redirects=5,
            ) as robots_client,
        ):
            if feeds is None:
                feeds = await build_feeds(
                    session_factory, settings, RobotsChecker(robots_client), search
                )
            succeeded = await _run_feeds(
                feeds, client, policy, session_factory, settings, counts, errors, deadline
            )
        analysed = await _analyse(
            session_factory, settings, llm, embedder, counts, errors, deadline
        )
        if not errors:
            status = "completed"
        else:
            status = "partial" if succeeded or (analysed and not feeds) else "failed"
    except asyncio.CancelledError:
        status = "interrupted"
        errors.append(_error("run", "run", "CancelledError", "the server stopped mid-run"))
        await asyncio.shield(_finish_run(session_factory, run_id, status, counts, errors))
        raise
    except Exception as exc:
        logger.error("Ingestion run %s failed: %s", run_id, type(exc).__name__)
        errors.append(_error("run", "run", type(exc).__name__, "unexpected error"))
    await _finish_run(session_factory, run_id, status, counts, errors)
    logger.info("Ingestion run %s %s: %d feed error(s)", run_id, status, len(errors))
    return RunSummary(
        run_id=run_id,
        status=status,
        counts={source: dict(c) for source, c in counts.items()},
        errors=errors,
    )


async def build_feeds(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    robots: RobotsChecker,
    search: SearchProvider | None = None,
) -> list[Feed]:
    """Watched companies' boards, then HN, then one feed per search query."""
    async with session_factory() as session:
        companies = (
            await session.scalars(
                select(Company)
                .where(Company.watch.is_(True), Company.ats != "none")
                .order_by(Company.name)
            )
        ).all()
    feeds: list[Feed] = [
        BOARD_FEEDS[c.ats](c.board_token or "", company_id=c.id, company_name=c.name)
        for c in companies
    ]
    if settings.jobs_hn_enabled:
        feeds.append(HnWhoIsHiring(keywords=settings.jobs_hn_keywords))
    if settings.jobs_search_queries and (search is not None or web_search_enabled(settings)):
        provider = search or build_search_provider(settings)
        denied = frozenset(d.lower().strip(".") for d in settings.jobs_denied_domains)
        feeds.extend(
            WebSearchFeed(
                query,
                number=n,
                provider=provider,
                robots=robots,
                max_results=settings.jobs_search_max_results,
                denied_domains=denied,
            )
            for n, query in enumerate(settings.jobs_search_queries, start=1)
        )
    return feeds


async def _run_feeds(
    feeds: list[Feed],
    client: httpx.AsyncClient,
    policy: HttpPolicy,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    counts: defaultdict[str, Counter[str]],
    errors: list[dict[str, str]],
    deadline: float,
) -> int:
    """Process feeds with bounded concurrency under the run deadline; return how many
    succeeded. Each feed's failure is recorded, never raised."""
    gate = asyncio.Semaphore(settings.jobs_feed_concurrency)
    succeeded = 0

    async def one(feed: Feed) -> None:
        nonlocal succeeded
        async with gate:
            try:
                async with asyncio.timeout(settings.jobs_feed_timeout_s):
                    await _process_feed(feed, client, policy, session_factory, counts)
                succeeded += 1
            except SourceError as exc:
                errors.append(_error(feed.key, feed.source, "SourceError", str(exc)))
            except TimeoutError:
                message = f"{feed.key}: no result within {settings.jobs_feed_timeout_s:g}s"
                errors.append(_error(feed.key, feed.source, "TimeoutError", message))
            except Exception as exc:  # a bug or the database: record, keep going
                logger.error("Feed %s failed: %s", feed.key, type(exc).__name__)
                errors.append(_error(feed.key, feed.source, type(exc).__name__, "unexpected error"))
            else:
                return
            counts[feed.source]["errors"] += 1

    tasks = [asyncio.create_task(one(feed)) for feed in feeds]
    try:
        async with asyncio.timeout_at(deadline):
            await asyncio.gather(*tasks)
    except TimeoutError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        unfinished = sum(1 for t in tasks if t.cancelled())
        errors.append(
            _error(
                "run",
                "run",
                "TimeoutError",
                f"run deadline of {settings.jobs_run_timeout_s:g}s reached; "
                f"{unfinished} feed(s) unfinished",
            )
        )
    return succeeded


async def _analyse(
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    llm: LLMProvider | None,
    embedder: EmbeddingProvider | None,
    counts: defaultdict[str, Counter[str]],
    errors: list[dict[str, str]],
    deadline: float,
) -> bool:
    """The analysis stage under what is left of the run deadline. True if it finished.

    Listings already extracted stay extracted when the deadline hits; the rest
    are picked up by the next run.
    """
    try:
        async with asyncio.timeout_at(deadline):
            outcome = await analyse_jobs(session_factory, settings, llm, embedder)
    except TimeoutError:
        message = "run deadline reached during analysis; the rest waits for the next run"
        errors.append(_error("analysis", "analysis", "TimeoutError", message))
        return False
    except Exception as exc:  # e.g. the database dropped: record, never crash the run
        logger.error("Analysis stage failed: %s", type(exc).__name__)
        errors.append(_error("analysis", "analysis", type(exc).__name__, "unexpected error"))
        return False
    counts["analysis"].update(outcome.counts)
    if outcome.llm_unavailable:
        errors.append(_error("analysis", "analysis", "LLMUnavailable", outcome.llm_unavailable))
    return True


async def _process_feed(
    feed: Feed,
    client: httpx.AsyncClient,
    policy: HttpPolicy,
    session_factory: async_sessionmaker[AsyncSession],
    counts: defaultdict[str, Counter[str]],
) -> None:
    """Fetch, upsert each listing in its own transaction, then close what disappeared."""
    result = await feed.fetch(client, policy)
    tally = counts[feed.source]
    tally["fetched"] += len(result.listings)
    tally["skipped"] += result.skipped
    seen: list[str] = []
    for listing in result.listings:
        async with session_factory() as session:
            stored = await upsert_listing(session, listing, feed.company_id)
            await session.commit()
        tally[stored.action] += 1
        seen.append(listing.external_id)
    if result.complete and feed.company_id is not None:
        tally["closed"] += await _close_missing(session_factory, feed, seen)


async def _close_missing(
    session_factory: async_sessionmaker[AsyncSession], feed: Feed, seen: list[str]
) -> int:
    """Mark this board's open jobs that are no longer listed as closed."""
    async with session_factory() as session:
        closed = await session.scalars(
            update(Job)
            .where(
                Job.source == feed.source,
                Job.company_id == feed.company_id,
                Job.closed_at.is_(None),
                Job.external_id.not_in(seen),
            )
            .values(closed_at=func.now())
            .returning(Job.id)
        )
        n = len(closed.all())
        await session.commit()
    return n


async def _start_run(session_factory: async_sessionmaker[AsyncSession], trigger: str) -> uuid.UUID:
    async with session_factory() as session:
        run = IngestionRun(trigger=trigger, status="running")
        session.add(run)
        try:
            await session.commit()
        except IntegrityError as exc:
            if "uq_ingestion_runs_one_running" in str(exc.orig):
                raise RunAlreadyRunningError("an ingestion run is already in progress") from None
            raise
        return run.id


async def _finish_run(
    session_factory: async_sessionmaker[AsyncSession],
    run_id: uuid.UUID,
    status: str,
    counts: defaultdict[str, Counter[str]],
    errors: list[dict[str, str]],
) -> None:
    """Record the terminal state. If even this fails, the reaper will close the run."""
    try:
        async with session_factory() as session:
            await session.execute(
                update(IngestionRun)
                .where(IngestionRun.id == run_id, IngestionRun.status == "running")
                .values(
                    status=status,
                    finished_at=func.now(),
                    counts={source: dict(c) for source, c in counts.items()},
                    errors=errors,
                )
            )
            await session.commit()
    except Exception as exc:
        logger.error("Could not record the end of run %s: %s", run_id, type(exc).__name__)


def _error(feed: str, source: str, error_type: str, message: str) -> dict[str, str]:
    return {"feed": feed, "source": source, "error_type": error_type, "message": message}
