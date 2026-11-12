"""The analysis stage against the real test database: extraction once per listing version,
failures and give-up, LLM outages, re-judging without the LLM, and private data in logs."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from sqlalchemy import select, update

from app.core.config import Settings
from app.db.models import CandidateProfile, Job, JobAnalysis
from app.db.session import SessionLocal
from app.jobs.analyse import analyse_jobs
from app.jobs.dedup import RawListing, upsert_listing
from app.jobs.ingest import run_ingestion
from app.jobs.rules import RULES_VERSION
from app.jobs.schemas import EXTRACTOR_VERSION
from app.llm.base import (
    ChatMessage,
    LLMError,
    LLMProvider,
    StreamEnd,
    StreamEvent,
    TextDelta,
    ToolSpec,
)
from tests.test_jobs_ingest import StaticFeed

pytestmark = pytest.mark.usefixtures("db")

SENTINEL = "SENTINEL-profile-9c41"

BENGALURU = {
    "locations": [{"city": "Bengaluru", "region": "Karnataka", "country": "India"}],
    "work_mode": "onsite",
    "role_family": "engineering",
    "employment_type": "full_time",
    "min_years_experience": 0,
    "graduation_years": [2025],
    "technologies": ["Python", "PostgreSQL"],
    "evidence": {"locations": "Bengaluru", "graduation_years": "2025 batch"},
}
AUSTIN = {
    "locations": [{"city": "Austin", "country": "United States"}],
    "role_family": "engineering",
    "evidence": {"locations": "Austin, Texas"},
}


class KeyedLLM(LLMProvider):
    """Answers by listing title: JSON (dict), raw text (str), an exception, or a hang."""

    name = "keyed"
    model = "keyed-model"

    def __init__(self, replies: dict[str, Any]) -> None:
        self.replies = replies
        self.calls: list[str] = []

    async def stream_chat(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[StreamEvent]:
        title = messages[-1].content.split("\n", 1)[0].removeprefix("Title: ")
        self.calls.append(title)
        reply = self.replies[title]
        if isinstance(reply, Exception):
            raise reply
        if reply == "hang":
            await asyncio.sleep(3600)
        text = reply if isinstance(reply, str) else json.dumps(reply)
        yield TextDelta(text)
        yield StreamEnd(finish_reason="stop")


def settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "jobs_enabled": True,
        "jobs_hn_enabled": False,
        "jobs_extract_concurrency": 3,
        "jobs_extract_timeout_s": 5,
        "jobs_run_timeout_s": 30,
    }
    return Settings(**{**base, **overrides})


async def add_job(title: str, text: str = "Python role. Bengaluru. 2025 batch.") -> int:
    async with SessionLocal() as session:
        result = await upsert_listing(
            session,
            RawListing(
                source="greenhouse",
                external_id=title,
                title=title,
                url=f"https://example.test/{title}",
                raw_text=text,
            ),
        )
        await session.commit()
        return result.job_id


async def set_profile(**fields: Any) -> None:
    values = {
        "graduation_year": 2025,
        "years_experience": 0,
        "allowed_locations": ["India"],
        "summary": SENTINEL,
        **fields,
    }
    async with SessionLocal() as session:
        profile = await session.get(CandidateProfile, 1)
        if profile is None:
            session.add(CandidateProfile(id=1, **values))
        else:
            for key, value in values.items():
                setattr(profile, key, value)
        await session.commit()


async def analyses() -> dict[str, JobAnalysis]:
    async with SessionLocal() as session:
        rows = await session.execute(
            select(Job.title, JobAnalysis).join(JobAnalysis, JobAnalysis.job_id == Job.id)
        )
        return {title: analysis for title, analysis in rows.all()}


async def test_extracts_and_judges_new_jobs() -> None:
    await set_profile()
    await add_job("Backend Engineer")
    await add_job("Platform Engineer", "Python. Join us in Austin, Texas.")
    llm = KeyedLLM({"Backend Engineer": BENGALURU, "Platform Engineer": AUSTIN})

    outcome = await analyse_jobs(SessionLocal, settings(), llm)

    assert outcome.counts["extracted"] == 2 and outcome.llm_unavailable is None
    rows = await analyses()
    good, bad = rows["Backend Engineer"], rows["Platform Engineer"]
    assert (good.verdict, bad.verdict) == ("eligible", "rejected")
    assert good.extractor_version == EXTRACTOR_VERSION and good.rules_version == RULES_VERSION
    assert good.extraction is not None and good.extraction["technologies"] == [
        "Python",
        "PostgreSQL",
    ]
    rejected = next(r for r in bad.reasons if r["outcome"] == "reject")
    assert rejected["rule"] == "location"
    assert rejected["evidence"] == "Join us in Austin, Texas."


async def test_unchanged_listings_are_never_sent_to_the_llm_again() -> None:
    await set_profile()
    await add_job("Backend Engineer")
    llm = KeyedLLM({"Backend Engineer": BENGALURU})
    await analyse_jobs(SessionLocal, settings(), llm)
    await analyse_jobs(SessionLocal, settings(), llm)
    assert llm.calls == ["Backend Engineer"]

    await add_job("Backend Engineer", "Edited: now Python and Go. Bengaluru. 2025 batch.")
    await analyse_jobs(SessionLocal, settings(), llm)
    assert llm.calls == ["Backend Engineer"] * 2  # an edit is a new version
    async with SessionLocal() as session:
        assert len((await session.scalars(select(JobAnalysis))).all()) == 2


async def test_profile_edits_rejudge_without_the_llm() -> None:
    await set_profile()
    await add_job("Backend Engineer")
    llm = KeyedLLM({"Backend Engineer": BENGALURU})
    await analyse_jobs(SessionLocal, settings(), llm)
    assert (await analyses())["Backend Engineer"].verdict == "eligible"

    await set_profile(allowed_locations=["United States"])
    outcome = await analyse_jobs(SessionLocal, settings(), None)  # no LLM at all
    assert outcome.counts["rejudged"] == 1
    assert (await analyses())["Backend Engineer"].verdict == "rejected"
    assert llm.calls == ["Backend Engineer"]

    outcome = await analyse_jobs(SessionLocal, settings(), None)
    assert outcome.counts["rejudged"] == 0  # nothing changed: no writes


async def test_no_profile_means_no_verdicts() -> None:
    await add_job("Backend Engineer")
    outcome = await analyse_jobs(
        SessionLocal, settings(), KeyedLLM({"Backend Engineer": BENGALURU})
    )
    assert outcome.counts["extracted"] == 1 and outcome.counts["not_judged_no_profile"] == 1
    assert (await analyses())["Backend Engineer"].verdict is None


async def test_bad_output_counts_attempts_then_gives_up() -> None:
    await set_profile()
    await add_job("Weird")
    llm = KeyedLLM({"Weird": "I cannot help with that."})
    for _ in range(4):
        await analyse_jobs(SessionLocal, settings(jobs_extract_max_attempts=3), llm)
    assert len(llm.calls) == 3  # the fourth run did not try again
    row = (await analyses())["Weird"]
    assert (row.verdict, row.extraction_attempts) == ("error", 3)
    assert row.extraction_error == "ExtractionError: no JSON object in the reply"


async def test_a_later_success_clears_the_error() -> None:
    await set_profile()
    await add_job("Backend Engineer")
    llm = KeyedLLM({"Backend Engineer": "{broken"})
    await analyse_jobs(SessionLocal, settings(), llm)
    llm.replies["Backend Engineer"] = BENGALURU
    await analyse_jobs(SessionLocal, settings(), llm)
    row = (await analyses())["Backend Engineer"]
    assert (row.verdict, row.extraction_error) == ("eligible", None)


async def test_a_timed_out_extraction_counts_as_an_attempt() -> None:
    await set_profile()
    await add_job("Slow")
    await add_job("Backend Engineer")
    llm = KeyedLLM({"Slow": "hang", "Backend Engineer": BENGALURU})
    outcome = await analyse_jobs(SessionLocal, settings(jobs_extract_timeout_s=0.2), llm)
    assert (outcome.counts["extracted"], outcome.counts["extraction_failed"]) == (1, 1)
    row = (await analyses())["Slow"]
    assert (row.extraction_error, row.extraction_attempts) == ("TimeoutError", 1)


async def test_an_llm_outage_defers_the_rest_without_counting_attempts() -> None:
    await set_profile()
    for n in range(5):
        await add_job(f"Job {n}")
    outage = LLMError("The model is temporarily unavailable", transient=True)
    llm = KeyedLLM({f"Job {n}": outage for n in range(5)})

    outcome = await analyse_jobs(SessionLocal, settings(jobs_extract_concurrency=1), llm)

    assert outcome.llm_unavailable is not None
    assert len(llm.calls) == 1  # stopped after the first transient failure
    assert outcome.counts["deferred"] == 5
    assert await analyses() == {}  # nothing recorded against the listings

    llm.replies = {f"Job {n}": BENGALURU for n in range(5)}
    outcome = await analyse_jobs(SessionLocal, settings(), llm)
    assert outcome.counts["extracted"] == 5


async def test_an_unexpected_error_fails_only_that_listing() -> None:
    await set_profile()
    await add_job("Odd")
    await add_job("Backend Engineer")
    llm = KeyedLLM({"Odd": RuntimeError(SENTINEL), "Backend Engineer": BENGALURU})
    outcome = await analyse_jobs(SessionLocal, settings(), llm)
    assert (outcome.counts["extracted"], outcome.counts["extraction_failed"]) == (1, 1)
    assert (await analyses())["Odd"].extraction_error == "RuntimeError"


async def test_a_bad_request_is_the_listings_attempt() -> None:
    await set_profile()
    await add_job("Huge")
    llm = KeyedLLM({"Huge": LLMError("context length exceeded", transient=False)})
    outcome = await analyse_jobs(SessionLocal, settings(), llm)
    assert outcome.llm_unavailable is None
    assert (await analyses())["Huge"].extraction_error == "LLMError"


async def test_per_run_cap_takes_the_newest_first() -> None:
    await set_profile()
    for n in range(4):
        await add_job(f"Job {n}")
    llm = KeyedLLM({f"Job {n}": BENGALURU for n in range(4)})
    await analyse_jobs(SessionLocal, settings(jobs_extract_max_per_run=2), llm)
    assert sorted(llm.calls) == ["Job 2", "Job 3"]


async def test_closed_jobs_are_not_extracted() -> None:
    await set_profile()
    job_id = await add_job("Gone")
    async with SessionLocal() as session:
        await session.execute(
            update(Job).where(Job.id == job_id).values(closed_at=Job.first_seen_at)
        )
        await session.commit()
    llm = KeyedLLM({})
    outcome = await analyse_jobs(SessionLocal, settings(), llm)
    assert llm.calls == [] and outcome.counts["to_extract"] == 0


async def test_the_run_records_analysis_counts_and_an_llm_outage() -> None:
    await set_profile()
    feed = StaticFeed(
        "greenhouse:acme",
        [
            RawListing(
                source="greenhouse",
                external_id="1",
                title="Backend Engineer",
                url="https://example.test/1",
                raw_text="Python role. Bengaluru. 2025 batch.",
            )
        ],
    )
    ok = await run_ingestion(
        SessionLocal, settings(), feeds=[feed], llm=KeyedLLM({"Backend Engineer": BENGALURU})
    )
    assert ok.status == "completed"
    assert ok.counts["analysis"]["extracted"] == 1 and ok.counts["analysis"]["eligible"] == 1

    await add_job("New one")
    down = KeyedLLM({"New one": LLMError("circuit open", transient=True)})
    partial = await run_ingestion(SessionLocal, settings(), feeds=[feed], llm=down)
    assert partial.status == "partial"
    assert partial.errors == [
        {
            "feed": "analysis",
            "source": "analysis",
            "error_type": "LLMUnavailable",
            "message": "the LLM is unavailable; extraction deferred",
        }
    ]


async def test_run_deadline_during_analysis_is_recorded() -> None:
    await set_profile()
    await add_job("Slow")
    summary = await run_ingestion(
        SessionLocal,
        settings(jobs_run_timeout_s=0.3, jobs_extract_timeout_s=60),
        feeds=[],
        llm=KeyedLLM({"Slow": "hang"}),
    )
    assert summary.status == "failed"
    assert summary.errors[0]["feed"] == "analysis"
    assert summary.errors[0]["error_type"] == "TimeoutError"


async def test_database_failure_during_analysis_does_not_crash_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def broken(*args: Any, **kwargs: Any) -> Any:
        raise ConnectionResetError("connection was closed in the middle of operation")

    monkeypatch.setattr("app.jobs.ingest.analyse_jobs", broken)
    summary = await run_ingestion(SessionLocal, settings(), feeds=[])
    assert summary.status == "failed"
    assert summary.errors[0]["error_type"] == "ConnectionResetError"
    assert summary.errors[0]["message"] == "unexpected error"


async def test_private_data_stays_out_of_logs_and_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    await set_profile(allowed_locations=[SENTINEL])
    await add_job("Backend Engineer", f"Python. {SENTINEL}-in-listing")
    await add_job("Broken")
    llm = KeyedLLM({"Backend Engineer": BENGALURU, "Broken": f"{SENTINEL} no json"})
    summary = await run_ingestion(SessionLocal, settings(), feeds=[], llm=llm)
    assert SENTINEL not in caplog.text
    assert SENTINEL not in json.dumps(summary.errors) + json.dumps(summary.counts)
    row = (await analyses())["Broken"]
    assert row.extraction_error is not None and SENTINEL not in row.extraction_error
