"""Ingestion runs: per-feed isolation, idempotent re-runs, deadlines, one run at a time,
no private data in errors or logs, and the reaper closing orphaned runs."""

import asyncio
import logging
import uuid
from datetime import timedelta
from typing import Any

import httpx
import pytest
from sqlalchemy import func, select, update

from app.core.config import Settings
from app.core.reaper import reap_orphans
from app.db.models import Company, IngestionRun, Job
from app.db.session import SessionLocal
from app.jobs.dedup import RawListing
from app.jobs.ingest import RunAlreadyRunningError, build_feeds, run_ingestion
from app.jobs.robots import RobotsChecker
from app.jobs.sources.base import Feed, FeedResult, HttpPolicy, SourceError
from app.jobs.sources.greenhouse import GreenhouseBoard
from app.jobs.sources.hn import HnWhoIsHiring
from app.jobs.sources.web_search import WebSearchFeed
from app.web.search import SearchHit, SearchProvider
from tests.test_jobs_sources import fixture

pytestmark = pytest.mark.usefixtures("db")

SENTINEL = "SENTINEL-private-query-7f3a"


def settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "jobs_enabled": True,
        "jobs_hn_enabled": False,
        "jobs_feed_timeout_s": 5,
        "jobs_run_timeout_s": 10,
    }
    return Settings(**{**base, **overrides})


def raw(external_id: str, text: str = "Build APIs in Python.") -> RawListing:
    return RawListing(
        source="greenhouse",
        external_id=external_id,
        title=f"Engineer {external_id}",
        url=f"https://example.test/jobs/{external_id}",
        raw_text=text,
    )


class StaticFeed(Feed):
    """Returns fixed listings, or raises, or hangs."""

    source = "greenhouse"

    def __init__(
        self,
        key: str,
        listings: list[RawListing] | None = None,
        *,
        error: Exception | None = None,
        hang: bool = False,
        complete: bool = False,
        company_id: uuid.UUID | None = None,
    ) -> None:
        super().__init__(key=key, company_id=company_id)
        self.listings = listings or []
        self.error = error
        self.hang = hang
        self.complete = complete

    async def fetch(self, client: httpx.AsyncClient, policy: HttpPolicy) -> FeedResult:
        if self.hang:
            await asyncio.sleep(3600)
        if self.error is not None:
            raise self.error
        return FeedResult(listings=self.listings, complete=self.complete)


async def run(feeds: list[Feed], **overrides: Any) -> Any:
    return await run_ingestion(SessionLocal, settings(**overrides), feeds=feeds)


async def stored_run(run_id: uuid.UUID) -> IngestionRun:
    async with SessionLocal() as session:
        row = await session.get(IngestionRun, run_id)
        assert row is not None
        return row


async def job_count() -> int:
    async with SessionLocal() as session:
        return int(await session.scalar(select(func.count()).select_from(Job)) or 0)


async def make_company(name: str, ats: str = "greenhouse", token: str | None = "acme") -> uuid.UUID:
    async with SessionLocal() as session:
        company = Company(name=name, ats=ats, board_token=token, watch=True)
        session.add(company)
        await session.commit()
        return company.id


# ---- Isolation and terminal states ----------------------------------------------------


async def test_one_failing_feed_does_not_fail_the_run() -> None:
    summary = await run(
        [
            StaticFeed("greenhouse:good", [raw("1"), raw("2")]),
            StaticFeed("greenhouse:bad", error=SourceError("greenhouse:bad: HTTP 404")),
            StaticFeed("greenhouse:slow", hang=True),
        ],
        jobs_feed_timeout_s=0.2,
    )
    assert summary.status == "partial"
    assert await job_count() == 2
    row = await stored_run(summary.run_id)
    assert row.status == "partial" and row.finished_at is not None
    assert row.counts["greenhouse"] == {"fetched": 2, "skipped": 0, "new": 2, "errors": 2}
    by_feed = {e["feed"]: e for e in row.errors}
    assert by_feed["greenhouse:bad"]["message"] == "greenhouse:bad: HTTP 404"
    assert by_feed["greenhouse:slow"]["error_type"] == "TimeoutError"


async def test_every_feed_failing_is_a_failed_run() -> None:
    summary = await run([StaticFeed("greenhouse:x", error=SourceError("greenhouse:x: HTTP 500"))])
    assert summary.status == "failed"
    assert (await stored_run(summary.run_id)).status == "failed"


async def test_a_clean_run_is_completed_and_a_run_with_no_feeds_too() -> None:
    assert (await run([StaticFeed("greenhouse:a", [raw("1")])])).status == "completed"
    assert (await run([])).status == "completed"


async def test_unexpected_feed_errors_are_recorded_by_type_only() -> None:
    summary = await run([StaticFeed("greenhouse:bug", error=KeyError(SENTINEL))])
    row = await stored_run(summary.run_id)
    assert row.errors == [
        {
            "feed": "greenhouse:bug",
            "source": "greenhouse",
            "error_type": "KeyError",
            "message": "unexpected error",
        }
    ]


async def test_run_deadline_still_records_a_terminal_state() -> None:
    summary = await run(
        [StaticFeed("greenhouse:fast", [raw("1")]), StaticFeed("greenhouse:hang", hang=True)],
        jobs_run_timeout_s=0.3,
        jobs_feed_timeout_s=60,
    )
    row = await stored_run(summary.run_id)
    assert row.status == "partial" and row.finished_at is not None
    assert any(e["feed"] == "run" and "1 feed(s) unfinished" in e["message"] for e in row.errors)
    assert await job_count() == 1  # what finished before the deadline is kept


async def test_database_failure_mid_feed_is_isolated_and_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.jobs.dedup import upsert_listing as real

    async def flaky(session: Any, listing: RawListing, company_id: Any = None) -> Any:
        if listing.external_id == "boom":
            raise ConnectionResetError("connection was closed in the middle of operation")
        return await real(session, listing, company_id)

    monkeypatch.setattr("app.jobs.ingest.upsert_listing", flaky)
    summary = await run(
        [
            StaticFeed("greenhouse:a", [raw("1"), raw("boom"), raw("3")]),
            StaticFeed("greenhouse:b", [raw("4")]),
        ]
    )
    assert summary.status == "partial"
    assert await job_count() == 2  # "1" committed before the drop, "4" in its own feed
    assert {e["feed"]: e["error_type"] for e in summary.errors} == {
        "greenhouse:a": "ConnectionResetError"
    }


# ---- Idempotency, edits and take-downs -----------------------------------------------


async def test_rerunning_creates_no_duplicates_and_detects_edits() -> None:
    first = await run([StaticFeed("greenhouse:a", [raw("1"), raw("2")])])
    second = await run([StaticFeed("greenhouse:a", [raw("1"), raw("2", "Now in Go.")])])
    assert first.counts["greenhouse"]["new"] == 2
    assert second.counts["greenhouse"] == {"fetched": 2, "skipped": 0, "unchanged": 1, "updated": 1}
    assert await job_count() == 2


async def test_complete_boards_close_jobs_that_disappeared_and_reopen_returns() -> None:
    company = await make_company("Acme")
    board = [raw("1"), raw("2")]
    await run([StaticFeed("greenhouse:acme", board, complete=True, company_id=company)])
    summary = await run(
        [StaticFeed("greenhouse:acme", [raw("1")], complete=True, company_id=company)]
    )
    assert summary.counts["greenhouse"]["closed"] == 1
    async with SessionLocal() as session:
        closed = {j.external_id: j.closed_at for j in await session.scalars(select(Job))}
    assert closed["1"] is None and closed["2"] is not None

    await run([StaticFeed("greenhouse:acme", board, complete=True, company_id=company)])
    async with SessionLocal() as session:
        assert all(j.closed_at is None for j in await session.scalars(select(Job)))


async def test_incomplete_feeds_and_failed_boards_never_close_jobs() -> None:
    company = await make_company("Acme")
    await run([StaticFeed("greenhouse:acme", [raw("1")], complete=True, company_id=company)])
    await run([StaticFeed("greenhouse:acme", [], complete=False, company_id=company)])
    await run([StaticFeed("greenhouse:acme", error=SourceError("x"), company_id=company)])
    async with SessionLocal() as session:
        job = await session.scalar(select(Job))
    assert job is not None and job.closed_at is None


# ---- One run at a time ----------------------------------------------------------------


async def test_a_second_run_while_one_is_running_is_refused() -> None:
    started = asyncio.Event()

    class Gate(StaticFeed):
        async def fetch(self, client: httpx.AsyncClient, policy: HttpPolicy) -> FeedResult:
            started.set()
            await asyncio.sleep(0.3)
            return FeedResult(listings=[])

    first = asyncio.create_task(run([Gate("greenhouse:gate")]))
    await started.wait()
    with pytest.raises(RunAlreadyRunningError):
        await run([])
    assert (await first).status == "completed"
    assert (await run([])).status == "completed"  # free again afterwards


async def test_a_cancelled_run_is_recorded_as_interrupted() -> None:
    started = asyncio.Event()

    class Gate(StaticFeed):
        async def fetch(self, client: httpx.AsyncClient, policy: HttpPolicy) -> FeedResult:
            started.set()
            await asyncio.sleep(3600)
            raise AssertionError("unreachable")

    task = asyncio.create_task(run([Gate("greenhouse:gate")]))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with SessionLocal() as session:
        row = await session.scalar(select(IngestionRun))
    assert row is not None and row.status == "interrupted" and row.finished_at is not None


async def test_reaper_interrupts_only_runs_older_than_the_run_timeout() -> None:
    async with SessionLocal() as session:
        old = IngestionRun(trigger="cron", status="running")
        session.add(old)
        await session.commit()
        await session.execute(
            update(IngestionRun)
            .where(IngestionRun.id == old.id)
            .values(started_at=func.now() - timedelta(hours=2))
        )
        await session.commit()
    result = await reap_orphans(SessionLocal, settings(jobs_run_timeout_s=900))
    assert result.job_runs == 1
    row = await stored_run(old.id)
    assert row.status == "interrupted" and row.errors[-1]["error_type"] == "Interrupted"

    fresh = await run([])  # the slot is free, and a fresh run is not reaped
    assert (await reap_orphans(SessionLocal, settings())).job_runs == 0
    assert (await stored_run(fresh.run_id)).status == "completed"


# ---- Real connectors end to end, through a mock transport -----------------------------


class CannedSearch(SearchProvider):
    name = "canned"

    def __init__(self, hits: list[SearchHit], fail: bool = False) -> None:
        self.hits = hits
        self.fail = fail

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        if self.fail:
            from app.web.search import SearchError

            raise SearchError("Tavily timed out after 10s")
        return self.hits


def hit(url: str) -> SearchHit:
    return SearchHit(title="AI engineer", url=url, snippet="0-2 years, Python.")


async def test_linkedin_hits_never_become_job_rows() -> None:
    search = CannedSearch(
        [
            hit("https://www.linkedin.com/jobs/view/4000"),
            hit("https://in.linkedin.com/jobs/view/4001"),
            hit("https://lnkd.in/abcd"),
            hit("https://careers.example.test/jobs/1"),
        ]
    )
    robots_hosts: list[str] = []

    def robots(request: httpx.Request) -> httpx.Response:
        robots_hosts.append(request.url.host)
        return httpx.Response(404)

    summary = await run_ingestion(
        SessionLocal,
        settings(jobs_search_queries=["ai engineer"], web_search_provider="fake"),
        search=search,
        robots_transport=httpx.MockTransport(robots),
    )
    async with SessionLocal() as session:
        urls = [j.url for j in await session.scalars(select(Job))]
    assert urls == ["https://careers.example.test/jobs/1"]
    assert not any("linkedin" in u or "lnkd.in" in u for u in urls)
    assert robots_hosts == ["careers.example.test"]  # LinkedIn was never contacted
    assert summary.counts["web_search"]["skipped"] == 3


async def test_watchlist_boards_and_hn_run_end_to_end() -> None:
    await make_company("Airbnb", "greenhouse", "airbnb")
    async with SessionLocal() as session:  # not watched: not fetched
        session.add(Company(name="Ignored", ats="lever", board_token="nope", watch=False))
        await session.commit()

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == "boards-api.greenhouse.io":
            return httpx.Response(200, json=fixture("greenhouse_airbnb.json"))
        if request.url.path.endswith("/search_by_date"):
            return httpx.Response(200, json=fixture("hn_threads.json"))
        if request.url.host == "hn.algolia.com":
            return httpx.Response(200, json=fixture("hn_comments.json"))
        return httpx.Response(404)

    summary = await run_ingestion(
        SessionLocal,
        settings(jobs_hn_enabled=True),
        transport=httpx.MockTransport(handle),
    )
    assert summary.status == "completed", summary.errors
    assert summary.counts["greenhouse"]["new"] == 3
    assert summary.counts["hn"]["new"] == 3
    again = await run_ingestion(
        SessionLocal, settings(jobs_hn_enabled=True), transport=httpx.MockTransport(handle)
    )
    assert again.counts["greenhouse"]["unchanged"] == 3 and await job_count() == 6
    async with SessionLocal() as session:
        airbnb = await session.scalars(select(Job).where(Job.source == "greenhouse"))
        assert all(j.company_id is not None for j in airbnb)


async def test_build_feeds_from_watchlist_and_settings() -> None:
    await make_company("Acme", "greenhouse", "acme")
    await make_company("Beta", "ashby", "beta")
    async with httpx.AsyncClient() as client:
        feeds = await build_feeds(
            SessionLocal,
            settings(jobs_hn_enabled=True, jobs_search_queries=["q1", "q2"]),
            RobotsChecker(client),
            search=CannedSearch([]),
        )
    assert [f.key for f in feeds] == [
        "greenhouse:acme",
        "ashby:beta",
        "hn:who-is-hiring",
        "web_search:#1",
        "web_search:#2",
    ]
    assert isinstance(feeds[0], GreenhouseBoard) and isinstance(feeds[2], HnWhoIsHiring)
    assert isinstance(feeds[3], WebSearchFeed)


async def test_search_queries_are_skipped_without_a_search_provider() -> None:
    async with httpx.AsyncClient() as client:
        feeds = await build_feeds(
            SessionLocal,
            settings(
                jobs_search_queries=["q"], web_search_provider="tavily", web_search_api_key=None
            ),
            RobotsChecker(client),
        )
    assert feeds == []


async def test_private_query_text_never_reaches_errors_or_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    summary = await run_ingestion(
        SessionLocal,
        settings(jobs_search_queries=[SENTINEL], web_search_provider="fake"),
        search=CannedSearch([], fail=True),
        robots_transport=httpx.MockTransport(lambda r: httpx.Response(404)),
    )
    assert summary.status == "failed"
    row = await stored_run(summary.run_id)
    assert SENTINEL not in str(row.errors) and SENTINEL not in str(row.counts)
    assert row.errors[0]["feed"] == "web_search:#1"
    assert SENTINEL not in caplog.text
