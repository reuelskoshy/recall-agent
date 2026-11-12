"""Job source connectors against recorded fixtures: parsing, retries, the denylist, robots.

No network: `httpx.MockTransport` answers with responses recorded from the live
public APIs on 2026-11-12 (tests/fixtures/jobs/, trimmed to a few items).
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.jobs.robots import RobotsChecker, host_of, is_denied_domain
from app.jobs.sources.ashby import AshbyBoard
from app.jobs.sources.base import (
    FeedResult,
    HttpPolicy,
    SourceError,
    get_json,
    html_to_text,
    parse_time,
)
from app.jobs.sources.greenhouse import GreenhouseBoard
from app.jobs.sources.hn import HnWhoIsHiring
from app.jobs.sources.lever import LeverBoard
from app.jobs.sources.web_search import WebSearchFeed
from app.web.search import FakeSearch, SearchError, SearchHit, SearchProvider

FIXTURES = Path(__file__).parent / "fixtures" / "jobs"
Handler = Callable[[httpx.Request], httpx.Response]


def fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class Recorder:
    """A mock transport that records every request it answers."""

    def __init__(self, handler: Handler) -> None:
        self.requests: list[httpx.Request] = []
        self._handler = handler

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))


def no_sleep_policy(attempts: int = 3) -> tuple[HttpPolicy, list[float]]:
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    return HttpPolicy(max_attempts=attempts, backoff_s=1.0, sleep=sleep), slept


async def fetch(feed: Any, handler: Handler) -> tuple[FeedResult, Recorder]:
    recorder = Recorder(handler)
    policy, _ = no_sleep_policy()
    async with recorder.client() as client:
        return await feed.fetch(client, policy), recorder


def serve(name: str) -> Handler:
    return lambda request: httpx.Response(200, json=fixture(name))


def assert_clean(result: FeedResult) -> None:
    """Every listing is usable: ids, titles, URLs, readable text, aware timestamps."""
    assert result.listings
    for listing in result.listings:
        assert listing.external_id and listing.title and listing.url.startswith("https://")
        assert listing.raw_text.strip()
        assert "<p>" not in listing.raw_text and "&lt;" not in listing.raw_text
        assert listing.posted_at is None or listing.posted_at.tzinfo is not None


# ---- Board connectors on recorded data ------------------------------------------------


async def test_greenhouse_parses_recorded_board() -> None:
    result, recorder = await fetch(
        GreenhouseBoard("airbnb", company_name="Airbnb"), serve("greenhouse_airbnb.json")
    )
    assert_clean(result)
    raw = fixture("greenhouse_airbnb.json")["jobs"]
    assert [x.external_id for x in result.listings] == [str(j["id"]) for j in raw]
    assert result.complete and result.skipped == 0
    assert result.listings[0].company_name == "Airbnb"
    assert result.listings[0].location_raw == raw[0]["location"]["name"]
    request = recorder.requests[0]
    assert request.url.path == "/v1/boards/airbnb/jobs"
    assert request.url.params["content"] == "true"


async def test_lever_parses_recorded_board_and_combines_sections() -> None:
    result, recorder = await fetch(LeverBoard("leverdemo"), serve("lever_leverdemo.json"))
    assert_clean(result)
    raw = fixture("lever_leverdemo.json")
    # The recorded third posting has a title and no text in any field: skipped.
    assert [x.external_id for x in result.listings] == [p["id"] for p in raw[:2]]
    assert result.skipped == 1
    assert result.listings[0].posted_at is not None  # createdAt, epoch milliseconds
    assert result.listings[0].posted_at.year >= 2020
    section_titles = [s["text"] for s in raw[0].get("lists", []) if s.get("text")]
    for title in section_titles:
        assert title in result.listings[0].raw_text
    assert recorder.requests[0].url.host == "api.lever.co"


async def test_lever_eu_token_uses_the_eu_host() -> None:
    _, recorder = await fetch(LeverBoard("eu:leverdemo"), serve("lever_leverdemo.json"))
    assert recorder.requests[0].url.host == "api.eu.lever.co"
    assert recorder.requests[0].url.path == "/v0/postings/leverdemo"


async def test_ashby_parses_recorded_board() -> None:
    result, _ = await fetch(AshbyBoard("ashby"), serve("ashby_ashby.json"))
    assert_clean(result)
    raw = fixture("ashby_ashby.json")["jobs"]
    assert [x.external_id for x in result.listings] == [j["id"] for j in raw]
    assert result.listings[0].location_raw.startswith(raw[0]["location"])


async def test_ashby_skips_unlisted_and_falls_back_to_the_url_id() -> None:
    data = fixture("ashby_ashby.json")
    first, second = data["jobs"][0], data["jobs"][1]
    first.pop("id")
    second["isListed"] = False
    result, _ = await fetch(AshbyBoard("ashby"), lambda r: httpx.Response(200, json=data))
    assert result.listings[0].external_id == first["jobUrl"].rstrip("/").rsplit("/", 1)[1]
    assert result.skipped == 1
    assert len(result.listings) == len(data["jobs"]) - 1
    assert second["id"] not in [x.external_id for x in result.listings]


async def test_items_missing_required_fields_are_skipped_not_fatal() -> None:
    data = fixture("greenhouse_airbnb.json")
    data["jobs"][0]["title"] = ""
    data["jobs"][1]["content"] = ""
    data["jobs"].append("not an object")
    result, _ = await fetch(GreenhouseBoard("airbnb"), lambda r: httpx.Response(200, json=data))
    assert (len(result.listings), result.skipped) == (1, 3)


# ---- HN Who is hiring ----------------------------------------------------------------


def hn_handler(threads: Any = None, comments: Any = None) -> Handler:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/search_by_date"):
            return httpx.Response(200, json=threads or fixture("hn_threads.json"))
        return httpx.Response(200, json=comments or fixture("hn_comments.json"))

    return handle


async def test_hn_reads_top_level_posts_of_the_latest_hiring_thread() -> None:
    result, recorder = await fetch(HnWhoIsHiring(), hn_handler())
    threads = fixture("hn_threads.json")["hits"]
    story = next(h for h in threads if h["title"].startswith("Ask HN: Who is hiring?"))
    comments = fixture("hn_comments.json")["hits"]
    top_level = [c for c in comments if str(c["parent_id"]) == story["objectID"]]
    assert len(top_level) < len(comments)  # the fixture includes a reply
    assert_clean(result)
    assert [x.external_id for x in result.listings] == [c["objectID"] for c in top_level]
    assert not result.complete  # a thread is not a full board: never close from it
    assert recorder.requests[1].url.params["tags"] == f"comment,story_{story['objectID']}"
    assert result.listings[0].url == (
        f"https://news.ycombinator.com/item?id={top_level[0]['objectID']}"
    )


async def test_hn_ignores_who_wants_to_be_hired_threads() -> None:
    threads = {
        "hits": [
            {"objectID": "2", "title": "Ask HN: Who wants to be hired? (December 2026)"},
            {"objectID": "1", "title": "Ask HN: Who is hiring? (November 2026)"},
        ]
    }
    _, recorder = await fetch(HnWhoIsHiring(), hn_handler(threads=threads))
    assert recorder.requests[1].url.params["tags"] == "comment,story_1"


async def test_hn_without_a_hiring_thread_is_a_source_error() -> None:
    threads = {"hits": [{"objectID": "2", "title": "Ask HN: Freelancer? Seeking freelancer?"}]}
    with pytest.raises(SourceError, match="no recent"):
        await fetch(HnWhoIsHiring(), hn_handler(threads=threads))


async def test_hn_keyword_filter_and_company_from_the_first_line() -> None:
    comments = {
        "nbPages": 1,
        "hits": [
            {"objectID": "11", "parent_id": 1, "comment_text": "Acme | ML Engineer | Bengaluru"},
            {"objectID": "12", "parent_id": 1, "comment_text": "Other | SRE | Berlin<p>Onsite"},
            {"objectID": "13", "parent_id": 1, "comment_text": ""},  # deleted
        ],
    }
    threads = {"hits": [{"objectID": "1", "title": "Ask HN: Who is hiring? (Nov)"}]}
    result, _ = await fetch(
        HnWhoIsHiring(keywords=["BENGALURU"]), hn_handler(threads=threads, comments=comments)
    )
    assert [(x.external_id, x.company_name) for x in result.listings] == [("11", "Acme")]
    # HN's bare <p> separators start new lines, so the title is only the first line.
    lines = html_to_text("Other | SRE | Berlin<p>Onsite<p>Python").splitlines()
    assert lines == ["Other | SRE | Berlin", "Onsite", "Python"]
    assert result.skipped == 1


# ---- Retries: transient only, bounded -------------------------------------------------


def sequence(*responses: httpx.Response | Exception) -> Handler:
    queue = list(responses)

    def handle(request: httpx.Request) -> httpx.Response:
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return handle


async def get(handler: Handler, attempts: int = 3) -> tuple[Any, Recorder, list[float]]:
    recorder = Recorder(handler)
    policy, slept = no_sleep_policy(attempts)
    async with recorder.client() as client:
        try:
            data: Any = await get_json(client, "https://feed.test/x", policy, label="t:feed")
        except SourceError as exc:
            data = exc
    return data, recorder, slept


async def test_transient_failures_are_retried_with_backoff() -> None:
    data, recorder, slept = await get(
        sequence(
            httpx.ConnectError("refused"),
            httpx.Response(503),
            httpx.Response(200, json={"ok": True}),
        )
    )
    assert data == {"ok": True}
    assert len(recorder.requests) == 3
    assert slept == [1.0, 2.0]


async def test_timeouts_are_retried_then_reported() -> None:
    error, recorder, _ = await get(sequence(*[httpx.ReadTimeout("slow")] * 3))
    assert isinstance(error, SourceError)
    assert str(error) == "t:feed: timed out (ReadTimeout) after 3 attempts"
    assert len(recorder.requests) == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410])
async def test_permanent_http_errors_are_not_retried(status: int) -> None:
    error, recorder, slept = await get(sequence(httpx.Response(status)))
    assert isinstance(error, SourceError) and str(error) == f"t:feed: HTTP {status}"
    assert len(recorder.requests) == 1 and slept == []


async def test_malformed_json_fails_at_once() -> None:
    error, recorder, _ = await get(sequence(httpx.Response(200, text="<html>oops</html>")))
    assert isinstance(error, SourceError) and "not valid JSON" in str(error)
    assert len(recorder.requests) == 1


async def test_retry_after_is_honoured_but_capped() -> None:
    _, _, slept = await get(
        sequence(
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(429, headers={"Retry-After": "3600"}),
            httpx.Response(200, json=[]),
        )
    )
    assert slept == [7.0, 30.0]


@pytest.mark.parametrize(
    "body",
    [{"jobs": "nope"}, {"unexpected": []}, ["a", "list"], None, "text"],
)
async def test_unexpected_shapes_are_source_errors(body: Any) -> None:
    content = json.dumps(body).encode()
    with pytest.raises(SourceError, match="unexpected response shape"):
        await fetch(GreenhouseBoard("acme"), lambda r: httpx.Response(200, content=content))


# ---- Small helpers --------------------------------------------------------------------


def test_html_to_text_decodes_entities_and_keeps_paragraphs() -> None:
    encoded = (
        "&lt;p&gt;Build &amp;amp; ship&lt;/p&gt;&lt;ul&gt;&lt;li&gt;Python&lt;/li&gt;&lt;/ul&gt;"
    )
    assert html_to_text(encoded) == "Build & ship\nPython"
    assert html_to_text("<p>A</p><p>B <b>bold</b></p>") == "A\nB bold"


@pytest.mark.parametrize(
    ("value", "year"),
    [
        ("2026-11-02T10:00:00-04:00", 2026),
        ("2024-03-04T14:29:08.532+00:00", 2024),
        ("2025-01-01T00:00:00Z", 2025),
        (1788199939703, 2026),  # epoch ms (Lever)
        (1762000000, 2025),  # epoch s (HN)
    ],
)
def test_parse_time_accepts_each_source_format(value: Any, year: int) -> None:
    parsed = parse_time(value)
    assert parsed is not None and parsed.year == year and parsed.tzinfo is not None


@pytest.mark.parametrize("value", [None, "", "yesterday", True, 10**30, {}])
def test_parse_time_rejects_junk(value: Any) -> None:
    assert parse_time(value) is None


# ---- Denylist and robots.txt ----------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "denied"),
    [
        ("https://www.linkedin.com/jobs/view/123", True),
        ("https://in.linkedin.com/jobs/view/123", True),
        ("https://LINKEDIN.COM./jobs", True),
        ("https://lnkd.in/abc", True),
        ("https://notlinkedin.com/jobs", False),
        ("https://linkedin.com.evil.test/jobs", False),
        ("https://jobs.example.com/1", False),
        ("ftp://example.com/x", True),  # not http(s): fail closed
        ("not a url", True),
    ],
)
def test_denylist(url: str, denied: bool) -> None:
    assert is_denied_domain(url) is denied


def test_extra_denied_domains_add_to_the_list() -> None:
    assert is_denied_domain("https://www.glassdoor.com/job/1", frozenset({"glassdoor.com"}))
    assert not is_denied_domain("https://www.glassdoor.com/job/1")


def test_host_of() -> None:
    assert host_of("https://Jobs.Example.com:8443/x") == "jobs.example.com"
    assert host_of("mailto:a@b.c") is None


ROBOTS = "User-agent: *\nDisallow: /private/\n"
# RFC 9309: a crawler obeys only the most specific group naming it, so once a
# RecallAgent group exists, the * group no longer applies to us.
ROBOTS_NAMED = ROBOTS + "\nUser-agent: RecallAgent\nDisallow: /no-bots/\n"


def robots_handler(table: dict[str, httpx.Response | Exception]) -> Handler:
    def handle(request: httpx.Request) -> httpx.Response:
        item = table.get(request.url.host, httpx.Response(404))
        if isinstance(item, Exception):
            raise item
        return item

    return handle


async def check(urls: list[str], table: dict[str, httpx.Response | Exception]) -> list[bool]:
    recorder = Recorder(robots_handler(table))
    async with recorder.client() as client:
        robots = RobotsChecker(client)
        return [await robots.allowed(u) for u in urls]


async def test_robots_rules_are_applied_to_our_agent() -> None:
    urls = ["https://a.test/jobs/1", "https://a.test/private/x", "https://a.test/no-bots/y"]
    generic: dict[str, httpx.Response | Exception] = {"a.test": httpx.Response(200, text=ROBOTS)}
    assert await check(urls, generic) == [True, False, True]
    named: dict[str, httpx.Response | Exception] = {
        "a.test": httpx.Response(200, text=ROBOTS_NAMED)
    }
    assert await check(urls, named) == [True, True, False]


@pytest.mark.parametrize(
    ("answer", "allowed"),
    [
        (httpx.Response(404), True),  # no robots.txt: no restrictions
        (httpx.Response(410), True),
        (httpx.Response(401), False),  # fail closed
        (httpx.Response(403), False),
        (httpx.Response(503), False),
        (httpx.ConnectTimeout("slow"), False),
    ],
)
async def test_robots_failures_fail_closed(
    answer: httpx.Response | Exception, allowed: bool
) -> None:
    assert await check(["https://b.test/job"], {"b.test": answer}) == [allowed]


async def test_robots_is_fetched_once_per_host() -> None:
    recorder = Recorder(robots_handler({"c.test": httpx.Response(200, text=ROBOTS)}))
    async with recorder.client() as client:
        robots = RobotsChecker(client)
        for n in range(3):
            await robots.allowed(f"https://c.test/jobs/{n}")
    assert [r.url.path for r in recorder.requests] == ["/robots.txt"]


# ---- Generic search connector ---------------------------------------------------------


class CannedSearch(SearchProvider):
    name = "canned"

    def __init__(self, hits: list[SearchHit] | Exception) -> None:
        self._hits = hits
        self.queries: list[str] = []

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        self.queries.append(query)
        if isinstance(self._hits, Exception):
            raise self._hits
        return self._hits[:max_results]


def hit(url: str) -> SearchHit:
    return SearchHit(title=f"Job at {url}", url=url, snippet="Python engineer, 0-2 years.")


async def search_feed(
    provider: SearchProvider, robots_table: dict[str, httpx.Response | Exception]
) -> tuple[FeedResult, Recorder]:
    robots_recorder = Recorder(robots_handler(robots_table))
    async with robots_recorder.client() as robots_client:
        feed = WebSearchFeed(
            "private query text",
            number=2,
            provider=provider,
            robots=RobotsChecker(robots_client),
            max_results=10,
        )
        policy, _ = no_sleep_policy()
        async with httpx.AsyncClient() as client:
            return await feed.fetch(client, policy), robots_recorder


async def test_search_drops_linkedin_before_any_request() -> None:
    provider = CannedSearch(
        [
            hit("https://www.linkedin.com/jobs/view/1"),
            hit("https://lnkd.in/xyz"),
            hit("https://careers.example.com/jobs/7"),
        ]
    )
    result, robots_recorder = await search_feed(provider, {})
    assert [x.url for x in result.listings] == ["https://careers.example.com/jobs/7"]
    assert result.skipped == 2
    # LinkedIn was never contacted, not even for robots.txt.
    assert {r.url.host for r in robots_recorder.requests} == {"careers.example.com"}


async def test_search_drops_hits_robots_disallows_or_cannot_confirm() -> None:
    provider = CannedSearch(
        [
            hit("https://a.test/jobs/1"),
            hit("https://a.test/private/2"),
            hit("https://down.test/jobs/3"),
        ]
    )
    table: dict[str, httpx.Response | Exception] = {
        "a.test": httpx.Response(200, text=ROBOTS),
        "down.test": httpx.ConnectError("refused"),
    }
    result, _ = await search_feed(provider, table)
    assert [x.url for x in result.listings] == ["https://a.test/jobs/1"]
    assert (result.skipped, result.complete) == (2, False)
    listing = result.listings[0]
    assert (listing.source, len(listing.external_id)) == ("web_search", 64)


async def test_search_failure_hides_the_query() -> None:
    provider = CannedSearch(SearchError("Tavily timed out after 10s"))
    with pytest.raises(SourceError) as info:
        await search_feed(provider, {})
    assert "web_search:#2" in str(info.value)
    assert "private query text" not in str(info.value)


async def test_fake_search_provider_works_as_a_feed() -> None:
    result, _ = await search_feed(FakeSearch(), {})
    assert len(result.listings) == 2
