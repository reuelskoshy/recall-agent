"""Generic search connector: the candidate's own queries through Recall's web search provider.

Each hit becomes a listing only if it passes, in order:
1. the domain denylist (LinkedIn always; see app/jobs/robots.py), checked
   before anything else, so a LinkedIn URL is never stored or even requested;
2. the site's robots.txt for our user agent (fail closed).

Search results are not a full board, so nothing is ever closed from this feed.
The hit's snippet is stored as the listing text; Phase 3 extraction decides
whether it describes a job at all.

The query text is private (it describes the candidate). It is never logged and
never used in the feed key or error messages: the feed is "web_search:#2".
"""

import hashlib

import httpx

from app.jobs.dedup import RawListing
from app.jobs.robots import RobotsChecker, is_denied_domain
from app.jobs.sources.base import Feed, FeedResult, HttpPolicy, SourceError
from app.web.search import SearchError, SearchProvider


def url_id(url: str) -> str:
    """Stable external id for a URL (URLs can exceed the column; ids must not)."""
    return hashlib.sha256(url.strip().encode()).hexdigest()


class WebSearchFeed(Feed):
    """One configured search query."""

    source = "web_search"

    def __init__(
        self,
        query: str,
        *,
        number: int,
        provider: SearchProvider,
        robots: RobotsChecker,
        max_results: int,
        denied_domains: frozenset[str] = frozenset(),
    ) -> None:
        super().__init__(key=f"web_search:#{number}")
        self._query = query
        self._provider = provider
        self._robots = robots
        self._max_results = max_results
        self._denied = denied_domains

    async def fetch(self, client: httpx.AsyncClient, policy: HttpPolicy) -> FeedResult:
        # The provider has its own timeout and single retry (app/web/search.py).
        try:
            hits = await self._provider.search(self._query, self._max_results)
        except SearchError as exc:
            raise SourceError(f"{self.key}: search failed ({exc})") from exc
        listings: list[RawListing] = []
        skipped = 0
        for hit in hits:
            if is_denied_domain(hit.url, self._denied) or not await self._robots.allowed(hit.url):
                skipped += 1
                continue
            listings.append(
                RawListing(
                    source=self.source,
                    external_id=url_id(hit.url),
                    title=hit.title[:500],
                    url=hit.url[:2000],
                    raw_text=hit.snippet,
                )
            )
        return FeedResult(listings=listings, skipped=skipped)
