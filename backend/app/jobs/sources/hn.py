"""Hacker News "Ask HN: Who is hiring?" via the Algolia HN Search API
(https://hn.algolia.com/api, read 2026-11-12; limit: 10,000 requests/hour per IP).

1. `search_by_date?tags=story,author_whoishiring` finds the account's recent
   threads; the newest titled "Ask HN: Who is hiring?" wins (the same account
   posts "Who wants to be hired?" and "Freelancer?" threads).
2. `search?tags=comment,story_{id}&hitsPerPage=1000` pages through its
   comments. Only top-level ones (`parent_id == story id`) are job posts.

Two or three requests a day. A thread is never a complete board (posts stay
up after the job is filled), so nothing is closed from this feed.
"""

from typing import Any

import httpx

from app.jobs.dedup import RawListing
from app.jobs.sources.base import (
    Feed,
    FeedResult,
    HttpPolicy,
    SourceError,
    get_json,
    html_to_text,
    parse_time,
    require_list,
)

API = "https://hn.algolia.com/api/v1"
THREAD_TITLE = "ask hn: who is hiring?"
PAGE_SIZE = 1000
MAX_PAGES = 5
TITLE_CHARS = 300


class HnWhoIsHiring(Feed):
    source = "hn"

    def __init__(self, keywords: list[str] | None = None) -> None:
        super().__init__(key="hn:who-is-hiring")
        self._keywords = [k.casefold() for k in keywords or [] if k.strip()]

    async def fetch(self, client: httpx.AsyncClient, policy: HttpPolicy) -> FeedResult:
        story_id = await self._latest_thread(client, policy)
        listings: list[RawListing] = []
        skipped = 0
        for page in range(MAX_PAGES):
            data = await get_json(
                client,
                f"{API}/search",
                policy,
                params={
                    "tags": f"comment,story_{story_id}",
                    "hitsPerPage": str(PAGE_SIZE),
                    "page": str(page),
                },
                label=self.key,
            )
            for hit in require_list(data, "hits", self.key):
                if not isinstance(hit, dict) or str(hit.get("parent_id")) != story_id:
                    continue  # a reply, not a job post
                listing = self._to_listing(hit)
                if listing is None:
                    skipped += 1
                elif self._wanted(listing):
                    listings.append(listing)
            pages = data.get("nbPages") if isinstance(data, dict) else None
            if not isinstance(pages, int) or page + 1 >= pages:
                break
        return FeedResult(listings=listings, skipped=skipped)

    async def _latest_thread(self, client: httpx.AsyncClient, policy: HttpPolicy) -> str:
        data = await get_json(
            client,
            f"{API}/search_by_date",
            policy,
            params={"tags": "story,author_whoishiring", "hitsPerPage": "10"},
            label=self.key,
        )
        for hit in require_list(data, "hits", self.key):  # newest first
            title = str(hit.get("title", "")) if isinstance(hit, dict) else ""
            if title.casefold().startswith(THREAD_TITLE) and hit.get("objectID"):
                return str(hit["objectID"])
        raise SourceError(f"{self.key}: no recent 'Who is hiring?' thread found")

    def _to_listing(self, hit: dict[str, Any]) -> RawListing | None:
        text = html_to_text(str(hit.get("comment_text") or ""))
        comment_id = hit.get("objectID")
        if not text or not comment_id:
            return None  # deleted or dead comment
        first_line = text.splitlines()[0][:TITLE_CHARS]
        # Convention: "Company | Role | Location | ...". Not enforced, so best effort.
        company = first_line.split("|")[0].strip() if "|" in first_line else ""
        return RawListing(
            source=self.source,
            external_id=str(comment_id),
            title=first_line,
            url=f"https://news.ycombinator.com/item?id={comment_id}",
            raw_text=text,
            company_name=company[:200],
            posted_at=parse_time(hit.get("created_at_i") or hit.get("created_at")),
        )

    def _wanted(self, listing: RawListing) -> bool:
        if not self._keywords:
            return True
        haystack = listing.raw_text.casefold()
        return any(k in haystack for k in self._keywords)
