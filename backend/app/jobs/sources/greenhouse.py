"""Greenhouse Job Board API (https://docs.greenhouse.io/job-board.html, read 2026-11-12).

`GET https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs?content=true`
needs no auth ("authentication is not required for any GET endpoints"), has no
documented pagination or rate limit, and answers 404 for an unknown board.
`content` is HTML-entity-encoded HTML. External id: the job *post* `id`.
Pay ranges are only on the single-job endpoint; Phase 3 takes salary from text.
"""

import httpx

from app.jobs.dedup import RawListing
from app.jobs.sources.base import (
    BoardFeed,
    FeedResult,
    HttpPolicy,
    get_json,
    html_to_text,
    parse_time,
    require_list,
)

API = "https://boards-api.greenhouse.io/v1/boards"


class GreenhouseBoard(BoardFeed):
    source = "greenhouse"

    async def fetch(self, client: httpx.AsyncClient, policy: HttpPolicy) -> FeedResult:
        data = await get_json(
            client,
            f"{API}/{self.board_token}/jobs",
            policy,
            params={"content": "true"},
            label=self.key,
        )
        listings: list[RawListing] = []
        skipped = 0
        for job in require_list(data, "jobs", self.key):
            if not isinstance(job, dict):
                skipped += 1
                continue
            text = html_to_text(str(job.get("content") or ""))
            if not (job.get("id") and job.get("title") and job.get("absolute_url") and text):
                skipped += 1
                continue
            location = job.get("location")
            listings.append(
                RawListing(
                    source=self.source,
                    external_id=str(job["id"]),
                    title=str(job["title"])[:500],
                    url=str(job["absolute_url"])[:2000],
                    raw_text=text,
                    company_name=self.company_name or str(job.get("company_name") or ""),
                    location_raw=(
                        str(location.get("name") or "")[:500] if isinstance(location, dict) else ""
                    ),
                    posted_at=parse_time(job.get("first_published") or job.get("updated_at")),
                )
            )
        return FeedResult(listings=listings, skipped=skipped, complete=True)
