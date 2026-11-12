"""Ashby public job posting API (https://developers.ashbyhq.com/docs/public-job-posting-api,
read 2026-11-12).

`GET https://api.ashbyhq.com/posting-api/job-board/{name}` needs no auth. An
unknown board answers 404 with a plain-text body. The docs omit `jobs[].id`
but live responses include it (a UUID, also the last part of `jobUrl`); the
id falls back to that path segment. Unlisted postings (`isListed: false`) are
skipped: they are reachable only by direct link, so the company is not
advertising them.
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

API = "https://api.ashbyhq.com/posting-api/job-board"


class AshbyBoard(BoardFeed):
    source = "ashby"

    async def fetch(self, client: httpx.AsyncClient, policy: HttpPolicy) -> FeedResult:
        data = await get_json(client, f"{API}/{self.board_token}", policy, label=self.key)
        listings: list[RawListing] = []
        skipped = 0
        for job in require_list(data, "jobs", self.key):
            if not isinstance(job, dict) or job.get("isListed") is False:
                skipped += 1
                continue
            url = str(job.get("jobUrl") or "")
            external_id = str(job.get("id") or url.rstrip("/").rpartition("/")[2])
            text = str(job.get("descriptionPlain") or "").strip() or html_to_text(
                str(job.get("descriptionHtml") or "")
            )
            if not (external_id and job.get("title") and url and text):
                skipped += 1
                continue
            locations = [str(job.get("location") or "")] + [
                str(s.get("location") or "")
                for s in job.get("secondaryLocations") or []
                if isinstance(s, dict)
            ]
            listings.append(
                RawListing(
                    source=self.source,
                    external_id=external_id,
                    title=str(job["title"])[:500],
                    url=url[:2000],
                    raw_text=text,
                    company_name=self.company_name,
                    location_raw="; ".join(loc for loc in locations if loc)[:500],
                    posted_at=parse_time(job.get("publishedAt")),
                )
            )
        return FeedResult(listings=listings, skipped=skipped, complete=True)
