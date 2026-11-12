"""Lever Postings API (https://github.com/lever/postings-api, read 2026-11-12).

`GET https://api.lever.co/v0/postings/{site}?mode=json` lists published
postings without auth ("publicly viewable"). Accounts hosted in the EU use
`api.eu.lever.co`; write the board token as `eu:{site}` for those. An unknown
site answers 404. Lever's robots.txt asks for `Crawl-delay: 1`; we make one
request per board per day. `createdAt` (epoch ms) is not in the README but is
in live responses, so it is used when present and ignored otherwise.
"""

from typing import Any

import httpx

from app.jobs.dedup import RawListing
from app.jobs.sources.base import (
    BoardFeed,
    FeedResult,
    HttpPolicy,
    SourceError,
    get_json,
    html_to_text,
    parse_time,
    require_list,
)

HOSTS = {"": "https://api.lever.co", "eu": "https://api.eu.lever.co"}


def posting_text(posting: dict[str, Any]) -> str:
    """Description, each titled list (responsibilities, requirements...), then the rest."""
    parts = [str(posting.get("descriptionPlain") or "")]
    for section in posting.get("lists") or []:
        if isinstance(section, dict):
            parts.append(str(section.get("text") or ""))
            parts.append(html_to_text(str(section.get("content") or "")))
    parts.append(str(posting.get("additionalPlain") or ""))
    return "\n".join(p.strip() for p in parts if p and p.strip())


class LeverBoard(BoardFeed):
    source = "lever"

    async def fetch(self, client: httpx.AsyncClient, policy: HttpPolicy) -> FeedResult:
        region, _, site = self.board_token.rpartition(":")
        if region not in HOSTS:
            raise SourceError(f"{self.key}: unknown Lever region (use 'eu:<site>' or '<site>')")
        data = await get_json(
            client,
            f"{HOSTS[region]}/v0/postings/{site}",
            policy,
            params={"mode": "json"},
            label=self.key,
        )
        listings: list[RawListing] = []
        skipped = 0
        for posting in require_list(data, None, self.key):
            if not isinstance(posting, dict):
                skipped += 1
                continue
            text = posting_text(posting)
            if not (
                posting.get("id") and posting.get("text") and posting.get("hostedUrl") and text
            ):
                skipped += 1
                continue
            categories = posting.get("categories")
            location = categories.get("location") if isinstance(categories, dict) else None
            listings.append(
                RawListing(
                    source=self.source,
                    external_id=str(posting["id"]),
                    title=str(posting["text"])[:500],
                    url=str(posting["hostedUrl"])[:2000],
                    raw_text=text,
                    company_name=self.company_name,
                    location_raw=str(location or "")[:500],
                    posted_at=parse_time(posting.get("createdAt")),
                )
            )
        return FeedResult(listings=listings, skipped=skipped, complete=True)
