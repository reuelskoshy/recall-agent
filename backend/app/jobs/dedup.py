"""Store listings idempotently: one row per (source, external_id), edits detected by hash.

Deduplication is done by the database, not by "look it up, then insert": a
unique constraint plus `INSERT ... ON CONFLICT DO UPDATE` is atomic, so two
overlapping runs (or a retried one) can never create a duplicate row.
"""

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from sqlalchemy import case, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Job

UpsertAction = Literal["new", "updated", "unchanged"]


@dataclass(frozen=True, slots=True)
class RawListing:
    """A listing as a source connector returns it, before any analysis."""

    source: str
    external_id: str
    title: str
    url: str
    raw_text: str
    company_name: str = ""
    location_raw: str = ""
    posted_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class UpsertResult:
    """What storing one listing did."""

    job_id: int
    action: UpsertAction


def normalize(text: str) -> str:
    """Collapse whitespace and case, so re-flowed or re-cased text hashes the same."""
    return " ".join(text.split()).casefold()


def content_hash(listing: RawListing) -> str:
    """SHA-256 over the parts whose change means the job itself changed.

    The URL and posting date are left out: a board moving its URLs or bumping a
    date does not change the role, and should not trigger a new LLM analysis.
    """
    parts = (normalize(listing.title), normalize(listing.location_raw), normalize(listing.raw_text))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


async def upsert_listing(
    session: AsyncSession, listing: RawListing, company_id: uuid.UUID | None = None
) -> UpsertResult:
    """Insert a new listing or refresh a known one. Does not commit.

    `now()` is fixed for the whole transaction, so "this row's timestamp equals
    now()" tells, in the same statement, whether the row was just inserted
    (`first_seen_at`) or its content just changed (`content_changed_at`).
    Callers commit once per listing, so each listing gets its own `now()`.
    """
    digest = content_hash(listing)
    values = insert(Job).values(
        source=listing.source,
        external_id=listing.external_id,
        company_id=company_id,
        company_name=listing.company_name,
        title=listing.title,
        url=listing.url,
        location_raw=listing.location_raw,
        raw_text=listing.raw_text,
        content_hash=digest,
        posted_at=listing.posted_at,
    )
    new = values.excluded  # the row we tried to insert
    changed = Job.content_hash != new.content_hash
    stmt = values.on_conflict_do_update(
        constraint="uq_jobs_source_external_id",
        set_={
            "last_seen_at": func.now(),
            "closed_at": None,  # it is back in the feed
            "title": new.title,
            "url": new.url,
            "location_raw": new.location_raw,
            "raw_text": new.raw_text,
            "content_hash": new.content_hash,
            # Never forget what we knew because a source left a field out.
            "posted_at": func.coalesce(new.posted_at, Job.posted_at),
            "company_name": func.coalesce(func.nullif(new.company_name, ""), Job.company_name),
            "company_id": func.coalesce(new.company_id, Job.company_id),
            "content_changed_at": case((changed, func.now()), else_=Job.content_changed_at),
        },
    ).returning(
        Job.id,
        (Job.first_seen_at == func.now()).label("inserted"),
        (Job.content_changed_at == func.now()).label("changed"),
    )
    row = (await session.execute(stmt)).one()
    action: UpsertAction = "new" if row.inserted else "updated" if row.changed else "unchanged"
    return UpsertResult(job_id=row.id, action=action)
