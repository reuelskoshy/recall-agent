"""Recall Jobs schema: dedup, the contact-policy block, one running run, the status machine."""

import uuid

import pytest
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from app.db.models import (
    ApplicationEvent,
    CandidateProfile,
    Company,
    Draft,
    IngestionRun,
    Job,
)
from app.db.session import SessionLocal
from app.jobs.applications import (
    TRANSITIONS,
    InvalidTransitionError,
    check_transition,
    get_or_create_application,
    transition,
)
from app.jobs.dedup import RawListing, UpsertResult, content_hash, upsert_listing


def listing(**overrides: str) -> RawListing:
    fields = {
        "source": "greenhouse",
        "external_id": "4001",
        "title": "Software Engineer",
        "url": "https://boards.example.com/acme/jobs/4001",
        "raw_text": "Build APIs in Python.",
        "company_name": "Acme",
        "location_raw": "Bengaluru",
    }
    return RawListing(**{**fields, **overrides})  # type: ignore[arg-type]


async def store(item: RawListing, company_id: uuid.UUID | None = None) -> UpsertResult:
    """Upsert in its own transaction, as the pipeline does (one commit per listing)."""
    async with SessionLocal() as session:
        result = await upsert_listing(session, item, company_id)
        await session.commit()
        return result


async def make_company(policy: str, name: str | None = None) -> uuid.UUID:
    async with SessionLocal() as session:
        company = Company(name=name or f"Co {uuid.uuid4().hex[:8]}", contact_policy=policy)
        session.add(company)
        await session.commit()
        return company.id


async def make_job(company_id: uuid.UUID | None) -> int:
    return (await store(listing(external_id=uuid.uuid4().hex), company_id)).job_id


async def add_draft(job_id: int, kind: str = "outreach_email") -> uuid.UUID:
    async with SessionLocal() as session:
        draft = Draft(job_id=job_id, kind=kind, body="Hello")
        session.add(draft)
        await session.commit()
        return draft.id


# ---- Content hash (pure) ----


def test_hash_ignores_whitespace_and_case() -> None:
    a = listing(raw_text="Build APIs\n\n in   Python.")
    b = listing(raw_text="build apis in python.")
    assert content_hash(a) == content_hash(b)


def test_hash_changes_with_title_location_or_text() -> None:
    base = content_hash(listing())
    assert content_hash(listing(title="Senior Software Engineer")) != base
    assert content_hash(listing(location_raw="Remote")) != base
    assert content_hash(listing(raw_text="Build APIs in Go.")) != base


def test_hash_ignores_url() -> None:
    assert content_hash(listing(url="https://elsewhere.example/4001")) == content_hash(listing())


# ---- Dedup (database) ----


async def test_same_listing_twice_is_one_row(db: None) -> None:
    first = await store(listing())
    second = await store(listing())
    assert first.action == "new"
    assert second == UpsertResult(job_id=first.job_id, action="unchanged")
    async with SessionLocal() as session:
        jobs = (await session.scalars(select(Job))).all()
    assert len(jobs) == 1
    assert jobs[0].last_seen_at > jobs[0].first_seen_at
    assert jobs[0].content_changed_at == jobs[0].first_seen_at


async def test_edited_listing_is_updated_in_place(db: None) -> None:
    first = await store(listing())
    edited = await store(listing(raw_text="Build APIs in Python and Go."))
    assert edited == UpsertResult(job_id=first.job_id, action="updated")
    async with SessionLocal() as session:
        job = await session.get(Job, first.job_id)
    assert job is not None
    assert job.raw_text == "Build APIs in Python and Go."
    assert job.content_hash == content_hash(listing(raw_text="Build APIs in Python and Go."))
    assert job.content_changed_at > job.first_seen_at


async def test_same_external_id_on_another_source_is_a_different_job(db: None) -> None:
    a = await store(listing(source="greenhouse"))
    b = await store(listing(source="lever"))
    assert a.action == b.action == "new"
    assert a.job_id != b.job_id


async def test_reappearing_listing_is_reopened(db: None) -> None:
    first = await store(listing())
    async with SessionLocal() as session:
        await session.execute(
            update(Job).where(Job.id == first.job_id).values(closed_at=Job.last_seen_at)
        )
        await session.commit()
    await store(listing())
    async with SessionLocal() as session:
        job = await session.get(Job, first.job_id)
    assert job is not None and job.closed_at is None


async def test_upsert_keeps_a_known_company_link(db: None) -> None:
    company_id = await make_company("allowed")
    first = await store(listing(), company_id)
    await store(listing(company_name=""), None)  # a source that cannot name the company
    async with SessionLocal() as session:
        job = await session.get(Job, first.job_id)
    assert job is not None
    assert job.company_id == company_id
    assert job.company_name == "Acme"


# ---- Contact policy: enforced by the database ----


async def test_new_company_defaults_to_application_form_only(db: None) -> None:
    async with SessionLocal() as session:
        company = Company(name="Fresh Co")
        session.add(company)
        await session.commit()
        await session.refresh(company)
    assert company.contact_policy == "application_form_only"


async def test_outreach_allowed_only_for_allowed_companies(db: None) -> None:
    job = await make_job(await make_company("allowed"))
    await add_draft(job)  # does not raise


@pytest.mark.parametrize("policy", ["do_not_contact", "application_form_only"])
async def test_outreach_draft_blocked_by_policy(db: None, policy: str) -> None:
    job = await make_job(await make_company(policy))
    with pytest.raises(IntegrityError, match="outreach drafts are not allowed"):
        await add_draft(job)


async def test_outreach_draft_blocked_without_a_company(db: None) -> None:
    job = await make_job(None)
    with pytest.raises(IntegrityError, match="no company"):
        await add_draft(job)


async def test_application_answers_allowed_for_do_not_contact(db: None) -> None:
    job = await make_job(await make_company("do_not_contact"))
    await add_draft(job, kind="application_answers")  # does not raise


async def test_cannot_turn_a_draft_into_outreach(db: None) -> None:
    job = await make_job(await make_company("do_not_contact"))
    draft_id = await add_draft(job, kind="application_answers")
    async with SessionLocal() as session:
        with pytest.raises(IntegrityError, match="outreach drafts are not allowed"):
            await session.execute(
                update(Draft).where(Draft.id == draft_id).values(kind="outreach_email")
            )


async def test_tightening_policy_discards_live_outreach_drafts(db: None) -> None:
    company_id = await make_company("allowed")
    job = await make_job(company_id)
    outreach = await add_draft(job)
    answers = await add_draft(job, kind="application_answers")
    async with SessionLocal() as session:
        await session.execute(
            update(Company).where(Company.id == company_id).values(contact_policy="do_not_contact")
        )
        await session.commit()
        statuses = {
            row.id: row.status for row in await session.execute(select(Draft.id, Draft.status))
        }
        assert statuses == {outreach: "discarded", answers: "needs_review"}
        # And the discarded outreach draft cannot be revived.
        with pytest.raises(IntegrityError, match="outreach drafts are not allowed"):
            await session.execute(
                update(Draft).where(Draft.id == outreach).values(status="approved")
            )


async def test_unknown_contact_policy_rejected(db: None) -> None:
    with pytest.raises(IntegrityError, match="ck_companies_contact_policy"):
        await make_company("maybe")


# ---- Ingestion runs ----


async def test_only_one_run_can_be_running(db: None) -> None:
    async with SessionLocal() as session:
        session.add(IngestionRun(trigger="cron"))
        await session.commit()
    async with SessionLocal() as session:
        session.add(IngestionRun(trigger="manual"))
        with pytest.raises(IntegrityError, match="uq_ingestion_runs_one_running"):
            await session.commit()


async def test_finished_run_allows_the_next(db: None) -> None:
    async with SessionLocal() as session:
        session.add(IngestionRun(trigger="cron"))
        await session.commit()
        await session.execute(update(IngestionRun).values(status="completed"))
        session.add(IngestionRun(trigger="cron"))
        await session.commit()  # does not raise


# ---- Profile ----


async def test_profile_is_a_single_row(db: None) -> None:
    async with SessionLocal() as session:
        session.add(CandidateProfile(id=1, graduation_year=2025))
        await session.commit()
    async with SessionLocal() as session:
        session.add(CandidateProfile(id=2))
        with pytest.raises(IntegrityError, match="ck_candidate_profile_single_row"):
            await session.commit()


# ---- Application state machine ----


def test_every_status_has_transitions() -> None:
    reachable = {target for targets in TRANSITIONS.values() for target in targets}
    assert reachable | {"new"} == set(TRANSITIONS)


@pytest.mark.parametrize("final", ["rejected", "withdrawn"])
def test_final_statuses_cannot_move(final: str) -> None:
    for target in TRANSITIONS:
        with pytest.raises(InvalidTransitionError):
            check_transition(final, target)


@pytest.mark.parametrize(
    ("current", "target"),
    [("new", "applied"), ("new", "offer"), ("applied", "drafted"), ("skipped", "applied")],
)
def test_illegal_moves_rejected(current: str, target: str) -> None:
    with pytest.raises(InvalidTransitionError, match=f"from '{current}' to '{target}'"):
        check_transition(current, target)


def test_unknown_status_rejected() -> None:
    with pytest.raises(InvalidTransitionError, match="Unknown status"):
        check_transition("new", "ghosted")


async def test_application_path_is_logged(db: None) -> None:
    job = await make_job(None)
    path = ["shortlisted", "applied", "interview", "offer"]
    async with SessionLocal() as session:
        application = await get_or_create_application(session, job)
        for status in path:
            await transition(session, application, status)
        await session.commit()
        events = (
            await session.scalars(
                select(ApplicationEvent)
                .where(ApplicationEvent.application_id == application.id)
                .order_by(ApplicationEvent.id)
            )
        ).all()
    assert [(e.from_status, e.to_status) for e in events] == [
        (None, "new"),
        ("new", "shortlisted"),
        ("shortlisted", "applied"),
        ("applied", "interview"),
        ("interview", "offer"),
    ]


async def test_get_or_create_is_idempotent(db: None) -> None:
    job = await make_job(None)
    async with SessionLocal() as session:
        first = await get_or_create_application(session, job)
        second = await get_or_create_application(session, job)
        await session.commit()
        events = (await session.scalars(select(ApplicationEvent))).all()
    assert first.id == second.id
    assert len(events) == 1


async def test_illegal_transition_leaves_status_unchanged(db: None) -> None:
    job = await make_job(None)
    async with SessionLocal() as session:
        application = await get_or_create_application(session, job)
        await session.commit()
        with pytest.raises(InvalidTransitionError):
            await transition(session, application, "offer")
        await session.rollback()
        await session.refresh(application)
    assert application.status == "new"
