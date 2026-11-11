"""The application status state machine.

    new -> shortlisted -> drafted -> applied -> interview -> offer
     |         |  ^          |         |           |          |
     |         |  +----------+         |           |          |
     +---------+--> skipped (not interested; can be reconsidered -> shortlisted)
    rejected / withdrawn: reachable from any non-terminal state (withdrawn also
    from offer, i.e. declining it). rejected and withdrawn are final.

shortlisted -> applied skips drafting on purpose: a company with an
application-form-only policy is applied to without an outreach draft.
The database only checks that a status is a known value; which moves are
legal is decided here, in one table, and every move is logged as an event.
"""

import uuid
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import APPLICATION_STATUSES, Application, ApplicationEvent

_ANY_ACTIVE_EXIT = frozenset({"rejected", "withdrawn"})

TRANSITIONS: dict[str, frozenset[str]] = {
    "new": frozenset({"shortlisted", "skipped"}) | _ANY_ACTIVE_EXIT,
    "shortlisted": frozenset({"drafted", "applied", "skipped"}) | _ANY_ACTIVE_EXIT,
    "drafted": frozenset({"shortlisted", "applied"}) | _ANY_ACTIVE_EXIT,
    "applied": frozenset({"interview"}) | _ANY_ACTIVE_EXIT,
    "interview": frozenset({"offer"}) | _ANY_ACTIVE_EXIT,
    "offer": frozenset({"withdrawn"}),
    "skipped": frozenset({"shortlisted"}),
    "rejected": frozenset(),
    "withdrawn": frozenset(),
}
assert set(TRANSITIONS) == set(APPLICATION_STATUSES), "every status needs a transition entry"


class InvalidTransitionError(ValueError):
    """The requested status change is not allowed from the current status."""


def allowed_next(status: str) -> frozenset[str]:
    """Statuses reachable in one move from `status`."""
    return TRANSITIONS[status]


def check_transition(current: str, target: str) -> None:
    """Raise InvalidTransitionError unless `current -> target` is a legal move."""
    if target not in TRANSITIONS:
        raise InvalidTransitionError(f"Unknown status {target!r}.")
    if target not in TRANSITIONS[current]:
        options = ", ".join(sorted(TRANSITIONS[current])) or "none (final status)"
        raise InvalidTransitionError(
            f"Cannot move an application from {current!r} to {target!r}. Allowed: {options}."
        )


async def get_or_create_application(session: AsyncSession, job_id: int) -> Application:
    """The job's application, created as `new` (with its first event) if missing.

    Insert-or-nothing on the unique job_id, so concurrent callers share one row.
    """
    inserted = await session.scalar(
        insert(Application)
        .values(id=uuid.uuid4(), job_id=job_id, status="new")
        .on_conflict_do_nothing(index_elements=["job_id"])
        .returning(Application.id)
    )
    if inserted is not None:
        session.add(ApplicationEvent(application_id=inserted, from_status=None, to_status="new"))
        await session.flush()
    application = await session.scalar(select(Application).where(Application.job_id == job_id))
    assert application is not None
    return application


async def transition(
    session: AsyncSession,
    application: Application,
    target: str,
    follow_up_on: date | None = None,
) -> ApplicationEvent:
    """Move `application` to `target`, logging the change. Does not commit.

    The row is locked first (SELECT ... FOR UPDATE), so two concurrent moves
    from the same state cannot both pass the check.
    """
    current = await session.scalar(
        select(Application.status).where(Application.id == application.id).with_for_update()
    )
    assert current is not None
    check_transition(current, target)
    application.status = target
    application.status_changed_at = func.now()
    if follow_up_on is not None:
        application.follow_up_on = follow_up_on
    event = ApplicationEvent(application_id=application.id, from_status=current, to_status=target)
    session.add(event)
    await session.flush()
    return event
