"""Reaper: fail work orphaned by a crash, restart, or scale-to-zero.

A turn or an ingestion that is cut off by the process dying leaves its row in
`running` / `processing` forever, because the code that would have finished it
no longer exists. Live work always ends within its own timeout, so anything
still unfinished well past that timeout is dead and safe to fail. Age-based
(rather than "everything at startup") also keeps it correct with several
instances sharing one database.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import func, literal, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.models import AgentRun, Document, IngestionRun

logger = logging.getLogger(__name__)

GRACE_S = 60.0
"""Slack on top of each timeout, so live work is never reaped."""

RUN_INTERRUPTED = "The server stopped before this run finished."
INGEST_INTERRUPTED = "Ingestion was interrupted (the server restarted). Retry it."
JOB_RUN_INTERRUPTED = {
    "feed": "run",
    "source": "run",
    "error_type": "Interrupted",
    "message": "the server stopped before this run finished",
}


@dataclass(frozen=True, slots=True)
class ReapResult:
    """How many rows one sweep failed."""

    runs: int
    documents: int
    job_runs: int = 0


async def reap_orphans(
    session_factory: async_sessionmaker[AsyncSession], settings: Settings
) -> ReapResult:
    """Fail runs and ingestions that have outlived their timeout."""
    run_cutoff = func.now() - timedelta(seconds=settings.agent_turn_timeout_s + GRACE_S)
    doc_cutoff = func.now() - timedelta(seconds=settings.ingest_timeout_s + GRACE_S)
    async with session_factory() as session:
        runs = await session.scalars(
            update(AgentRun)
            .where(AgentRun.status == "running", AgentRun.created_at < run_cutoff)
            .values(status="interrupted", error=RUN_INTERRUPTED, finished_at=func.now())
            .returning(AgentRun.id)
        )
        run_ids = runs.all()
        documents = await session.scalars(
            update(Document)
            .where(Document.status == "processing", Document.updated_at < doc_cutoff)
            .values(status="failed", error=INGEST_INTERRUPTED)
            .returning(Document.id)
        )
        document_ids = documents.all()
        # Job-search ingestion runs: always reaped (cheap), whether or not jobs are enabled.
        job_cutoff = func.now() - timedelta(seconds=settings.jobs_run_timeout_s + GRACE_S)
        job_runs = await session.scalars(
            update(IngestionRun)
            .where(IngestionRun.status == "running", IngestionRun.started_at < job_cutoff)
            .values(
                status="interrupted",
                finished_at=func.now(),
                errors=IngestionRun.errors.op("||")(literal([JOB_RUN_INTERRUPTED], JSONB)),
            )
            .returning(IngestionRun.id)
        )
        job_run_ids = job_runs.all()
        await session.commit()
    result = ReapResult(runs=len(run_ids), documents=len(document_ids), job_runs=len(job_run_ids))
    if result.runs or result.documents or result.job_runs:
        logger.warning(
            "Reaper failed %d orphaned run(s), %d ingestion(s) and %d job run(s)",
            result.runs,
            result.documents,
            result.job_runs,
        )
    return result


async def run_reaper(session_factory: async_sessionmaker[AsyncSession], settings: Settings) -> None:
    """Sweep now, then every `reaper_interval_s`, until cancelled. Never raises otherwise.

    A failed sweep (e.g. the database is briefly down) is logged and retried on
    the next tick; it must not take the app down with it.
    """
    while True:
        try:
            await reap_orphans(session_factory, settings)
        except Exception:
            logger.exception("Reaper sweep failed; retrying in %gs", settings.reaper_interval_s)
        await asyncio.sleep(settings.reaper_interval_s)
