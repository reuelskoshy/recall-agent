"""Observability endpoints: inspect agent runs and their steps."""

import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import AgentRun
from app.db.session import get_session

router = APIRouter(tags=["observability"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]


class RunSummary(BaseModel):
    """One agent turn: outcome, cost, and latency."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    conversation_id: uuid.UUID
    user_input: str
    status: str
    provider: str
    model: str
    iterations: int
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int | None
    error: str | None
    created_at: datetime
    finished_at: datetime | None


class StepOut(BaseModel):
    """One LLM call or tool call inside a run."""

    model_config = ConfigDict(from_attributes=True)

    index: int
    iteration: int
    kind: str
    name: str
    arguments: dict[str, Any] | None
    result_summary: str
    is_error: bool
    latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None


class RunDetail(RunSummary):
    """A run with its ordered steps."""

    steps: list[StepOut]


@router.get("/runs", response_model=list[RunSummary])
async def list_runs(
    session: SessionDep,
    conversation_id: uuid.UUID | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[AgentRun]:
    """Most recent runs first, optionally for one conversation."""
    query = select(AgentRun).order_by(AgentRun.created_at.desc()).limit(limit)
    if conversation_id is not None:
        query = query.where(AgentRun.conversation_id == conversation_id)
    return list((await session.scalars(query)).all())


@router.get("/runs/{run_id}", response_model=RunDetail)
async def get_run(run_id: uuid.UUID, session: SessionDep) -> AgentRun:
    """One run with every step (tool arguments, result summaries, latency, tokens)."""
    run = await session.scalar(
        select(AgentRun).where(AgentRun.id == run_id).options(selectinload(AgentRun.steps))
    )
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
    return run
