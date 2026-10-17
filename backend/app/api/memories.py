"""Memory endpoints: inspect, search (with score breakdown), add, edit, and delete.

Users can see and correct everything the agent remembers about them, which is
the main safeguard against wrong or unwanted memories.
"""

import uuid
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.models import Memory
from app.db.session import get_session
from app.memory.long_term import (
    MemoryKind,
    MemoryRejectedError,
    RankWeights,
    edit_memory,
    recall_memories,
    save_memory,
)
from app.rag.embeddings import EmbeddingProvider, get_embedder

router = APIRouter(prefix="/memories", tags=["memory"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
EmbedderDep = Annotated[EmbeddingProvider, Depends(get_embedder)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


class MemoryOut(BaseModel):
    """A stored memory."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    content: str
    kind: str
    importance: int
    source: str
    source_conversation_id: uuid.UUID | None
    access_count: int
    created_at: datetime
    updated_at: datetime
    last_accessed_at: datetime


class ScoreBreakdown(BaseModel):
    """Each term of the ranking formula, all in 0-1."""

    score: float
    similarity: float
    relevance: float
    recency: float
    importance: float


class ScoredMemoryOut(MemoryOut):
    """A memory returned by search, with why it ranked where it did."""

    ranking: ScoreBreakdown


class MemoryCreate(BaseModel):
    """Body of POST /api/memories."""

    content: str = Field(min_length=3, max_length=500)
    kind: MemoryKind = "fact"
    importance: int = Field(default=3, ge=1, le=5)


class MemoryUpdate(BaseModel):
    """Body of PATCH /api/memories/{id}; omitted fields are unchanged."""

    content: str | None = Field(default=None, min_length=3, max_length=500)
    kind: MemoryKind | None = None
    importance: int | None = Field(default=None, ge=1, le=5)


class MemorySaved(BaseModel):
    """Result of POST: the memory, and whether it merged into a near-duplicate."""

    action: Literal["created", "updated"]
    previous_content: str | None
    memory: MemoryOut


async def _get_or_404(session: AsyncSession, memory_id: uuid.UUID) -> Memory:
    memory = await session.get(Memory, memory_id)
    if memory is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Memory not found")
    return memory


@router.get("", response_model=list[MemoryOut] | list[ScoredMemoryOut])
async def list_memories(
    session: SessionDep,
    embedder: EmbedderDep,
    settings: SettingsDep,
    q: Annotated[str | None, Query(max_length=300)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
) -> list[MemoryOut] | list[ScoredMemoryOut]:
    """All memories (most recently used first), or ranked search results when `q` is given.

    Searching here does not count as the agent "using" a memory, so it does not
    change recency.
    """
    if q and q.strip():
        found = await recall_memories(
            session,
            embedder,
            q,
            RankWeights.from_settings(settings),
            limit=limit,
            min_similarity=settings.memory_min_similarity,
            touch=False,
        )
        return [
            ScoredMemoryOut.model_validate(
                {
                    **MemoryOut.model_validate(s.memory).model_dump(),
                    "ranking": ScoreBreakdown(
                        score=s.score,
                        similarity=s.similarity,
                        relevance=s.relevance,
                        recency=s.recency,
                        importance=s.importance,
                    ),
                }
            )
            for s in found
        ]
    rows = await session.scalars(
        select(Memory).order_by(Memory.last_accessed_at.desc()).limit(limit)
    )
    return [MemoryOut.model_validate(m) for m in rows.all()]


@router.post("", response_model=MemorySaved)
async def create_memory(
    body: MemoryCreate,
    response: Response,
    session: SessionDep,
    embedder: EmbedderDep,
    settings: SettingsDep,
) -> MemorySaved:
    """Add a memory by hand. 201 if new; 200 if it updated a near-duplicate."""
    try:
        outcome = await save_memory(
            session,
            embedder,
            body.content,
            body.kind,
            body.importance,
            source="manual",
            dedup_similarity=settings.memory_dedup_similarity,
        )
    except MemoryRejectedError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    response.status_code = (
        status.HTTP_201_CREATED if outcome.action == "created" else status.HTTP_200_OK
    )
    return MemorySaved(
        action=outcome.action,
        previous_content=outcome.previous_content,
        memory=MemoryOut.model_validate(outcome.memory),
    )


@router.patch("/{memory_id}", response_model=MemoryOut)
async def update_memory(
    memory_id: uuid.UUID, body: MemoryUpdate, session: SessionDep, embedder: EmbedderDep
) -> Memory:
    """Correct a memory's text, kind, or importance (re-embeds when the text changes)."""
    memory = await _get_or_404(session, memory_id)
    try:
        return await edit_memory(
            session, embedder, memory, body.content, body.kind, body.importance
        )
    except MemoryRejectedError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(memory_id: uuid.UUID, session: SessionDep) -> None:
    """Forget one memory."""
    await session.delete(await _get_or_404(session, memory_id))
    await session.commit()
