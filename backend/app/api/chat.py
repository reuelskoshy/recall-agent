"""Chat endpoints: stream a turn over SSE, and browse/delete conversations."""

import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.agent.events import AgentEvent
from app.agent.runner import run_turn
from app.core.config import Settings, get_settings
from app.db.models import Conversation
from app.db.session import get_session
from app.llm.base import LLMProvider
from app.llm.factory import get_llm_provider

router = APIRouter(tags=["chat"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]


# ---- Schemas -------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Body of POST /api/chat. Omit `conversation_id` to start a new conversation."""

    message: str = Field(min_length=1, max_length=8000)
    conversation_id: uuid.UUID | None = None


class ConversationSummary(BaseModel):
    """A conversation in the sidebar list."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    updated_at: datetime


class MessageOut(BaseModel):
    """One stored message."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    tool_calls: list[dict[str, Any]] | None
    tool_call_id: str | None
    created_at: datetime


class ConversationDetail(ConversationSummary):
    """A conversation with its full message history."""

    messages: list[MessageOut]


# ---- Dependencies --------------------------------------------------------------


async def get_conversation_for_chat(body: ChatRequest, session: SessionDep) -> Conversation | None:
    """Load the conversation named in the request, or 404.

    This runs as a dependency (before streaming starts) because once an SSE
    response has begun, an HTTP error status can no longer be sent.
    """
    if body.conversation_id is None:
        return None
    conversation = await session.get(Conversation, body.conversation_id)
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
    return conversation


# ---- Routes --------------------------------------------------------------------


@router.post(
    "/chat",
    response_class=EventSourceResponse,
    responses={404: {"description": "Conversation not found"}},
)
async def chat(
    body: ChatRequest,
    session: SessionDep,
    conversation: Annotated[Conversation | None, Depends(get_conversation_for_chat)],
    provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncIterator[ServerSentEvent]:
    """Run one chat turn, streaming events as SSE.

    Event names: `conversation`, `token`, `done`, `error`. Each event's data is
    the JSON of the matching model in `app.agent.events`.
    """
    events: AsyncIterator[AgentEvent] = run_turn(
        session, provider, conversation, body.message, settings.chat_history_limit
    )
    async for event in events:
        yield ServerSentEvent(event=event.type, data=event)


@router.get("/conversations", response_model=list[ConversationSummary])
async def list_conversations(session: SessionDep, limit: int = 50) -> list[Conversation]:
    """List conversations, most recently active first."""
    rows = await session.scalars(
        select(Conversation).order_by(Conversation.updated_at.desc()).limit(min(limit, 200))
    )
    return list(rows.all())


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: uuid.UUID, session: SessionDep) -> Conversation:
    """Return one conversation with all its messages."""
    conversation = await session.scalar(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .options(selectinload(Conversation.messages))
    )
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
    return conversation


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: uuid.UUID, session: SessionDep) -> None:
    """Delete a conversation and (via ON DELETE CASCADE) its messages."""
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
    await session.delete(conversation)
    await session.commit()
