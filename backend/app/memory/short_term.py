"""Short-term memory: the recent message history of one conversation.

Phase 1 uses a simple "last N messages" window. Phase 5 replaces it with a
token-aware window plus a running summary of older turns.
"""

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Message
from app.llm.base import ChatMessage, Role, ToolCall


def to_chat_message(row: Message) -> ChatMessage:
    """Convert a stored `Message` row into a provider-neutral `ChatMessage`."""
    calls = tuple(
        ToolCall(id=c["id"], name=c["name"], arguments=c["arguments"]) for c in row.tool_calls or []
    )
    role: Role = row.role  # type: ignore[assignment]  # constrained on write
    return ChatMessage(
        role=role, content=row.content, tool_calls=calls, tool_call_id=row.tool_call_id
    )


def trim_orphan_tool_results(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    """Drop leading `tool` messages whose assistant tool call was cut off by the window.

    Providers reject a tool result that does not follow the tool call it answers.
    """
    start = 0
    while start < len(messages) and messages[start].role == "tool":
        start += 1
    return list(messages[start:])


async def load_history(
    session: AsyncSession, conversation_id: uuid.UUID, limit: int
) -> list[ChatMessage]:
    """Return the last `limit` messages of a conversation, oldest first."""
    rows = (
        await session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
    ).all()
    return trim_orphan_tool_results([to_chat_message(r) for r in reversed(rows)])
