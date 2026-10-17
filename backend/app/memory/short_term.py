"""Short-term memory: the recent message history of one conversation.

Phase 2 uses a simple "last N messages" window. Phase 5 replaces it with a
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


def sanitize_tool_pairs(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    """Keep only well-formed tool exchanges.

    Providers reject (HTTP 400) a tool result that does not directly follow its
    call, and an assistant tool call that is not followed by all its results.
    Both happen when the history window cuts a tool exchange in half, or when a
    turn was interrupted. Broken calls are dropped (keeping any assistant text),
    and orphan results are dropped.
    """
    out: list[ChatMessage] = []
    i = 0
    while i < len(messages):
        message = messages[i]
        if message.role == "tool":  # results are consumed together with their call below
            i += 1
            continue
        if message.role == "assistant" and message.tool_calls:
            j = i + 1
            results: list[ChatMessage] = []
            while j < len(messages) and messages[j].role == "tool":
                results.append(messages[j])
                j += 1
            call_ids = {call.id for call in message.tool_calls}
            if call_ids == {r.tool_call_id for r in results}:
                out.append(message)
                out.extend(results)
            elif message.content:
                out.append(ChatMessage(role="assistant", content=message.content))
            i = j
            continue
        out.append(message)
        i += 1
    return out


async def load_history(
    session: AsyncSession, conversation_id: uuid.UUID, limit: int
) -> list[ChatMessage]:
    """Return the last `limit` messages of a conversation, oldest first, made provider-safe."""
    rows = (
        await session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.seq.desc())
            .limit(limit)
        )
    ).all()
    return sanitize_tool_pairs([to_chat_message(r) for r in reversed(rows)])
