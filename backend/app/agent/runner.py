"""Handles one user turn: persist, call the model, stream events, persist the reply.

Phase 1 makes a single model call. Phase 2 turns this into the tool-calling loop.
"""

import logging
from collections.abc import AsyncIterator

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.events import AgentEvent, ConversationStarted, Done, Error, Token, UsageOut
from app.agent.prompts import SYSTEM_PROMPT
from app.db.models import Conversation, Message
from app.llm.base import (
    ChatMessage,
    FinishReason,
    LLMError,
    LLMProvider,
    StreamEnd,
    TextDelta,
    UsageEvent,
)
from app.memory.short_term import load_history

logger = logging.getLogger(__name__)

TITLE_MAX_CHARS = 60


def make_title(first_message: str) -> str:
    """Derive a conversation title from its first user message."""
    text = " ".join(first_message.split())
    return text if len(text) <= TITLE_MAX_CHARS else text[: TITLE_MAX_CHARS - 1].rstrip() + "…"


async def run_turn(
    session: AsyncSession,
    provider: LLMProvider,
    conversation: Conversation | None,
    user_text: str,
    history_limit: int,
) -> AsyncIterator[AgentEvent]:
    """Run one turn and yield events. Creates the conversation when `conversation` is None.

    The user message is committed before the model is called, so it survives an
    LLM failure. The assistant message is only saved if the stream completes.
    """
    if conversation is None:
        conversation = Conversation(title=make_title(user_text))
        session.add(conversation)
    conversation.updated_at = func.now()
    session.add(Message(conversation=conversation, role="user", content=user_text))
    await session.commit()
    yield ConversationStarted(conversation_id=conversation.id, title=conversation.title)

    history = await load_history(session, conversation.id, history_limit)
    messages = [ChatMessage(role="system", content=SYSTEM_PROMPT), *history]

    parts: list[str] = []
    usage: UsageOut | None = None
    finish_reason: FinishReason = "stop"
    try:
        async for event in provider.stream_chat(messages):
            if isinstance(event, TextDelta):
                parts.append(event.text)
                yield Token(text=event.text)
            elif isinstance(event, UsageEvent):
                usage = UsageOut(
                    prompt_tokens=event.usage.prompt_tokens,
                    completion_tokens=event.usage.completion_tokens,
                )
            elif isinstance(event, StreamEnd):
                finish_reason = event.finish_reason
    except LLMError as exc:
        logger.warning("LLM call failed for conversation %s: %s", conversation.id, exc)
        yield Error(message=str(exc))
        return

    reply = Message(conversation=conversation, role="assistant", content="".join(parts))
    session.add(reply)
    await session.commit()
    yield Done(message_id=reply.id, finish_reason=finish_reason, usage=usage)
