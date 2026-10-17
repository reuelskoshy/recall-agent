"""Events the agent emits while handling one user turn.

They are transport-agnostic: the API sends each one as a Server-Sent Event whose
`event:` name equals `type`, and the frontend mirrors them as a TypeScript union.
"""

import uuid
from typing import Literal

from pydantic import BaseModel

from app.llm.base import FinishReason


class UsageOut(BaseModel):
    """Token usage summed over every model call in the turn."""

    prompt_tokens: int
    completion_tokens: int


class ConversationStarted(BaseModel):
    """First event of every turn: which conversation it belongs to."""

    type: Literal["conversation"] = "conversation"
    conversation_id: uuid.UUID
    title: str


class Token(BaseModel):
    """A chunk of assistant text."""

    type: Literal["token"] = "token"
    text: str


class Done(BaseModel):
    """The turn finished and the assistant message was saved."""

    type: Literal["done"] = "done"
    message_id: uuid.UUID
    finish_reason: FinishReason
    usage: UsageOut | None


class Error(BaseModel):
    """The turn failed. Partial text already streamed is not saved."""

    type: Literal["error"] = "error"
    message: str


AgentEvent = ConversationStarted | Token | Done | Error
