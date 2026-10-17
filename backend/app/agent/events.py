"""Events the agent emits while handling one user turn.

They are transport-agnostic: the API sends each one as a Server-Sent Event whose
`event:` name equals `type`, and the frontend mirrors them as a TypeScript union.
"""

import uuid
from typing import Any, Literal

from pydantic import BaseModel

from app.llm.base import FinishReason


class UsageOut(BaseModel):
    """Token usage summed over every model call in the turn."""

    prompt_tokens: int
    completion_tokens: int


class ConversationStarted(BaseModel):
    """First event of every turn: which conversation and run it belongs to."""

    type: Literal["conversation"] = "conversation"
    conversation_id: uuid.UUID
    title: str
    run_id: uuid.UUID


class Token(BaseModel):
    """A chunk of assistant text."""

    type: Literal["token"] = "token"
    text: str


class ToolCallStarted(BaseModel):
    """The model asked for a tool; it is now running."""

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any]


class ToolResultEvent(BaseModel):
    """A tool finished (successfully or not). `content` is exactly what the model sees."""

    type: Literal["tool_result"] = "tool_result"
    id: str
    name: str
    content: str
    is_error: bool
    latency_ms: int
    data: dict[str, Any] | None = None


class Done(BaseModel):
    """The turn finished and the final assistant message was saved."""

    type: Literal["done"] = "done"
    message_id: uuid.UUID
    run_id: uuid.UUID
    finish_reason: FinishReason
    iterations: int
    usage: UsageOut


class Error(BaseModel):
    """The turn failed. Partial text already streamed is not saved."""

    type: Literal["error"] = "error"
    message: str
    run_id: uuid.UUID | None = None


AgentEvent = ConversationStarted | Token | ToolCallStarted | ToolResultEvent | Done | Error
