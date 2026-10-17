"""Provider-neutral LLM contract.

Every provider (OpenAI, Anthropic, Ollama, a test fake) translates its own wire
format into these types, so the rest of the app never imports a vendor SDK.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A request from the model to run one tool. `arguments` is the raw JSON string."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message in a conversation, in provider-neutral form.

    - assistant messages may carry `tool_calls` (and empty `content`)
    - assistant messages may carry `reasoning`: a reasoning model's hidden
      thinking, which some providers require to be sent back in history
    - tool messages carry the `tool_call_id` they answer
    """

    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    reasoning: str = ""


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool the model may call. `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


# ---- Stream events: what `stream_chat` yields ---------------------------------


@dataclass(frozen=True, slots=True)
class TextDelta:
    """A chunk of assistant text, to forward to the UI as it arrives."""

    text: str


@dataclass(frozen=True, slots=True)
class ReasoningDelta:
    """A chunk of a reasoning model's thinking (not part of the answer)."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolCallEvent:
    """A complete tool call (providers buffer argument fragments before emitting)."""

    call: ToolCall


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage for one model call."""

    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class UsageEvent:
    """Token usage, emitted once per call when the provider reports it."""

    usage: Usage


FinishReason = Literal["stop", "tool_calls", "length", "content_filter"]


@dataclass(frozen=True, slots=True)
class StreamEnd:
    """Always the last event of a successful stream."""

    finish_reason: FinishReason


StreamEvent = TextDelta | ReasoningDelta | ToolCallEvent | UsageEvent | StreamEnd


class LLMError(Exception):
    """Raised by providers for any upstream failure (auth, rate limit, network, ...).

    Providers wrap vendor exceptions in this so callers handle one error type.
    """


class LLMProvider(ABC):
    """Interface every chat model provider implements."""

    name: str
    model: str

    @abstractmethod
    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
    ) -> AsyncIterator[StreamEvent]:
        """Stream one model response.

        Yields any number of `TextDelta` / `ReasoningDelta` / `ToolCallEvent` /
        `UsageEvent`, then
        exactly one `StreamEnd`. Raises `LLMError` on upstream failure.
        """
