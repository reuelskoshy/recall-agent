"""A deterministic, scripted provider for tests and key-less local demos."""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from app.llm.base import (
    ChatMessage,
    LLMError,
    LLMProvider,
    StreamEnd,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolSpec,
    Usage,
    UsageEvent,
)


@dataclass
class ScriptedTurn:
    """What the fake model 'says' on one call: text, tool calls, or an error."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    error: str | None = None


class FakeProvider(LLMProvider):
    """Replays scripted turns in order; echoes the last user message when the script runs out.

    Every call is recorded in `calls` so tests can assert on what the model was sent.
    """

    name = "fake"

    def __init__(self, script: Sequence[ScriptedTurn] = (), model: str = "fake-model") -> None:
        self.model = model
        self._script = list(script)
        self.calls: list[tuple[list[ChatMessage], list[ToolSpec]]] = []

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
    ) -> AsyncIterator[StreamEvent]:
        """Yield the next scripted turn word by word, mimicking real streaming."""
        self.calls.append((list(messages), list(tools)))
        if self._script:
            turn = self._script.pop(0)
        else:
            last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            turn = ScriptedTurn(text=f"(fake model) You said: {last_user}")

        words = turn.text.split(" ") if turn.text else []
        for i, word in enumerate(words):
            yield TextDelta(word if i == 0 else f" {word}")
        if turn.error is not None:
            raise LLMError(turn.error)
        for call in turn.tool_calls:
            yield ToolCallEvent(call)
        prompt_tokens = sum(len(m.content.split()) for m in messages)
        yield UsageEvent(Usage(prompt_tokens=prompt_tokens, completion_tokens=len(words)))
        yield StreamEnd("tool_calls" if turn.tool_calls else "stop")
