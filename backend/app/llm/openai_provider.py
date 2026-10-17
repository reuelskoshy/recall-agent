"""OpenAI Chat Completions provider (also works with OpenAI-compatible servers like Ollama)."""

from collections.abc import AsyncIterator, Sequence
from typing import Any, cast, get_args

import openai
from openai import AsyncOpenAI

from app.llm.base import (
    ChatMessage,
    FinishReason,
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


def to_openai_message(message: ChatMessage) -> dict[str, Any]:
    """Convert a neutral `ChatMessage` into an OpenAI chat message dict."""
    if message.role == "tool":
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
    out: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        out["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in message.tool_calls
        ]
    return out


def to_openai_tool(tool: ToolSpec) -> dict[str, Any]:
    """Convert a neutral `ToolSpec` into OpenAI's function-tool format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


class ToolCallBuffer:
    """Reassembles tool calls that OpenAI streams as fragments keyed by `index`.

    The first fragment of a call carries its id and name; later fragments carry
    pieces of the JSON arguments string, which must be concatenated.
    """

    def __init__(self) -> None:
        self._parts: dict[int, dict[str, str]] = {}

    def add(self, index: int, call_id: str | None, name: str | None, args: str | None) -> None:
        """Merge one streamed fragment into the call at `index`."""
        part = self._parts.setdefault(index, {"id": "", "name": "", "arguments": ""})
        if call_id:
            part["id"] = call_id
        if name:
            part["name"] += name
        if args:
            part["arguments"] += args

    def drain(self) -> list[ToolCall]:
        """Return the completed calls in index order and reset the buffer."""
        calls = [
            ToolCall(id=p["id"], name=p["name"], arguments=p["arguments"] or "{}")
            for _, p in sorted(self._parts.items())
        ]
        self._parts.clear()
        return calls


def normalize_finish_reason(reason: str) -> FinishReason:
    """Map OpenAI's finish reasons onto the neutral set."""
    if reason == "function_call":
        return "tool_calls"
    if reason in get_args(FinishReason):
        return cast(FinishReason, reason)
    return "stop"


class OpenAIProvider(LLMProvider):
    """Streams chat completions from OpenAI."""

    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        if not api_key:
            raise LLMError("LLM_API_KEY is not set (or use LLM_PROVIDER=fake)")
        self.model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
    ) -> AsyncIterator[StreamEvent]:
        """Stream one completion, normalizing chunks into neutral events."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [to_openai_message(m) for m in messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = [to_openai_tool(t) for t in tools]

        buffer = ToolCallBuffer()
        finish_reason: FinishReason = "stop"
        try:
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if chunk.usage is not None:
                    usage = Usage(chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
                    yield UsageEvent(usage)
                for choice in chunk.choices:
                    delta = choice.delta
                    if delta.content:
                        yield TextDelta(delta.content)
                    for tc in delta.tool_calls or []:
                        fn = tc.function
                        buffer.add(
                            tc.index,
                            tc.id,
                            fn.name if fn else None,
                            fn.arguments if fn else None,
                        )
                    if choice.finish_reason:
                        finish_reason = normalize_finish_reason(choice.finish_reason)
        except openai.OpenAIError as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc

        for call in buffer.drain():
            yield ToolCallEvent(call)
        yield StreamEnd(finish_reason)
