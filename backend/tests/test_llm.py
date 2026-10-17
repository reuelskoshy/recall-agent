"""Unit tests for the LLM layer: OpenAI normalization, factory, fake provider."""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from openai.types.chat import ChatCompletionChunk

from app.core.config import Settings
from app.llm.base import (
    ChatMessage,
    LLMError,
    ReasoningDelta,
    StreamEnd,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    UsageEvent,
)
from app.llm.factory import build_provider
from app.llm.fake import FakeProvider
from app.llm.openai_provider import (
    OpenAIProvider,
    ToolCallBuffer,
    normalize_finish_reason,
    to_openai_message,
)


def chunk(
    *,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> ChatCompletionChunk:
    """Build a real SDK chunk object, exactly as the OpenAI client would yield it."""
    choices = []
    if content is not None or tool_calls is not None or finish_reason is not None:
        delta: dict[str, Any] = {}
        if content is not None:
            delta["content"] = content
        if tool_calls is not None:
            delta["tool_calls"] = tool_calls
        choices.append({"index": 0, "delta": delta, "finish_reason": finish_reason})
    data: dict[str, Any] = {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": choices,
    }
    if usage is not None:
        data["usage"] = {**usage, "total_tokens": sum(usage.values())}
    return ChatCompletionChunk.model_validate(data)


def provider_with_chunks(chunks: list[ChatCompletionChunk]) -> OpenAIProvider:
    provider = OpenAIProvider(api_key="test-key", model="m")

    async def fake_create(**_kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        async def gen() -> AsyncIterator[ChatCompletionChunk]:
            for c in chunks:
                yield c

        return gen()

    provider._client.chat.completions.create = fake_create  # type: ignore[method-assign, assignment]
    return provider


async def collect(provider: OpenAIProvider | FakeProvider) -> list[StreamEvent]:
    return [e async for e in provider.stream_chat([ChatMessage(role="user", content="hi")])]


async def test_openai_stream_normalizes_text_and_usage() -> None:
    provider = provider_with_chunks(
        [
            chunk(content="Hel"),
            chunk(content="lo"),
            chunk(finish_reason="stop"),
            chunk(usage={"prompt_tokens": 5, "completion_tokens": 2}),
        ]
    )

    events = await collect(provider)

    assert events[:2] == [TextDelta("Hel"), TextDelta("lo")]
    assert isinstance(events[2], UsageEvent)
    assert events[2].usage.total_tokens == 7
    assert events[-1] == StreamEnd("stop")


async def test_openai_stream_reassembles_fragmented_parallel_tool_calls() -> None:
    def tc(index: int, **fields: Any) -> dict[str, Any]:
        out: dict[str, Any] = {"index": index}
        if "id" in fields:
            out["id"] = fields["id"]
            out["type"] = "function"
        out["function"] = {k: v for k, v in fields.items() if k in ("name", "arguments")}
        return out

    provider = provider_with_chunks(
        [
            chunk(tool_calls=[tc(0, id="call_a", name="calculator", arguments="")]),
            chunk(tool_calls=[tc(1, id="call_b", name="get_time", arguments="{}")]),
            chunk(tool_calls=[tc(0, arguments='{"expr": ')]),
            chunk(tool_calls=[tc(0, arguments='"2+2"}')]),
            chunk(finish_reason="tool_calls"),
        ]
    )

    events = await collect(provider)

    assert events == [
        ToolCallEvent(ToolCall("call_a", "calculator", '{"expr": "2+2"}')),
        ToolCallEvent(ToolCall("call_b", "get_time", "{}")),
        StreamEnd("tool_calls"),
    ]


def test_tool_call_buffer_defaults_empty_arguments_to_empty_object() -> None:
    buffer = ToolCallBuffer()
    buffer.add(0, "id1", "now", None)

    assert buffer.drain() == [ToolCall("id1", "now", "{}")]
    assert buffer.drain() == []


def test_to_openai_message_formats_tool_traffic() -> None:
    assistant = ChatMessage(
        role="assistant", tool_calls=(ToolCall("c1", "calculator", '{"expr":"1"}'),)
    )
    tool = ChatMessage(role="tool", content="1", tool_call_id="c1")

    assert to_openai_message(assistant)["tool_calls"][0] == {
        "id": "c1",
        "type": "function",
        "function": {"name": "calculator", "arguments": '{"expr":"1"}'},
    }
    assert to_openai_message(tool) == {"role": "tool", "tool_call_id": "c1", "content": "1"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("stop", "stop"), ("function_call", "tool_calls"), ("length", "length"), ("weird", "stop")],
)
def test_normalize_finish_reason(raw: str, expected: str) -> None:
    assert normalize_finish_reason(raw) == expected


def test_factory_builds_named_provider_and_rejects_unknown() -> None:
    assert isinstance(build_provider(Settings(llm_provider="fake")), FakeProvider)
    with pytest.raises(LLMError, match="Unknown LLM_PROVIDER"):
        build_provider(Settings(llm_provider="nope"))


def test_openai_provider_requires_api_key() -> None:
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        build_provider(Settings(llm_provider="openai", llm_api_key=""))


async def test_fake_provider_echoes_when_script_is_exhausted() -> None:
    events = await collect(FakeProvider())

    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text == "(fake model) You said: hi"
    assert events[-1] == StreamEnd("stop")


def test_blank_base_url_is_treated_as_unset() -> None:
    assert Settings(llm_base_url="").llm_base_url is None


async def test_openai_stream_emits_reasoning_from_nonstandard_delta_field() -> None:
    reasoning_chunk = ChatCompletionChunk.model_validate(
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "m",
            "choices": [{"index": 0, "delta": {"reasoning": "Think first."}}],
        }
    )
    provider = provider_with_chunks(
        [reasoning_chunk, chunk(content="Hi"), chunk(finish_reason="stop")]
    )

    events = await collect(provider)

    assert events[:2] == [ReasoningDelta("Think first."), TextDelta("Hi")]


def test_reasoning_field_is_sent_back_on_every_assistant_message_when_configured() -> None:
    with_reasoning = ChatMessage(role="assistant", content="Hi", reasoning="greet")
    without = ChatMessage(role="assistant", content="Old reply")  # stored before capture existed
    user = ChatMessage(role="user", content="hi")

    assert to_openai_message(with_reasoning, "reasoning")["reasoning"] == "greet"
    assert to_openai_message(without, "reasoning")["reasoning"] == ""  # IFM accepts ""
    assert "reasoning" not in to_openai_message(user, "reasoning")
    assert "reasoning" not in to_openai_message(with_reasoning)  # off by default (OpenAI)


def test_factory_passes_reasoning_field_and_blank_means_unset() -> None:
    settings = Settings(llm_provider="openai", llm_api_key="k", llm_reasoning_field="reasoning")
    provider = build_provider(settings)
    assert isinstance(provider, OpenAIProvider)
    assert provider.reasoning_field == "reasoning"
    assert Settings(llm_reasoning_field="").llm_reasoning_field is None
