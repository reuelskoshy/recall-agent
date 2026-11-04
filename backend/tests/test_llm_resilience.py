"""Circuit breaker, model fallback, and transient-error classification."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Sequence

import httpx2 as httpx  # the OpenAI SDK is built on the httpx2 fork
import openai
import pytest

from app.core.config import Settings
from app.llm.base import (
    ChatMessage,
    LLMError,
    LLMProvider,
    StreamEnd,
    StreamEvent,
    TextDelta,
    ToolSpec,
)
from app.llm.factory import build_provider
from app.llm.fake import FakeProvider, ScriptedTurn
from app.llm.openai_provider import is_transient
from app.llm.resilience import CircuitBreaker, ResilientProvider
from tests.test_llm import provider_with_chunks

OUTAGE = ScriptedTurn(error="HTTP 503", transient=True)
BAD_REQUEST = ScriptedTurn(error="HTTP 400")


class Clock:
    """A manual clock, so cool-downs pass instantly."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def ask(provider: LLMProvider) -> list[StreamEvent]:
    return [e async for e in provider.stream_chat([ChatMessage(role="user", content="hi")])]


def text_of(events: list[StreamEvent]) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDelta))


def state(breaker: CircuitBreaker) -> str:
    """Read the state afresh (mypy would otherwise narrow it between calls)."""
    return breaker.state


# ---- CircuitBreaker ------------------------------------------------------------


def test_breaker_opens_after_threshold_then_half_opens_after_cool_down() -> None:
    clock = Clock()
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout_s=30, clock=clock)

    breaker.record_failure()
    assert state(breaker) == "closed"
    breaker.record_failure()
    assert state(breaker) == "open"
    assert not breaker.allow()
    assert breaker.retry_in_s == 30

    clock.now = 30
    assert state(breaker) == "half_open"
    assert breaker.allow()  # the probe
    assert not breaker.allow()  # only one probe at a time


def test_failed_probe_reopens_and_successful_probe_closes() -> None:
    clock = Clock()
    breaker = CircuitBreaker(failure_threshold=1, reset_timeout_s=10, clock=clock)
    breaker.record_failure()

    clock.now = 10
    assert breaker.allow()
    breaker.record_failure()
    assert state(breaker) == "open"  # a full new cool-down
    assert breaker.retry_in_s == 10

    clock.now = 20
    assert breaker.allow()
    breaker.record_success()
    assert state(breaker) == "closed"


def test_success_resets_the_consecutive_count() -> None:
    breaker = CircuitBreaker(failure_threshold=2)
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    assert state(breaker) == "closed"


# ---- ResilientProvider ---------------------------------------------------------


async def test_open_breaker_fails_fast_without_calling_the_provider() -> None:
    primary = FakeProvider([OUTAGE, OUTAGE])
    provider = ResilientProvider(primary, failure_threshold=2, clock=Clock())

    for _ in range(2):
        with pytest.raises(LLMError, match="HTTP 503"):
            await ask(provider)
    with pytest.raises(LLMError, match="requests are paused for 30s") as info:
        await ask(provider)

    assert info.value.transient
    assert len(primary.calls) == 2  # the third request never reached it


async def test_provider_recovers_after_the_cool_down() -> None:
    clock = Clock()
    primary = FakeProvider([OUTAGE, ScriptedTurn(text="back")])
    provider = ResilientProvider(primary, failure_threshold=1, reset_timeout_s=5, clock=clock)

    with pytest.raises(LLMError):
        await ask(provider)
    clock.now = 5

    assert text_of(await ask(provider)) == "back"
    assert state(provider.primary_breaker) == "closed"


async def test_non_transient_errors_neither_trip_the_breaker_nor_fail_over() -> None:
    fallback = FakeProvider([ScriptedTurn(text="unused")])
    provider = ResilientProvider(
        FakeProvider([BAD_REQUEST] * 3), fallback, failure_threshold=1, clock=Clock()
    )

    for _ in range(3):
        with pytest.raises(LLMError, match="HTTP 400"):
            await ask(provider)

    assert state(provider.primary_breaker) == "closed"
    assert fallback.calls == []


async def test_falls_back_when_the_primary_fails_before_any_output() -> None:
    primary = FakeProvider([OUTAGE], model="primary")
    fallback = FakeProvider([ScriptedTurn(text="from fallback")], model="backup")
    provider = ResilientProvider(primary, fallback, clock=Clock())

    events = await ask(provider)

    assert text_of(events) == "from fallback"
    assert events[-1] == StreamEnd("stop", model="backup")


async def test_open_primary_goes_straight_to_the_fallback() -> None:
    primary = FakeProvider([OUTAGE], model="primary")
    fallback = FakeProvider([ScriptedTurn(text="one"), ScriptedTurn(text="two")], model="backup")
    provider = ResilientProvider(primary, fallback, failure_threshold=1, clock=Clock())

    await ask(provider)  # primary fails, opens, fallback answers
    events = await ask(provider)

    assert text_of(events) == "two"
    assert len(primary.calls) == 1


async def test_no_failover_once_text_has_been_streamed() -> None:
    primary = FakeProvider([ScriptedTurn(text="half an", error="dropped", transient=True)])
    fallback = FakeProvider([ScriptedTurn(text="unused")])
    provider = ResilientProvider(primary, fallback, clock=Clock())

    with pytest.raises(LLMError, match="dropped"):
        await ask(provider)
    assert fallback.calls == []  # splicing two answers together would be worse


async def test_all_routes_down_raises_the_last_error() -> None:
    provider = ResilientProvider(
        FakeProvider([OUTAGE]),
        FakeProvider([ScriptedTurn(error="fallback 502", transient=True)]),
        clock=Clock(),
    )
    with pytest.raises(LLMError, match="fallback 502"):
        await ask(provider)


class Hanging(LLMProvider):
    name = "hanging"
    model = "hanging"

    async def stream_chat(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[StreamEvent]:
        await asyncio.sleep(3600)
        yield TextDelta("never")


async def test_a_cancelled_probe_frees_the_probe_slot() -> None:
    clock = Clock()
    provider = ResilientProvider(Hanging(), failure_threshold=1, reset_timeout_s=1, clock=clock)
    provider.primary_breaker.record_failure()
    clock.now = 1

    task = asyncio.create_task(ask(provider))
    await asyncio.sleep(0.01)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert provider.primary_breaker.allow()  # the next request may probe


def test_factory_builds_the_fallback_from_the_primary_settings() -> None:
    provider = build_provider(
        Settings(
            llm_provider="openai",
            llm_api_key="k",
            llm_base_url="https://api.ifm.ai/v1",
            llm_fallback_model="small-model",
        )
    )
    assert isinstance(provider, ResilientProvider)
    assert provider.fallback_model == "small-model"


# ---- What counts as transient --------------------------------------------------


def _status_error(code: int) -> openai.APIStatusError:
    request = httpx.Request("POST", "https://x")
    return openai.APIStatusError("x", response=httpx.Response(code, request=request), body=None)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (openai.APITimeoutError(request=httpx.Request("POST", "https://x")), True),
        (openai.APIConnectionError(request=httpx.Request("POST", "https://x")), True),
        (_status_error(429), True),
        (_status_error(503), True),
        (_status_error(400), False),
        (_status_error(401), False),
    ],
)
def test_is_transient(exc: openai.OpenAIError, expected: bool) -> None:
    assert is_transient(exc) is expected


async def test_connection_dropping_mid_stream_becomes_a_transient_llm_error() -> None:
    provider = provider_with_chunks([])

    async def broken_create(**_kwargs: object) -> AsyncIterator[object]:
        async def gen() -> AsyncIterator[object]:
            raise httpx.RemoteProtocolError("peer closed connection")
            yield  # pragma: no cover

        return gen()

    provider._client.chat.completions.create = broken_create  # type: ignore[method-assign, assignment]
    with pytest.raises(LLMError, match="dropped mid-answer") as info:
        await ask(provider)
    assert info.value.transient
