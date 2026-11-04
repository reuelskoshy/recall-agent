"""Circuit breaker and model fallback around any `LLMProvider`.

Without a breaker, every request during a provider outage waits through the
SDK's full retry cycle (timeouts x retries) before failing, so an outage also
makes the app slow and ties up workers. The breaker notices consecutive
transient failures and then fails fast (or goes straight to the fallback model)
until a cool-down passes; then one probe request decides whether to close again.

    closed --(N transient failures in a row)--> open --(cool-down)--> half-open
       ^                                                                  |
       +------------------- probe succeeds ---------- probe fails -> open +

Failover only happens before the first event is streamed: once tokens have
reached the user, switching models would splice two different answers.
"""

import logging
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from app.llm.base import ChatMessage, LLMError, LLMProvider, StreamEnd, StreamEvent, ToolSpec

logger = logging.getLogger(__name__)

BreakerState = Literal["closed", "open", "half_open"]


class CircuitBreaker:
    """Consecutive-failure circuit breaker. Not thread-safe; fine on one event loop."""

    def __init__(
        self,
        failure_threshold: int = 3,
        reset_timeout_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False

    @property
    def state(self) -> BreakerState:
        """closed: normal. open: failing fast. half_open: cool-down over, next call probes."""
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at >= self.reset_timeout_s:
            return "half_open"
        return "open"

    @property
    def retry_in_s(self) -> float:
        """Seconds until an open breaker lets a probe through (0 when not open)."""
        if self._opened_at is None:
            return 0.0
        return max(0.0, self.reset_timeout_s - (self._clock() - self._opened_at))

    def allow(self) -> bool:
        """Whether a call may go through now. In half-open, only one probe at a time."""
        state = self.state
        if state == "closed":
            return True
        if state == "half_open" and not self._probe_in_flight:
            self._probe_in_flight = True
            return True
        return False

    def record_success(self) -> None:
        """The provider answered: close the breaker."""
        if self._opened_at is not None:
            logger.info("Circuit breaker closed: the provider recovered")
        self._failures = 0
        self._opened_at = None
        self._probe_in_flight = False

    def record_failure(self) -> None:
        """A transient failure: count it, and open (or re-open) the breaker if needed."""
        self._failures += 1
        was_probe = self._probe_in_flight
        self._probe_in_flight = False
        if was_probe or self._failures >= self.failure_threshold:
            if self._opened_at is None or was_probe:
                logger.warning(
                    "Circuit breaker opened after %d failure(s); failing fast for %gs",
                    self._failures,
                    self.reset_timeout_s,
                )
            self._opened_at = self._clock()

    def release(self) -> None:
        """A call ended with no verdict (e.g. the client disconnected): free the probe slot."""
        self._probe_in_flight = False


@dataclass
class _Route:
    provider: LLMProvider
    breaker: CircuitBreaker


class ResilientProvider(LLMProvider):
    """Wraps a primary provider (and optionally a fallback), each behind its own breaker."""

    def __init__(
        self,
        primary: LLMProvider,
        fallback: LLMProvider | None = None,
        failure_threshold: int = 3,
        reset_timeout_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = primary.name
        self.model = primary.model
        self._routes = [
            _Route(p, CircuitBreaker(failure_threshold, reset_timeout_s, clock))
            for p in (primary, fallback)
            if p is not None
        ]

    @property
    def primary(self) -> LLMProvider:
        """The wrapped primary provider."""
        return self._routes[0].provider

    @property
    def primary_breaker(self) -> CircuitBreaker:
        """The primary model's breaker (for health reporting)."""
        return self._routes[0].breaker

    @property
    def fallback_breaker(self) -> CircuitBreaker | None:
        """The fallback model's breaker, if a fallback is configured."""
        return self._routes[1].breaker if len(self._routes) > 1 else None

    @property
    def fallback_model(self) -> str | None:
        """The fallback model's name, if one is configured."""
        return self._routes[1].provider.model if len(self._routes) > 1 else None

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
    ) -> AsyncIterator[StreamEvent]:
        """Stream from the first healthy route; fail over only before any output."""
        last_error: LLMError | None = None
        for route in self._routes:
            breaker = route.breaker
            if not breaker.allow():
                logger.info("Skipping %s: circuit open", route.provider.model)
                last_error = LLMError(
                    "The model provider is failing repeatedly, so requests are paused for "
                    f"{breaker.retry_in_s:.0f}s. Please try again shortly.",
                    transient=True,
                )
                continue
            streamed = False
            verdict = False
            try:
                async for event in route.provider.stream_chat(messages, tools):
                    streamed = True
                    if isinstance(event, StreamEnd) and event.model is None:
                        event = StreamEnd(event.finish_reason, model=route.provider.model)
                    yield event
                breaker.record_success()
                verdict = True
                return
            except LLMError as exc:
                verdict = True
                if not exc.transient:
                    breaker.record_success()  # it answered; the request itself was refused
                    raise
                breaker.record_failure()
                if streamed:
                    raise
                logger.warning("%s failed (%s); trying the next model", route.provider.model, exc)
                last_error = exc
            finally:
                if not verdict:
                    breaker.release()
        assert last_error is not None  # at least one route always exists
        raise last_error
