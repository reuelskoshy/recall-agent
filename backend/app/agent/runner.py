"""The agent loop: one user turn, from message in to final answer out.

    user message
        -> LLM (with tool specs)
            -> no tool calls?   final answer: save, done
            -> tool calls?      run them in parallel, append results, call the LLM again
        ... at most `max_iterations` LLM calls; the last one gets no tools, which
        forces the model to answer with what it has.

Every LLM call and tool call is recorded as an `AgentStep` of an `AgentRun`.
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, TypeVar

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.events import (
    AgentEvent,
    ConversationStarted,
    Done,
    Error,
    MemoriesRecalled,
    Token,
    ToolCallStarted,
    ToolResultEvent,
    UsageOut,
)
from app.agent.prompts import TOOL_BUDGET_EXHAUSTED, build_system_prompt
from app.core.config import Settings
from app.db.models import AgentRun, AgentStep, Conversation, Document, Message
from app.db.session import SessionLocal
from app.llm.base import (
    ChatMessage,
    FinishReason,
    LLMError,
    LLMProvider,
    ReasoningDelta,
    StreamEnd,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    Usage,
    UsageEvent,
)
from app.memory.extraction import schedule_extraction
from app.memory.long_term import RankWeights, core_memories, recall_memories
from app.memory.short_term import load_history
from app.rag.embeddings import EmbeddingProvider
from app.tools.base import ToolContext, ToolResult
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

T = TypeVar("T")

TITLE_MAX_CHARS = 60
SUMMARY_MAX_CHARS = 500
EMPTY_ANSWER_FALLBACK = "I wasn't able to finish an answer within this turn's step limit."
UNEXPECTED_ERROR = "Something went wrong while answering. It was logged; please try again."
DB_UNAVAILABLE = "Could not save your message: the database is unavailable. Please retry."
CLIENT_DISCONNECTED = "The client disconnected before the answer finished."


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Limits for one turn."""

    max_iterations: int = 6
    turn_timeout_s: float = 120.0
    history_limit: int = 20

    @classmethod
    def from_settings(cls, settings: Settings) -> "AgentConfig":
        """Build from application settings."""
        return cls(
            max_iterations=settings.agent_max_iterations,
            turn_timeout_s=settings.agent_turn_timeout_s,
            history_limit=settings.chat_history_limit,
        )


class AgentTimeoutError(Exception):
    """The turn exceeded its wall-clock budget."""


def _timeout_message(config: AgentConfig) -> str:
    return f"The agent exceeded its {config.turn_timeout_s:g}s time limit."


def make_title(first_message: str) -> str:
    """Derive a conversation title from its first user message."""
    text = " ".join(first_message.split())
    return text if len(text) <= TITLE_MAX_CHARS else text[: TITLE_MAX_CHARS - 1].rstrip() + "…"


def parse_arguments(raw: str) -> dict[str, Any]:
    """Best-effort parse of a tool call's JSON arguments, for logging and the UI."""
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_raw": raw}
    return value if isinstance(value, dict) else {"_raw": raw}


def summarize(text: str) -> str:
    """Clip text for the observability log."""
    return text if len(text) <= SUMMARY_MAX_CHARS else text[: SUMMARY_MAX_CHARS - 1] + "…"


def _elapsed_ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


@dataclass
class _ModelReply:
    text: str
    reasoning: str
    calls: list[ToolCall]
    usage: Usage | None
    finish_reason: FinishReason
    model: str | None = None


class _RunRecorder:
    """Accumulates steps and totals onto an `AgentRun` (flushed with the session)."""

    def __init__(self, session: AsyncSession, run: AgentRun) -> None:
        self._session = session
        self._run = run
        self._next_index = 0
        self._started = time.perf_counter()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self._started

    def _add(self, **fields: Any) -> None:
        self._session.add(AgentStep(run=self._run, index=self._next_index, **fields))
        self._next_index += 1

    def llm_step(self, iteration: int, model: str, latency_ms: int, reply: _ModelReply) -> None:
        if reply.usage:
            self.prompt_tokens += reply.usage.prompt_tokens
            self.completion_tokens += reply.usage.completion_tokens
        if reply.calls:
            summary = "tool calls: " + ", ".join(c.name for c in reply.calls)
        else:
            summary = summarize(reply.text)
        self._add(
            iteration=iteration,
            kind="llm",
            name=model,
            result_summary=summary,
            latency_ms=latency_ms,
            prompt_tokens=reply.usage.prompt_tokens if reply.usage else None,
            completion_tokens=reply.usage.completion_tokens if reply.usage else None,
        )

    def tool_step(
        self, iteration: int, call: ToolCall, result: ToolResult, latency_ms: int
    ) -> None:
        self._add(
            iteration=iteration,
            kind="tool",
            name=call.name,
            arguments=parse_arguments(call.arguments),
            result_summary=summarize(result.content),
            is_error=result.is_error,
            latency_ms=latency_ms,
        )

    def finish(self, status: str, iterations: int, error: str | None = None) -> None:
        run = self._run
        run.status = status
        run.iterations = iterations
        run.error = error
        run.prompt_tokens = self.prompt_tokens
        run.completion_tokens = self.completion_tokens
        run.latency_ms = round(self.elapsed_s * 1000)
        run.finished_at = func.now()

    def usage(self) -> UsageOut:
        return UsageOut(prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens)


async def _timed_execute(
    registry: ToolRegistry, call: ToolCall, ctx: ToolContext
) -> tuple[ToolResult, int]:
    start = time.perf_counter()
    result = await registry.execute(call, ctx)
    return result, _elapsed_ms(start)


async def _before_deadline(events: AsyncIterator[T], deadline: float) -> AsyncIterator[T]:
    """Re-yield `events`, raising TimeoutError if the next one is not ready by `deadline`.

    The timeout wraps each `__anext__` only, never a `yield`, so it cannot cancel
    the consumer. This is what bounds a model call that stalls mid-stream: the
    SDK's own timeout is per read, and its retries multiply it.
    """
    iterator = aiter(events)
    try:
        while True:
            try:
                async with asyncio.timeout_at(deadline):
                    event = await anext(iterator)
            except StopAsyncIteration:
                return
            yield event
    finally:
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()


# Writes that must outlive a cancelled request (see `_record_cancellation`).
_background: set[asyncio.Task[None]] = set()


async def _mark_cancelled(run_id: uuid.UUID, iterations: int, latency_ms: int) -> None:
    try:
        async with SessionLocal() as session:
            await session.execute(
                update(AgentRun)
                .where(AgentRun.id == run_id, AgentRun.status == "running")
                .values(
                    status="cancelled",
                    iterations=iterations,
                    error=CLIENT_DISCONNECTED,
                    latency_ms=latency_ms,
                    finished_at=func.now(),
                )
            )
            await session.commit()
    except Exception:
        # The reaper marks it "interrupted" later instead.
        logger.exception("Run %s: could not record cancellation", run_id)


def _record_cancellation(run_id: uuid.UUID, iterations: int, latency_ms: int) -> None:
    """Mark a run cancelled from a separate task.

    The request's own task is being cancelled (the client went away), and under
    anyio every further `await` in it is cancelled too, so the write runs in its
    own task, on a fresh session.
    """
    task = asyncio.create_task(_mark_cancelled(run_id, iterations, latency_ms))
    _background.add(task)
    task.add_done_callback(_background.discard)


async def drain_background_writes() -> None:
    """Wait for pending cancellation writes (app shutdown, and tests)."""
    while _background:
        await asyncio.gather(*list(_background), return_exceptions=True)


async def _finish_failed(
    session: AsyncSession,
    recorder: _RunRecorder,
    run_id: uuid.UUID,
    status: str,
    iterations: int,
    error: str,
) -> None:
    """Record a failed run without raising: the session may be broken, the DB may be down.

    The rollback discards the half-finished round, so stored history never holds
    a tool call without its results. If even this fails, the reaper closes the run.
    """
    try:
        await session.rollback()
        recorder.finish(status, iterations, error=error)
        await session.commit()
    except Exception:
        logger.exception("Run %s: could not record status %r", run_id, status)


async def run_turn(
    session: AsyncSession,
    provider: LLMProvider,
    registry: ToolRegistry,
    conversation: Conversation | None,
    user_text: str,
    config: AgentConfig,
    embedder: EmbeddingProvider,
    settings: Settings,
) -> AsyncIterator[AgentEvent]:
    """Run one agent turn and yield events. Creates the conversation when it is None.

    The user message and run row are committed before the model is called, so
    they survive failures. Each tool round (assistant call + results) is
    committed atomically, so stored history never holds a call without results.

    Failure contract: the stream always ends with exactly one `Done` or `Error`
    (unless the client is gone), and the run never stays "running": it ends
    completed | failed | timeout here, cancelled if the client disconnects, or
    interrupted (set by the reaper) if the process dies.
    """
    if conversation is None:
        conversation = Conversation(title=make_title(user_text))
        session.add(conversation)
    conversation.updated_at = func.now()
    session.add(Message(conversation=conversation, role="user", content=user_text))
    run = AgentRun(
        conversation=conversation,
        user_input=user_text,
        provider=provider.name,
        model=provider.model,
        status="running",
    )
    session.add(run)
    try:
        await session.commit()
    except Exception:
        logger.exception("Could not start a run: database write failed")
        yield Error(message=DB_UNAVAILABLE)
        return
    # Plain values: after a rollback, reading an expired ORM attribute needs a
    # lazy load, which an async session cannot do.
    run_id, conversation_id = run.id, conversation.id
    yield ConversationStarted(
        conversation_id=conversation_id, title=conversation.title, run_id=run_id
    )

    deadline = asyncio.get_running_loop().time() + config.turn_timeout_s
    recorder = _RunRecorder(session, run)
    iteration = 0

    try:
        history = await load_history(session, conversation_id, config.history_limit)
        document_names = (
            await session.scalars(
                select(Document.filename)
                .where(Document.status == "ready")
                .order_by(Document.created_at.desc())
            )
        ).all()
        # Long-term memory: core memories (instructions, important facts) always, plus
        # the best-ranked others (similarity + recency + importance). Never fatal, and
        # time-boxed: a slow embedder or database degrades the turn instead of stalling it.
        memory_lines: list[str] = []
        try:
            async with asyncio.timeout(settings.memory_recall_timeout_s):
                core = await core_memories(
                    session, settings.memory_core_min_importance, settings.memory_core_limit
                )
                recalled = await recall_memories(
                    session,
                    embedder,
                    user_text,
                    RankWeights.from_settings(settings),
                    limit=settings.memory_inject_k,
                    min_similarity=settings.memory_min_similarity,
                    exclude=[m.id for m in core],
                )
            used = [
                {
                    "id": str(m.id),
                    "content": m.content,
                    "kind": m.kind,
                    "importance_level": m.importance,
                    "reason": "core",
                }
                for m in core
            ] + [{**s.to_dict(), "reason": "relevant"} for s in recalled]
            memory_lines = [m.content for m in core] + [s.memory.content for s in recalled]
        except Exception:
            logger.exception("Run %s: memory recall failed; continuing without memories", run_id)
            await session.rollback()
            used = []
        if used:
            run.memories_used = used
            await session.commit()
            yield MemoriesRecalled(memories=used)
        system_prompt = build_system_prompt(document_names, memory_lines)
        messages = [ChatMessage(role="system", content=system_prompt), *history]
        ctx = ToolContext(
            conversation_id=conversation_id,
            session_factory=SessionLocal,
            embedder=embedder,
            settings=settings,
        )

        for iteration in range(1, config.max_iterations + 1):
            tools_allowed = iteration < config.max_iterations
            prompt = (
                messages
                if tools_allowed
                else [*messages, ChatMessage(role="system", content=TOOL_BUDGET_EXHAUSTED)]
            )

            # 1. Call the model, streaming its text to the client as it arrives. The
            #    turn deadline bounds the call too, even if it stalls mid-stream.
            started = time.perf_counter()
            reply = _ModelReply(text="", reasoning="", calls=[], usage=None, finish_reason="stop")
            parts: list[str] = []
            reasoning_parts: list[str] = []
            try:
                async for event in _before_deadline(
                    provider.stream_chat(prompt, registry.specs() if tools_allowed else ()),
                    deadline,
                ):
                    if isinstance(event, TextDelta):
                        parts.append(event.text)
                        yield Token(text=event.text)
                    elif isinstance(event, ReasoningDelta):
                        reasoning_parts.append(event.text)
                    elif isinstance(event, ToolCallEvent):
                        reply.calls.append(event.call)
                    elif isinstance(event, UsageEvent):
                        reply.usage = event.usage
                    elif isinstance(event, StreamEnd):
                        reply.finish_reason = event.finish_reason
                        reply.model = event.model
            except TimeoutError as exc:
                raise AgentTimeoutError(_timeout_message(config)) from exc
            reply.text = "".join(parts)
            reply.reasoning = "".join(reasoning_parts)
            # The fallback model may have answered instead of the primary.
            model = reply.model or provider.model
            recorder.llm_step(iteration, model, _elapsed_ms(started), reply)

            # 2. No tool calls (or no tools allowed): this is the final answer.
            if not reply.calls or not tools_allowed:
                text = reply.text if reply.text.strip() else EMPTY_ANSWER_FALLBACK
                final = Message(
                    conversation=conversation,
                    role="assistant",
                    content=text,
                    reasoning=reply.reasoning or None,
                )
                session.add(final)
                recorder.finish("completed", iteration)
                await session.commit()
                if settings.memory_auto_extract:
                    schedule_extraction(
                        SessionLocal, provider, embedder, settings, conversation_id, user_text, text
                    )
                yield Done(
                    message_id=final.id,
                    run_id=run_id,
                    finish_reason=reply.finish_reason,
                    iterations=iteration,
                    usage=recorder.usage(),
                )
                return

            # 3. Run the requested tools concurrently, then feed results back. Each
            #    tool has its own timeout; the turn deadline caps them together.
            for call in reply.calls:
                yield ToolCallStarted(
                    id=call.id, name=call.name, arguments=parse_arguments(call.arguments)
                )
            try:
                async with asyncio.timeout_at(deadline):
                    outcomes = await asyncio.gather(
                        *(_timed_execute(registry, call, ctx) for call in reply.calls)
                    )
            except TimeoutError as exc:
                raise AgentTimeoutError(_timeout_message(config)) from exc

            assistant = ChatMessage(
                role="assistant",
                content=reply.text,
                tool_calls=tuple(reply.calls),
                reasoning=reply.reasoning,
            )
            messages.append(assistant)
            session.add(
                Message(
                    conversation=conversation,
                    role="assistant",
                    content=reply.text,
                    reasoning=reply.reasoning or None,
                    tool_calls=[
                        {"id": c.id, "name": c.name, "arguments": c.arguments} for c in reply.calls
                    ],
                )
            )
            for call, (result, latency_ms) in zip(reply.calls, outcomes, strict=True):
                recorder.tool_step(iteration, call, result, latency_ms)
                messages.append(
                    ChatMessage(role="tool", content=result.content, tool_call_id=call.id)
                )
                session.add(
                    Message(
                        conversation=conversation,
                        role="tool",
                        content=result.content,
                        tool_call_id=call.id,
                        tool_data=result.data,
                    )
                )
                yield ToolResultEvent(
                    id=call.id,
                    name=call.name,
                    content=result.content,
                    is_error=result.is_error,
                    latency_ms=latency_ms,
                    data=result.data,
                )
            await session.commit()

    except LLMError as exc:
        logger.warning("Run %s: LLM call failed: %s", run_id, exc)
        await _finish_failed(session, recorder, run_id, "failed", iteration, str(exc))
        yield Error(message=str(exc), run_id=run_id)
    except AgentTimeoutError as exc:
        logger.warning("Run %s: %s", run_id, exc)
        await _finish_failed(session, recorder, run_id, "timeout", iteration, str(exc))
        yield Error(message=str(exc), run_id=run_id)
    except (asyncio.CancelledError, GeneratorExit):
        # The client disconnected (or the server is shutting down). Nothing more can
        # be sent; just make sure the run does not stay "running".
        logger.info("Run %s: cancelled at iteration %d", run_id, iteration)
        _record_cancellation(run_id, iteration, round(recorder.elapsed_s * 1000))
        raise
    except Exception as exc:
        # A bug, or the database failing mid-turn: still close the run and tell the user.
        logger.exception("Run %s: unexpected failure", run_id)
        await _finish_failed(
            session, recorder, run_id, "failed", iteration, f"{type(exc).__name__}: {exc}"
        )
        yield Error(message=UNEXPECTED_ERROR, run_id=run_id)
