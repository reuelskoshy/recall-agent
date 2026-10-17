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
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.events import (
    AgentEvent,
    ConversationStarted,
    Done,
    Error,
    Token,
    ToolCallStarted,
    ToolResultEvent,
    UsageOut,
)
from app.agent.prompts import SYSTEM_PROMPT, TOOL_BUDGET_EXHAUSTED
from app.core.config import Settings
from app.db.models import AgentRun, AgentStep, Conversation, Message
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
from app.memory.short_term import load_history
from app.tools.base import ToolContext, ToolResult
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

TITLE_MAX_CHARS = 60
SUMMARY_MAX_CHARS = 500
EMPTY_ANSWER_FALLBACK = "I wasn't able to finish an answer within this turn's step limit."


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


async def run_turn(
    session: AsyncSession,
    provider: LLMProvider,
    registry: ToolRegistry,
    conversation: Conversation | None,
    user_text: str,
    config: AgentConfig,
) -> AsyncIterator[AgentEvent]:
    """Run one agent turn and yield events. Creates the conversation when it is None.

    The user message and run row are committed before the model is called, so
    they survive failures. Each tool round (assistant call + results) is
    committed atomically, so stored history never holds a call without results.
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
    await session.commit()
    yield ConversationStarted(
        conversation_id=conversation.id, title=conversation.title, run_id=run.id
    )

    history = await load_history(session, conversation.id, config.history_limit)
    messages = [ChatMessage(role="system", content=SYSTEM_PROMPT), *history]
    ctx = ToolContext(conversation_id=conversation.id, session_factory=SessionLocal)
    recorder = _RunRecorder(session, run)
    iteration = 0

    try:
        for iteration in range(1, config.max_iterations + 1):
            if recorder.elapsed_s > config.turn_timeout_s:
                raise AgentTimeoutError(
                    f"The agent exceeded its {config.turn_timeout_s:g}s time limit."
                )
            tools_allowed = iteration < config.max_iterations
            prompt = (
                messages
                if tools_allowed
                else [*messages, ChatMessage(role="system", content=TOOL_BUDGET_EXHAUSTED)]
            )

            # 1. Call the model, streaming its text to the client as it arrives.
            started = time.perf_counter()
            reply = _ModelReply(text="", reasoning="", calls=[], usage=None, finish_reason="stop")
            parts: list[str] = []
            reasoning_parts: list[str] = []
            async for event in provider.stream_chat(
                prompt, registry.specs() if tools_allowed else ()
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
            reply.text = "".join(parts)
            reply.reasoning = "".join(reasoning_parts)
            recorder.llm_step(iteration, provider.model, _elapsed_ms(started), reply)

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
                yield Done(
                    message_id=final.id,
                    run_id=run.id,
                    finish_reason=reply.finish_reason,
                    iterations=iteration,
                    usage=recorder.usage(),
                )
                return

            # 3. Run the requested tools concurrently, then feed results back.
            for call in reply.calls:
                yield ToolCallStarted(
                    id=call.id, name=call.name, arguments=parse_arguments(call.arguments)
                )
            outcomes = await asyncio.gather(
                *(_timed_execute(registry, call, ctx) for call in reply.calls)
            )

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
        logger.warning("Run %s: LLM call failed: %s", run.id, exc)
        recorder.finish("failed", iteration, error=str(exc))
        await session.commit()
        yield Error(message=str(exc), run_id=run.id)
    except AgentTimeoutError as exc:
        logger.warning("Run %s: %s", run.id, exc)
        recorder.finish("timeout", iteration, error=str(exc))
        await session.commit()
        yield Error(message=str(exc), run_id=run.id)
