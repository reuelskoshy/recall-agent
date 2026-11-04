"""Failure injection: every way a turn or an ingestion can break must end cleanly.

The contract under test: the SSE stream ends with `done` or `error`, the run row
never stays "running", stored history stays valid, and orphaned work is reaped.
"""

import asyncio
import contextlib
import hashlib
import uuid
from collections.abc import AsyncIterator, Sequence
from datetime import timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError

import app.agent.runner as runner
from app.agent.events import AgentEvent, ConversationStarted, Done, Error
from app.agent.runner import (
    DB_UNAVAILABLE,
    UNEXPECTED_ERROR,
    AgentConfig,
    drain_background_writes,
    run_turn,
)
from app.core.config import Settings, get_settings
from app.core.reaper import INGEST_INTERRUPTED, RUN_INTERRUPTED, reap_orphans
from app.db.models import AgentRun, Conversation, Document, Message
from app.db.session import SessionLocal
from app.llm.base import (
    ChatMessage,
    LLMProvider,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolSpec,
)
from app.llm.factory import get_llm_provider
from app.llm.fake import FakeProvider, ScriptedTurn
from app.rag.embeddings import FakeEmbedder
from app.rag.ingestion import ingest_document
from app.tools.base import Tool, ToolContext, ToolResult
from app.tools.registry import ToolRegistry, get_tool_registry
from tests.conftest import parse_sse

pytestmark = pytest.mark.usefixtures("db")


class StallingProvider(LLMProvider):
    """Streams a little text, then hangs forever, like an upstream that stops mid-answer."""

    name = "stalling"
    model = "stalling-model"

    async def stream_chat(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[StreamEvent]:
        yield TextDelta("Thinking")
        await asyncio.sleep(3600)


class CrashingProvider(LLMProvider):
    """Raises something that is not an LLMError: a bug, not an upstream failure."""

    name = "crashing"
    model = "crashing-model"

    async def stream_chat(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[StreamEvent]:
        yield TextDelta("Partial")
        raise RuntimeError("provider bug")


class CallsSlowTool(LLMProvider):
    """Always asks for the slow tool."""

    name = "slow-tool-caller"
    model = "slow-tool-caller"

    async def stream_chat(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec] = ()
    ) -> AsyncIterator[StreamEvent]:
        yield ToolCallEvent(ToolCall(id="call_slow", name="slow", arguments="{}"))


class NoInput(BaseModel):
    pass


class SlowTool(Tool[NoInput]):
    name = "slow"
    description = "Takes far longer than any turn may."
    input_model = NoInput
    timeout_s = 3600.0

    async def run(self, args: NoInput, ctx: ToolContext) -> ToolResult:
        await asyncio.sleep(3600)
        return ToolResult(content="never")


def override(app: FastAPI, provider: LLMProvider, **settings: Any) -> None:
    app.dependency_overrides[get_llm_provider] = lambda: provider
    app.dependency_overrides[get_settings] = lambda: Settings(**settings)


async def chat(client: AsyncClient, message: str = "hi") -> list[tuple[str, dict[str, Any]]]:
    response = await client.post("/api/chat", json={"message": message})
    assert response.status_code == 200
    return parse_sse(response.text)


async def load_run(run_id: str | uuid.UUID) -> AgentRun:
    async with SessionLocal() as session:
        run = await session.get(AgentRun, uuid.UUID(str(run_id)))
        assert run is not None
        return run


async def stored_roles(conversation_id: str) -> list[str]:
    async with SessionLocal() as session:
        rows = await session.scalars(
            select(Message.role)
            .where(Message.conversation_id == uuid.UUID(conversation_id))
            .order_by(Message.seq)
        )
        return list(rows.all())


# ---- The agent loop ------------------------------------------------------------


async def test_model_stalling_mid_stream_hits_the_turn_deadline(
    app: FastAPI, client: AsyncClient
) -> None:
    override(app, StallingProvider(), agent_turn_timeout_s=0.3)

    events = await asyncio.wait_for(chat(client), timeout=10)

    name, data = events[-1]
    assert name == "error"
    assert "0.3s time limit" in data["message"]
    run = await load_run(data["run_id"])
    assert run.status == "timeout"
    assert run.finished_at is not None


async def test_tools_running_past_the_deadline_leave_no_half_round_in_history(
    app: FastAPI, client: AsyncClient
) -> None:
    registry = ToolRegistry()
    registry.register(SlowTool())
    app.dependency_overrides[get_tool_registry] = lambda: registry
    override(app, CallsSlowTool(), agent_turn_timeout_s=0.3)

    events = await asyncio.wait_for(chat(client), timeout=10)

    assert [name for name, _ in events][-2:] == ["tool_call", "error"]
    run = await load_run(events[-1][1]["run_id"])
    assert run.status == "timeout"
    # The assistant's tool call was rolled back with the round: no call without a result.
    assert await stored_roles(events[0][1]["conversation_id"]) == ["user"]


async def test_unexpected_exception_still_ends_with_an_error_event(
    app: FastAPI, client: AsyncClient
) -> None:
    override(app, CrashingProvider())

    events = await chat(client)

    assert events[-1][1]["message"] == UNEXPECTED_ERROR
    run = await load_run(events[-1][1]["run_id"])
    assert run.status == "failed"
    assert run.error == "RuntimeError: provider bug"


async def test_database_failure_mid_turn_still_closes_the_run(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken_history(*_args: Any, **_kwargs: Any) -> list[ChatMessage]:
        raise OperationalError("SELECT ...", {}, Exception("connection reset"))

    monkeypatch.setattr(runner, "load_history", broken_history)
    override(app, FakeProvider())

    events = await chat(client)

    assert [name for name, _ in events] == ["conversation", "error"]
    run = await load_run(events[-1][1]["run_id"])
    assert run.status == "failed"
    assert run.error is not None and run.error.startswith("OperationalError")


async def test_slow_memory_recall_degrades_instead_of_stalling(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def hanging_core_memories(*_args: Any, **_kwargs: Any) -> list[Any]:
        await asyncio.sleep(3600)
        return []

    monkeypatch.setattr(runner, "core_memories", hanging_core_memories)
    override(
        app, FakeProvider([ScriptedTurn(text="Answered anyway.")]), memory_recall_timeout_s=0.2
    )

    events = await asyncio.wait_for(chat(client), timeout=10)

    assert events[-1][0] == "done"
    assert (await load_run(events[-1][1]["run_id"])).status == "completed"


async def _start_turn(
    session: Any, provider: LLMProvider, config: AgentConfig | None = None
) -> AsyncIterator[AgentEvent]:
    settings = Settings()
    return run_turn(
        session,
        provider,
        ToolRegistry(),
        None,
        "hello",
        config or AgentConfig(),
        FakeEmbedder(),
        settings,
    )


async def test_client_disconnect_marks_the_run_cancelled() -> None:
    events: list[AgentEvent] = []
    started = asyncio.Event()

    async def consume() -> None:
        async with SessionLocal() as session:
            async for event in await _start_turn(session, StallingProvider()):
                events.append(event)
                started.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=10)
    await asyncio.sleep(0.05)  # let it reach the hanging model call
    task.cancel()  # what the server does when the client goes away
    with contextlib.suppress(asyncio.CancelledError):
        await task
    await drain_background_writes()

    assert isinstance(events[0], ConversationStarted)
    assert not any(isinstance(e, Done | Error) for e in events)
    run = await load_run(events[0].run_id)
    assert run.status == "cancelled"
    assert run.finished_at is not None


async def test_database_down_at_start_yields_an_error_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with SessionLocal() as session:

        async def broken_commit() -> None:
            raise OperationalError("INSERT ...", {}, Exception("connection refused"))

        monkeypatch.setattr(session, "commit", broken_commit)
        events = [event async for event in await _start_turn(session, FakeProvider())]

    assert len(events) == 1
    assert isinstance(events[0], Error)
    assert events[0].message == DB_UNAVAILABLE
    assert events[0].run_id is None


# ---- Ingestion and the reaper --------------------------------------------------


async def _document(status: str = "processing", age: timedelta = timedelta(0)) -> uuid.UUID:
    async with SessionLocal() as session:
        doc = Document(
            filename="notes.txt",
            content_type="text/plain",
            size_bytes=5,
            sha256=hashlib.sha256(uuid.uuid4().bytes).hexdigest(),
            status=status,
            embedding_model="fake-embedder",
        )
        session.add(doc)
        await session.flush()
        doc.updated_at = func.now() - age
        await session.commit()
        return doc.id


class HangingEmbedder(FakeEmbedder):
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        await asyncio.sleep(3600)
        return []


async def test_ingestion_past_its_timeout_is_marked_failed() -> None:
    doc_id = await _document()

    await asyncio.wait_for(
        ingest_document(
            SessionLocal, HangingEmbedder(), doc_id, "notes.txt", b"hello world", 1000, 100, 0.2
        ),
        timeout=10,
    )

    async with SessionLocal() as session:
        doc = await session.get(Document, doc_id)
        assert doc is not None
        assert doc.status == "failed"
        assert doc.error == "Ingestion took longer than 0.2s; retry it."


async def test_reaper_fails_only_work_older_than_its_timeout() -> None:
    settings = Settings(agent_turn_timeout_s=120, ingest_timeout_s=600)
    async with SessionLocal() as session:
        conversation = Conversation(title="t")
        old, fresh, done = (
            AgentRun(conversation=conversation, user_input="x", provider="p", model="m")
            for _ in range(3)
        )
        done.status = "completed"
        session.add_all([old, fresh, done])
        await session.flush()
        old.created_at = func.now() - timedelta(hours=1)
        done.created_at = func.now() - timedelta(hours=1)
        await session.commit()
        old_id, fresh_id, done_id = old.id, fresh.id, done.id
    stale_doc = await _document(age=timedelta(hours=1))
    fresh_doc = await _document()
    ready_doc = await _document(status="ready", age=timedelta(hours=1))

    result = await reap_orphans(SessionLocal, settings)

    assert (result.runs, result.documents) == (1, 1)
    old_run = await load_run(old_id)
    assert (old_run.status, old_run.error) == ("interrupted", RUN_INTERRUPTED)
    assert (await load_run(fresh_id)).status == "running"
    assert (await load_run(done_id)).status == "completed"
    async with SessionLocal() as session:
        statuses = {d.id: (d.status, d.error) for d in await session.scalars(select(Document))}
    assert statuses[stale_doc] == ("failed", INGEST_INTERRUPTED)
    assert statuses[fresh_doc] == ("processing", None)
    assert statuses[ready_doc] == ("ready", None)
