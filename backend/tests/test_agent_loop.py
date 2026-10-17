"""Integration tests for the agent loop: real Postgres + registry, scripted fake LLM."""

import uuid
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.config import Settings, get_settings
from app.db.models import AgentRun, Message
from app.db.session import SessionLocal
from app.llm.base import ToolCall
from app.llm.factory import get_llm_provider
from app.llm.fake import FakeProvider, ScriptedTurn
from tests.conftest import parse_sse

pytestmark = pytest.mark.usefixtures("db")


def calc(expression: str, call_id: str = "call_1") -> ToolCall:
    return ToolCall(id=call_id, name="calculator", arguments=f'{{"expression": "{expression}"}}')


def use_fake(app: FastAPI, *turns: ScriptedTurn, **settings: Any) -> FakeProvider:
    fake = FakeProvider(turns)
    app.dependency_overrides[get_llm_provider] = lambda: fake
    if settings:
        app.dependency_overrides[get_settings] = lambda: Settings(**settings)
    return fake


async def chat(client: AsyncClient, message: str) -> list[tuple[str, dict[str, Any]]]:
    response = await client.post("/api/chat", json={"message": message})
    assert response.status_code == 200
    return parse_sse(response.text)


async def load_run(run_id: str) -> AgentRun:
    async with SessionLocal() as session:
        run = await session.scalar(
            select(AgentRun)
            .where(AgentRun.id == uuid.UUID(run_id))
            .options(selectinload(AgentRun.steps))
        )
        assert run is not None
        return run


async def test_tool_round_trip(app: FastAPI, client: AsyncClient) -> None:
    fake = use_fake(
        app,
        ScriptedTurn(text="Let me calculate.", tool_calls=[calc("1234 * 5678")]),
        ScriptedTurn(text="The answer is 7006652."),
    )

    events = await chat(client, "What is 1234 * 5678?")

    names = [name for name, _ in events]
    assert names[0] == "conversation"
    assert names[-1] == "done"
    tool_call = next(data for name, data in events if name == "tool_call")
    assert tool_call == {
        "type": "tool_call",
        "id": "call_1",
        "name": "calculator",
        "arguments": {"expression": "1234 * 5678"},
    }
    tool_result = next(data for name, data in events if name == "tool_result")
    assert tool_result["content"] == "1234 * 5678 = 7006652"
    assert tool_result["is_error"] is False
    assert names.index("tool_call") < names.index("tool_result") < names.index("done")

    # The second model call saw the tool exchange and was offered the tools.
    second_messages, second_tools = fake.calls[1]
    assert [m.role for m in second_messages[-3:]] == ["user", "assistant", "tool"]
    assert second_messages[-1].content == "1234 * 5678 = 7006652"
    assert {t.name for t in second_tools} >= {"calculator", "get_current_datetime"}

    done = events[-1][1]
    assert done["iterations"] == 2

    # History persisted in order, and the run log has llm -> tool -> llm.
    conversation_id = uuid.UUID(events[0][1]["conversation_id"])
    async with SessionLocal() as session:
        rows = (
            await session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.seq)
            )
        ).all()
    assert [(m.role, m.tool_call_id) for m in rows] == [
        ("user", None),
        ("assistant", None),
        ("tool", "call_1"),
        ("assistant", None),
    ]
    assert rows[1].tool_calls == [
        {"id": "call_1", "name": "calculator", "arguments": '{"expression": "1234 * 5678"}'}
    ]

    run = await load_run(done["run_id"])
    assert run.status == "completed"
    assert run.iterations == 2
    assert [(s.kind, s.name) for s in run.steps] == [
        ("llm", "fake-model"),
        ("tool", "calculator"),
        ("llm", "fake-model"),
    ]
    assert run.steps[1].arguments == {"expression": "1234 * 5678"}
    assert run.prompt_tokens > 0


async def test_parallel_tool_calls_all_run(app: FastAPI, client: AsyncClient) -> None:
    use_fake(
        app,
        ScriptedTurn(tool_calls=[calc("2+2", "a"), calc("3*3", "b")]),
        ScriptedTurn(text="4 and 9."),
    )

    events = await chat(client, "2+2 and 3*3?")

    results = {data["id"]: data["content"] for name, data in events if name == "tool_result"}
    assert results == {"a": "2+2 = 4", "b": "3*3 = 9"}


async def test_tool_error_is_fed_back_and_the_model_recovers(
    app: FastAPI, client: AsyncClient
) -> None:
    fake = use_fake(
        app,
        ScriptedTurn(tool_calls=[calc("2 ^ 10")]),
        ScriptedTurn(tool_calls=[calc("2 ** 10", "call_2")]),
        ScriptedTurn(text="1024"),
    )

    events = await chat(client, "2 to the 10th?")

    results = [data for name, data in events if name == "tool_result"]
    assert results[0]["is_error"] is True
    assert "Use **" in results[0]["content"]
    assert results[1]["content"] == "2 ** 10 = 1024"
    assert "Use **" in fake.calls[1][0][-1].content  # the model saw the error
    run = await load_run(events[-1][1]["run_id"])
    assert [s.is_error for s in run.steps if s.kind == "tool"] == [True, False]


async def test_iteration_limit_forces_a_final_answer_without_tools(
    app: FastAPI, client: AsyncClient
) -> None:
    looping = ScriptedTurn(tool_calls=[calc("1+1")])
    fake = use_fake(
        app,
        looping,
        ScriptedTurn(text="Best effort: 2.", tool_calls=[calc("1+1", "ignored")]),
        agent_max_iterations=2,
    )

    events = await chat(client, "loop forever")

    final_messages, final_tools = fake.calls[1]
    assert final_tools == []  # no tools offered on the last iteration
    assert "used all available tool calls" in final_messages[-1].content
    assert [name for name, _ in events].count("tool_call") == 1
    assert events[-1][1]["iterations"] == 2
    detail = (await client.get(f"/api/conversations/{events[0][1]['conversation_id']}")).json()
    assert detail["messages"][-1]["content"] == "Best effort: 2."


async def test_llm_failure_mid_run_marks_run_failed(app: FastAPI, client: AsyncClient) -> None:
    use_fake(
        app,
        ScriptedTurn(tool_calls=[calc("1+1")]),
        ScriptedTurn(error="upstream 500"),
    )

    events = await chat(client, "hi")

    assert events[-1] == ("error", {"type": "error", "message": "upstream 500", "run_id": ANY})
    run = await load_run(events[-1][1]["run_id"])
    assert run.status == "failed"
    assert run.error == "upstream 500"
    assert run.finished_at is not None


async def test_runs_and_tools_endpoints(app: FastAPI, client: AsyncClient) -> None:
    use_fake(app, ScriptedTurn(tool_calls=[calc("6*7")]), ScriptedTurn(text="42"))
    events = await chat(client, "6*7")
    run_id = events[-1][1]["run_id"]
    conversation_id = events[0][1]["conversation_id"]

    runs = (await client.get("/api/runs", params={"conversation_id": conversation_id})).json()
    assert [r["id"] for r in runs] == [run_id]
    assert runs[0]["status"] == "completed"

    detail = (await client.get(f"/api/runs/{run_id}")).json()
    assert [s["kind"] for s in detail["steps"]] == ["llm", "tool", "llm"]
    assert detail["steps"][1]["result_summary"] == "6*7 = 42"
    assert (await client.get(f"/api/runs/{uuid.uuid4()}")).status_code == 404

    tools = (await client.get("/api/tools")).json()
    calculator = next(t for t in tools if t["name"] == "calculator")
    assert calculator["parameters"]["required"] == ["expression"]


class _Any:
    def __eq__(self, other: object) -> bool:
        return True


ANY: Any = _Any()
