"""Integration tests for query_database: dataset, read-only role, limits, and the agent."""

import json
import uuid
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from app.core.config import Settings
from app.datasets.acme import ACME, generate
from app.db.session import SessionLocal
from app.llm.base import ToolCall
from app.llm.factory import get_llm_provider
from app.llm.fake import FakeProvider, ScriptedTurn
from app.rag.embeddings import FakeEmbedder
from app.sql.executor import get_readonly_engine, run_readonly_query
from app.tools.base import ToolContext, ToolError
from app.tools.query_database import QueryDatabaseTool, describe_schema
from app.tools.registry import build_registry
from tests.conftest import parse_sse

pytestmark = pytest.mark.usefixtures("db")

SETTINGS = Settings()


def ctx(settings: Settings = SETTINGS) -> ToolContext:
    return ToolContext(
        conversation_id=uuid.uuid4(),
        session_factory=SessionLocal,
        embedder=FakeEmbedder(),
        settings=settings,
    )


async def query(sql: str, settings: Settings = SETTINGS) -> tuple[str, dict[str, Any] | None]:
    """Run through the registry, exactly as the agent loop does."""
    registry = build_registry(settings)
    result = await registry.execute(
        ToolCall("q1", "query_database", json.dumps({"sql": sql})), ctx(settings)
    )
    return result.content, result.data


def reader(settings: Settings = SETTINGS) -> Any:
    return get_readonly_engine(settings.readonly_url, settings.sql_statement_timeout_ms, "acme")


def test_dataset_is_deterministic() -> None:
    assert generate(seed=42) == generate(seed=42)
    assert len(ACME.rows["departments"]) == 6
    assert len(ACME.rows["orders"]) == 300


def test_schema_description_lists_tables_keys_and_comments() -> None:
    text = describe_schema(ACME.metadata)

    assert "employees -- Current employees" in text
    assert "department_id INTEGER -> departments.id" in text
    assert "'completed' | 'shipped' | 'pending' | 'cancelled'" in text
    assert "acme" not in QueryDatabaseTool.description.split("Tables")[1].split("\n")[0]


async def test_answers_an_aggregate_question() -> None:
    content, data = await query(
        "SELECT d.name, count(*) AS headcount FROM employees e "
        "JOIN departments d ON d.id = e.department_id GROUP BY d.name ORDER BY headcount DESC"
    )

    assert data is not None
    assert data["columns"] == ["name", "headcount"]
    assert data["rows"][0] == ["Engineering", 14]
    assert sum(row[1] for row in data["rows"]) == len(ACME.rows["employees"])
    assert content.startswith("name | headcount\nEngineering | 14")
    assert "Rows returned: 6." in content


async def test_caps_rows_and_tells_the_model() -> None:
    content, data = await query("SELECT id FROM orders ORDER BY id", Settings(sql_max_rows=5))

    assert data is not None
    assert data["rows"] == [[1], [2], [3], [4], [5]]
    assert data["truncated"] is True
    assert "capped at 5 rows" in content


async def test_guard_and_sql_errors_are_returned_for_the_model_to_fix() -> None:
    content, data = await query("DELETE FROM orders")
    assert content.startswith("Error: Only read-only SELECT")
    assert data is None

    content, _ = await query("SELECT salry FROM employees")
    assert content.startswith("Error: SQL error:")
    assert "salry" in content


# ---- The role is the real boundary: these bypass the guard on purpose. ----


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO acme.departments (id, name, location, annual_budget) VALUES (99, 'x', 'y', 1)",
        "UPDATE acme.employees SET salary = 0",
        "SELECT * FROM public.conversations",
        "CREATE TABLE acme.stolen (id int)",
    ],
)
async def test_readonly_role_blocks_writes_and_app_tables(sql: str) -> None:
    with pytest.raises(ToolError, match=r"read-only transaction|permission denied"):
        await run_readonly_query(reader(), sql, max_rows=10)


async def test_multiple_statements_are_rejected_by_postgres() -> None:
    with pytest.raises(ToolError, match="multiple commands"):
        await run_readonly_query(reader(), "SELECT 1; SELECT 2", max_rows=10)


async def test_statement_timeout_cancels_slow_queries() -> None:
    fast = Settings(sql_statement_timeout_ms=200)
    with pytest.raises(ToolError, match="statement timeout"):
        await run_readonly_query(reader(fast), "SELECT pg_sleep(3)", max_rows=1)


async def test_agent_answers_with_sql(app: FastAPI, client: AsyncClient) -> None:
    sql = "SELECT count(*) AS n FROM employees WHERE is_remote"
    remote = sum(e["is_remote"] for e in ACME.rows["employees"])
    fake = FakeProvider(
        [
            ScriptedTurn(tool_calls=[ToolCall("c1", "query_database", f'{{"sql": "{sql}"}}')]),
            ScriptedTurn(text=f"{remote} employees work remotely."),
        ]
    )
    app.dependency_overrides[get_llm_provider] = lambda: fake

    events = parse_sse((await client.post("/api/chat", json={"message": "Remote?"})).text)

    result = next(data for name, data in events if name == "tool_result")
    assert result["is_error"] is False
    assert result["data"]["rows"] == [[remote]]
    assert result["data"]["sql"] == "SELECT COUNT(*) AS n FROM employees WHERE is_remote"
    # The model saw the schema in the tool spec.
    spec = next(s for s in fake.calls[0][1] if s.name == "query_database")
    assert "leave_balances" in spec.description
