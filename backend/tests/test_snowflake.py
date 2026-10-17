"""Snowflake backend: dialect-aware guard, setup script, and execution against a fake connection.

No Snowflake account is needed: `aio.connect` is replaced by a fake that records
the SQL it receives and replays scripted results or errors.
"""

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import sqlglot
from snowflake.connector.errors import ProgrammingError

from app.core.config import Settings
from app.datasets.acme import ACME
from app.datasets.snowflake_setup import build_setup_sql, ensure_key_pair, literal
from app.llm.base import ToolCall
from app.rag.embeddings import FakeEmbedder
from app.sql import snowflake as sf
from app.sql.guard import UnsafeQueryError, validate_select, with_row_limit
from app.tools.base import ToolContext, ToolError
from app.tools.registry import build_registry

TABLES = [t.name for t in ACME.metadata.sorted_tables]

# ---- Guard in Snowflake's dialect ------------------------------------------------


def test_guard_accepts_snowflake_syntax() -> None:
    query = validate_select(
        "SELECT DATE_TRUNC('month', order_date)::date AS m, COUNT(*) FROM orders "
        "WHERE status <> 'cancelled' GROUP BY 1 ORDER BY 1",
        "ACME",
        TABLES,
        dialect="snowflake",
    )

    assert query.tables == {"orders"}
    assert query.sql.startswith("SELECT CAST(DATE_TRUNC('MONTH', order_date) AS DATE) AS m")


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        ("SELECT SYSTEM$WAIT(60)", r"system\$wait"),
        ("SELECT * FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()))", "result_scan"),
        ("SELECT * FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY())", "query_history"),
        ("SELECT GET_DDL('table', 'orders')", "get_ddl"),
        ("SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.USERS", "is not available"),
        ("SELECT * FROM other_schema.orders", "is not available"),
        ("USE ROLE ACCOUNTADMIN", "Only read-only SELECT"),
        ("EXECUTE IMMEDIATE 'DROP TABLE orders'", "Only read-only SELECT"),
        ("CALL some_procedure()", "Only read-only SELECT"),
    ],
)
def test_guard_blocks_snowflake_specific_escapes(sql: str, message: str) -> None:
    with pytest.raises(UnsafeQueryError, match=message):
        validate_select(sql, "ACME", TABLES, dialect="snowflake")


def test_schema_qualifier_is_case_insensitive() -> None:
    assert validate_select("SELECT id FROM acme.orders", "ACME", TABLES, dialect="snowflake")


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT a FROM t", "SELECT a FROM t LIMIT 51"),
        ("SELECT a FROM t LIMIT 5", "SELECT a FROM t LIMIT 5"),
        ("SELECT a FROM t LIMIT 1000", "SELECT a FROM t LIMIT 51"),
        (
            "SELECT a FROM t UNION ALL SELECT b FROM u",
            "SELECT a FROM t UNION ALL SELECT b FROM u LIMIT 51",
        ),
    ],
)
def test_row_limit_is_pushed_to_the_warehouse(sql: str, expected: str) -> None:
    assert with_row_limit(sql, 51, "snowflake") == expected


# ---- Setup script ---------------------------------------------------------------


@pytest.fixture
def key_path(tmp_path: Path) -> Path:
    path = tmp_path / "key.p8"
    ensure_key_pair(path)
    return path


def test_key_pair_is_created_once_and_reused(tmp_path: Path) -> None:
    path = tmp_path / "secrets" / "key.p8"

    first = ensure_key_pair(path)
    second = ensure_key_pair(path)

    assert first == second  # same public key: the existing private key was reused
    assert path.read_text().startswith("-----BEGIN PRIVATE KEY-----")
    assert "BEGIN" not in first  # Snowflake wants the bare base64 body


def test_setup_sql_loads_every_row_and_grants_read_only(key_path: Path) -> None:
    sql = build_setup_sql(ACME, Settings(), ensure_key_pair(key_path))

    statements = [s for s in sqlglot.parse(sql, read="snowflake") if s is not None]
    inserts = [s for s in statements if isinstance(s, sqlglot.exp.Insert)]
    assert sum(len(s.expression.expressions) for s in inserts) == sum(
        len(rows) for rows in ACME.rows.values()
    )
    grants = [line for line in sql.splitlines() if line.startswith("GRANT ")]
    assert grants == [
        "GRANT USAGE ON WAREHOUSE RECALL_WH TO ROLE RECALL_READER;",
        "GRANT USAGE ON DATABASE RECALL TO ROLE RECALL_READER;",
        "GRANT USAGE ON SCHEMA RECALL.ACME TO ROLE RECALL_READER;",
        "GRANT SELECT ON ALL TABLES IN SCHEMA RECALL.ACME TO ROLE RECALL_READER;",
        "GRANT ROLE RECALL_READER TO USER RECALL_AGENT;",
    ]
    assert "TYPE = SERVICE" in sql
    assert "PASSWORD =" not in sql.upper()  # no password is ever set


def test_literals_escape_quotes_and_render_types() -> None:
    assert literal("O'Brien") == "'O''Brien'"
    assert literal(True) == "TRUE"
    assert literal(None) == "NULL"
    assert literal(Decimal("12.50")) == "12.50"


# ---- Execution against a fake Snowflake --------------------------------------------


@dataclass
class Column:
    name: str


@dataclass
class FakeSnowflake:
    """Scripted stand-in for the async connector: results or errors, in order."""

    script: list[Any] = field(default_factory=list)
    executed: list[str] = field(default_factory=list)
    connects: list[dict[str, Any]] = field(default_factory=list)

    async def connect(self, **kwargs: Any) -> "FakeConnection":
        self.connects.append(kwargs)
        return FakeConnection(self)


class FakeConnection:
    def __init__(self, server: FakeSnowflake) -> None:
        self.server = server
        self.closed = False

    def is_closed(self) -> bool:
        return self.closed

    async def close(self) -> None:
        self.closed = True

    def cursor(self) -> "FakeCursor":
        return FakeCursor(self.server)


class FakeCursor:
    def __init__(self, server: FakeSnowflake) -> None:
        self.server = server
        self.description: list[Column] = []
        self.rows: list[tuple[Any, ...]] = []
        self.sfqid = "01b2-fake-query-id"

    async def execute(self, sql: str, num_statements: int | None = None) -> None:
        assert num_statements == 1  # Snowflake must reject multi-statement batches
        self.server.executed.append(sql)
        step = self.server.script.pop(0)
        if isinstance(step, Exception):
            raise step
        columns, self.rows = step
        self.description = [Column(c) for c in columns]

    async def fetchmany(self, size: int) -> list[tuple[Any, ...]]:
        return self.rows[:size]

    def close(self) -> None:
        pass


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeSnowflake]:
    server = FakeSnowflake()
    monkeypatch.setattr("snowflake.connector.aio.connect", server.connect)
    sf._backends.clear()
    yield server
    sf._backends.clear()


def settings_for(key_path: Path, **overrides: Any) -> Settings:
    return Settings(
        sql_backend="snowflake",
        snowflake_account="myorg-myaccount",
        snowflake_private_key_path=str(key_path),
        **overrides,
    )


async def test_runs_capped_query_and_normalizes_results(
    fake: FakeSnowflake, key_path: Path
) -> None:
    fake.script.append(
        (
            ["NAME", "Total Revenue"],
            [("Meridian Labs", Decimal("190890.00")), ("Foxglove", Decimal("1.5"))],
        )
    )
    backend = sf.SnowflakeBackend.from_settings(settings_for(key_path))

    result = await backend.run("SELECT name FROM customers", max_rows=1)

    assert fake.executed == ["SELECT name FROM customers LIMIT 2"]
    assert result.columns == ["name", "Total Revenue"]  # unquoted lower-cased, quoted kept
    assert result.rows == [["Meridian Labs", 190890]]
    assert result.truncated is True
    assert result.query_id == "01b2-fake-query-id"
    options = fake.connects[0]
    assert options["authenticator"] == "SNOWFLAKE_JWT"
    assert options["role"] == "RECALL_READER"
    assert options["query_tag"] == "recall-agent"
    assert options["session_parameters"]["STATEMENT_TIMEOUT_IN_SECONDS"] == 5
    assert "password" not in options


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (ProgrammingError(msg="Statement reached its statement timeout", errno=630), "timeout"),
        (
            ProgrammingError(msg="invalid identifier 'SALRY'", errno=904),
            r"SQL error: .*invalid identifier",
        ),
    ],
)
async def test_errors_become_messages_the_model_can_act_on(
    fake: FakeSnowflake, key_path: Path, error: Exception, message: str
) -> None:
    fake.script.append(error)
    backend = sf.SnowflakeBackend.from_settings(settings_for(key_path))

    with pytest.raises(ToolError, match=message):
        await backend.run("SELECT salry FROM employees", max_rows=5)


async def test_expired_session_reconnects_once(fake: FakeSnowflake, key_path: Path) -> None:
    fake.script += [ProgrammingError(msg="Session expired", errno=390112), (["N"], [(1,)])]
    backend = sf.SnowflakeBackend.from_settings(settings_for(key_path))

    result = await backend.run("SELECT 1 AS n", max_rows=5)

    assert result.rows == [[1]]
    assert len(fake.connects) == 2


def test_missing_configuration_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match="SNOWFLAKE_ACCOUNT"):
        sf.SnowflakeBackend.from_settings(Settings(sql_backend="snowflake"))
    with pytest.raises(ToolError, match="private key not found"):
        sf.SnowflakeBackend.from_settings(
            Settings(snowflake_account="a-b", snowflake_private_key_path=str(tmp_path / "x.p8"))
        )


async def test_query_database_tool_runs_on_snowflake(fake: FakeSnowflake, key_path: Path) -> None:
    fake.script.append((["N"], [(9,)]))
    settings = settings_for(key_path)
    ctx = ToolContext(uuid.uuid4(), None, FakeEmbedder(), settings)  # type: ignore[arg-type]
    sql = "SELECT COUNT(*) AS n FROM employees WHERE is_remote"

    result = await build_registry(settings).execute(
        ToolCall("q1", "query_database", json.dumps({"sql": sql})), ctx
    )

    assert not result.is_error, result.content
    assert result.data is not None
    assert result.data["backend"] == "snowflake"
    assert result.data["query_id"] == "01b2-fake-query-id"
    assert result.data["rows"] == [[9]]
    assert fake.executed == ["SELECT COUNT(*) AS n FROM employees WHERE is_remote LIMIT 51"]
