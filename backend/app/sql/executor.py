"""Execute validated SQL as the read-only role, with a row cap and a timeout.

Connection-level safety (independent of the SQL guard):
- the login role only has SELECT on the dataset schema;
- every transaction is opened READ ONLY;
- `statement_timeout` cancels long queries server-side;
- statements are sent as a prepared statement, which Postgres only accepts
  for a single statement;
- rows are read through a cursor, so a huge result is never materialized:
  we fetch max_rows + 1 and only use the extra row to report truncation.
"""

import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from datetime import time as dt_time
from decimal import Decimal
from typing import Any
from uuid import UUID

import asyncpg
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.tools.base import ToolError

_engines: dict[tuple[str, int, str], AsyncEngine] = {}


@dataclass(frozen=True, slots=True)
class QueryResult:
    """Rows from one query, JSON-friendly."""

    columns: list[str]
    rows: list[list[Any]]
    truncated: bool
    """True if more than `max_rows` rows matched (only `max_rows` are returned)."""
    elapsed_ms: int
    query_id: str | None = None
    """The database's own ID for the query, when it has one (Snowflake Query History)."""


def get_readonly_engine(url: str, statement_timeout_ms: int, search_path: str) -> AsyncEngine:
    """A pooled engine for the read-only role (one per distinct configuration)."""
    key = (url, statement_timeout_ms, search_path)
    if key not in _engines:
        _engines[key] = create_async_engine(
            url,
            pool_pre_ping=True,
            pool_size=3,
            max_overflow=2,
            connect_args={
                "server_settings": {
                    "search_path": str(search_path),  # may be a SQLAlchemy quoted_name
                    "statement_timeout": str(statement_timeout_ms),
                    "default_transaction_read_only": "on",
                    "idle_in_transaction_session_timeout": str(statement_timeout_ms * 2),
                }
            },
        )
    return _engines[key]


async def dispose_readonly_engines() -> None:
    """Close every pooled read-only connection (app shutdown)."""
    for engine in _engines.values():
        await engine.dispose()
    _engines.clear()


def to_jsonable(value: Any) -> Any:
    """Convert a Postgres value to something JSON (and the model) can read."""
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, datetime | date | dt_time):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


async def run_readonly_query(engine: AsyncEngine, sql: str, max_rows: int) -> QueryResult:
    """Run one SELECT in a read-only transaction and return at most `max_rows` rows.

    Database errors are raised as ToolError with Postgres' message, so the model
    can see what was wrong (unknown column, bad cast, timeout) and fix its query.
    """
    started = time.perf_counter()
    try:
        async with engine.connect() as conn:
            raw = await conn.get_raw_connection()
            driver: asyncpg.Connection = raw.driver_connection
            async with driver.transaction(readonly=True):
                statement = await driver.prepare(sql)
                columns = [attr.name for attr in statement.get_attributes()]
                cursor = await statement.cursor()
                records = await cursor.fetch(max_rows + 1)
    except asyncpg.QueryCanceledError as exc:
        raise ToolError(
            "The query was cancelled because it exceeded the statement timeout. "
            "Add filters or aggregate instead of scanning everything."
        ) from exc
    except asyncpg.PostgresError as exc:
        hint = f" Hint: {exc.hint}" if getattr(exc, "hint", None) else ""
        raise ToolError(f"SQL error: {exc}{hint}") from exc

    return QueryResult(
        columns=columns,
        rows=[[to_jsonable(v) for v in record] for record in records[:max_rows]],
        truncated=len(records) > max_rows,
        elapsed_ms=round((time.perf_counter() - started) * 1000),
    )
