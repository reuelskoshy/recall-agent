"""Snowflake backend for query_database (async connector, key-pair auth).

Safety, layer by layer (the guard still runs first, in Snowflake's dialect):
- Identity: a SERVICE user (no password, no interactive login) authenticating
  with an RSA key pair. The app holds no credential that can write anything.
- Role: RECALL_READER has USAGE on one warehouse/database/schema and SELECT on
  its tables, nothing else (see app/datasets/snowflake_setup.py). Snowflake has
  no read-only session mode, so this grant set is the security boundary.
- Single statement: `num_statements=1` makes Snowflake itself reject batches.
- Time: STATEMENT_TIMEOUT_IN_SECONDS on the session (and on the user and the
  warehouse, set by the setup script), plus a client-side query timeout.
- Cost: the query gets a server-side LIMIT max_rows + 1, so the warehouse never
  produces more rows than we read; the warehouse is XSMALL with auto-suspend.
- Audit: every query carries QUERY_TAG='recall-agent' and its query ID is
  shown in the UI, so it can be found in Snowflake's Query History.
"""

import asyncio
import logging
import math
import time
from pathlib import Path
from typing import Any

from snowflake.connector import aio
from snowflake.connector.errors import DatabaseError, Error, ProgrammingError

from app.core.config import Settings
from app.sql.backends import SqlBackend
from app.sql.executor import QueryResult, to_jsonable
from app.sql.guard import with_row_limit
from app.tools.base import ToolError

logger = logging.getLogger(__name__)

QUERY_TAG = "recall-agent"
TIMEOUT_ERRNOS = frozenset({604, 630})
"""604: statement cancelled; 630: statement reached its statement/warehouse timeout."""
SESSION_EXPIRED_ERRNOS = frozenset({390112, 390114})
REPO_ROOT = Path(__file__).resolve().parents[3]

_backends: dict[tuple[str, ...], "SnowflakeBackend"] = {}


def resolve_key_path(path: str) -> Path:
    """Find the private key: as given (absolute or from the CWD), else from the repo root."""
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return REPO_ROOT / path


def column_name(name: str) -> str:
    """Snowflake upper-cases unquoted identifiers; show them lower-case like Postgres does.

    Quoted mixed-case aliases ("Total Revenue") are kept as written.
    """
    return name.lower() if name == name.upper() else name


class SnowflakeBackend(SqlBackend):
    """Runs validated SELECTs on a Snowflake warehouse as a read-only service user."""

    name = "snowflake"
    dialect = "snowflake"
    label = "Snowflake"

    def __init__(self, connect_kwargs: dict[str, Any], schema: str, timeout_s: int) -> None:
        self.schema = schema
        self._connect_kwargs = connect_kwargs
        self.timeout_s = timeout_s
        self._conn: aio.SnowflakeConnection | None = None
        # One session, one query at a time: keeps session state simple, and the
        # agent rarely runs more than one SQL query per step.
        self._lock = asyncio.Lock()

    @classmethod
    def from_settings(cls, settings: Settings) -> "SnowflakeBackend":
        """Build connection options from SNOWFLAKE_* settings (raises ToolError if incomplete)."""
        if not settings.snowflake_account:
            raise ToolError("SQL_BACKEND=snowflake but SNOWFLAKE_ACCOUNT is not set.")
        key = resolve_key_path(settings.snowflake_private_key_path)
        if not key.is_file():
            raise ToolError(
                f"Snowflake private key not found at {settings.snowflake_private_key_path}. "
                "Run `python -m app.datasets.snowflake_setup` to create it."
            )
        timeout_s = max(1, math.ceil(settings.sql_statement_timeout_ms / 1000))
        kwargs: dict[str, Any] = {
            "account": settings.snowflake_account,
            "user": settings.snowflake_user,
            "authenticator": "SNOWFLAKE_JWT",
            "private_key_file": str(key),
            "role": settings.snowflake_role,
            "warehouse": settings.snowflake_warehouse,
            "database": settings.snowflake_database,
            "schema": settings.snowflake_schema,
            "query_tag": QUERY_TAG,
            "login_timeout": 20,
            "query_timeout": timeout_s,  # client-side cancel, on top of the server timeout
            "client_session_keep_alive": True,
            # Always log out on close (we never leave async queries running).
            "enable_server_session_keep_alive_auto_detection": False,
            "session_parameters": {
                "STATEMENT_TIMEOUT_IN_SECONDS": timeout_s,
                "TIMEZONE": "UTC",
            },
        }
        if settings.snowflake_private_key_passphrase:
            kwargs["private_key_password"] = settings.snowflake_private_key_passphrase
        return cls(kwargs, schema=settings.snowflake_schema, timeout_s=timeout_s)

    async def _connection(self) -> aio.SnowflakeConnection:
        if self._conn is None or self._conn.is_closed():
            started = time.perf_counter()
            self._conn = await aio.connect(**self._connect_kwargs)
            logger.info(
                "Connected to Snowflake as %s in %.1fs",
                self._connect_kwargs["user"],
                time.perf_counter() - started,
            )
        return self._conn

    async def close(self) -> None:
        """Log the session out (app shutdown)."""
        if self._conn is not None and not self._conn.is_closed():
            await self._conn.close()
        self._conn = None

    async def run(self, sql: str, max_rows: int) -> QueryResult:
        """Run one SELECT (capped server-side) and fetch at most max_rows + 1 rows."""
        capped = with_row_limit(sql, max_rows + 1, self.dialect)
        started = time.perf_counter()
        async with self._lock:
            try:
                columns, records, query_id = await self._execute(capped, max_rows + 1)
            except ProgrammingError as exc:
                if exc.errno in SESSION_EXPIRED_ERRNOS:
                    self._conn = None
                    columns, records, query_id = await self._execute(capped, max_rows + 1)
                else:
                    raise _to_tool_error(exc) from exc
            except Error as exc:
                raise _to_tool_error(exc) from exc
        return QueryResult(
            columns=columns,
            rows=[[to_jsonable(v) for v in record] for record in records[:max_rows]],
            truncated=len(records) > max_rows,
            elapsed_ms=round((time.perf_counter() - started) * 1000),
            query_id=query_id,
        )

    async def _execute(self, sql: str, fetch: int) -> tuple[list[str], list[Any], str | None]:
        conn = await self._connection()
        cursor = conn.cursor()
        try:
            await cursor.execute(sql, num_statements=1)
            columns = [column_name(c.name) for c in cursor.description or []]
            records = list(await cursor.fetchmany(fetch))
            return columns, records, cursor.sfqid
        finally:
            cursor.close()


def _to_tool_error(exc: Error) -> ToolError:
    """Turn connector errors into messages the model (or the user) can act on."""
    if exc.errno in TIMEOUT_ERRNOS:
        return ToolError(
            "The query was cancelled because it exceeded the statement timeout. "
            "Add filters or aggregate instead of scanning everything."
        )
    message = (exc.msg or str(exc)).strip()
    if isinstance(exc, ProgrammingError):
        return ToolError(f"SQL error: {message}")
    if isinstance(exc, DatabaseError):
        return ToolError(f"Snowflake is unavailable or rejected the login: {message}")
    return ToolError(f"Snowflake error: {message}")


def get_snowflake_backend(settings: Settings) -> SnowflakeBackend:
    """One backend (and one session) per distinct configuration."""
    key = (
        settings.snowflake_account,
        settings.snowflake_user,
        settings.snowflake_role,
        settings.snowflake_warehouse,
        settings.snowflake_database,
        settings.snowflake_schema,
        str(settings.sql_statement_timeout_ms),
    )
    if key not in _backends:
        _backends[key] = SnowflakeBackend.from_settings(settings)
    return _backends[key]


async def close_snowflake_backends() -> None:
    """Close every open Snowflake session."""
    for backend in _backends.values():
        try:
            await backend.close()
        except Error:
            logger.warning("Failed to close a Snowflake session", exc_info=True)
    _backends.clear()
