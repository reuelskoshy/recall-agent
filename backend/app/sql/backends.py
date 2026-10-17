"""Pluggable SQL backends for query_database.

A backend knows its SQL dialect (so the guard parses and renders the right
syntax), where the dataset lives (schema name), and how to run one validated
SELECT with a row cap. The tool itself is backend-agnostic: same schema
description, same guard, same result format, whether the rows come from the
local Postgres or a Snowflake warehouse.

Adding a backend (BigQuery, Databricks SQL, ...) means one class here plus a
`SQL_BACKEND` value; nothing in the agent or the UI changes.
"""

from abc import ABC, abstractmethod

from app.core.config import Settings
from app.sql.executor import QueryResult, get_readonly_engine, run_readonly_query


class SqlBackend(ABC):
    """Runs validated, read-only SQL against one data source."""

    name: str
    """Shown to the user in the trace card, e.g. 'postgres' or 'snowflake'."""
    dialect: str
    """sqlglot dialect used to parse, check, and render queries."""
    label: str
    """How the tool describes the database to the model, e.g. 'PostgreSQL'."""
    schema: str
    """Schema holding the dataset; the only qualifier the guard accepts."""

    @abstractmethod
    async def run(self, sql: str, max_rows: int) -> QueryResult:
        """Execute one validated SELECT and return at most `max_rows` rows.

        Raise ToolError with a message the model can act on (bad column,
        timeout, warehouse unavailable).
        """


class PostgresBackend(SqlBackend):
    """The local dataset, queried as the read-only `recall_reader` role."""

    name = "postgres"
    dialect = "postgres"
    label = "PostgreSQL"

    def __init__(self, url: str, statement_timeout_ms: int, schema: str) -> None:
        self.schema = schema
        self._url = url
        self._timeout_ms = statement_timeout_ms

    async def run(self, sql: str, max_rows: int) -> QueryResult:
        """Run in a READ ONLY transaction, streaming at most max_rows + 1 rows."""
        engine = get_readonly_engine(self._url, self._timeout_ms, self.schema)
        return await run_readonly_query(engine, sql, max_rows)


def build_sql_backend(settings: Settings, schema: str) -> SqlBackend:
    """The backend selected by `SQL_BACKEND`.

    `schema` is the dataset's schema name; Snowflake uses its own configured
    schema (identifiers there are case-insensitive, so 'ACME' matches 'acme').
    """
    if settings.sql_backend == "snowflake":
        from app.sql.snowflake import get_snowflake_backend  # optional heavy import

        return get_snowflake_backend(settings)
    return PostgresBackend(settings.readonly_url, settings.sql_statement_timeout_ms, schema)
