"""query_database: answer data questions with read-only SQL over the sample dataset.

The model sees the schema in the tool description, writes one SELECT, and gets
back a compact table. Safety is layered: SQL guard -> read-only role -> timeout
and row cap (see app/sql). Errors go back to the model so it can fix and retry.
"""

from typing import Any, ClassVar

from pydantic import BaseModel, Field
from sqlalchemy import MetaData, Table

from app.datasets.acme import ACME, Dataset
from app.sql.executor import QueryResult, get_readonly_engine, run_readonly_query
from app.sql.guard import validate_select
from app.tools.base import Tool, ToolContext, ToolResult
from app.tools.registry import register_tool

MAX_CELL_CHARS = 60


def describe_table(table: Table) -> str:
    """One table as `name(col TYPE PK, col TYPE -> other.id -- comment, ...)`."""
    columns: list[str] = []
    for column in table.columns:
        parts = [column.name, str(column.type).split("(")[0]]
        if column.primary_key:
            parts.append("PK")
        for fk in column.foreign_keys:
            parts.append(f"-> {fk.column.table.name}.{fk.column.name}")
        text = " ".join(parts)
        if column.comment:
            text += f" -- {column.comment}"
        columns.append(f"  {text}")
    header = f"{table.name}" + (f" -- {table.comment}" if table.comment else "")
    return header + "\n" + "\n".join(columns)


def describe_schema(metadata: MetaData) -> str:
    """Every table of a dataset, in foreign-key order."""
    return "\n".join(describe_table(t) for t in metadata.sorted_tables)


def _cell(value: Any) -> str:
    text = "NULL" if value is None else str(value)
    return text if len(text) <= MAX_CELL_CHARS else text[: MAX_CELL_CHARS - 1] + "…"


def format_for_model(result: QueryResult, budget_chars: int) -> str:
    """A pipe-separated table that stays within `budget_chars`."""
    if not result.rows:
        return "The query ran successfully and returned 0 rows."
    lines = [" | ".join(result.columns)]
    used = len(lines[0])
    shown = 0
    for row in result.rows:
        line = " | ".join(_cell(v) for v in row)
        if used + len(line) + 1 > budget_chars:
            break
        lines.append(line)
        used += len(line) + 1
        shown += 1

    count = f"{len(result.rows)}+" if result.truncated else str(len(result.rows))
    notes = [f"Rows returned: {count}."]
    if shown < len(result.rows):
        notes.append(f"Only the first {shown} rows are shown here.")
    if result.truncated:
        notes.append(
            f"The result was capped at {len(result.rows)} rows; more rows matched. "
            "Use aggregation (COUNT, SUM, GROUP BY) or filters rather than listing rows."
        )
    return "\n".join(lines) + "\n\n" + " ".join(notes)


class QueryDatabaseInput(BaseModel):
    """Arguments for query_database."""

    sql: str = Field(
        description="One PostgreSQL SELECT statement (WITH clauses allowed). Prefer "
        "aggregates and explicit columns; add ORDER BY when order matters.",
        min_length=1,
        max_length=4000,
    )


def _build_description(dataset: Dataset) -> str:
    return (
        "Run a read-only SQL query against the company database (PostgreSQL) and get "
        "the result rows. Use it for questions about Acme's employees, departments, "
        "salaries, leave balances, customers, products, orders, and revenue. Only one "
        "SELECT statement is allowed; results are capped, so aggregate in SQL instead "
        "of fetching raw rows. Tables (already on the search path, no schema prefix "
        "needed):\n" + describe_schema(dataset.metadata)
    )


@register_tool
class QueryDatabaseTool(Tool[QueryDatabaseInput]):
    """Natural-language questions over the sample dataset, answered via SQL."""

    name = "query_database"
    description = _build_description(ACME)
    input_model = QueryDatabaseInput
    dataset: ClassVar[Dataset] = ACME

    async def run(self, args: QueryDatabaseInput, ctx: ToolContext) -> ToolResult:
        """Validate, execute as the read-only role, and format the rows."""
        settings = ctx.settings
        schema = self.dataset.metadata.schema or "public"
        query = validate_select(
            args.sql, schema, [t.name for t in self.dataset.metadata.sorted_tables]
        )
        engine = get_readonly_engine(
            settings.readonly_url, settings.sql_statement_timeout_ms, schema
        )
        result = await run_readonly_query(engine, query.sql, settings.sql_max_rows)
        return ToolResult(
            content=format_for_model(result, budget_chars=settings.tool_max_result_chars - 300),
            data={
                "sql": query.sql,
                "columns": result.columns,
                "rows": result.rows,
                "row_count": len(result.rows),
                "truncated": result.truncated,
                "elapsed_ms": result.elapsed_ms,
            },
        )
