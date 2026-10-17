"""Static validation of model-written SQL before it reaches the database.

This is the first of three layers (guard -> read-only role -> timeouts/limits).
It gives the model fast, specific feedback ("only SELECT is allowed") and
narrows what can run, but it is NOT the security boundary: the database role
has only SELECT on one schema, so even SQL that slips past this check cannot
write or read app tables.

The query is parsed into an AST (sqlglot), checked, and re-rendered from that
AST. The rendered SQL is what executes, so the database runs exactly what was
validated: no parser differential between this check and Postgres.
"""

from collections.abc import Collection
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from app.tools.base import ToolError

MAX_SQL_CHARS = 4000

_ALLOWED_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)
_FORBIDDEN_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
    exp.Copy,
    exp.Into,  # SELECT ... INTO creates a table
    exp.Lock,  # SELECT ... FOR UPDATE takes row locks
)
_FORBIDDEN_FUNCTION_PREFIXES = ("pg_", "lo_", "dblink", "txid_")
_FORBIDDEN_FUNCTIONS = frozenset(
    {
        "set_config",
        "current_setting",
        "nextval",
        "setval",
        "query_to_xml",
        "query_to_xml_and_xmlschema",
        "table_to_xml",
        "cursor_to_xml",
        "database_to_xml",
        "schema_to_xml",
    }
)


class UnsafeQueryError(ToolError):
    """The SQL was rejected before execution; the message tells the model why."""


@dataclass(frozen=True, slots=True)
class ValidatedQuery:
    """A query that passed the guard, rendered from its validated AST."""

    sql: str
    tables: frozenset[str]


def _function_name(node: exp.Func) -> str:
    return (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()


def validate_select(sql: str, schema: str, allowed_tables: Collection[str]) -> ValidatedQuery:
    """Accept exactly one read-only query over `allowed_tables`, or raise UnsafeQueryError."""
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise UnsafeQueryError("The query is empty.")
    if len(sql) > MAX_SQL_CHARS:
        raise UnsafeQueryError(f"The query is too long (max {MAX_SQL_CHARS} characters).")

    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except ParseError as exc:
        first = str(exc).splitlines()[0]
        raise UnsafeQueryError(f"Could not parse the SQL: {first}") from exc
    if len(statements) != 1:
        raise UnsafeQueryError("Send exactly one SQL statement (no semicolon-separated batch).")
    tree = statements[0]

    if not isinstance(tree, _ALLOWED_ROOTS):
        raise UnsafeQueryError("Only read-only SELECT queries (optionally WITH ...) are allowed.")
    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            kind = node.key.upper()
            raise UnsafeQueryError(f"{kind} is not allowed; only plain SELECT queries can run.")
        if isinstance(node, exp.Func):
            name = _function_name(node)
            if name in _FORBIDDEN_FUNCTIONS or name.startswith(_FORBIDDEN_FUNCTION_PREFIXES):
                raise UnsafeQueryError(f"Function {name}() is not allowed.")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    allowed = {t.lower() for t in allowed_tables}
    tables: set[str] = set()
    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            continue  # a set-returning function such as generate_series(); checked above
        name = table.name.lower()
        qualifier = table.db.lower()
        if table.catalog or (qualifier and qualifier != schema):
            raise UnsafeQueryError(
                f"Table {table.sql(dialect='postgres')} is not available. "
                f"Only tables in schema {schema!r} can be queried."
            )
        if not qualifier and name in cte_names:
            continue
        if name not in allowed:
            raise UnsafeQueryError(
                f"Unknown table {name!r}. Available tables: {', '.join(sorted(allowed))}."
            )
        tables.add(name)

    return ValidatedQuery(sql=tree.sql(dialect="postgres"), tables=frozenset(tables))
