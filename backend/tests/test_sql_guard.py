"""Unit tests for the SQL guard (no database needed)."""

import pytest

from app.sql.guard import UnsafeQueryError, validate_select

TABLES = ["employees", "departments", "orders"]


def check(sql: str) -> str:
    return validate_select(sql, "acme", TABLES).sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(*) FROM employees",
        "select e.first_name, d.name from acme.employees e join departments d "
        "on d.id = e.department_id order by 1 limit 5;",
        "WITH big AS (SELECT department_id, avg(salary) a FROM employees GROUP BY 1) "
        "SELECT * FROM big WHERE a > 60000",
        "SELECT id FROM orders UNION SELECT id FROM employees",
        "SELECT date_trunc('month', order_date)::date, count(*) FROM orders "
        "WHERE status <> 'cancelled' AND order_date >= '2026-01-01' GROUP BY 1",
        "SELECT n FROM generate_series(1, 3) AS g(n)",
    ],
)
def test_accepts_read_only_queries(sql: str) -> None:
    assert check(sql)


@pytest.mark.parametrize(
    ("sql", "message"),
    [
        ("", "empty"),
        ("SELECT 1; DROP TABLE employees", "exactly one"),
        ("DELETE FROM employees", "Only read-only SELECT"),
        ("UPDATE employees SET salary = 0", "Only read-only SELECT"),
        ("INSERT INTO employees (id) VALUES (1)", "Only read-only SELECT"),
        ("DROP TABLE employees", "Only read-only SELECT"),
        ("SET ROLE postgres", "Only read-only SELECT"),
        ("COPY employees TO '/tmp/x'", "Only read-only SELECT"),
        ("WITH d AS (DELETE FROM employees RETURNING *) SELECT * FROM d", "DELETE"),
        ("SELECT * INTO stolen FROM employees", "INTO"),
        ("SELECT * FROM employees FOR UPDATE", "LOCK"),
        ("SELECT pg_sleep(10)", "pg_sleep"),
        ("SELECT pg_read_file('/etc/passwd')", "pg_read_file"),
        ("SELECT query_to_xml('delete from employees', true, true, '')", "query_to_xml"),
        ("SELECT set_config('statement_timeout', '0', false)", "set_config"),
        ("SELECT * FROM conversations", "Unknown table 'conversations'"),
        ("SELECT * FROM public.messages", "schema 'acme'"),
        ("SELECT * FROM pg_catalog.pg_user", "schema 'acme'"),
        ("SELECT * FROM information_schema.tables", "schema 'acme'"),
        ("SELECT FROM WHERE", "parse"),
    ],
)
def test_rejects_unsafe_or_invalid_queries(sql: str, message: str) -> None:
    with pytest.raises(UnsafeQueryError, match=message):
        check(sql)


def test_cte_names_are_not_mistaken_for_tables() -> None:
    query = validate_select(
        "WITH recent AS (SELECT * FROM orders) SELECT count(*) FROM recent", "acme", TABLES
    )

    assert query.tables == {"orders"}


def test_executes_the_sql_it_validated() -> None:
    # The output is re-rendered from the checked AST, normalized and without the semicolon.
    assert check("select  id from employees ;") == "SELECT id FROM employees"
