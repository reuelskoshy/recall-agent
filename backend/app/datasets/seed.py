"""Provision the read-only SQL role and load the sample dataset.

Run with `python -m app.datasets.seed` (docker compose does this on start).
Idempotent: the schema is tagged with the dataset version and rebuilt only
when that changes (or with `--force`); grants are re-applied every time.

The dataset is managed here rather than in Alembic migrations on purpose: it
is demo content in its own schema, not application state, and it is rebuilt
from code instead of migrated in place.
"""

import argparse
import asyncio
import logging

from sqlalchemy import insert, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from app.core.config import Settings, get_settings
from app.datasets.acme import ACME, Dataset

logger = logging.getLogger(__name__)


def _literal(value: str) -> str:
    """Quote a string as a SQL literal (DDL such as CREATE ROLE cannot take bind params)."""
    return "'" + value.replace("'", "''") + "'"


async def provision_readonly_role(conn: AsyncConnection, settings: Settings) -> None:
    """Create or update the login role used by the SQL tool.

    The role is the real security boundary of query_database: it owns nothing,
    cannot create anything, and is only granted SELECT on the dataset schema
    (see `grant_readonly`), so it cannot read app tables such as conversations.
    Defaults make every session read-only with a statement timeout.
    """
    role = settings.readonly_db_user  # validated as a plain identifier in Settings
    exists = await conn.scalar(text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": role})
    verb = "ALTER" if exists else "CREATE"
    await conn.execute(
        text(
            f"{verb} ROLE {role} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION "
            f"PASSWORD {_literal(settings.readonly_db_password)}"
        )
    )
    await conn.execute(text(f"ALTER ROLE {role} SET default_transaction_read_only = on"))
    await conn.execute(
        text(f"ALTER ROLE {role} SET statement_timeout = {settings.sql_statement_timeout_ms}")
    )


async def grant_readonly(conn: AsyncConnection, dataset: Dataset, role: str) -> None:
    """Allow `role` to read the dataset schema and nothing else."""
    schema = dataset.metadata.schema
    await conn.execute(text(f"GRANT USAGE ON SCHEMA {schema} TO {role}"))
    await conn.execute(text(f"GRANT SELECT ON ALL TABLES IN SCHEMA {schema} TO {role}"))
    # Belt and braces: app tables in `public` were never granted, but make it explicit.
    await conn.execute(text(f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {role}"))
    await conn.execute(text(f"REVOKE CREATE ON SCHEMA public FROM {role}"))


async def _installed_version(conn: AsyncConnection, schema: str) -> str | None:
    version: str | None = await conn.scalar(
        text("SELECT obj_description(oid, 'pg_namespace') FROM pg_namespace WHERE nspname = :s"),
        {"s": schema},
    )
    return version


async def load_dataset(conn: AsyncConnection, dataset: Dataset) -> None:
    """Drop and recreate the dataset schema, then insert all rows."""
    schema = dataset.metadata.schema
    await conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
    await conn.execute(text(f"CREATE SCHEMA {schema}"))
    await conn.run_sync(dataset.metadata.create_all)
    for table in dataset.metadata.sorted_tables:  # foreign-key order
        rows = dataset.rows.get(table.name, [])
        if rows:
            await conn.execute(insert(table), rows)
    await conn.execute(text(f"COMMENT ON SCHEMA {schema} IS {_literal(dataset.version)}"))


async def seed(
    engine: AsyncEngine, settings: Settings, dataset: Dataset = ACME, force: bool = False
) -> bool:
    """Provision the role and (re)load the dataset if needed. Returns True if data was loaded."""
    schema = dataset.metadata.schema
    assert schema, "datasets live in their own schema"
    async with engine.begin() as conn:
        await provision_readonly_role(conn, settings)
        loaded = force or await _installed_version(conn, schema) != dataset.version
        if loaded:
            await load_dataset(conn, dataset)
        await grant_readonly(conn, dataset, settings.readonly_db_user)
    return loaded


async def _main(force: bool) -> None:
    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    try:
        loaded = await seed(engine, settings, force=force)
    finally:
        await engine.dispose()
    rows = sum(len(r) for r in ACME.rows.values())
    print(
        f"Dataset {ACME.version}: loaded {rows} rows into schema '{ACME.metadata.schema}'."
        if loaded
        else f"Dataset {ACME.version} already loaded; role and grants refreshed."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="Rebuild even if up to date.")
    asyncio.run(_main(parser.parse_args().force))
