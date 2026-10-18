"""Provisioning the read-only role on managed Postgres, where the app is not a superuser."""

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.config import Settings
from app.datasets.seed import provision_readonly_role
from app.db.session import engine

pytestmark = pytest.mark.usefixtures("db")

OWNER = "recall_test_owner"  # like Neon's neondb_owner: CREATEROLE + CREATEDB, not superuser
READER = "recall_test_reader"


async def _drop_roles() -> None:
    async with engine.begin() as conn:
        for role in (READER, OWNER):
            await conn.execute(text(f"DROP ROLE IF EXISTS {role}"))


@pytest.fixture
async def owner_engine() -> AsyncIterator[AsyncEngine]:
    """An engine connected as a non-superuser that may create roles."""
    await _drop_roles()
    async with engine.begin() as conn:
        await conn.execute(
            text(f"CREATE ROLE {OWNER} LOGIN CREATEROLE CREATEDB PASSWORD 'owner-pass'")
        )
    url = engine.url.set(username=OWNER, password="owner-pass")
    owner = create_async_engine(url)
    yield owner
    await owner.dispose()
    await _drop_roles()


def settings(password: str) -> Settings:
    return Settings(readonly_db_user=READER, readonly_db_password=password)


async def test_non_superuser_can_create_then_update_the_role(owner_engine: AsyncEngine) -> None:
    # The first boot creates the role; every later boot updates it. Both must work
    # without superuser (Postgres 16+ rejects ALTER ROLE ... NOSUPERUSER otherwise).
    for password in ("first-pass", "rotated-pass"):
        async with owner_engine.begin() as conn:
            await provision_readonly_role(conn, settings(password))

    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT rolcanlogin, rolsuper, rolcreatedb, rolcreaterole, rolconfig "
                    "FROM pg_roles WHERE rolname = :r"
                ),
                {"r": READER},
            )
        ).one()
    assert row.rolcanlogin
    assert not (row.rolsuper or row.rolcreatedb or row.rolcreaterole)
    assert "default_transaction_read_only=on" in row.rolconfig

    reader = create_async_engine(engine.url.set(username=READER, password="rotated-pass"))
    try:
        async with reader.connect() as conn:
            assert await conn.scalar(text("SELECT current_user")) == READER
    finally:
        await reader.dispose()


async def test_refuses_a_reader_role_with_privileges() -> None:
    await _drop_roles()
    async with engine.begin() as conn:
        await conn.execute(text(f"CREATE ROLE {READER} LOGIN CREATEDB"))
    try:
        with pytest.raises(RuntimeError, match="rolcreatedb"):
            async with engine.begin() as conn:
                await provision_readonly_role(conn, settings("pass"))
    finally:
        await _drop_roles()
