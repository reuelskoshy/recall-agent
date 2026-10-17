"""Shared pytest fixtures.

DB-backed tests run against a separate `<db>_test` database (created and migrated
on first use), so they never touch development data.
"""

import asyncio
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.engine import make_url

from app.core.config import get_settings

# Point the app at the test database BEFORE anything creates the engine.
_dev_url = make_url(get_settings().database_url)
TEST_DB_NAME = f"{_dev_url.database}_test"
os.environ["DATABASE_URL"] = _dev_url.set(database=TEST_DB_NAME).render_as_string(
    hide_password=False
)
os.environ["ENVIRONMENT"] = "test"
# Never download/load the real embedding model in tests.
os.environ["EMBEDDING_PROVIDER"] = "fake"
get_settings.cache_clear()

import asyncpg  # noqa: E402
from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.db.session import engine  # noqa: E402
from app.main import create_app  # noqa: E402
from app.rag.embeddings import FakeEmbedder, get_embedder  # noqa: E402

BACKEND_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture
def app() -> FastAPI:
    """A fresh app per test, so dependency overrides never leak between tests."""
    app = create_app()
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    return app


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """HTTP client that calls the ASGI app in-process (no server needed)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def _create_test_database() -> None:
    admin = await asyncpg.connect(
        host=_dev_url.host,
        port=_dev_url.port,
        user=_dev_url.username,
        password=_dev_url.password,
        database=_dev_url.database,
    )
    try:
        exists = await admin.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", TEST_DB_NAME)
        if not exists:
            await admin.execute(f'CREATE DATABASE "{TEST_DB_NAME}"')
    finally:
        await admin.close()


@pytest.fixture(scope="session")
async def test_db() -> None:
    """Create and migrate the test database once per run.

    Skips DB tests when Postgres is down, unless REQUIRE_DB=1 (set in CI).
    """
    try:
        await _create_test_database()
    except (OSError, asyncpg.PostgresError) as exc:
        if os.getenv("REQUIRE_DB") == "1":
            raise
        pytest.skip(f"Postgres not reachable: {exc}")

    # Alembic's env.py calls asyncio.run(), which cannot nest inside this loop.
    config = Config(str(BACKEND_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    await asyncio.to_thread(command.upgrade, config, "head")


@pytest.fixture
async def db(test_db: None) -> AsyncIterator[None]:
    """Use the test database and wipe its data afterwards."""
    yield
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE conversations, documents CASCADE"))


def parse_sse(body: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse an SSE response body into (event name, JSON data) pairs."""
    events: list[tuple[str, dict[str, Any]]] = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        name, data_lines = "message", []
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if data_lines:
            events.append((name, json.loads("\n".join(data_lines))))
    return events
