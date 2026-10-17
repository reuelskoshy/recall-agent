"""Shared pytest fixtures."""

import os
from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from app.db.session import engine
from app.main import create_app


@pytest.fixture
def app() -> FastAPI:
    """A fresh app per test, so dependency overrides never leak between tests."""
    return create_app()


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """HTTP client that calls the ASGI app in-process (no server needed)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture(scope="session")
async def require_db() -> None:
    """Skip DB-backed tests when Postgres is down, unless REQUIRE_DB=1 (set in CI)."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        if os.getenv("REQUIRE_DB") == "1":
            raise
        pytest.skip(f"Postgres not reachable: {exc}")
