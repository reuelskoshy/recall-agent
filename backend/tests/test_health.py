"""Tests for GET /api/health."""

from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy.exc import OperationalError

from app.db.session import get_session


class _BrokenSession:
    """Stand-in session whose queries fail the way a lost DB connection does."""

    async def execute(self, *_args: object, **_kwargs: object) -> None:
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))


async def test_health_returns_503_when_database_unreachable(
    app: FastAPI, client: AsyncClient
) -> None:
    async def broken_session() -> AsyncIterator[_BrokenSession]:
        yield _BrokenSession()

    app.dependency_overrides[get_session] = broken_session

    response = await client.get("/api/health")

    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "database": "unreachable", "pgvector": None}


@pytest.mark.usefixtures("require_db")
async def test_health_ok_with_pgvector(client: AsyncClient) -> None:
    response = await client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["pgvector"]  # e.g. "0.8.0"; requires `alembic upgrade head`
