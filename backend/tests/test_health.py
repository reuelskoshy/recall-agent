"""Tests for the liveness and readiness endpoints."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy.exc import OperationalError

from app.api.health import llm_provider_or_error
from app.core.config import Settings, get_settings
from app.db.session import get_session
from app.llm.base import LLMError
from app.llm.fake import FakeProvider
from app.llm.resilience import ResilientProvider


class _BrokenSession:
    """Stand-in session whose queries fail the way a lost DB connection does."""

    async def execute(self, *_args: object, **_kwargs: object) -> None:
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))


class _HangingSession:
    """Stand-in session whose queries never return (a half-open TCP connection)."""

    async def execute(self, *_args: object, **_kwargs: object) -> None:
        await asyncio.sleep(3600)


def use_session(app: FastAPI, session: object) -> None:
    async def override() -> AsyncIterator[object]:
        yield session

    app.dependency_overrides[get_session] = override


def components(body: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["name"]: c for c in body["components"]}


def tripped(provider: ResilientProvider, breaker_name: str = "primary") -> None:
    breaker = provider.primary_breaker if breaker_name == "primary" else provider.fallback_breaker
    assert breaker is not None
    for _ in range(breaker.failure_threshold):
        breaker.record_failure()


async def test_liveness_ignores_dependencies(app: FastAPI, client: AsyncClient) -> None:
    use_session(app, _BrokenSession())

    response = await client.get("/api/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("path", ["/api/health", "/api/health/ready"])
async def test_readiness_is_503_when_the_database_is_unreachable(
    app: FastAPI, client: AsyncClient, path: str
) -> None:
    use_session(app, _BrokenSession())

    response = await client.get(path)

    assert response.status_code == 503
    body = response.json()
    assert (body["status"], body["database"], body["pgvector"]) == ("down", "unreachable", None)
    assert components(body)["database"] == {
        "name": "database",
        "status": "down",
        "detail": "unreachable (OperationalError)",
        "latency_ms": None,
    }


async def test_a_hanging_database_is_reported_down_within_the_timeout(
    app: FastAPI, client: AsyncClient
) -> None:
    use_session(app, _HangingSession())
    app.dependency_overrides[get_settings] = lambda: Settings(health_db_timeout_s=0.1)

    response = await asyncio.wait_for(client.get("/api/health/ready"), timeout=5)

    assert response.status_code == 503
    assert components(response.json())["database"]["detail"] == "no answer within 0.1s"


@pytest.mark.usefixtures("db")
async def test_ready_when_everything_is_up(app: FastAPI, client: AsyncClient) -> None:
    app.dependency_overrides[llm_provider_or_error] = lambda: ResilientProvider(FakeProvider())

    response = await client.get("/api/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["pgvector"]  # e.g. "0.8.0"; requires `alembic upgrade head`
    parts = components(body)
    assert parts["database"]["latency_ms"] is not None
    assert parts["llm"] == {
        "name": "llm",
        "status": "ok",
        "detail": "fake-model",
        "latency_ms": None,
    }
    assert parts["embeddings"]["status"] == "ok"
    assert parts["web_search"]["status"] == "disabled"  # no key in tests


@pytest.mark.usefixtures("db")
async def test_open_llm_breaker_is_degraded_not_down(app: FastAPI, client: AsyncClient) -> None:
    provider = ResilientProvider(FakeProvider())
    tripped(provider)
    app.dependency_overrides[llm_provider_or_error] = lambda: provider

    response = await client.get("/api/health/ready")

    # Still 200: history, documents and memories keep working without the LLM.
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    llm = components(body)["llm"]
    assert llm["status"] == "down"
    assert llm["detail"].startswith("fake-model failing; requests paused for")


@pytest.mark.usefixtures("db")
async def test_fallback_in_use_is_reported(app: FastAPI, client: AsyncClient) -> None:
    provider = ResilientProvider(FakeProvider(), FakeProvider(model="backup"))
    tripped(provider)
    app.dependency_overrides[llm_provider_or_error] = lambda: provider

    llm = components((await client.get("/api/health/ready")).json())["llm"]

    assert llm["status"] == "degraded"
    assert llm["detail"] == "fake-model failing; answering with backup"


@pytest.mark.usefixtures("db")
async def test_unbuildable_llm_is_reported_without_failing_the_probe(
    app: FastAPI, client: AsyncClient
) -> None:
    app.dependency_overrides[llm_provider_or_error] = lambda: LLMError("LLM_API_KEY is not set")

    response = await client.get("/api/health/ready")

    assert response.status_code == 200
    assert components(response.json())["llm"]["detail"] == "LLM_API_KEY is not set"
