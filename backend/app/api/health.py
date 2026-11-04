"""Health checks: liveness (is the process up?) and readiness (can it do its job?).

They answer different questions, and mixing them causes outages of their own:
a platform that restarts containers whenever the database blips turns a short
database hiccup into a fleet of cold starts. So:

- GET /api/health/live: the process answers. No dependencies are touched. Point
  the platform's restart/health check here.
- GET /api/health/ready (and /api/health): every dependency, each with its own
  status. 503 only when the database (without which nothing works) is down; a
  sick LLM or web search is reported as "degraded", because the rest of the app
  (history, documents, memories) still works.

Checks are cheap and bounded: the database gets a few seconds, and the LLM is
never called (that would cost tokens on every probe); its circuit breaker
already knows whether it is healthy.
"""

import asyncio
import logging
import time
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.session import get_session
from app.llm.base import LLMError, LLMProvider
from app.llm.factory import get_llm_provider
from app.llm.resilience import ResilientProvider
from app.rag.embeddings import EmbeddingProvider, get_embedder
from app.web.search import web_search_enabled

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

ComponentStatus = Literal["ok", "degraded", "down", "disabled"]


class ComponentHealth(BaseModel):
    """One dependency. `detail` says what is wrong (or what is in use), for humans."""

    name: str
    status: ComponentStatus
    detail: str | None = None
    latency_ms: int | None = None


class LiveResponse(BaseModel):
    """The process is up."""

    status: Literal["ok"] = "ok"


class HealthResponse(BaseModel):
    """Overall status plus each dependency.

    `status`: ok (all fine), degraded (works, with something impaired), or down
    (the database is unreachable: nothing works). `database` and `pgvector` are
    kept for older clients.
    """

    status: Literal["ok", "degraded", "down"]
    database: Literal["ok", "unreachable"]
    pgvector: str | None
    components: list[ComponentHealth]


def llm_provider_or_error() -> LLMProvider | LLMError:
    """The provider, or why it cannot be built (e.g. no API key), without failing the probe."""
    try:
        return get_llm_provider()
    except LLMError as exc:
        return exc


async def check_database(
    session: AsyncSession, timeout_s: float
) -> tuple[ComponentHealth, str | None]:
    """Query Postgres (and the pgvector version) within `timeout_s`."""
    started = time.perf_counter()
    try:
        async with asyncio.timeout(timeout_s):
            result = await session.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
            version = result.scalar_one_or_none()
    except TimeoutError:
        logger.warning("Health check: database did not answer within %gs", timeout_s)
        return ComponentHealth(
            name="database", status="down", detail=f"no answer within {timeout_s:g}s"
        ), None
    except Exception as exc:
        logger.warning("Health check: database unreachable: %r", exc)
        return ComponentHealth(
            name="database", status="down", detail=f"unreachable ({type(exc).__name__})"
        ), None
    latency_ms = round((time.perf_counter() - started) * 1000)
    if version is None:
        return ComponentHealth(
            name="database",
            status="degraded",
            detail="pgvector extension missing (run migrations)",
            latency_ms=latency_ms,
        ), None
    return ComponentHealth(name="database", status="ok", latency_ms=latency_ms), version


def check_llm(provider: LLMProvider | LLMError) -> ComponentHealth:
    """Report the LLM from its circuit breaker(s), without calling it."""
    if isinstance(provider, LLMError):
        return ComponentHealth(name="llm", status="down", detail=str(provider))
    if not isinstance(provider, ResilientProvider):
        return ComponentHealth(name="llm", status="ok", detail=provider.model)

    primary = provider.primary_breaker
    fallback = provider.fallback_breaker
    if primary.state == "closed":
        return ComponentHealth(name="llm", status="ok", detail=provider.model)
    if primary.state == "half_open":
        return ComponentHealth(
            name="llm", status="degraded", detail=f"{provider.model} recovering (probing)"
        )
    if fallback is not None and fallback.state != "open":
        return ComponentHealth(
            name="llm",
            status="degraded",
            detail=f"{provider.model} failing; answering with {provider.fallback_model}",
        )
    return ComponentHealth(
        name="llm",
        status="down",
        detail=f"{provider.model} failing; requests paused for {primary.retry_in_s:.0f}s",
    )


def check_embeddings(embedder: EmbeddingProvider, expected_dim: int) -> ComponentHealth:
    """The model loads at startup; report it, and catch a dimension mismatch."""
    if embedder.dimension != expected_dim:
        return ComponentHealth(
            name="embeddings",
            status="down",
            detail=f"{embedder.model} gives {embedder.dimension} dims, expected {expected_dim}",
        )
    return ComponentHealth(name="embeddings", status="ok", detail=embedder.model)


def check_web_search(settings: Settings) -> ComponentHealth:
    """Configured or not. (Not probed: every Tavily call costs a credit.)"""
    if not web_search_enabled(settings):
        return ComponentHealth(name="web_search", status="disabled", detail="no API key")
    return ComponentHealth(name="web_search", status="ok", detail=settings.web_search_provider)


def overall_status(components: list[ComponentHealth]) -> Literal["ok", "degraded", "down"]:
    """down if the database is down; degraded if anything else is impaired; else ok."""
    if any(c.name == "database" and c.status == "down" for c in components):
        return "down"
    if any(c.status in ("degraded", "down") for c in components):
        return "degraded"
    return "ok"


@router.get("/health/live", response_model=LiveResponse)
async def live() -> LiveResponse:
    """Liveness: the process is up and serving. Touches no dependency."""
    return LiveResponse()


@router.get("/health/ready", response_model=HealthResponse)
@router.get("/health", response_model=HealthResponse)
async def ready(
    response: Response,
    session: Annotated[AsyncSession, Depends(get_session)],
    provider: Annotated[LLMProvider | LLMError, Depends(llm_provider_or_error)],
    embedder: Annotated[EmbeddingProvider, Depends(get_embedder)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    """Readiness: each dependency's status. 503 when the database is down."""
    database, pgvector = await check_database(session, settings.health_db_timeout_s)
    components = [
        database,
        check_llm(provider),
        check_embeddings(embedder, settings.embedding_dim),
        check_web_search(settings),
    ]
    overall = overall_status(components)
    if overall == "down":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status=overall,
        database="unreachable" if database.status == "down" else "ok",
        pgvector=pgvector,
        components=components,
    )
