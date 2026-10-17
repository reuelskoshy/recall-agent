"""Health check endpoint used by Docker, CI, and the frontend."""

import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Service status. `pgvector` is the installed extension version, or None."""

    status: Literal["ok", "degraded"]
    database: Literal["ok", "unreachable"]
    pgvector: str | None


@router.get("/health", response_model=HealthResponse)
async def health(
    response: Response,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> HealthResponse:
    """Report whether the API can reach Postgres and whether pgvector is installed.

    Returns 503 when the database is unreachable, so orchestrators can detect it.
    """
    try:
        result = await session.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        )
        pgvector_version = result.scalar_one_or_none()
    except (SQLAlchemyError, OSError):
        logger.exception("Health check: database unreachable")
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="degraded", database="unreachable", pgvector=None)

    return HealthResponse(
        status="ok" if pgvector_version else "degraded",
        database="ok",
        pgvector=pgvector_version,
    )
