"""Async SQLAlchemy engine and session factory, hardened against a flaky database.

Managed Postgres (Neon) suspends idle computes and drops idle connections, and
any network can blip. So: new connections are retried with backoff (a cold
start or failover answers on the second or third try); every wait is bounded
(connect, checkout from the pool, each statement); pooled connections are
checked before use and recycled before the server would cut them.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import asyncpg
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Failures that a later attempt can fix: refused/reset/unreachable (OSError,
# which includes TimeoutError), "the database system is starting up", and
# "too many connections". Auth and unknown-database errors are not retried.
RETRYABLE_CONNECT_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    asyncpg.CannotConnectNowError,
    asyncpg.TooManyConnectionsError,
)
MAX_BACKOFF_S = 4.0

Connect = Callable[..., Awaitable[Any]]


async def connect_with_retry(
    connect: Connect,
    kwargs: dict[str, Any],
    retries: int,
    base_delay_s: float = 0.5,
) -> Any:
    """Call `connect(**kwargs)`, retrying transient failures with exponential backoff."""
    for attempt in range(retries + 1):
        try:
            return await connect(**kwargs)
        except RETRYABLE_CONNECT_ERRORS as exc:
            if attempt == retries:
                raise
            delay = min(base_delay_s * 2**attempt, MAX_BACKOFF_S)
            logger.warning(
                "Database connection failed (%s: %s); retry %d/%d in %.1fs",
                type(exc).__name__,
                exc,
                attempt + 1,
                retries,
                delay,
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


def asyncpg_connect_kwargs(url: str, settings: Settings) -> dict[str, Any]:
    """asyncpg.connect() arguments for a SQLAlchemy URL (as the asyncpg dialect builds them)."""
    parsed = make_url(url)
    kwargs: dict[str, Any] = parsed.translate_connect_args(username="user")
    kwargs.update(parsed.query)
    kwargs["timeout"] = settings.db_connect_timeout_s
    kwargs["command_timeout"] = settings.db_command_timeout_s
    return kwargs


def build_engine(settings: Settings) -> AsyncEngine:
    """The app's engine: retrying connects, bounded waits, recycled connections."""
    connect_kwargs = asyncpg_connect_kwargs(settings.database_url, settings)

    async def creator() -> Any:
        return await connect_with_retry(
            asyncpg.connect, connect_kwargs, retries=settings.db_connect_retries
        )

    return create_async_engine(
        settings.database_url,
        async_creator=creator,
        pool_pre_ping=True,  # replace connections the server closed while idle
        pool_recycle=settings.db_pool_recycle_s,
        pool_timeout=settings.db_pool_timeout_s,
    )


engine: AsyncEngine = build_engine(get_settings())

SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yield one session per request and always close it."""
    async with SessionLocal() as session:
        yield session
