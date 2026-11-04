"""Database connection resilience: retrying connects and bounded waits."""

from typing import Any

import asyncpg
import pytest

from app.core.config import Settings
from app.db.session import asyncpg_connect_kwargs, connect_with_retry


class FlakyConnect:
    """Fails with each of `errors` in turn, then returns a fake connection."""

    def __init__(self, *errors: BaseException) -> None:
        self.errors = list(errors)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return "connection"


@pytest.mark.parametrize(
    "error",
    [
        ConnectionRefusedError("refused"),
        TimeoutError(),
        asyncpg.CannotConnectNowError("the database system is starting up"),
        asyncpg.TooManyConnectionsError("too many connections"),
    ],
    ids=["refused", "timeout", "starting-up", "too-many"],
)
async def test_transient_connect_failures_are_retried(error: BaseException) -> None:
    connect = FlakyConnect(error, error)

    result = await connect_with_retry(connect, {"host": "db"}, retries=3, base_delay_s=0)

    assert result == "connection"
    assert len(connect.calls) == 3


async def test_gives_up_after_the_last_retry() -> None:
    connect = FlakyConnect(*(ConnectionResetError("reset") for _ in range(3)))

    with pytest.raises(ConnectionResetError):
        await connect_with_retry(connect, {}, retries=2, base_delay_s=0)
    assert len(connect.calls) == 3


async def test_a_wrong_password_is_not_retried() -> None:
    connect = FlakyConnect(asyncpg.InvalidPasswordError("bad password"))

    with pytest.raises(asyncpg.InvalidPasswordError):
        await connect_with_retry(connect, {}, retries=3, base_delay_s=0)
    assert len(connect.calls) == 1


def test_connect_kwargs_keep_ssl_and_add_timeouts() -> None:
    settings = Settings(
        database_url="postgresql://u:p@ep-x.neon.tech/neondb?sslmode=require&channel_binding=require",
        db_connect_timeout_s=7,
        db_command_timeout_s=20,
    )

    kwargs = asyncpg_connect_kwargs(settings.database_url, settings)

    assert kwargs == {
        "host": "ep-x.neon.tech",
        "user": "u",
        "password": "p",
        "database": "neondb",
        "ssl": "require",
        "timeout": 7,
        "command_timeout": 20,
    }
