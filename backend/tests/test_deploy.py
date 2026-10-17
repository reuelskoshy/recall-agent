"""Public-demo protections: password gate, rate limits, provider URLs, static frontend."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings, get_settings
from app.core.security import ChatRateLimiter, make_token
from app.llm.factory import get_llm_provider
from app.llm.fake import FakeProvider
from app.main import create_app

pytestmark = pytest.mark.usefixtures("db")


def protect(app: FastAPI) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(demo_password="s3cret")


async def test_api_requires_the_demo_token_when_a_password_is_set(
    app: FastAPI, client: AsyncClient
) -> None:
    protect(app)

    assert (await client.get("/api/health")).status_code == 200  # platform health checks
    assert (await client.get("/api/conversations")).status_code == 401
    assert (await client.post("/api/chat", json={"message": "hi"})).status_code == 401
    assert (await client.get("/api/auth/status")).json() == {
        "required": True,
        "authenticated": False,
    }

    wrong = await client.post("/api/auth/login", json={"password": "guess"})
    assert wrong.status_code == 401
    token = (await client.post("/api/auth/login", json={"password": "s3cret"})).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    assert (await client.get("/api/conversations", headers=headers)).status_code == 200
    status = (await client.get("/api/auth/status", headers=headers)).json()
    assert status == {"required": True, "authenticated": True}


async def test_changing_the_password_revokes_old_tokens(app: FastAPI, client: AsyncClient) -> None:
    old = make_token("old-password")
    protect(app)

    response = await client.get("/api/memories", headers={"Authorization": f"Bearer {old}"})

    assert response.status_code == 401


async def test_no_password_means_open_access(client: AsyncClient) -> None:
    assert (await client.get("/api/conversations")).status_code == 200
    assert (await client.get("/api/auth/status")).json() == {
        "required": False,
        "authenticated": True,
    }


async def test_chat_is_rate_limited_per_ip(app: FastAPI, client: AsyncClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(rate_limit_chat_per_ip=2)
    app.dependency_overrides[get_llm_provider] = lambda: FakeProvider()

    statuses = [
        (await client.post("/api/chat", json={"message": f"m{i}"})).status_code for i in range(3)
    ]

    assert statuses == [200, 200, 429]


def test_rate_window_slides_and_daily_cap_applies() -> None:
    from fastapi import HTTPException

    limiter = ChatRateLimiter()
    settings = Settings(rate_limit_chat_per_ip=1, rate_limit_window_s=60, daily_chat_limit=3)

    limiter.check("1.1.1.1", settings, now=0)
    with pytest.raises(HTTPException, match="Too many"):
        limiter.check("1.1.1.1", settings, now=30)
    limiter.check("1.1.1.1", settings, now=61)  # window slid
    limiter.check("2.2.2.2", settings, now=61)  # other IPs are independent
    with pytest.raises(HTTPException, match="today's message limit"):
        limiter.check("3.3.3.3", settings, now=62)


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (
            "postgresql://u:p@ep-x.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require",
            "postgresql+asyncpg://u:p@ep-x.eu-central-1.aws.neon.tech/neondb?ssl=require",
        ),
        ("postgres://u:p@host:5432/db", "postgresql+asyncpg://u:p@host:5432/db"),
        ("postgresql+asyncpg://u:p@host/db", "postgresql+asyncpg://u:p@host/db"),
    ],
)
def test_provider_database_urls_are_adapted_for_asyncpg(given: str, expected: str) -> None:
    settings = Settings(database_url=given)

    assert settings.database_url == expected
    # The read-only URL keeps the host and TLS settings, with the reader's credentials.
    assert settings.readonly_url.startswith("postgresql+asyncpg://recall_reader:")


async def test_built_frontend_is_served_next_to_the_api(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>Recall</title>")
    get_settings.cache_clear()
    import os

    os.environ["STATIC_DIR"] = str(tmp_path)
    try:
        app = create_app()
    finally:
        del os.environ["STATIC_DIR"]
        get_settings.cache_clear()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        page = await c.get("/")
        health = await c.get("/api/health")

    assert page.status_code == 200
    assert "<title>Recall</title>" in page.text
    assert health.json()["status"] == "ok"


async def test_sample_documents_are_ingested_once(tmp_path: Path) -> None:
    from app.rag.seed_samples import seed_samples

    (tmp_path / "guide.md").write_text("# Guide\n\nThe office opens at nine.")
    (tmp_path / "ignore.exe").write_bytes(b"MZ")

    assert await seed_samples(tmp_path) == ["guide.md"]
    assert await seed_samples(tmp_path) == []  # idempotent across restarts


def test_client_ip_ignores_spoofed_forwarded_for_entries() -> None:
    from starlette.requests import Request

    from app.core.security import client_ip

    def request(forwarded: str | None) -> Request:
        headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded else []
        return Request({"type": "http", "headers": headers, "client": ("10.0.0.9", 1234)})

    # A client claims to be 1.2.3.4; the proxy appended the real address 5.6.7.8.
    assert client_ip(request("1.2.3.4, 5.6.7.8"), trusted_hops=1) == "5.6.7.8"
    assert client_ip(request("1.2.3.4, 5.6.7.8"), trusted_hops=0) == "10.0.0.9"
    assert client_ip(request(None), trusted_hops=1) == "10.0.0.9"
