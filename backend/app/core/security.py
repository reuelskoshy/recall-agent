"""Protection for a public demo: a shared access password and LLM cost limits.

This is a demo gate, not user accounts. One password (DEMO_PASSWORD) is shared
with the people you send the link to; it is exchanged for a bearer token the
browser sends on every API call. The token is an HMAC of the password, so
changing DEMO_PASSWORD revokes every token at once, and no server-side session
store is needed. A header (not a cookie) keeps it working when a host embeds the
app in an iframe (e.g. Hugging Face Spaces), where third-party cookies are often blocked.

Rate limits protect the LLM bill: a sliding window per client IP, plus a global
daily cap. They are in-memory (one process), so they reset on restart, which is
acceptable for a demo; several replicas would need Redis instead.
"""

import hashlib
import hmac
import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from app.core.config import Settings, get_settings

TOKEN_CONTEXT = b"recall-demo-token-v1"


def make_token(password: str) -> str:
    """The bearer token for a password (deterministic, so it needs no storage)."""
    return hmac.new(password.encode(), TOKEN_CONTEXT, hashlib.sha256).hexdigest()


def password_matches(given: str, expected: str) -> bool:
    """Constant-time comparison, so response timing leaks nothing about the password."""
    return hmac.compare_digest(given.encode(), expected.encode())


def require_access(request: Request, settings: Annotated[Settings, Depends(get_settings)]) -> None:
    """Dependency: allow the request if no password is configured or the token is valid."""
    if not settings.demo_password:
        return
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    expected = make_token(settings.demo_password)
    if scheme.lower() != "bearer" or not hmac.compare_digest(token.encode(), expected.encode()):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "This demo is password-protected.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def client_ip(request: Request, trusted_hops: int = 0) -> str:
    """The caller's IP for rate limiting.

    Behind a reverse proxy the socket peer is the proxy, so the real client is
    in X-Forwarded-For. That header is client-controlled, though: each proxy
    *appends* the address it saw, so only the last `trusted_hops` entries were
    written by our own infrastructure. Reading the leftmost entry (a common
    mistake) would let anyone dodge the limit by sending a fake header.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if trusted_hops > 0 and forwarded:
        entries = [e.strip() for e in forwarded.split(",") if e.strip()]
        if entries:
            return entries[-min(trusted_hops, len(entries))]
    return request.client.host if request.client else "unknown"


class ChatRateLimiter:
    """Sliding-window limit per IP plus a global per-UTC-day cap."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._day = ""
        self._day_count = 0

    def reset(self) -> None:
        """Forget all recorded requests."""
        self._hits.clear()
        self._day, self._day_count = "", 0

    def check(self, ip: str, settings: Settings, now: float | None = None) -> None:
        """Record one chat turn for `ip`, or raise 429 if a limit is reached."""
        now = time.monotonic() if now is None else now
        today = datetime.now(UTC).date().isoformat()
        if today != self._day:
            self._day, self._day_count = today, 0
        if settings.daily_chat_limit and self._day_count >= settings.daily_chat_limit:
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "The demo has reached today's message limit. Please try again tomorrow.",
            )

        if settings.rate_limit_chat_per_ip:
            hits = self._hits[ip]
            while hits and now - hits[0] >= settings.rate_limit_window_s:
                hits.popleft()
            if len(hits) >= settings.rate_limit_chat_per_ip:
                retry = int(settings.rate_limit_window_s - (now - hits[0])) + 1
                raise HTTPException(
                    status.HTTP_429_TOO_MANY_REQUESTS,
                    f"Too many messages. Please wait {retry}s and try again.",
                    headers={"Retry-After": str(retry)},
                )
            hits.append(now)
        self._day_count += 1


chat_limiter = ChatRateLimiter()


def limit_chat(request: Request, settings: Annotated[Settings, Depends(get_settings)]) -> None:
    """Dependency for POST /chat: apply the chat rate limits."""
    chat_limiter.check(client_ip(request, settings.trusted_proxy_hops), settings)
