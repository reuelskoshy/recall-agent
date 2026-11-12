"""What every job source shares: the feed interface, its errors, and a careful HTTP GET.

A *feed* is the unit of isolation: one company's board, one HN thread, one
search query. The ingestion run fetches each feed under its own deadline, so a
board that times out or returns garbage costs that feed, never the whole run.

Retry policy (the same reasoning as `app/web/search.py`): only failures that a
second attempt can plausibly fix are retried (connection errors, timeouts, 429,
5xx), a bounded number of times with exponential backoff. A 404 or a malformed
body fails at once: retrying would return the same answer.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html import unescape
from typing import Any

import httpx
from selectolax.lexbor import LexborHTMLParser

from app.jobs.dedup import RawListing
from app.web.fetch import BLOCK_END, normalize_lines

logger = logging.getLogger(__name__)

RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_AFTER_S = 30.0
"""Never wait longer than this for a Retry-After header: the feed has its own deadline."""

Sleep = Callable[[float], Awaitable[None]]


class SourceError(Exception):
    """A feed failed. The message is stored on the run, so it names the feed and the
    failure only: never a search query, a profile value, or a response body."""


@dataclass(frozen=True, slots=True)
class FeedResult:
    """What one feed returned."""

    listings: list[RawListing]
    skipped: int = 0
    """Items present in the response but unusable (missing id, title or text)."""
    complete: bool = False
    """True when `listings` is the whole board: anything previously seen on this
    feed and missing now has been taken down, and can be marked closed."""


@dataclass(slots=True)
class HttpPolicy:
    """Bounded retries for one feed's requests."""

    max_attempts: int = 3
    backoff_s: float = 1.0
    sleep: Sleep = field(default=asyncio.sleep)


class Feed(ABC):
    """One unit of fetching. Subclasses implement `fetch`."""

    source: str
    """The `jobs.source` value its listings are stored under."""

    def __init__(self, key: str, company_id: Any = None) -> None:
        self.key = key
        """Stable, non-private label for counts and errors, e.g. "greenhouse:acme"."""
        self.company_id = company_id
        """The watched company this feed belongs to, if any."""

    @abstractmethod
    async def fetch(self, client: httpx.AsyncClient, policy: HttpPolicy) -> FeedResult:
        """Return the feed's listings, or raise SourceError."""


async def get_json(
    client: httpx.AsyncClient,
    url: str,
    policy: HttpPolicy,
    *,
    params: dict[str, str] | None = None,
    label: str,
) -> Any:
    """GET `url` and decode JSON, retrying transient failures only.

    `label` names the feed in error messages (the URL could carry a query).
    """
    for attempt in range(1, policy.max_attempts + 1):
        delay = policy.backoff_s * 2 ** (attempt - 1)
        try:
            response = await client.get(url, params=params)
        except httpx.TimeoutException as exc:
            reason = f"{label}: timed out ({type(exc).__name__})"
        except httpx.TransportError as exc:
            reason = f"{label}: unreachable ({type(exc).__name__})"
        else:
            if response.status_code < 400:
                try:
                    return response.json()
                except ValueError as exc:
                    raise SourceError(f"{label}: response is not valid JSON") from exc
            if response.status_code not in RETRYABLE_STATUSES:
                raise SourceError(f"{label}: HTTP {response.status_code}")
            reason = f"{label}: HTTP {response.status_code}"
            delay = max(delay, _retry_after(response))
        if attempt == policy.max_attempts:
            raise SourceError(f"{reason} after {attempt} attempts")
        logger.info("%s; retry %d in %.1fs", reason, attempt, delay)
        await policy.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def _retry_after(response: httpx.Response) -> float:
    """Seconds from a Retry-After header (seconds form only), capped; 0 if absent."""
    try:
        return min(float(response.headers.get("Retry-After", "0")), MAX_RETRY_AFTER_S)
    except ValueError:
        return 0.0


class BoardFeed(Feed):
    """A watched company's public ATS board (Greenhouse, Lever, Ashby).

    Boards return every open posting, so the result is `complete`: a posting
    that disappears was taken down and its job is marked closed.
    """

    def __init__(self, board_token: str, *, company_id: Any = None, company_name: str = "") -> None:
        if not board_token:
            raise ValueError("a board feed needs a board token")
        super().__init__(key=f"{self.source}:{board_token}", company_id=company_id)
        self.board_token = board_token
        self.company_name = company_name


def html_to_text(html: str) -> str:
    """Readable text from a posting's HTML: paragraphs and list items on their own lines."""
    marked = BLOCK_END.sub(r"\1\n", unescape(html) if "&lt;" in html else html)
    body = LexborHTMLParser(f"<body>{marked}</body>").body
    return normalize_lines(body.text(separator="") if body else "")


def parse_time(value: Any) -> datetime | None:
    """ISO-8601 strings and epoch seconds/milliseconds; None for anything else."""
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, int | float):
            seconds = value / 1000 if value > 10**11 else value
            return datetime.fromtimestamp(seconds, tz=UTC)
        if isinstance(value, str) and value:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except (ValueError, OverflowError, OSError):
        return None
    return None


def require_list(data: Any, key: str | None, label: str) -> list[Any]:
    """The list of postings in a response, or SourceError if the shape is wrong.

    `key` names the field holding the list (`{"jobs": [...]}`); None means the
    response itself is the list (Lever).
    """
    if key is not None:
        data = data.get(key) if isinstance(data, dict) else None
    if not isinstance(data, list):
        raise SourceError(f"{label}: unexpected response shape")
    return data
