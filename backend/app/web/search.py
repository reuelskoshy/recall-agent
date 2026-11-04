"""Web search providers behind one interface (like LLM providers and embedders)."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from app.core.config import Settings

logger = logging.getLogger(__name__)

# Tavily statuses worth one more try: a rate-limit blip or a server error. (432/433
# mean a plan or quota limit, which a retry within seconds cannot fix.)
RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})


class SearchError(Exception):
    """The search provider failed; the message is safe to show the model."""


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One web search result."""

    title: str
    url: str
    snippet: str
    score: float | None = None


class SearchProvider(ABC):
    """Turns a query into ranked web results."""

    name: str

    @abstractmethod
    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        """Return up to `max_results` hits, best first."""


class TavilySearch(SearchProvider):
    """Tavily (https://tavily.com): a search API built for agents, returning clean snippets."""

    name = "tavily"
    endpoint = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str,
        timeout_s: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
        retry_delay_s: float = 1.0,
    ) -> None:
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._transport = transport
        self._retry_delay_s = retry_delay_s

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        """POST /search with basic depth (1 credit per call), retrying one fast failure.

        Only failures that come back quickly (connection refused, 429, 5xx) are
        retried: a timed-out attempt has already used most of the tool's budget.
        """
        try:
            return await self._search_once(query, max_results)
        except _Retryable as exc:
            logger.info("Tavily failed (%s); retrying once", exc.reason)
            await asyncio.sleep(self._retry_delay_s)
        try:
            return await self._search_once(query, max_results)
        except _Retryable as exc:
            raise SearchError(exc.reason) from exc

    async def _search_once(self, query: str, max_results: int) -> list[SearchHit]:
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_s, transport=self._transport
            ) as client:
                response = await client.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={
                        "query": query,
                        "max_results": max_results,
                        "search_depth": "basic",
                        "include_answer": False,
                    },
                )
        except httpx.TimeoutException as exc:
            raise SearchError(f"Tavily timed out after {self._timeout_s:g}s") from exc
        except httpx.TransportError as exc:
            raise _Retryable(f"could not reach Tavily ({type(exc).__name__})") from exc
        except httpx.HTTPError as exc:
            raise SearchError(f"could not reach Tavily ({type(exc).__name__})") from exc

        if response.status_code in (401, 403):
            raise SearchError("Tavily rejected the API key (check WEB_SEARCH_API_KEY)")
        if response.status_code in (429, 432, 433):
            if response.status_code == 429:
                raise _Retryable("Tavily is rate-limiting requests; try again later")
            raise SearchError("the web search quota or rate limit was reached; try again later")
        if response.status_code in RETRYABLE_STATUSES:
            raise _Retryable(f"Tavily answered HTTP {response.status_code}")
        if response.status_code >= 400:
            raise SearchError(f"Tavily answered HTTP {response.status_code}")
        try:
            results = response.json().get("results") or []
        except ValueError as exc:
            raise SearchError("Tavily returned an unreadable response") from exc
        return [
            SearchHit(
                title=str(item.get("title") or item["url"]),
                url=str(item["url"]),
                snippet=str(item.get("content") or ""),
                score=item.get("score"),
            )
            for item in results[:max_results]
            if isinstance(item, dict) and item.get("url")
        ]


class _Retryable(Exception):
    """A failure worth one more attempt; becomes a SearchError if the retry fails too."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class FakeSearch(SearchProvider):
    """Deterministic results for tests and offline development."""

    name = "fake"

    async def search(self, query: str, max_results: int) -> list[SearchHit]:
        """Two canned hits that echo the query."""
        hits = [
            SearchHit(
                title=f"Result {n} for {query}",
                url=f"https://example.com/{n}",
                snippet=f"Snippet {n} about {query}.",
                score=1 - n / 10,
            )
            for n in (1, 2)
        ]
        return hits[:max_results]


def web_search_enabled(settings: Settings) -> bool:
    """web_search is offered only when a provider is usable (it has a key, or is the fake)."""
    return settings.web_search_provider == "fake" or bool(settings.web_search_api_key)


def build_search_provider(settings: Settings) -> SearchProvider:
    """The configured provider (see WEB_SEARCH_PROVIDER)."""
    if settings.web_search_provider == "fake":
        return FakeSearch()
    if not settings.web_search_api_key:
        raise SearchError("web search is not configured (WEB_SEARCH_API_KEY is empty)")
    return TavilySearch(settings.web_search_api_key, timeout_s=settings.web_timeout_s)
