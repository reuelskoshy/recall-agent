"""May we use this URL? A hard domain denylist, then the site's robots.txt.

Two different checks, on purpose:

- The **denylist** is policy. LinkedIn (and its link shortener) is never
  used, whatever its robots.txt says today. It cannot be configured away;
  `JOBS_DENIED_DOMAINS` can only add to it.
- **robots.txt** is the site's own permission. It is read with the standard
  library's `urllib.robotparser` (no new dependency), once per host per run.

Both fail closed: a URL we cannot parse, or a robots.txt we cannot read
(timeout, 5xx, 401/403), means "no". The one exception is the standard rule
that a missing robots.txt (404 and other 4xx) means everything is allowed.

robots.txt is fetched from hosts named in search results, i.e. chosen by a
third party, so it goes through the SSRF-safe transport like `fetch_url`.
"""

import logging
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from app.web.fetch import USER_AGENT

logger = logging.getLogger(__name__)

ALWAYS_DENIED = frozenset({"linkedin.com", "lnkd.in"})
ROBOTS_MAX_BYTES = 500_000
ROBOTS_AGENT = "RecallAgent"
"""The product token robots.txt rules are matched against (from USER_AGENT)."""


def host_of(url: str) -> str | None:
    """Lower-cased hostname of an http(s) URL, or None if it is not one."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    return parts.hostname.lower().rstrip(".")


def is_denied_domain(url: str, extra: frozenset[str] = frozenset()) -> bool:
    """True if the URL's host is a denied domain or any subdomain of one.

    Unparseable URLs are denied (fail closed). "in.linkedin.com" matches
    "linkedin.com"; "notlinkedin.com" does not.
    """
    host = host_of(url)
    if host is None:
        return True
    return any(host == domain or host.endswith("." + domain) for domain in ALWAYS_DENIED | extra)


class RobotsChecker:
    """Answers `allowed(url)` from each host's robots.txt, fetched once and cached.

    Make one per run: robots.txt can change, and a per-run cache needs no expiry.
    """

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client
        self._cache: dict[str, RobotFileParser | bool] = {}

    async def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        host = host_of(url)
        if host is None:
            return False
        origin = f"{parts.scheme}://{parts.netloc}"
        rules = self._cache.get(origin)
        if rules is None:
            rules = await self._load(origin)
            self._cache[origin] = rules
        if isinstance(rules, bool):
            return rules
        return rules.can_fetch(ROBOTS_AGENT, url)

    async def _load(self, origin: str) -> RobotFileParser | bool:
        """Parsed rules, or a blanket True (no robots.txt) / False (unreadable)."""
        try:
            response = await self._client.get(
                f"{origin}/robots.txt", headers={"User-Agent": USER_AGENT}
            )
        except Exception as exc:  # timeout, refused, blocked private address, ...
            logger.info("robots.txt for %s unreadable (%s): denying", origin, type(exc).__name__)
            return False
        status = response.status_code
        if status in (401, 403) or status >= 500:
            logger.info("robots.txt for %s answered HTTP %d: denying", origin, status)
            return False
        if status >= 400:
            return True  # no robots.txt: no restrictions
        parser = RobotFileParser()
        parser.parse(response.text[:ROBOTS_MAX_BYTES].splitlines())
        return parser
