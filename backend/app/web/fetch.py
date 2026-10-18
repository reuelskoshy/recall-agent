"""Fetch a web page and extract its readable text."""

import re
from dataclasses import dataclass

import httpx
from selectolax.lexbor import LexborHTMLParser

from app.web.ssrf import BlockedAddressError, SafeTransport

USER_AGENT = "RecallAgent/1.0 (+https://github.com/reuelskoshy/recall-agent)"
MAX_REDIRECTS = 5
HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
TEXT_TYPES = HTML_TYPES | {"text/plain", "text/markdown", "application/json"}

# Page furniture that is never the content.
DROP_SELECTOR = (
    "script, style, noscript, template, svg, canvas, iframe, form, button, "
    "nav, header, footer, aside, [role=navigation], [aria-hidden=true]"
)
CONTENT_SELECTORS = ("main", "article", "[role=main]")
# Closing tags of block elements: a line break goes after each, so paragraphs
# stay separate while inline tags (<b>, <a>) stay inside their sentence.
BLOCK_END = re.compile(
    r"(?i)(<br\s*/?>|</(?:p|div|h[1-6]|li|tr|td|th|dd|dt|section|article|blockquote|pre|"
    r"table|ul|ol|figcaption)>)"
)


class FetchError(Exception):
    """The page could not be fetched or read; the message is safe to show the model."""


@dataclass(frozen=True, slots=True)
class FetchedPage:
    """The readable text of a page."""

    url: str
    """Final URL, after redirects."""
    title: str
    text: str
    content_type: str
    truncated: bool


def normalize_lines(text: str) -> str:
    """Collapse whitespace inside lines and drop empty lines."""
    lines = (" ".join(line.split()) for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def extract_text(html: str) -> tuple[str, str]:
    """(title, main text) of an HTML document, without scripts and page furniture."""
    tree = LexborHTMLParser(html)
    title_node = tree.css_first("title")
    title = " ".join(title_node.text().split()) if title_node else ""
    for node in tree.css(DROP_SELECTOR):
        node.decompose()
    root = next(
        (node for node in map(tree.css_first, CONTENT_SELECTORS) if node is not None), tree.body
    )
    if root is None:
        return title, ""
    marked = BLOCK_END.sub(r"\1\n", root.html or "")
    body = LexborHTMLParser(marked).body
    return title, normalize_lines(body.text(separator="") if body else "")


async def fetch_page(
    url: str,
    *,
    max_bytes: int,
    max_chars: int,
    timeout_s: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FetchedPage:
    """GET `url` (public addresses only) and return its readable text.

    Reads at most `max_bytes` of the body and returns at most `max_chars` of
    text; `truncated` says whether anything was cut.
    """
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL as exc:
        raise FetchError(f"invalid URL ({exc})") from exc
    if parsed.scheme not in ("http", "https") or not parsed.host:
        raise FetchError("only http and https URLs can be fetched")
    if parsed.userinfo:
        raise FetchError("URLs with embedded credentials are not allowed")

    body = bytearray()
    truncated = False
    async with httpx.AsyncClient(
        transport=transport or SafeTransport(),
        follow_redirects=True,
        max_redirects=MAX_REDIRECTS,
        timeout=timeout_s,
        trust_env=False,  # never route through an environment proxy
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain;q=0.9,*/*;q=0.5"},
    ) as client:
        try:
            async with client.stream("GET", parsed) as response:
                if response.status_code >= 400:
                    raise FetchError(f"the server answered HTTP {response.status_code}")
                content_type = (
                    response.headers.get("content-type", "").split(";")[0].strip().lower()
                )
                if content_type and content_type not in TEXT_TYPES:
                    raise FetchError(f"unsupported content type {content_type!r} (not a web page)")
                async for chunk in response.aiter_bytes():
                    body += chunk
                    if len(body) >= max_bytes:
                        truncated = True
                        break
                encoding = response.encoding or "utf-8"
                final_url = str(response.url)
        except BlockedAddressError as exc:
            raise FetchError(f"blocked: {exc}") from exc
        except httpx.TooManyRedirects as exc:
            raise FetchError(f"more than {MAX_REDIRECTS} redirects") from exc
        except httpx.TimeoutException as exc:
            raise FetchError(f"timed out after {timeout_s:g}s") from exc
        except httpx.HTTPError as exc:
            raise FetchError(f"request failed ({type(exc).__name__})") from exc

    try:
        raw = bytes(body[:max_bytes]).decode(encoding, errors="replace")
    except LookupError:  # an unknown charset name in the headers
        raw = bytes(body[:max_bytes]).decode("utf-8", errors="replace")

    if content_type in HTML_TYPES or (not content_type and "<html" in raw[:2000].lower()):
        title, text = extract_text(raw)
    else:
        title, text = "", normalize_lines(raw)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]
        truncated = True
    return FetchedPage(
        url=final_url,
        title=title,
        text=text,
        content_type=content_type or "unknown",
        truncated=truncated,
    )
