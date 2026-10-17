"""Recursive text chunking.

Why chunk at all: embedding a whole document blurs many topics into one vector,
and retrieval would return far more text than the model needs. Small chunks are
precise but lose context; large chunks keep context but dilute the vector.

Defaults (1000 characters ~ 250 tokens, 150 overlap):
- well under BGE-small's 512-token input limit, so nothing is silently truncated;
- big enough to hold a full paragraph or a definition with its explanation;
- overlap means a sentence cut at a boundary still appears whole in one chunk.

The splitter prefers natural boundaries: paragraphs, then lines, then
sentences, then words, and only cuts mid-word as a last resort.

Structure first: Markdown headings split the text into sections *before*
size-based splitting, so unrelated sections never share a chunk (a chunk mixing
"sick leave" and "working hours" has a blurred embedding that matches neither
well). Each chunk is prefixed with its heading path, e.g.
"Handbook > Leave > Annual leave", so its vector carries the topic even when
the body never names it ("contextual chunk headers").
"""

import re
from dataclasses import dataclass

SEPARATORS = ("\n\n", "\n", ". ", " ", "")


_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$")


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown into (heading path, body) sections; text without headings is one section."""
    sections: list[tuple[str, str]] = []
    path: list[tuple[int, str]] = []  # (level, title) of the enclosing headings
    body: list[str] = []
    in_code = False

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            sections.append((" > ".join(title for _, title in path), content))
        body.clear()

    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_code = not in_code  # a "# comment" inside a code block is not a heading
        match = None if in_code else _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, match.group(2).strip()))
        else:
            body.append(line)
    flush()
    return sections


@dataclass(frozen=True, slots=True)
class Chunk:
    """A piece of a document, with where it came from."""

    content: str
    page: int | None
    index: int
    section: str | None = None
    """Innermost heading, used to cite documents that have no pages."""


def normalize_whitespace(text: str) -> str:
    """Tidy extracted text: unify newlines, trim line ends, collapse blank runs."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into chunks of at most `chunk_size` characters with ~`overlap` overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    text = text.strip()
    if not text:
        return []
    pieces = _split_recursive(text, SEPARATORS, chunk_size)
    return _merge(pieces, chunk_size, overlap)


def _split_recursive(text: str, separators: tuple[str, ...], chunk_size: int) -> list[str]:
    """Break text into pieces no longer than chunk_size, using the coarsest separator possible."""
    if len(text) <= chunk_size:
        return [text]
    separator = next((s for s in separators if s == "" or s in text), "")
    if separator == "":
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    remaining = separators[separators.index(separator) + 1 :]
    parts = text.split(separator)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        # Keep the separator attached so merged chunks read naturally.
        piece = part + separator if i < len(parts) - 1 else part
        if not piece:
            continue
        if len(piece) <= chunk_size:
            pieces.append(piece)
        else:
            pieces.extend(_split_recursive(piece, remaining, chunk_size))
    return pieces


def _merge(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack pieces into chunks, starting each new chunk with the previous tail."""
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for piece in pieces:
        if current and length + len(piece) > chunk_size:
            chunks.append("".join(current).strip())
            # Carry trailing pieces (up to `overlap` chars) into the next chunk.
            while current and (length > overlap or length + len(piece) > chunk_size):
                length -= len(current.pop(0))
        current.append(piece)
        length += len(piece)
    if current:
        chunks.append("".join(current).strip())
    return [c for c in chunks if c]


def chunk_pages(
    pages: list[tuple[int | None, str]], chunk_size: int = 1000, overlap: int = 150
) -> list[Chunk]:
    """Chunk each page and section separately, prefixing chunks with their heading path.

    Pages are never mixed, so every chunk maps to exactly one page (exact citations).
    """
    chunks: list[Chunk] = []
    for page, text in pages:
        for heading, body in split_sections(normalize_whitespace(text)):
            header = f"{heading}\n\n" if heading else ""
            # The header counts toward the size budget (it is embedded too).
            budget = max(chunk_size - len(header), overlap + 100)
            for piece in split_text(body, budget, min(overlap, budget - 1)):
                section = heading.split(" > ")[-1] if heading else None
                chunks.append(
                    Chunk(
                        content=header + piece,
                        page=page,
                        index=len(chunks),
                        section=section,
                    )
                )
    return chunks
