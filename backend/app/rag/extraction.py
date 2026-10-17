"""Text extraction from uploaded files."""

import io
from pathlib import PurePath

from pypdf import PdfReader
from pypdf.errors import PdfReadError

SUPPORTED_EXTENSIONS = {".pdf": "application/pdf", ".txt": "text/plain", ".md": "text/markdown"}

Page = tuple[int | None, str]
"""(1-based page number, or None for formats without pages; text)."""


class ExtractionError(Exception):
    """The file could not be turned into text."""


def file_extension(filename: str) -> str:
    """Lower-case extension including the dot, e.g. '.pdf'."""
    return PurePath(filename).suffix.lower()


def extract_pages(filename: str, data: bytes) -> list[Page]:
    """Extract text per page. Raises ExtractionError for unreadable or text-less files."""
    extension = file_extension(filename)
    if extension == ".pdf":
        pages = _extract_pdf(data)
    elif extension in (".txt", ".md"):
        pages = [(None, _decode(data))]
    else:
        raise ExtractionError(f"Unsupported file type {extension or '(none)'!r}")

    if not any(text.strip() for _, text in pages):
        raise ExtractionError(
            "No extractable text found. Scanned/image-only PDFs need OCR, which is not supported."
        )
    return pages


def _extract_pdf(data: bytes) -> list[Page]:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise ExtractionError("Encrypted PDFs are not supported.")
        return [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, 1)]
    except PdfReadError as exc:
        raise ExtractionError(f"Could not read PDF: {exc}") from exc


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Legacy encodings: keep going rather than reject the file.
        return data.decode("latin-1")
