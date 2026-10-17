"""Unit tests for chunking, text extraction, and embedders."""

import itertools
import math

import pytest

from app.rag.chunking import chunk_pages, normalize_whitespace, split_sections, split_text
from app.rag.embeddings import FakeEmbedder, check_dimension
from app.rag.extraction import ExtractionError, extract_pages
from tests.pdf_factory import make_pdf

PARAGRAPH = (
    "Retrieval augmented generation grounds a language model in external documents. "
    "The retriever finds relevant passages and the generator answers from them. "
)


def test_short_text_is_a_single_chunk() -> None:
    assert split_text("Hello world.", chunk_size=100, overlap=10) == ["Hello world."]


def test_empty_text_has_no_chunks() -> None:
    assert split_text("   \n\n ", chunk_size=100, overlap=10) == []


def test_chunks_respect_size_and_cover_all_text() -> None:
    text = "\n\n".join(PARAGRAPH * 3 for _ in range(10))

    chunks = split_text(text, chunk_size=300, overlap=50)

    assert len(chunks) > 1
    assert all(0 < len(c) <= 300 for c in chunks)
    # Every sentence of the source survives somewhere.
    for sentence in ("grounds a language model", "generator answers from them"):
        assert any(sentence in c for c in chunks)


def test_neighbouring_chunks_overlap() -> None:
    sentences = [f"Sentence number {i} is here." for i in range(40)]
    chunks = split_text(" ".join(sentences), chunk_size=200, overlap=60)

    for left, right in itertools.pairwise(chunks):
        tail_words = left.split()[-3:]
        assert " ".join(tail_words) in right


def test_prefers_paragraph_boundaries() -> None:
    first, second = "A" * 80, "B" * 80
    chunks = split_text(f"{first}\n\n{second}", chunk_size=100, overlap=0)

    assert chunks == [first, second]


def test_hard_cuts_text_without_separators() -> None:
    chunks = split_text("x" * 250, chunk_size=100, overlap=0)

    assert [len(c) for c in chunks] == [100, 100, 50]


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=100, overlap=100)


def test_chunks_never_span_pages_and_keep_page_numbers() -> None:
    chunks = chunk_pages([(1, "Page one text."), (2, "Page two text.")], chunk_size=100, overlap=10)

    assert [(c.page, c.index, c.content) for c in chunks] == [
        (1, 0, "Page one text."),
        (2, 1, "Page two text."),
    ]


MARKDOWN = """# Handbook
Intro text.

## Leave
### Annual leave
You get 24 days.

### Sick leave
You get 12 days.

## Code
```python
# not a heading
print("hi")
```
"""


def test_split_sections_tracks_the_heading_path() -> None:
    assert split_sections(MARKDOWN) == [
        ("Handbook", "Intro text."),
        ("Handbook > Leave > Annual leave", "You get 24 days."),
        ("Handbook > Leave > Sick leave", "You get 12 days."),
        ("Handbook > Code", '```python\n# not a heading\nprint("hi")\n```'),
    ]


def test_text_without_headings_is_one_section() -> None:
    assert split_sections("Just a paragraph.\n\nAnother.") == [
        ("", "Just a paragraph.\n\nAnother.")
    ]


def test_sections_never_share_a_chunk_and_carry_their_heading() -> None:
    chunks = chunk_pages([(None, MARKDOWN)], chunk_size=1000, overlap=100)

    assert [c.section for c in chunks] == ["Handbook", "Annual leave", "Sick leave", "Code"]
    assert [c.content for c in chunks[1:3]] == [
        "Handbook > Leave > Annual leave\n\nYou get 24 days.",
        "Handbook > Leave > Sick leave\n\nYou get 12 days.",
    ]


def test_heading_counts_toward_the_chunk_size() -> None:
    body = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_pages([(None, f"# A fairly long section title\n{body}")], 300, 50)

    assert len(chunks) > 1
    assert all(len(c.content) <= 300 for c in chunks)
    assert all(c.content.startswith("A fairly long section title\n\n") for c in chunks)


def test_normalize_whitespace() -> None:
    assert normalize_whitespace("a  b \r\n\n\n\nc\t\t d \n") == "a b\n\nc d"


def test_extracts_pdf_text_per_page() -> None:
    pdf = make_pdf(["Annual leave is 24 days.", "Remote work needs approval."])

    pages = extract_pages("policy.pdf", pdf)

    assert [p for p, _ in pages] == [1, 2]
    assert "Annual leave is 24 days." in pages[0][1]
    assert "Remote work" in pages[1][1]


def test_extracts_text_and_markdown_with_bom() -> None:
    assert extract_pages("notes.md", "﻿# Title\nBody".encode()) == [(None, "# Title\nBody")]


@pytest.mark.parametrize(
    ("filename", "data", "message"),
    [
        ("a.docx", b"x", "Unsupported file type"),
        ("a.pdf", b"not a pdf", "Could not read PDF"),
        ("a.pdf", make_pdf(["   "]), "No extractable text"),
        ("a.txt", b"   \n ", "No extractable text"),
    ],
)
def test_extraction_errors(filename: str, data: bytes, message: str) -> None:
    with pytest.raises(ExtractionError, match=message):
        extract_pages(filename, data)


async def test_fake_embedder_is_normalized_and_similarity_tracks_overlap() -> None:
    embedder = FakeEmbedder()
    a, b, c = await embedder.embed_documents(
        ["annual leave policy days", "annual leave days", "cafeteria lunch menu"]
    )

    def cosine(x: list[float], y: list[float]) -> float:
        return sum(i * j for i, j in zip(x, y, strict=True))

    assert math.isclose(cosine(a, a), 1.0, rel_tol=1e-9)
    assert cosine(a, b) > 0.7
    assert abs(cosine(a, c)) < 0.3


async def test_dimension_mismatch_fails_fast() -> None:
    await check_dimension(FakeEmbedder(384), 384)
    with pytest.raises(RuntimeError, match="vector\\(384\\)"):
        await check_dimension(FakeEmbedder(768), 384)
