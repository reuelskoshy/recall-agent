"""Integration tests: upload -> ingest -> hybrid search -> agent tool with citations."""

from typing import Any
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from app.llm.base import ToolCall
from app.llm.factory import get_llm_provider
from app.llm.fake import FakeProvider, ScriptedTurn
from tests.conftest import parse_sse
from tests.pdf_factory import make_pdf

pytestmark = pytest.mark.usefixtures("db")

HANDBOOK = make_pdf(
    [
        "Welcome to Acme. This handbook explains company policies for all employees.",
        "Annual leave: every employee receives 24 days of paid annual leave per year.",
        "Remote work: staff may work remotely up to three days per week with approval.",
    ]
)
FAQ_MD = (
    b"# Cafeteria FAQ\n\nThe cafeteria serves lunch from 12:30 to 14:00. "
    b"Error code ZX-42 means the card reader is offline."
)


async def upload(client: AsyncClient, name: str, data: bytes) -> dict[str, Any]:
    response = await client.post("/api/documents", files={"file": (name, data)})
    assert response.status_code in (200, 202), response.text
    body: dict[str, Any] = response.json()
    return body


async def test_upload_ingests_pdf_into_page_tagged_chunks(client: AsyncClient) -> None:
    created = await upload(client, "handbook.pdf", HANDBOOK)
    assert created["status"] == "processing"

    # Background ingestion has finished by the time the ASGI call returns.
    doc = (await client.get(f"/api/documents/{created['id']}")).json()
    assert doc["status"] == "ready"
    assert doc["page_count"] == 3
    assert doc["chunk_count"] == 3
    assert doc["content_type"] == "application/pdf"


async def test_duplicate_upload_returns_existing_document(client: AsyncClient) -> None:
    first = await upload(client, "handbook.pdf", HANDBOOK)
    response = await client.post("/api/documents", files={"file": ("copy.pdf", HANDBOOK)})

    assert response.status_code == 200
    assert response.json()["id"] == first["id"]
    assert len((await client.get("/api/documents")).json()) == 1


@pytest.mark.parametrize(
    ("name", "data", "status"),
    [("evil.exe", b"MZ", 415), ("empty.txt", b"", 422)],
)
async def test_upload_rejects_bad_files(
    client: AsyncClient, name: str, data: bytes, status: int
) -> None:
    response = await client.post("/api/documents", files={"file": (name, data)})

    assert response.status_code == status


async def test_upload_rejects_oversized_files(app: FastAPI, client: AsyncClient) -> None:
    from app.core.config import Settings, get_settings

    app.dependency_overrides[get_settings] = lambda: Settings(max_upload_mb=0)

    response = await client.post("/api/documents", files={"file": ("a.txt", b"hello")})

    assert response.status_code == 413


async def test_unreadable_file_is_marked_failed_with_reason(client: AsyncClient) -> None:
    created = await upload(client, "scan.pdf", make_pdf([" "]))

    doc = (await client.get(f"/api/documents/{created['id']}")).json()
    assert doc["status"] == "failed"
    assert "No extractable text" in doc["error"]


async def test_search_ranks_the_right_page_and_gates_irrelevant_queries(
    client: AsyncClient,
) -> None:
    await upload(client, "handbook.pdf", HANDBOOK)
    await upload(client, "faq.md", FAQ_MD)

    hits = (await client.get("/api/search", params={"q": "annual leave days"})).json()
    assert hits[0]["citation"] == "handbook.pdf p.2"
    assert hits[0]["vector_rank"] == 1

    # Exact token that only keyword search can pin down reliably.
    hits = (await client.get("/api/search", params={"q": "ZX-42"})).json()
    assert hits[0]["filename"] == "faq.md"
    assert hits[0]["keyword_rank"] == 1
    assert hits[0]["page"] is None
    assert hits[0]["section"] == "Cafeteria FAQ"
    assert hits[0]["citation"] == "faq.md § Cafeteria FAQ"

    # Nothing in the corpus is about this: the relevance gate returns nothing.
    assert (await client.get("/api/search", params={"q": "quantum chromodynamics"})).json() == []


async def test_deleting_a_document_removes_it_from_search(client: AsyncClient) -> None:
    doc = await upload(client, "faq.md", FAQ_MD)
    assert (await client.get("/api/search", params={"q": "cafeteria lunch"})).json()

    assert (await client.delete(f"/api/documents/{doc['id']}")).status_code == 204

    assert (await client.get("/api/search", params={"q": "cafeteria lunch"})).json() == []
    assert (await client.get(f"/api/documents/{doc['id']}")).status_code == 404


async def test_reindex_rebuilds_chunks_from_the_stored_original(
    app: FastAPI, client: AsyncClient
) -> None:
    from app.core.config import Settings, get_settings

    long_text = " ".join(f"Policy clause {i} covers topic {i}." for i in range(120)).encode()
    doc = await upload(client, "policies.txt", long_text)
    before = (await client.get(f"/api/documents/{doc['id']}")).json()["chunk_count"]

    # New chunking settings, then reindex from the stored original (no re-upload).
    app.dependency_overrides[get_settings] = lambda: Settings(
        rag_chunk_size=300, rag_chunk_overlap=30
    )
    response = await client.post(f"/api/documents/{doc['id']}/reindex")

    assert response.status_code == 202
    after = (await client.get(f"/api/documents/{doc['id']}")).json()
    assert after["status"] == "ready"
    assert after["chunk_count"] > before
    assert (await client.post(f"/api/documents/{uuid4()}/reindex")).status_code == 404


async def test_agent_cites_documents_via_search_tool(app: FastAPI, client: AsyncClient) -> None:
    await upload(client, "handbook.pdf", HANDBOOK)
    search = ToolCall("s1", "search_documents", '{"query": "annual leave days"}')
    fake = FakeProvider(
        [
            ScriptedTurn(tool_calls=[search]),
            ScriptedTurn(text="You get 24 days [handbook.pdf p.2]."),
        ]
    )
    app.dependency_overrides[get_llm_provider] = lambda: fake

    events = parse_sse((await client.post("/api/chat", json={"message": "Leave?"})).text)

    # The system prompt told the model which documents exist.
    system_prompt = fake.calls[0][0][0].content
    assert "- handbook.pdf" in system_prompt
    # The model received numbered, citable passages.
    tool_message = fake.calls[1][0][-1].content
    assert "[1] Source: handbook.pdf p.2" in tool_message
    # The UI received structured citations, and they survive a reload.
    result = next(data for name, data in events if name == "tool_result")
    assert result["data"]["results"][0]["filename"] == "handbook.pdf"
    assert result["data"]["results"][0]["page"] == 2
    detail = (await client.get(f"/api/conversations/{events[0][1]['conversation_id']}")).json()
    stored_tool = next(m for m in detail["messages"] if m["role"] == "tool")
    assert stored_tool["tool_data"]["results"][0]["page"] == 2


async def test_search_tool_reports_nothing_relevant(app: FastAPI, client: AsyncClient) -> None:
    await upload(client, "handbook.pdf", HANDBOOK)
    search = ToolCall("s1", "search_documents", '{"query": "quantum chromodynamics"}')
    fake = FakeProvider([ScriptedTurn(tool_calls=[search]), ScriptedTurn(text="I don't know.")])
    app.dependency_overrides[get_llm_provider] = lambda: fake

    events = parse_sse((await client.post("/api/chat", json={"message": "QCD?"})).text)

    result = next(data for name, data in events if name == "tool_result")
    assert result["content"].startswith("No relevant passages")
    assert result["data"]["results"] == []
