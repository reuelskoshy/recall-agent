"""Integration tests for the chat API: real Postgres, scripted fake LLM."""

import uuid

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import select

from app.db.models import Message
from app.db.session import SessionLocal
from app.llm.base import LLMError, LLMProvider
from app.llm.factory import get_llm_provider
from app.llm.fake import FakeProvider, ScriptedTurn
from tests.conftest import parse_sse

pytestmark = pytest.mark.usefixtures("db")


def use_fake(app: FastAPI, *turns: ScriptedTurn) -> FakeProvider:
    """Swap the real LLM for a scripted fake and return it for assertions."""
    fake = FakeProvider(turns)
    app.dependency_overrides[get_llm_provider] = lambda: fake
    return fake


async def stored_messages(conversation_id: str) -> list[tuple[str, str]]:
    async with SessionLocal() as session:
        rows = await session.scalars(
            select(Message)
            .where(Message.conversation_id == uuid.UUID(conversation_id))
            .order_by(Message.created_at)
        )
        return [(m.role, m.content) for m in rows]


async def test_new_chat_streams_tokens_and_persists_both_messages(
    app: FastAPI, client: AsyncClient
) -> None:
    use_fake(app, ScriptedTurn(text="Hello there, friend"))

    response = await client.post("/api/chat", json={"message": "Hi!"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names == ["conversation", "token", "token", "token", "done"]
    assert "".join(data["text"] for name, data in events if name == "token") == (
        "Hello there, friend"
    )

    conversation_id = events[0][1]["conversation_id"]
    assert events[0][1]["title"] == "Hi!"
    done = events[-1][1]
    assert done["finish_reason"] == "stop"
    assert done["usage"]["completion_tokens"] == 3
    assert await stored_messages(conversation_id) == [
        ("user", "Hi!"),
        ("assistant", "Hello there, friend"),
    ]


async def test_follow_up_sends_history_to_the_model(app: FastAPI, client: AsyncClient) -> None:
    fake = use_fake(app, ScriptedTurn(text="Nice to meet you, Asha."), ScriptedTurn(text="Asha."))

    first = parse_sse((await client.post("/api/chat", json={"message": "I'm Asha"})).text)
    conversation_id = first[0][1]["conversation_id"]
    await client.post(
        "/api/chat", json={"message": "What's my name?", "conversation_id": conversation_id}
    )

    sent, _tools = fake.calls[1]
    assert [(m.role, m.content) for m in sent[1:]] == [
        ("user", "I'm Asha"),
        ("assistant", "Nice to meet you, Asha."),
        ("user", "What's my name?"),
    ]
    assert sent[0].role == "system"


async def test_unknown_conversation_returns_404_before_streaming(
    app: FastAPI, client: AsyncClient
) -> None:
    use_fake(app)

    response = await client.post(
        "/api/chat", json={"message": "hi", "conversation_id": str(uuid.uuid4())}
    )

    assert response.status_code == 404


async def test_llm_failure_emits_error_and_keeps_only_user_message(
    app: FastAPI, client: AsyncClient
) -> None:
    use_fake(app, ScriptedTurn(text="partial", error="rate limited"))

    events = parse_sse((await client.post("/api/chat", json={"message": "hi"})).text)

    assert [name for name, _ in events] == ["conversation", "token", "error"]
    assert events[-1][1]["message"] == "rate limited"
    assert await stored_messages(events[0][1]["conversation_id"]) == [("user", "hi")]


async def test_missing_provider_config_returns_503(app: FastAPI, client: AsyncClient) -> None:
    def broken_provider() -> LLMProvider:
        raise LLMError("OPENAI_API_KEY is not set")

    app.dependency_overrides[get_llm_provider] = broken_provider

    response = await client.post("/api/chat", json={"message": "hi"})

    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


async def test_empty_message_is_rejected(app: FastAPI, client: AsyncClient) -> None:
    use_fake(app)

    response = await client.post("/api/chat", json={"message": ""})

    assert response.status_code == 422


async def test_list_get_and_delete_conversations(app: FastAPI, client: AsyncClient) -> None:
    use_fake(app)
    first = parse_sse((await client.post("/api/chat", json={"message": "first"})).text)
    second = parse_sse((await client.post("/api/chat", json={"message": "second"})).text)
    first_id = first[0][1]["conversation_id"]
    second_id = second[0][1]["conversation_id"]

    listing = (await client.get("/api/conversations")).json()
    assert [c["id"] for c in listing] == [second_id, first_id]  # most recent first

    detail = (await client.get(f"/api/conversations/{first_id}")).json()
    assert [(m["role"], m["content"]) for m in detail["messages"]] == [
        ("user", "first"),
        ("assistant", "(fake model) You said: first"),
    ]

    assert (await client.delete(f"/api/conversations/{first_id}")).status_code == 204
    assert (await client.get(f"/api/conversations/{first_id}")).status_code == 404
    assert await stored_messages(first_id) == []  # cascade removed the messages
