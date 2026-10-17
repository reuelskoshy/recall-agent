"""Long-term memory: ranking math, dedup, safety filter, extraction, tools, API, and agent."""

import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy import func, select, update

from app.core.config import Settings, get_settings
from app.db.models import AgentRun, Memory
from app.db.session import SessionLocal
from app.llm.base import ToolCall
from app.llm.factory import get_llm_provider
from app.llm.fake import FakeProvider, ScriptedTurn
from app.memory.extraction import (
    drain_pending_extractions,
    extract_and_store,
    parse_extraction,
)
from app.memory.long_term import (
    MemoryRejectedError,
    RankWeights,
    combined_score,
    embedding_text,
    importance_score,
    recall_memories,
    recency_score,
    relevance_score,
    save_memory,
)
from app.rag.embeddings import FakeEmbedder
from tests.conftest import parse_sse

EMBEDDER = FakeEmbedder()
WEIGHTS = RankWeights()
NOW = datetime(2026, 10, 17, tzinfo=UTC)


async def save(content: str, importance: int = 3, dedup: float = 0.9) -> Memory:
    async with SessionLocal() as session:
        outcome = await save_memory(
            session, EMBEDDER, content, "fact", importance, source="manual", dedup_similarity=dedup
        )
        return outcome.memory


# ---- Ranking formula (pure) -------------------------------------------------------


def test_recency_halves_every_half_life() -> None:
    assert recency_score(NOW, NOW, 30) == 1.0
    assert recency_score(NOW - timedelta(days=30), NOW, 30) == pytest.approx(0.5)
    assert recency_score(NOW - timedelta(days=60), NOW, 30) == pytest.approx(0.25)
    assert recency_score(NOW + timedelta(days=1), NOW, 30) == 1.0  # clock skew: never > 1


def test_importance_maps_one_to_five_onto_zero_to_one() -> None:
    assert [importance_score(i) for i in (1, 3, 5)] == [0.0, 0.5, 1.0]


def test_combined_score_is_a_weighted_mean() -> None:
    assert combined_score(1.0, 1.0, 1.0, WEIGHTS) == pytest.approx(1.0)
    assert combined_score(0.8, 0.5, 0.25, WEIGHTS) == pytest.approx(
        0.7 * 0.8 + 0.2 * 0.5 + 0.1 * 0.25
    )


def test_relevance_rescales_similarity_above_the_floor() -> None:
    assert relevance_score(0.5, 0.5) == 0.0
    assert relevance_score(0.75, 0.5) == pytest.approx(0.5)
    assert relevance_score(1.0, 0.5) == 1.0
    assert relevance_score(0.3, 0.5) == 0.0  # below the floor never goes negative


def test_memories_are_embedded_without_the_shared_subject() -> None:
    assert embedding_text("The user is vegetarian.") == "is vegetarian."
    assert embedding_text("The user's name is Ada.") == "name is Ada."
    assert embedding_text("Prefers tea") == "Prefers tea"


# ---- Parsing the extraction model's output (pure) ------------------------------


def test_parse_extraction_is_lenient_about_wrapping() -> None:
    reply = (
        "Sure! Here you go:\n```json\n"
        '{"memories": [{"content": "The user is vegetarian.", "kind": "preference",'
        ' "importance": 4, "replaces": null}, {"content": "x"},'
        ' {"content": "The user lives in Pune.", "kind": "bogus"}]}\n```'
    )

    memories = parse_extraction(reply)

    # Invalid items (too short, unknown kind) are dropped individually.
    assert [m.content for m in memories] == ["The user is vegetarian."]
    assert memories[0].kind == "preference"
    assert memories[0].importance == 4


@pytest.mark.parametrize("reply", ["", "nothing to save", "{not json}", '{"memories": "no"}'])
def test_parse_extraction_returns_nothing_for_garbage(reply: str) -> None:
    assert parse_extraction(reply) == []


@pytest.mark.parametrize(
    "content",
    [
        "My password is hunter2",
        "The user's API key is sk-abcdefghijklmnopqrstuvwxyz",
        "Card 4111 1111 1111 1111",
    ],
)
async def test_secrets_are_never_stored(content: str) -> None:
    async with SessionLocal() as session:
        with pytest.raises(MemoryRejectedError):
            await save_memory(
                session, EMBEDDER, content, "fact", 3, source="tool", dedup_similarity=0.9
            )


# ---- Storage and recall (DB) ------------------------------------------------------

db_tests = pytest.mark.usefixtures("db")


@db_tests
async def test_near_duplicates_update_instead_of_inserting() -> None:
    first = await save("The user is vegetarian", importance=2)
    async with SessionLocal() as session:
        outcome = await save_memory(
            session, EMBEDDER, "the user is VEGETARIAN!", "preference", 4, "auto", 0.9
        )
        count = await session.scalar(select(func.count()).select_from(Memory))

    assert outcome.action == "updated"
    assert outcome.memory.id == first.id
    assert outcome.previous_content == "The user is vegetarian"
    assert outcome.memory.importance == 4  # max of old and new
    assert count == 1


@db_tests
async def test_recency_and_importance_break_ties_between_relevant_memories() -> None:
    # Same text (dedup disabled) so similarity is identical; only age and importance differ.
    fresh = await save("The user works on the payments team", importance=5, dedup=1.01)
    stale = await save("The user works on the payments team", importance=1, dedup=1.01)
    async with SessionLocal() as session:
        await session.execute(
            update(Memory)
            .where(Memory.id == stale.id)
            .values(last_accessed_at=NOW - timedelta(days=90))
        )
        await session.execute(
            update(Memory).where(Memory.id == fresh.id).values(last_accessed_at=NOW)
        )
        await session.commit()

        ranked = await recall_memories(
            session, EMBEDDER, "payments team", WEIGHTS, min_similarity=0.3, now=NOW
        )

    assert [r.memory.id for r in ranked] == [fresh.id, stale.id]
    assert ranked[0].similarity == pytest.approx(ranked[1].similarity)
    assert ranked[0].recency == pytest.approx(1.0)
    assert ranked[1].recency == pytest.approx(0.125)  # three half-lives
    assert ranked[0].score - ranked[1].score == pytest.approx(0.2 * 0.875 + 0.1 * 1.0)


@db_tests
async def test_unrelated_memories_are_not_recalled_and_recall_counts_as_use() -> None:
    memory = await save("The user prefers dark roast coffee")
    async with SessionLocal() as session:
        assert await recall_memories(session, EMBEDDER, "quantum physics", WEIGHTS) == []

        found = await recall_memories(
            session, EMBEDDER, "dark roast coffee", WEIGHTS, min_similarity=0.3
        )
        await session.commit()
        refreshed = await session.get(Memory, memory.id, populate_existing=True)

    assert [f.memory.id for f in found] == [memory.id]
    assert refreshed is not None
    assert refreshed.access_count == 1


@db_tests
async def test_extraction_adds_new_and_replaces_contradicted_memories() -> None:
    settings = Settings(memory_min_similarity=0.2)
    old = await save("The user lives in a flat in central Berlin")
    fake = FakeProvider(
        [
            ScriptedTurn(
                text=json.dumps(
                    {
                        "memories": [
                            {
                                "content": "The user lives in a flat in central Lisbon",
                                "kind": "fact",
                                "replaces": 1,
                            },
                            {"content": "The user is learning Portuguese", "kind": "goal"},
                        ]
                    }
                )
            )
        ]
    )

    outcomes = await extract_and_store(
        SessionLocal,
        fake,
        EMBEDDER,
        settings,
        conversation_id=None,
        user_text="I moved from Berlin; the user lives in Lisbon now and is learning Portuguese",
        answer="Welcome to Lisbon!",
    )

    assert [o.action for o in outcomes] == ["updated", "created"]
    assert outcomes[0].memory.id == old.id
    # The model was shown the related existing memory, numbered, to reference it.
    prompt = fake.calls[0][0][1].content
    assert "[1] The user lives in a flat in central Berlin" in prompt
    async with SessionLocal() as session:
        contents = set((await session.scalars(select(Memory.content))).all())
    assert contents == {
        "The user lives in a flat in central Lisbon",
        "The user is learning Portuguese",
    }


@db_tests
async def test_extraction_survives_a_failing_model() -> None:
    fake = FakeProvider([ScriptedTurn(error="provider down")])

    outcomes = await extract_and_store(
        SessionLocal, fake, EMBEDDER, Settings(), None, "hi", "hello"
    )

    assert outcomes == []


# ---- Agent integration ------------------------------------------------------------


@db_tests
async def test_relevant_memories_are_injected_into_the_prompt(
    app: FastAPI, client: AsyncClient
) -> None:
    await save("The user is vegetarian and avoids fish")
    await save("The user drives a red bicycle")
    await save("The user's name is Ada", importance=5)  # core: in every prompt
    fake = FakeProvider([ScriptedTurn(text="Try a lentil curry.")])
    app.dependency_overrides[get_llm_provider] = lambda: fake
    app.dependency_overrides[get_settings] = lambda: Settings(memory_min_similarity=0.3)

    events = parse_sse(
        (
            await client.post(
                "/api/chat", json={"message": "dinner for a vegetarian who avoids fish?"}
            )
        ).text
    )

    names = [name for name, _ in events]
    assert names[:2] == ["conversation", "memories"]
    recalled = events[1][1]["memories"]
    assert [(m["content"], m["reason"]) for m in recalled] == [
        ("The user's name is Ada", "core"),
        ("The user is vegetarian and avoids fish", "relevant"),
    ]
    assert {"score", "similarity", "recency", "importance"} <= recalled[1].keys()
    system_prompt = fake.calls[0][0][0].content
    assert (
        "What you remember about the user:\n- The user's name is Ada\n- The user is vegetarian"
        in system_prompt
    )
    assert "bicycle" not in system_prompt
    async with SessionLocal() as session:
        run = await session.scalar(select(AgentRun))
    assert run is not None and run.memories_used is not None
    assert [m["reason"] for m in run.memories_used] == ["core", "relevant"]


@db_tests
async def test_agent_saves_a_memory_with_the_tool(app: FastAPI, client: AsyncClient) -> None:
    call = ToolCall(
        "m1",
        "save_memory",
        '{"content": "The user prefers answers in bullet points", "kind": "instruction",'
        ' "importance": 5}',
    )
    fake = FakeProvider([ScriptedTurn(tool_calls=[call]), ScriptedTurn(text="Noted!")])
    app.dependency_overrides[get_llm_provider] = lambda: fake

    events = parse_sse(
        (await client.post("/api/chat", json={"message": "Always use bullets"})).text
    )

    result = next(data for name, data in events if name == "tool_result")
    assert result["content"] == "Saved to long-term memory."
    memories = (await client.get("/api/memories")).json()
    assert memories[0]["content"] == "The user prefers answers in bullet points"
    assert memories[0]["source"] == "tool"
    assert memories[0]["kind"] == "instruction"
    assert memories[0]["source_conversation_id"] == events[0][1]["conversation_id"]


@db_tests
async def test_save_memory_flags_conflicts_and_replaces_outdated_memories(
    app: FastAPI, client: AsyncClient
) -> None:
    old = await save("The user lives in a flat in central Berlin")
    moved = ToolCall(
        "m1", "save_memory", '{"content": "The user lives in a flat in central Lisbon"}'
    )
    fix = ToolCall(
        "m2",
        "save_memory",
        '{"content": "The user lives in a flat in central Lisbon",'
        ' "replaces": "The user lives in a flat in central Berlin"}',
    )
    fake = FakeProvider(
        [
            ScriptedTurn(tool_calls=[moved]),
            ScriptedTurn(tool_calls=[fix]),
            ScriptedTurn(text="Welcome to Lisbon!"),
        ]
    )
    app.dependency_overrides[get_llm_provider] = lambda: fake

    events = parse_sse((await client.post("/api/chat", json={"message": "I moved"})).text)

    first, second = [data for name, data in events if name == "tool_result"]
    # The new fact was saved, and the model was told what it might contradict.
    assert (
        "Related memories on the same topic:\n- The user lives in a flat in central Berlin"
        in first["content"]
    )
    # The model then retired the outdated memory; the new fact is not duplicated.
    assert second["data"]["action"] == "replaced"
    assert second["data"]["previous_content"] == "The user lives in a flat in central Berlin"
    memories = [m["content"] for m in (await client.get("/api/memories")).json()]
    assert memories == ["The user lives in a flat in central Lisbon"]
    async with SessionLocal() as session:
        assert await session.get(Memory, old.id) is None


@db_tests
async def test_memories_are_extracted_automatically_after_a_turn(
    app: FastAPI, client: AsyncClient
) -> None:
    extracted = '{"memories": [{"content": "The user is training for a marathon", "kind": "goal"}]}'
    fake = FakeProvider([ScriptedTurn(text="Good luck!"), ScriptedTurn(text=extracted)])
    app.dependency_overrides[get_llm_provider] = lambda: fake
    app.dependency_overrides[get_settings] = lambda: Settings(memory_auto_extract=True)

    await client.post("/api/chat", json={"message": "I'm training for a marathon in May"})
    await drain_pending_extractions()

    memories = (await client.get("/api/memories")).json()
    assert [(m["content"], m["source"]) for m in memories] == [
        ("The user is training for a marathon", "auto")
    ]
    # The extraction call got the exchange, and no tools.
    assert "User: I'm training for a marathon in May" in fake.calls[1][0][1].content
    assert fake.calls[1][1] == []


# ---- REST API -------------------------------------------------------------------


@db_tests
async def test_memory_crud_and_ranked_search(client: AsyncClient) -> None:
    created = await client.post(
        "/api/memories", json={"content": "The user lives in Pune", "importance": 4}
    )
    assert created.status_code == 201
    memory_id = created.json()["memory"]["id"]

    duplicate = await client.post("/api/memories", json={"content": "the user lives in pune"})
    assert duplicate.status_code == 200
    assert duplicate.json()["action"] == "updated"

    edited = await client.patch(
        f"/api/memories/{memory_id}", json={"content": "The user lives in Mumbai"}
    )
    assert edited.json()["content"] == "The user lives in Mumbai"
    assert edited.json()["importance"] == 4

    # Re-embedded on edit: the new wording is what search finds.
    hits = (await client.get("/api/memories", params={"q": "lives in Mumbai"})).json()
    assert hits[0]["id"] == memory_id
    assert 0 < hits[0]["ranking"]["score"] <= 1
    assert (await client.get("/api/memories", params={"q": "Pune"})).json() == []

    secret = await client.post("/api/memories", json={"content": "my password is hunter2"})
    assert secret.status_code == 422

    assert (await client.delete(f"/api/memories/{memory_id}")).status_code == 204
    assert (await client.delete(f"/api/memories/{memory_id}")).status_code == 404
    assert (await client.get("/api/memories")).json() == []


@db_tests
async def test_save_memory_tool_requires_a_consistent_subject() -> None:
    import uuid

    from app.tools.base import ToolContext
    from app.tools.registry import build_registry

    settings = Settings()
    ctx = ToolContext(uuid.uuid4(), SessionLocal, EMBEDDER, settings)

    result = await build_registry(settings).execute(
        ToolCall("m1", "save_memory", '{"content": "Ada is vegetarian"}'), ctx
    )

    assert result.is_error
    assert "starting with 'The user'" in result.content
