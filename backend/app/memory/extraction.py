"""Automatic memory extraction: after each turn, ask the LLM what is worth remembering.

Runs as a background task once the answer has been streamed, so it never
slows the user down. The model sees the exchange plus related existing
memories, and returns JSON: new memories, or corrections that replace an
existing one ("I moved to Lisbon" replaces "The user lives in Berlin").
Everything is saved through `save_memory`, so dedup and the secret filter apply.

Failures (LLM down, malformed JSON) are logged and dropped: a missed memory is
acceptable; a broken chat turn is not.
"""

import asyncio
import json
import logging
import re
import uuid
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.llm.base import ChatMessage, LLMProvider, TextDelta
from app.memory.long_term import (
    MemoryKind,
    MemoryRejectedError,
    RankWeights,
    SaveOutcome,
    edit_memory,
    recall_memories,
    save_memory,
)
from app.rag.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)

MAX_PER_TURN = 5
MAX_ANSWER_CHARS = 2000
RELATED_MEMORIES = 8

EXTRACTION_PROMPT = """You maintain long-term memory about the USER of an AI assistant.
Read one exchange and decide what is worth remembering for FUTURE conversations.

Save only durable, user-specific information:
- facts about the user (name, role, team, location, skills, projects)
- preferences (tools, formats, style, diet, likes and dislikes)
- goals and ongoing plans
- standing instructions for the assistant ("always answer in metric units")

Do NOT save:
- the question itself, one-off tasks, or small talk
- general knowledge, document content, or data returned by tools
- anything about the assistant
- passwords, API keys, card numbers, or other secrets

Write each memory as one short sentence starting with "The user" (never the user's
name), one fact per memory, e.g. "The user is vegetarian." and, separately,
"The user's name is Ada."
importance: 5 = identity or standing instruction; 4 = constraints that matter in
many conversations (diet, allergies, accessibility, role); 3 = typical preference or
fact; 1 = minor detail. Importance 4-5 memories are shown in every conversation.

Existing memories are listed with numbers. Do not repeat them. If the exchange
updates or contradicts one, output the corrected memory and set "replaces" to
its number; otherwise "replaces" is null. Also fix existing memories that the
exchange shows are outdated, even if a newer memory already covers the change
(e.g. "[2] The user lives in Berlin" when the user said they moved to Lisbon:
output the corrected full fact with "replaces": 2).

Reply with ONLY this JSON, no other text:
{"memories": [{"content": "...", "kind": "fact|preference|goal|instruction",
"importance": 3, "replaces": null}]}
Most exchanges contain nothing worth remembering: then reply {"memories": []}."""


class ExtractedMemory(BaseModel):
    """One memory proposed by the extraction model."""

    content: str = Field(min_length=3, max_length=500)
    kind: MemoryKind = "fact"
    importance: int = Field(default=3, ge=1, le=5)
    replaces: int | None = None


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

_pending: set[asyncio.Task[list[SaveOutcome]]] = set()


def parse_extraction(text: str) -> list[ExtractedMemory]:
    """Parse the model's reply leniently: code fences and surrounding prose are ignored.

    Invalid items are skipped individually; unparseable output yields [].
    """
    cleaned = _FENCE.sub("", text.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return []
    try:
        payload: Any = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return []
    items = payload.get("memories") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    memories: list[ExtractedMemory] = []
    for item in items[:MAX_PER_TURN]:
        try:
            memories.append(ExtractedMemory.model_validate(item))
        except ValidationError:
            continue
    return memories


def build_extraction_messages(
    user_text: str, answer: str, existing: list[str]
) -> list[ChatMessage]:
    """The prompt: instructions, then numbered related memories and the exchange."""
    listed = "\n".join(f"[{i}] {c}" for i, c in enumerate(existing, 1)) or "(none)"
    clipped = answer if len(answer) <= MAX_ANSWER_CHARS else answer[:MAX_ANSWER_CHARS] + "…"
    return [
        ChatMessage(role="system", content=EXTRACTION_PROMPT),
        ChatMessage(
            role="user",
            content=f"Existing memories:\n{listed}\n\n"
            f"Exchange:\nUser: {user_text}\nAssistant: {clipped}",
        ),
    ]


async def _complete(provider: LLMProvider, messages: list[ChatMessage]) -> str:
    parts = [
        event.text async for event in provider.stream_chat(messages) if isinstance(event, TextDelta)
    ]
    return "".join(parts)


async def extract_and_store(
    session_factory: async_sessionmaker[AsyncSession],
    provider: LLMProvider,
    embedder: EmbeddingProvider,
    settings: Settings,
    conversation_id: uuid.UUID | None,
    user_text: str,
    answer: str,
) -> list[SaveOutcome]:
    """Extract memories from one exchange and save them. Never raises."""
    try:
        async with session_factory() as session:
            related = await recall_memories(
                session,
                embedder,
                user_text,
                RankWeights.from_settings(settings),
                limit=RELATED_MEMORIES,
                min_similarity=settings.memory_min_similarity,
                touch=False,
            )
            reply = await _complete(
                provider,
                build_extraction_messages(user_text, answer, [s.memory.content for s in related]),
            )
            proposals = parse_extraction(reply)
            if not proposals and "memories" not in reply:
                logger.warning("Memory extraction returned unparseable output: %.200s", reply)

            outcomes: list[SaveOutcome] = []
            for proposal in proposals:
                try:
                    target = (
                        related[proposal.replaces - 1].memory
                        if proposal.replaces and 1 <= proposal.replaces <= len(related)
                        else None
                    )
                    if target is not None:
                        previous = target.content
                        updated = await edit_memory(
                            session,
                            embedder,
                            target,
                            content=proposal.content,
                            kind=proposal.kind,
                            importance=proposal.importance,
                        )
                        outcomes.append(SaveOutcome(updated, "updated", previous))
                        continue
                    outcomes.append(
                        await save_memory(
                            session,
                            embedder,
                            proposal.content,
                            proposal.kind,
                            proposal.importance,
                            source="auto",
                            dedup_similarity=settings.memory_dedup_similarity,
                            conversation_id=conversation_id,
                        )
                    )
                except MemoryRejectedError as exc:
                    logger.info("Skipped extracted memory: %s", exc)
            logger.info(
                "Memory extraction for %s: %d proposed, %d saved%s",
                conversation_id,
                len(proposals),
                len(outcomes),
                "".join(f"\n  {o.action}: {o.memory.content}" for o in outcomes),
            )
            return outcomes
    except Exception:
        logger.exception("Memory extraction failed for conversation %s", conversation_id)
        return []


def schedule_extraction(
    session_factory: async_sessionmaker[AsyncSession],
    provider: LLMProvider,
    embedder: EmbeddingProvider,
    settings: Settings,
    conversation_id: uuid.UUID | None,
    user_text: str,
    answer: str,
) -> None:
    """Run `extract_and_store` in the background (keeping a reference so it is not GC'd)."""
    task = asyncio.create_task(
        extract_and_store(
            session_factory, provider, embedder, settings, conversation_id, user_text, answer
        )
    )
    _pending.add(task)
    task.add_done_callback(_pending.discard)


async def drain_pending_extractions() -> None:
    """Wait for in-flight extractions (app shutdown, and tests)."""
    while _pending:
        await asyncio.gather(*list(_pending), return_exceptions=True)
