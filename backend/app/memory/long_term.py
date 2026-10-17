"""Long-term memory: durable facts about the user that persist across conversations.

Storage: one row per memory with a pgvector embedding (see `Memory`).

Ranking. Recall scores every candidate memory as a weighted sum of three terms,
each normalized to [0, 1]:

    score = (w_sim * relevance + w_rec * recency + w_imp * importance) / (w_sim + w_rec + w_imp)

    relevance   (cosine - floor) / (1 - floor): cosine similarity to the query,
                rescaled so the relevance floor maps to 0 and identical to 1
    recency     0.5 ** (days since the memory was last used / half_life_days)
                (exponential decay: 1.0 now, 0.5 after one half-life, 0.25 after two)
    importance  (importance - 1) / 4, mapping the 1-5 scale to 0-1

Defaults: w_sim=0.7, w_rec=0.2, w_imp=0.1, half-life 30 days, floor 0.5.
Relevance dominates, so the memory must be about the question. Importance is
weighted lightly because it already has its own channel: important memories
are "core" and always in the prompt (below), so in the ranked layer it only
needs to break ties. At 0.2 it overturned correct-but-weak matches ("which
city am I in" ranked the user's name above their city); at 0.1 the right
memory ranked first for all 5 live probe queries.

Why rescale: bge-small's cosine for short facts sits in a compressed band
(roughly 0.45-0.65), so raw differences between a right and a wrong memory are
~0.05, small enough for the importance term to overturn. Rescaling above the
floor stretches that band to use the whole 0-1 range. Recency and importance
break ties between relevant memories: a fresh, important fact beats a stale
trivial one. A hard similarity floor runs first, so an important but unrelated
memory is never pulled in just for being important.

"Last used" is bumped whenever a memory is recalled, so memories that keep
being useful stay fresh, and ones that never come up again slowly fade.

Core vs. ranked. Short queries embed poorly against short facts ("suggest
dinner" vs. "The user is vegetarian" scores only ~0.45 with bge-small), so
similarity alone would miss memories that matter everywhere. Standing
instructions and high-importance facts (name, diet, role) are therefore "core":
always in the prompt. Everything else competes on the ranked score.

Embedding text. Every memory starts with "The user ..." (enforced for the tool,
requested of the extractor), and that shared subject carries no information but
inflates every similarity. Memories are embedded without it ("is vegetarian"),
which measurably separated right from wrong memories (top-1 6/8 -> 7/8) and
same-topic from different-topic pairs (0.82/0.70 -> 0.76/0.58).

Dedup. Before inserting, the nearest existing memory is checked; at or above
`dedup_similarity` the existing row is updated (newer wording wins, importance
takes the max) instead of a near-duplicate being added.
"""

import asyncio
import math
import re
import uuid
from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, get_args

from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import Memory
from app.rag.embeddings import EmbeddingProvider

MemoryKind = Literal["fact", "preference", "goal", "instruction"]
MemorySource = Literal["tool", "auto", "manual"]
MEMORY_KINDS: tuple[str, ...] = get_args(MemoryKind)
MAX_MEMORY_CHARS = 500
RECALL_CANDIDATES = 50

# One process-wide lock makes dedup + insert atomic, so two concurrent saves
# (e.g. a save_memory tool call and post-turn extraction) cannot both insert
# the same fact. Enough for a single-process app; multi-process would need
# an advisory lock in Postgres instead.
_save_lock = asyncio.Lock()

_SUBJECT = re.compile(r"^the user(?:'s)?\s+", re.IGNORECASE)

_SENSITIVE = [
    (re.compile(r"\b(sk|pk|rk)-[A-Za-z0-9_-]{16,}"), "an API key"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "a card or account number"),
    (re.compile(r"(?i)\b(password|passwd|pin code|api key|secret key)\b\s*(is|:|=)"), "a secret"),
]


class MemoryRejectedError(ValueError):
    """The content cannot be stored as a memory (e.g. it contains a secret)."""


@dataclass(frozen=True, slots=True)
class RankWeights:
    """Weights and decay for the ranking formula (see module docstring)."""

    similarity: float = 0.7
    recency: float = 0.2
    importance: float = 0.1
    half_life_days: float = 30.0

    @classmethod
    def from_settings(cls, settings: Settings) -> "RankWeights":
        """Read weights from configuration."""
        return cls(
            similarity=settings.memory_weight_similarity,
            recency=settings.memory_weight_recency,
            importance=settings.memory_weight_importance,
            half_life_days=settings.memory_half_life_days,
        )


@dataclass(frozen=True, slots=True)
class ScoredMemory:
    """A recalled memory with every term of its score, for ranking and for the UI."""

    memory: Memory
    similarity: float
    """Raw cosine similarity to the query."""
    relevance: float
    """Similarity rescaled above the floor; the term used in the score."""
    recency: float
    importance: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form (tool data, API responses, run logs)."""
        m = self.memory
        return {
            "id": str(m.id),
            "content": m.content,
            "kind": m.kind,
            "importance_level": m.importance,
            "score": round(self.score, 3),
            "similarity": round(self.similarity, 3),
            "relevance": round(self.relevance, 3),
            "recency": round(self.recency, 3),
            "importance": round(self.importance, 3),
        }


@dataclass(frozen=True, slots=True)
class SaveOutcome:
    """Result of `save_memory`: a new row, or an existing near-duplicate updated."""

    memory: Memory
    action: Literal["created", "updated"]
    previous_content: str | None = None


def recency_score(last_used: datetime, now: datetime, half_life_days: float) -> float:
    """Exponential decay by age: 1.0 if just used, 0.5 after one half-life."""
    age_days = max((now - last_used).total_seconds(), 0.0) / 86_400
    return math.pow(0.5, age_days / half_life_days)


def importance_score(importance: int) -> float:
    """Map the 1-5 importance scale onto 0-1."""
    return (min(max(importance, 1), 5) - 1) / 4


def relevance_score(similarity: float, floor: float) -> float:
    """Rescale cosine similarity so `floor` maps to 0 and 1.0 stays 1.0."""
    if floor >= 1:
        return 1.0 if similarity >= 1 else 0.0
    return min(max((similarity - floor) / (1 - floor), 0.0), 1.0)


def combined_score(relevance: float, recency: float, importance: float, w: RankWeights) -> float:
    """Weighted mean of the three terms (normalized by the weight sum, so it stays in 0-1)."""
    total = w.similarity + w.recency + w.importance
    if total == 0:
        return relevance
    return (w.similarity * relevance + w.recency * recency + w.importance * importance) / total


def embedding_text(content: str) -> str:
    """What gets embedded for a memory: its text without the shared 'The user' subject."""
    return _SUBJECT.sub("", content.strip()) or content


def normalize_content(text: str) -> str:
    """Collapse whitespace and validate length and content; raise MemoryRejectedError."""
    content = " ".join(text.split())
    if len(content) < 3:
        raise MemoryRejectedError("The memory is empty.")
    if len(content) > MAX_MEMORY_CHARS:
        raise MemoryRejectedError(f"Memories must be at most {MAX_MEMORY_CHARS} characters.")
    for pattern, label in _SENSITIVE:
        if pattern.search(content):
            raise MemoryRejectedError(f"Not saved: it looks like it contains {label}.")
    return content


async def recall_memories(
    session: AsyncSession,
    embedder: EmbeddingProvider,
    query: str,
    weights: RankWeights,
    limit: int = 5,
    min_similarity: float = 0.5,
    touch: bool = True,
    now: datetime | None = None,
    exclude: Collection[uuid.UUID] = (),
) -> list[ScoredMemory]:
    """Return the best `limit` memories for `query`, ranked by the combined score.

    With `touch`, returned memories count as used: their recency resets and
    their access count grows. (The scores returned reflect the state before.)
    """
    query = query.strip()
    if not query or limit <= 0:
        return []
    now = now or datetime.now(UTC)
    vector = await embedder.embed_query(query)
    distance = Memory.embedding.cosine_distance(vector)
    stmt = select(Memory, (1 - distance).label("similarity"))
    if exclude:
        stmt = stmt.where(Memory.id.not_in(list(exclude)))
    rows = (await session.execute(stmt.order_by(distance).limit(RECALL_CANDIDATES))).all()

    scored: list[ScoredMemory] = []
    for memory, similarity in rows:
        if similarity < min_similarity:
            continue
        rel = relevance_score(float(similarity), min_similarity)
        rec = recency_score(memory.last_accessed_at, now, weights.half_life_days)
        imp = importance_score(memory.importance)
        scored.append(
            ScoredMemory(
                memory=memory,
                similarity=float(similarity),
                relevance=rel,
                recency=rec,
                importance=imp,
                score=combined_score(rel, rec, imp, weights),
            )
        )
    scored.sort(key=lambda s: s.score, reverse=True)
    top = scored[:limit]

    if touch and top:
        await session.execute(
            update(Memory)
            .where(Memory.id.in_([s.memory.id for s in top]))
            .values(last_accessed_at=func.now(), access_count=Memory.access_count + 1)
            .execution_options(synchronize_session=False)
        )
    return top


async def find_similar(
    session: AsyncSession,
    embedder: EmbeddingProvider,
    text: str,
    min_similarity: float,
    limit: int = 3,
    exclude: Collection[uuid.UUID] = (),
) -> list[tuple[Memory, float]]:
    """Memories whose wording is close to `text` (statement vs. statement, no ranking).

    Used to find the memory a correction refers to, and to surface memories
    a new one may contradict.
    """
    [vector] = await embedder.embed_documents([embedding_text(text)])
    distance = Memory.embedding.cosine_distance(vector)
    stmt = select(Memory, (1 - distance).label("similarity"))
    if exclude:
        stmt = stmt.where(Memory.id.not_in(list(exclude)))
    rows = (await session.execute(stmt.order_by(distance).limit(limit))).all()
    return [(m, float(s)) for m, s in rows if s >= min_similarity]


async def core_memories(session: AsyncSession, min_importance: int, limit: int) -> list[Memory]:
    """Always-on memories: every instruction, and facts at or above `min_importance`."""
    if limit <= 0:
        return []
    rows = await session.scalars(
        select(Memory)
        .where(or_(Memory.kind == "instruction", Memory.importance >= min_importance))
        .order_by(Memory.importance.desc(), Memory.last_accessed_at.desc())
        .limit(limit)
    )
    return list(rows.all())


async def save_memory(
    session: AsyncSession,
    embedder: EmbeddingProvider,
    content: str,
    kind: str,
    importance: int,
    source: MemorySource,
    dedup_similarity: float,
    conversation_id: uuid.UUID | None = None,
) -> SaveOutcome:
    """Store a memory, or update a near-duplicate in place. Commits.

    Raises MemoryRejectedError for empty, oversized, or secret-looking content.
    """
    content = normalize_content(content)
    if kind not in MEMORY_KINDS:
        kind = "fact"
    importance = min(max(importance, 1), 5)
    [vector] = await embedder.embed_documents([embedding_text(content)])

    async with _save_lock:
        distance = Memory.embedding.cosine_distance(vector)
        nearest = (
            await session.execute(
                select(Memory, (1 - distance).label("similarity")).order_by(distance).limit(1)
            )
        ).first()
        if nearest is not None and nearest.similarity >= dedup_similarity:
            memory: Memory = nearest.Memory
            previous = memory.content
            memory.content = content
            memory.kind = kind
            memory.importance = max(memory.importance, importance)
            memory.embedding = vector
            memory.last_accessed_at = func.now()
            await session.commit()
            await session.refresh(memory)
            return SaveOutcome(memory=memory, action="updated", previous_content=previous)

        memory = Memory(
            content=content,
            kind=kind,
            importance=importance,
            source=source,
            embedding=vector,
            source_conversation_id=conversation_id,
        )
        session.add(memory)
        await session.commit()
        await session.refresh(memory)
        return SaveOutcome(memory=memory, action="created")


async def edit_memory(
    session: AsyncSession,
    embedder: EmbeddingProvider,
    memory: Memory,
    content: str | None = None,
    kind: str | None = None,
    importance: int | None = None,
) -> Memory:
    """Change a memory in place, re-embedding it if the text changed. Commits."""
    if content is not None:
        normalized = normalize_content(content)
        if normalized != memory.content:
            memory.content = normalized
            [memory.embedding] = await embedder.embed_documents([embedding_text(normalized)])
    if kind is not None and kind in MEMORY_KINDS:
        memory.kind = kind
    if importance is not None:
        memory.importance = min(max(importance, 1), 5)
    await session.commit()
    await session.refresh(memory)
    return memory
