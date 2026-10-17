"""Hybrid retrieval: vector similarity + Postgres full-text search, fused with RRF.

Why hybrid: embeddings capture meaning ("vacation" ~ "annual leave") but are
weak on exact tokens (error codes, names, numbers); keyword search is the
opposite. Running both and fusing them covers each one's blind spot.

Reciprocal Rank Fusion: score(chunk) = sum over lists of 1 / (K + rank).
It only uses ranks, so the two incomparable score scales (cosine vs ts_rank)
never need normalizing. K = 60 is the value from the original RRF paper; it
damps the advantage of being ranked first in a single list.

Relevance gate: a chunk is kept only if its cosine similarity clears
`min_similarity` OR it matched the keyword query. When nothing passes, search
returns nothing and the agent says it does not know.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, DocumentChunk
from app.rag.embeddings import EmbeddingProvider

RRF_K = 60


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A search hit with its provenance and both rankings."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page: int | None
    section: str | None
    chunk_index: int
    content: str
    similarity: float
    """Cosine similarity to the query (1 = identical direction)."""
    vector_rank: int | None
    keyword_rank: int | None
    score: float
    """Fused RRF score, used for ordering."""

    @property
    def citation(self) -> str:
        """Source label: 'handbook.pdf p.3', 'notes.md § Setup', or just the filename."""
        if self.page:
            return f"{self.filename} p.{self.page}"
        if self.section:
            return f"{self.filename} § {self.section}"
        return self.filename


async def search_chunks(
    session: AsyncSession,
    embedder: EmbeddingProvider,
    query: str,
    top_k: int = 5,
    min_similarity: float = 0.5,
    candidates: int = 20,
) -> list[RetrievedChunk]:
    """Return up to `top_k` relevant chunks from ready documents, best first."""
    query = query.strip()
    if not query:
        return []
    query_vector = await embedder.embed_query(query)
    distance = DocumentChunk.embedding.cosine_distance(query_vector)
    ready = Document.status == "ready"

    # 1. Semantic candidates (uses the HNSW index: ORDER BY distance LIMIT n).
    vector_rows = (
        await session.execute(
            select(DocumentChunk.id, distance.label("distance"))
            .join(Document)
            .where(ready)
            .order_by(distance)
            .limit(candidates)
        )
    ).all()

    # 2. Keyword candidates (uses the GIN index). websearch_to_tsquery accepts
    #    free text safely: quotes, OR, and -exclusions, never a syntax error.
    tsquery = func.websearch_to_tsquery("english", query)
    keyword_rows = (
        await session.execute(
            select(DocumentChunk.id)
            .join(Document)
            .where(ready, DocumentChunk.content_tsv.op("@@")(tsquery))
            .order_by(func.ts_rank_cd(DocumentChunk.content_tsv, tsquery).desc())
            .limit(candidates)
        )
    ).all()

    vector_rank = {row.id: rank for rank, row in enumerate(vector_rows, 1)}
    keyword_rank = {row.id: rank for rank, row in enumerate(keyword_rows, 1)}
    candidate_ids = set(vector_rank) | set(keyword_rank)
    if not candidate_ids:
        return []

    # 3. Load candidates with their exact similarity (keyword-only hits need it too).
    rows = (
        await session.execute(
            select(DocumentChunk, Document.filename, (1 - distance).label("similarity"))
            .join(Document)
            .where(DocumentChunk.id.in_(candidate_ids))
        )
    ).all()

    results: list[RetrievedChunk] = []
    for chunk, filename, similarity in rows:
        v_rank = vector_rank.get(chunk.id)
        k_rank = keyword_rank.get(chunk.id)
        if similarity < min_similarity and k_rank is None:
            continue  # neither semantically close nor a keyword match
        score = sum(1 / (RRF_K + r) for r in (v_rank, k_rank) if r is not None)
        results.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                filename=filename,
                page=chunk.page,
                section=chunk.section,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                similarity=float(similarity),
                vector_rank=v_rank,
                keyword_rank=k_rank,
                score=score,
            )
        )
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]
