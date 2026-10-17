"""Embedding providers: turn text into vectors for similarity search.

The chat model and the embedding model are independent choices. Here the chat
model is remote (IFM K2) while embeddings run locally, free, on CPU.
"""

import asyncio
import hashlib
import logging
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

Vector = list[float]


class EmbeddingProvider(ABC):
    """Interface for embedding models. Vectors are L2-normalized."""

    name: str
    model: str
    dimension: int

    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """Embed passages to be stored and searched."""

    @abstractmethod
    async def embed_query(self, text: str) -> Vector:
        """Embed a search query (some models encode queries differently)."""


class SentenceTransformerEmbedder(EmbeddingProvider):
    """Local Hugging Face model via sentence-transformers (default: BAAI/bge-small-en-v1.5).

    Encoding is CPU-bound, so it runs in a worker thread to keep the event loop
    responsive. The model loads lazily (or at startup via `warm_up`).
    """

    name = "sentence_transformers"

    def __init__(self, model: str, query_prefix: str = "", batch_size: int = 32) -> None:
        self.model = model
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self._model: Any = None
        self._lock = asyncio.Lock()
        self.dimension = 0

    async def warm_up(self) -> None:
        """Load the model (downloading it on first use) and record its dimension."""
        async with self._lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load)
                self.dimension = int(self._model.get_embedding_dimension())
                logger.info("Loaded embedding model %s (dim=%d)", self.model, self.dimension)

    def _load(self) -> Any:
        from sentence_transformers import SentenceTransformer  # heavy import, deferred

        return SentenceTransformer(self.model, device="cpu")

    async def _encode(self, texts: Sequence[str], prompt: str | None) -> list[Vector]:
        await self.warm_up()
        array = await asyncio.to_thread(
            self._model.encode,
            list(texts),
            prompt=prompt,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [row.tolist() for row in array]

    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """Embed passages (no prefix)."""
        if not texts:
            return []
        return await self._encode(texts, prompt=None)

    async def embed_query(self, text: str) -> Vector:
        """Embed a query with the model's retrieval instruction prefix (BGE recommends one)."""
        return (await self._encode([text], prompt=self.query_prefix or None))[0]


_WORD = re.compile(r"[a-z0-9]+")


class FakeEmbedder(EmbeddingProvider):
    """Deterministic hashed bag-of-words vectors, for tests (no model download).

    Texts that share words get high cosine similarity; unrelated texts get ~0.
    That is enough to test retrieval logic end to end.
    """

    name = "fake"

    def __init__(self, dimension: int = 384) -> None:
        self.model = "fake-embedder"
        self.dimension = dimension

    def _vector(self, text: str) -> Vector:
        vec = [0.0] * self.dimension
        for word in _WORD.findall(text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            vec[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """Embed each text independently."""
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> Vector:
        """Same as documents: the fake model has no query instruction."""
        return self._vector(text)


def build_embedder(settings: Settings) -> EmbeddingProvider:
    """Instantiate the embedder named by `settings.embedding_provider`."""
    if settings.embedding_provider == "sentence_transformers":
        return SentenceTransformerEmbedder(
            model=settings.embedding_model, query_prefix=settings.embedding_query_prefix
        )
    if settings.embedding_provider == "fake":
        return FakeEmbedder(dimension=settings.embedding_dim)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER {settings.embedding_provider!r}")


async def check_dimension(embedder: EmbeddingProvider, expected: int) -> None:
    """Fail fast if the model's output size does not match the vector column.

    Storing vectors of the wrong size would fail at insert time (or, worse, a
    model swap of the same size would silently make old vectors meaningless).
    """
    if isinstance(embedder, SentenceTransformerEmbedder):
        await embedder.warm_up()
    if embedder.dimension != expected:
        raise RuntimeError(
            f"Embedding model {embedder.model} outputs {embedder.dimension} dimensions, but "
            f"the database column is vector({expected}). Change EMBEDDING_MODEL, or migrate "
            "the column and re-embed all documents."
        )


@lru_cache
def get_embedder() -> EmbeddingProvider:
    """FastAPI dependency: one embedder (and loaded model) per process."""
    return build_embedder(get_settings())
