"""Document ingestion: bytes -> pages -> chunks -> embeddings -> rows."""

import logging
import uuid

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import Document, DocumentChunk
from app.rag.chunking import chunk_pages
from app.rag.embeddings import EmbeddingProvider
from app.rag.extraction import ExtractionError, extract_pages

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 64


async def ingest_document(
    session_factory: async_sessionmaker[AsyncSession],
    embedder: EmbeddingProvider,
    document_id: uuid.UUID,
    filename: str,
    data: bytes,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Extract, chunk, embed, and store one document, then mark it ready (or failed).

    Runs after the upload response is sent, so it owns its own DB session and
    never raises: every failure is recorded on the document row instead.
    """
    async with session_factory() as session:
        document = await session.get(Document, document_id)
        if document is None:
            logger.warning("Document %s vanished before ingestion", document_id)
            return
        try:
            pages = extract_pages(filename, data)
            chunks = chunk_pages(pages, chunk_size, chunk_overlap)
            vectors: list[list[float]] = []
            for start in range(0, len(chunks), EMBED_BATCH_SIZE):
                batch = chunks[start : start + EMBED_BATCH_SIZE]
                vectors.extend(await embedder.embed_documents([c.content for c in batch]))

            # Idempotent: a retried ingestion replaces any partial earlier attempt.
            await session.execute(
                delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
            )
            session.add_all(
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=chunk.index,
                    page=chunk.page,
                    section=chunk.section,
                    content=chunk.content,
                    embedding=vector,
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            )
            document.status = "ready"
            document.error = None
            document.chunk_count = len(chunks)
            document.page_count = sum(1 for page, _ in pages if page is not None) or None
            await session.commit()
            logger.info("Ingested %s: %d chunks", filename, len(chunks))
        except ExtractionError as exc:
            await _mark_failed(session, document_id, str(exc))
        except Exception as exc:
            logger.exception("Ingestion of %s failed", filename)
            await _mark_failed(session, document_id, f"Ingestion failed ({type(exc).__name__}).")


async def _mark_failed(session: AsyncSession, document_id: uuid.UUID, error: str) -> None:
    await session.rollback()
    document = await session.get(Document, document_id)
    if document is not None:
        document.status = "failed"
        document.error = error
        await session.commit()
