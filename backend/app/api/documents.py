"""Document endpoints: upload (ingested in the background), list, inspect, delete, search."""

import hashlib
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import undefer

from app.core.config import Settings, get_settings
from app.db.models import Document
from app.db.session import SessionLocal, get_session
from app.rag.embeddings import EmbeddingProvider, get_embedder
from app.rag.extraction import SUPPORTED_EXTENSIONS, file_extension
from app.rag.ingestion import ingest_document
from app.rag.retrieval import search_chunks

router = APIRouter(tags=["documents"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
EmbedderDep = Annotated[EmbeddingProvider, Depends(get_embedder)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


class DocumentOut(BaseModel):
    """A document and its ingestion status."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    content_type: str
    size_bytes: int
    status: str
    error: str | None
    page_count: int | None
    chunk_count: int
    embedding_model: str
    created_at: datetime


class SearchHit(BaseModel):
    """A retrieval result (GET /api/search), with both rankings for debugging."""

    document_id: uuid.UUID
    filename: str
    page: int | None
    section: str | None
    chunk_index: int
    citation: str
    similarity: float
    vector_rank: int | None
    keyword_rank: int | None
    score: float
    content: str


@router.post(
    "/documents",
    response_model=DocumentOut,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        200: {"description": "Identical file already uploaded; existing document returned"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported file type"},
    },
)
async def upload_document(
    file: UploadFile,
    response: Response,
    background: BackgroundTasks,
    session: SessionDep,
    embedder: EmbedderDep,
    settings: SettingsDep,
) -> Document:
    """Upload a PDF, TXT, or MD file. Returns immediately; poll the document for status."""
    filename = (file.filename or "upload").strip()
    extension = file_extension(filename)
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type {extension or '(none)'!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.",
        )
    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE, f"File exceeds {settings.max_upload_mb} MB."
        )
    if not data:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "File is empty.")

    sha256 = hashlib.sha256(data).hexdigest()
    existing = await session.scalar(select(Document).where(Document.sha256 == sha256))
    if existing is not None and existing.status != "failed":
        response.status_code = status.HTTP_200_OK
        return existing
    if existing is not None:  # a failed earlier attempt: retry it in place
        document = existing
        document.filename = filename
        document.status = "processing"
        document.error = None
        document.data = data
    else:
        document = Document(
            filename=filename,
            content_type=SUPPORTED_EXTENSIONS[extension],
            size_bytes=len(data),
            sha256=sha256,
            status="processing",
            embedding_model=embedder.model,
            data=data,
        )
        session.add(document)
    await session.commit()

    background.add_task(
        ingest_document,
        SessionLocal,
        embedder,
        document.id,
        filename,
        data,
        settings.rag_chunk_size,
        settings.rag_chunk_overlap,
    )
    return document


@router.get("/documents", response_model=list[DocumentOut])
async def list_documents(session: SessionDep) -> list[Document]:
    """All documents, newest first."""
    rows = await session.scalars(select(Document).order_by(Document.created_at.desc()))
    return list(rows.all())


@router.get("/documents/{document_id}", response_model=DocumentOut)
async def get_document(document_id: uuid.UUID, session: SessionDep) -> Document:
    """One document's status and stats."""
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return document


@router.post(
    "/documents/{document_id}/reindex",
    response_model=DocumentOut,
    status_code=status.HTTP_202_ACCEPTED,
    responses={409: {"description": "Original file not stored; re-upload it"}},
)
async def reindex_document(
    document_id: uuid.UUID,
    background: BackgroundTasks,
    session: SessionDep,
    embedder: EmbedderDep,
    settings: SettingsDep,
) -> Document:
    """Re-chunk and re-embed from the stored original (after changing chunking or model)."""
    document = await session.get(Document, document_id, options=[undefer(Document.data)])
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    if document.data is None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Original file not stored; re-upload it.")
    data = document.data
    document.status = "processing"
    document.error = None
    document.embedding_model = embedder.model
    await session.commit()
    background.add_task(
        ingest_document,
        SessionLocal,
        embedder,
        document.id,
        document.filename,
        data,
        settings.rag_chunk_size,
        settings.rag_chunk_overlap,
    )
    return document


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: uuid.UUID, session: SessionDep) -> None:
    """Delete a document and (via ON DELETE CASCADE) all its chunks."""
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    await session.delete(document)
    await session.commit()


@router.get("/search", response_model=list[SearchHit])
async def search(
    session: SessionDep,
    embedder: EmbedderDep,
    settings: SettingsDep,
    q: Annotated[str, Query(min_length=1, max_length=500)],
    top_k: Annotated[int, Query(ge=1, le=20)] = 5,
    min_similarity: float | None = None,
) -> list[SearchHit]:
    """Run retrieval directly (no LLM). Useful for debugging and evaluation."""
    results = await search_chunks(
        session,
        embedder,
        q,
        top_k=top_k,
        min_similarity=settings.rag_min_similarity if min_similarity is None else min_similarity,
        candidates=settings.rag_candidates,
    )
    return [
        SearchHit(
            document_id=r.document_id,
            filename=r.filename,
            page=r.page,
            section=r.section,
            chunk_index=r.chunk_index,
            citation=r.citation,
            similarity=r.similarity,
            vector_rank=r.vector_rank,
            keyword_rank=r.keyword_rank,
            score=r.score,
            content=r.content,
        )
        for r in results
    ]
