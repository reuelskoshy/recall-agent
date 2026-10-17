"""Ingest the sample documents on first boot, so a fresh deployment demos RAG immediately.

Run with `python -m app.rag.seed_samples [directory]` (the production image does
this on start). Files already ingested (same SHA-256) are skipped, so it is safe
to run on every boot.
"""

import asyncio
import hashlib
import sys
from pathlib import Path

from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import Document
from app.db.session import SessionLocal, engine
from app.rag.embeddings import build_embedder
from app.rag.extraction import SUPPORTED_EXTENSIONS
from app.rag.ingestion import ingest_document

DEFAULT_DIR = Path(__file__).resolve().parents[3] / "samples"


def read_samples(directory: Path) -> list[tuple[str, bytes]]:
    """(filename, bytes) for every supported file in `directory`, sorted by name."""
    return [
        (path.name, path.read_bytes())
        for path in sorted(directory.glob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


async def seed_samples(directory: Path) -> list[str]:
    """Ingest every supported file in `directory` that is not stored yet; return their names."""
    settings = get_settings()
    embedder = build_embedder(settings)
    added: list[str] = []
    for name, data in await asyncio.to_thread(read_samples, directory):
        extension = Path(name).suffix.lower()
        sha256 = hashlib.sha256(data).hexdigest()
        async with SessionLocal() as session:
            if await session.scalar(select(Document.id).where(Document.sha256 == sha256)):
                continue
            document = Document(
                filename=name,
                content_type=SUPPORTED_EXTENSIONS[extension],
                size_bytes=len(data),
                sha256=sha256,
                status="processing",
                embedding_model=embedder.model,
                data=data,
            )
            session.add(document)
            await session.commit()
            document_id = document.id
        await ingest_document(
            SessionLocal,
            embedder,
            document_id,
            name,
            data,
            settings.rag_chunk_size,
            settings.rag_chunk_overlap,
        )
        added.append(name)
    return added


async def _main(directory: Path) -> None:
    try:
        added = await seed_samples(directory)
        print(f"Sample documents ingested: {', '.join(added) if added else 'none (up to date)'}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    samples = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    if samples.is_dir():
        asyncio.run(_main(samples))
    else:
        print(f"No samples directory at {samples}; skipping.")
