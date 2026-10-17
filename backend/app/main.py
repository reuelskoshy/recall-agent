"""FastAPI application factory and entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.memories import router as memories_router
from app.api.runs import router as runs_router
from app.api.tools import router as tools_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.models import EMBEDDING_DIM
from app.db.session import engine
from app.llm.base import LLMError
from app.memory.extraction import drain_pending_extractions
from app.rag.embeddings import check_dimension, get_embedder
from app.sql.executor import dispose_readonly_engines


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup: load the embedding model and verify its dimension. Shutdown: close the pool."""
    settings = get_settings()
    if settings.embedding_dim != EMBEDDING_DIM:
        raise RuntimeError(
            f"EMBEDDING_DIM={settings.embedding_dim} but the schema uses vector({EMBEDDING_DIM})."
        )
    await check_dimension(get_embedder(), settings.embedding_dim)
    yield
    await drain_pending_extractions()
    await dispose_readonly_engines()
    await engine.dispose()


def create_app() -> FastAPI:
    """Build the FastAPI app. A factory keeps tests free to build fresh instances."""
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=f"{settings.app_name} API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router, prefix="/api")
    app.include_router(chat_router, prefix="/api")
    app.include_router(tools_router, prefix="/api")
    app.include_router(runs_router, prefix="/api")
    app.include_router(documents_router, prefix="/api")
    app.include_router(memories_router, prefix="/api")

    @app.exception_handler(LLMError)
    async def llm_error_handler(_request: Request, exc: LLMError) -> JSONResponse:
        # E.g. a missing API key while building the provider, before any streaming starts.
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": str(exc)}
        )

    return app


app = create_app()
