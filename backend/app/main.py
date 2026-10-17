"""FastAPI application factory and entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.runs import router as runs_router
from app.api.tools import router as tools_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import engine
from app.llm.base import LLMError


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Release pooled DB connections on shutdown."""
    yield
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

    @app.exception_handler(LLMError)
    async def llm_error_handler(_request: Request, exc: LLMError) -> JSONResponse:
        # E.g. a missing API key while building the provider, before any streaming starts.
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": str(exc)}
        )

    return app


app = create_app()
