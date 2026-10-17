"""Application settings loaded from environment variables (and an optional .env file)."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated configuration. Every value can be overridden by an env var."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Recall"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    database_url: str = Field(
        default="postgresql+asyncpg://recall:recall@localhost:5433/recall",
        description="SQLAlchemy async URL (asyncpg driver).",
    )
    cors_origins: list[str] = ["http://localhost:5173"]

    llm_provider: str = Field(default="openai", description="openai | fake (see llm/factory.py)")
    llm_timeout_s: float = 60.0
    llm_api_key: str = ""
    llm_chat_model: str = "gpt-5.4-mini"
    llm_base_url: str | None = Field(
        default=None, description="Point at any OpenAI-compatible server, e.g. Ollama's /v1."
    )

    chat_history_limit: int = Field(
        default=20, description="Max past messages sent to the model (token-aware in Phase 5)."
    )

    @field_validator("llm_base_url", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        """Treat `LLM_BASE_URL=` (blank in .env) as unset."""
        return None if value == "" else value


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read the environment once per process)."""
    return Settings()
