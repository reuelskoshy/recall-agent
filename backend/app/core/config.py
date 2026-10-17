"""Application settings loaded from environment variables (and an optional .env file)."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url


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

    # ---- query_database (read-only SQL) ----
    readonly_db_user: str = Field(
        default="recall_reader",
        pattern=r"^[a-z_][a-z0-9_]{0,62}$",
        description="Login role the SQL tool connects as; it may only SELECT the dataset.",
    )
    readonly_db_password: str = "recall_reader"
    database_readonly_url: str | None = Field(
        default=None,
        description="Explicit URL for the SQL tool. Default: DATABASE_URL with the "
        "read-only user and password swapped in.",
    )
    sql_max_rows: int = Field(default=50, ge=1, le=500, description="Rows returned per query.")
    sql_statement_timeout_ms: int = Field(
        default=5000, ge=100, description="Postgres statement_timeout for the SQL tool."
    )

    llm_provider: str = Field(default="openai", description="openai | fake (see llm/factory.py)")
    llm_timeout_s: float = 60.0
    llm_max_retries: int = Field(
        default=4, ge=0, description="Retries on 429/5xx/network errors, with exponential backoff."
    )
    llm_api_key: str = ""
    llm_chat_model: str = "gpt-5.4-mini"
    llm_base_url: str | None = Field(
        default=None, description="Point at any OpenAI-compatible server, e.g. Ollama's /v1."
    )

    llm_reasoning_field: str | None = Field(
        default=None,
        description="Send each assistant message's reasoning back under this key "
        "(e.g. 'reasoning' for IFM K2). Leave unset for OpenAI.",
    )

    chat_history_limit: int = Field(
        default=20, description="Max past messages sent to the model (token-aware in Phase 5)."
    )

    agent_max_iterations: int = Field(
        default=6, ge=1, description="Max LLM calls per turn; the last one gets no tools."
    )
    agent_turn_timeout_s: float = Field(default=120.0, description="Wall-clock budget per turn.")
    tool_timeout_s: float = Field(default=15.0, description="Default per-tool-call timeout.")
    tool_max_result_chars: int = Field(
        default=4000, description="Tool output longer than this is truncated before the LLM."
    )

    # ---- Embeddings & RAG ----
    embedding_provider: str = Field(
        default="sentence_transformers", description="sentence_transformers | fake"
    )
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = Field(
        default=384, description="Must match the vector(N) column; changing it needs a migration."
    )
    embedding_query_prefix: str = Field(
        default="Represent this sentence for searching relevant passages: ",
        description="BGE's retrieval instruction, prepended to queries (not documents).",
    )
    rag_chunk_size: int = Field(default=1000, description="Max characters per chunk.")
    rag_chunk_overlap: int = Field(default=150, description="Characters shared by neighbours.")
    rag_top_k: int = Field(default=5, description="Passages returned by search_documents.")
    rag_min_similarity: float = Field(
        default=0.5,
        description="Cosine similarity below which a passage counts as irrelevant "
        "(calibrated for bge-small-en-v1.5: on-topic ~0.55-0.8, off-topic <0.48).",
    )
    rag_candidates: int = Field(
        default=20, description="Candidates fetched from each of vector and keyword search."
    )
    max_upload_mb: int = 20

    # ---- Long-term memory ----
    # score = w_sim * similarity + w_rec * recency + w_imp * importance, each term in [0, 1].
    memory_weight_similarity: float = Field(default=0.7, ge=0)
    memory_weight_recency: float = Field(default=0.2, ge=0)
    memory_weight_importance: float = Field(default=0.1, ge=0)
    memory_half_life_days: float = Field(
        default=30.0, gt=0, description="Recency halves every N days since last use."
    )
    memory_min_similarity: float = Field(
        default=0.5, description="Memories less similar than this to the query are never used."
    )
    memory_dedup_similarity: float = Field(
        default=0.9, description="A new memory this similar to an existing one updates it instead."
    )
    memory_inject_k: int = Field(
        default=5, ge=0, description="Top-ranked memories added to the system prompt each turn."
    )
    memory_core_min_importance: int = Field(
        default=4,
        ge=1,
        le=6,
        description="Memories this important (and all instructions) are always in the prompt.",
    )
    memory_core_limit: int = Field(default=10, ge=0, description="Max always-on memories.")
    memory_auto_extract: bool = Field(
        default=True, description="After each turn, ask the LLM to extract memories."
    )

    @property
    def readonly_url(self) -> str:
        """Connection URL for the read-only SQL role."""
        if self.database_readonly_url:
            return self.database_readonly_url
        return (
            make_url(self.database_url)
            .set(username=self.readonly_db_user, password=self.readonly_db_password)
            .render_as_string(hide_password=False)
        )

    @field_validator("llm_base_url", "llm_reasoning_field", "database_readonly_url", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        """Treat blank values in .env (e.g. `LLM_BASE_URL=`) as unset."""
        return None if value == "" else value


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read the environment once per process)."""
    return Settings()
