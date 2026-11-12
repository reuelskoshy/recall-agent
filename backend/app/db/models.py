"""ORM models. Alembic autogenerate compares these against the live schema."""

import uuid
from datetime import date, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Computed,
    Date,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    LargeBinary,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Conversation(Base):
    """A chat session. Its messages are the agent's short-term memory."""

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.seq",
    )


class Message(Base):
    """One message. Mirrors `llm.base.ChatMessage` so history round-trips losslessly."""

    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_conversation_seq", "conversation_id", "seq"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    # Strict insertion order. Timestamps can tie when several rows are inserted in
    # one statement (an assistant tool call and its results), so order by this.
    seq: Mapped[int] = mapped_column(BigInteger, Identity(), unique=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text, default="")
    # Assistant tool requests, as [{"id", "name", "arguments"}].
    tool_calls: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)
    tool_call_id: Mapped[str | None] = mapped_column(String(100))
    # A reasoning model's hidden thinking for this assistant message.
    reasoning: Mapped[str | None] = mapped_column(Text)
    # Structured tool output for the UI (e.g. search citations); never sent to the LLM.
    tool_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.clock_timestamp()
    )

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class AgentRun(Base):
    """Observability: one row per agent turn (status, totals, latency)."""

    __tablename__ = "agent_runs"
    __table_args__ = (Index("ix_agent_runs_conversation_created", "conversation_id", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    user_input: Mapped[str] = mapped_column(Text)
    # running | completed | failed | timeout | cancelled (client left) | interrupted (reaper)
    status: Mapped[str] = mapped_column(String(20), default="running")
    provider: Mapped[str] = mapped_column(String(50))
    model: Mapped[str] = mapped_column(String(200))
    iterations: Mapped[int] = mapped_column(Integer, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
    # Long-term memories injected into this run's prompt: [{id, content, score}].
    memories_used: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    conversation: Mapped[Conversation] = relationship()
    steps: Mapped[list["AgentStep"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="AgentStep.index",
    )


class AgentStep(Base):
    """Observability: one LLM call or one tool call within a run."""

    __tablename__ = "agent_steps"
    __table_args__ = (Index("ix_agent_steps_run_index", "run_id", "index"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"))
    index: Mapped[int] = mapped_column(Integer)
    iteration: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(10))  # llm | tool
    name: Mapped[str] = mapped_column(String(200))  # model name or tool name
    arguments: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    result_summary: Mapped[str] = mapped_column(Text, default="")
    is_error: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[int] = mapped_column(Integer)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.clock_timestamp()
    )

    run: Mapped[AgentRun] = relationship(back_populates="steps")


# Keep in sync with Settings.embedding_dim; changing it requires a migration + re-embedding.
EMBEDDING_DIM = 384


class Document(Base):
    """An uploaded file. Its text lives in `DocumentChunk` rows."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(Integer)
    # Identical uploads are detected by content hash, not by name.
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    # processing | ready | failed
    status: Mapped[str] = mapped_column(String(20), default="processing")
    error: Mapped[str | None] = mapped_column(Text)
    page_count: Mapped[int | None] = mapped_column(Integer)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding_model: Mapped[str] = mapped_column(String(200))
    # The original upload, kept so documents can be re-chunked or re-embedded later
    # (new chunking settings or embedding model). Deferred: never loaded unless asked.
    data: Mapped[bytes | None] = mapped_column(LargeBinary, deferred=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="DocumentChunk.chunk_index",
    )


class DocumentChunk(Base):
    """One searchable piece of a document: text, page, vector, and keyword index."""

    __tablename__ = "document_chunks"
    __table_args__ = (
        # HNSW: approximate nearest-neighbour graph index. Fast queries without a
        # training step (unlike IVFFlat), at the cost of slower inserts and more memory.
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_document_chunks_content_tsv", "content_tsv", postgresql_using="gin"),
        Index("ix_document_chunks_document_index", "document_id", "chunk_index"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    chunk_index: Mapped[int] = mapped_column(Integer)
    page: Mapped[int | None] = mapped_column(Integer)
    # Innermost heading (Markdown); cites pageless documents precisely.
    section: Mapped[str | None] = mapped_column(String(300))
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    # Maintained by Postgres from `content`: powers the keyword half of hybrid search.
    content_tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', content)", persisted=True)
    )

    document: Mapped[Document] = relationship(back_populates="chunks")


class Memory(Base):
    """Long-term memory: one durable fact about the user, shared across conversations.

    Ranked at recall time by similarity, recency of use, and importance
    (see app/memory/long_term.py).
    """

    __tablename__ = "memories"
    __table_args__ = (
        CheckConstraint("importance BETWEEN 1 AND 5", name="importance"),
        Index(
            "ix_memories_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    content: Mapped[str] = mapped_column(Text)
    # fact | preference | goal | instruction
    kind: Mapped[str] = mapped_column(String(20))
    importance: Mapped[int] = mapped_column(Integer, default=3)
    # tool (agent called save_memory) | auto (extracted after a turn) | manual (UI/API)
    source: Mapped[str] = mapped_column(String(10))
    # Deferred: only compared inside SQL, so listing memories never ships vectors.
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), deferred=True)
    # Memories outlive the conversation they came from.
    source_conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL")
    )
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    # Recency is measured from the last time the memory was recalled (or saved).
    last_accessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ---------------------------------------------------------------------------
# Recall Jobs: a private job-search pipeline (see docs/jobs-plan.md).
#
# Allowed values are plain strings guarded by CHECK constraints rather than
# Postgres ENUM types: adding a value is then a one-line constraint change,
# not an `ALTER TYPE` that cannot run inside a transaction.
# ---------------------------------------------------------------------------

ENTRY_KINDS = ("education", "experience", "project", "link", "achievement")
ATS_KINDS = ("greenhouse", "lever", "ashby", "none")
CONTACT_POLICIES = ("allowed", "application_form_only", "do_not_contact")
JOB_SOURCES = ("greenhouse", "lever", "ashby", "hn", "web_search")
VERDICTS = ("eligible", "flagged", "rejected", "error")
DRAFT_KINDS = ("outreach_email", "application_answers")
DRAFT_STATUSES = ("needs_review", "approved", "discarded")
APPLICATION_STATUSES = (
    "new",
    "shortlisted",
    "drafted",
    "applied",
    "interview",
    "offer",
    "rejected",
    "withdrawn",
    "skipped",
)
RUN_TRIGGERS = ("cron", "manual")
RUN_STATUSES = ("running", "completed", "partial", "failed", "interrupted")


def _one_of(column: str, values: tuple[str, ...], nullable: bool = False) -> CheckConstraint:
    """CHECK that `column` holds one of `values` (named ck_<table>_<column>)."""
    listed = ", ".join(f"'{v}'" for v in values)
    condition = f"{column} IN ({listed})"
    if nullable:
        condition = f"{column} IS NULL OR {condition}"
    return CheckConstraint(condition, name=column)


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now())


def _updated_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class CandidateProfile(Base):
    """The one candidate's preferences and headline facts (exactly one row, id = 1).

    Details (education, jobs, projects) are `ProfileEntry` rows and skills are
    `ProfileSkill` rows, so the claims check can match a draft against
    individual facts instead of free text.
    """

    __tablename__ = "candidate_profile"
    __table_args__ = (
        CheckConstraint("id = 1", name="single_row"),
        CheckConstraint("years_experience >= 0", name="years_experience"),
        CheckConstraint("min_salary IS NULL OR min_salary >= 0", name="min_salary"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1, autoincrement=False)
    full_name: Mapped[str] = mapped_column(String(200), default="")
    headline: Mapped[str] = mapped_column(String(300), default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    graduation_year: Mapped[int | None] = mapped_column(Integer)
    degree: Mapped[str | None] = mapped_column(String(100))
    degree_field: Mapped[str | None] = mapped_column(String(200))
    years_experience: Mapped[float] = mapped_column(Numeric(4, 1, asdecimal=False), default=0)
    allowed_locations: Mapped[list[str]] = mapped_column(
        ARRAY(String(200)), default=list, server_default="{}"
    )
    remote_ok: Mapped[bool] = mapped_column(Boolean, default=True)
    role_families: Mapped[list[str]] = mapped_column(
        ARRAY(String(30)), default=lambda: ["engineering"], server_default="{engineering}"
    )
    employment_types: Mapped[list[str]] = mapped_column(
        ARRAY(String(30)), default=list, server_default="{}"
    )
    min_salary: Mapped[int | None] = mapped_column(Integer)
    salary_currency: Mapped[str | None] = mapped_column(String(3))
    # Profile summary embedded for the semantic match (Phase 4); refreshed on edit.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), deferred=True)
    updated_at: Mapped[datetime] = _updated_at()


class ProfileEntry(Base):
    """One dated fact block: a degree, a job, a project, a link, an achievement."""

    __tablename__ = "profile_entries"
    __table_args__ = (
        _one_of("kind", ENTRY_KINDS),
        CheckConstraint(
            "end_date IS NULL OR start_date IS NULL OR end_date >= start_date", name="dates"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    kind: Mapped[str] = mapped_column(String(20))
    title: Mapped[str] = mapped_column(String(300))
    organization: Mapped[str | None] = mapped_column(String(300))
    location: Mapped[str | None] = mapped_column(String(200))
    start_date: Mapped[date | None] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)  # NULL = ongoing
    description: Mapped[str] = mapped_column(Text, default="")
    technologies: Mapped[list[str]] = mapped_column(
        ARRAY(String(100)), default=list, server_default="{}"
    )
    url: Mapped[str | None] = mapped_column(String(2000))
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()


class ProfileSkill(Base):
    """A skill the candidate can claim. Names are unique ignoring case."""

    __tablename__ = "profile_skills"
    __table_args__ = (
        Index("uq_profile_skills_lower_name", func.lower(text("name")), unique=True),
        CheckConstraint("years IS NULL OR years >= 0", name="years"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(100))
    category: Mapped[str | None] = mapped_column(String(50))
    years: Mapped[float | None] = mapped_column(Numeric(4, 1, asdecimal=False))
    created_at: Mapped[datetime] = _created_at()


class ResumeVariant(Base):
    """A resume tailored to a kind of role. The file itself is an uploaded `Document`."""

    __tablename__ = "resume_variants"
    __table_args__ = (
        Index(
            "uq_resume_variants_one_default",
            "is_default",
            unique=True,
            postgresql_where=text("is_default"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    target_role_family: Mapped[str | None] = mapped_column(String(30))
    # SET NULL: deleting the document must not fail; the variant just loses its file.
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL")
    )
    notes: Mapped[str] = mapped_column(Text, default="")
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()


class Company(Base):
    """An employer: where its jobs come from, and whether it may be contacted.

    `contact_policy` defaults to application_form_only, so a company discovered
    by search is never contactable until the candidate decides it is.
    """

    __tablename__ = "companies"
    __table_args__ = (
        _one_of("ats", ATS_KINDS),
        _one_of("contact_policy", CONTACT_POLICIES),
        CheckConstraint("ats = 'none' OR board_token IS NOT NULL", name="board_token"),
        Index("uq_companies_lower_name", func.lower(text("name")), unique=True),
        Index(
            "uq_companies_ats_board_token",
            "ats",
            "board_token",
            unique=True,
            postgresql_where=text("board_token IS NOT NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200))
    website: Mapped[str | None] = mapped_column(String(500))
    ats: Mapped[str] = mapped_column(String(20), default="none")
    # The company's id on its job board, e.g. "stripe" in boards.greenhouse.io/stripe.
    board_token: Mapped[str | None] = mapped_column(String(200))
    # True = fetch its board on every daily run.
    watch: Mapped[bool] = mapped_column(Boolean, default=False)
    contact_policy: Mapped[str] = mapped_column(
        String(30), default="application_form_only", server_default="application_form_only"
    )
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()


class Job(Base):
    """One listing, deduplicated by (source, external_id).

    `content_hash` covers the normalised title, location and text: when a
    listing is edited the hash changes and the job is analysed again. The id is
    a small integer so the candidate can say "draft outreach for job 12".
    """

    __tablename__ = "jobs"
    __table_args__ = (
        _one_of("source", JOB_SOURCES),
        UniqueConstraint("source", "external_id", name="uq_jobs_source_external_id"),
        Index("ix_jobs_first_seen_at", "first_seen_at"),
        Index("ix_jobs_company_id", "company_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    source: Mapped[str] = mapped_column(String(20))
    external_id: Mapped[str] = mapped_column(String(300))
    company_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("companies.id", ondelete="SET NULL")
    )
    # As the source names it; `company_id` is set once it is matched to a Company.
    company_name: Mapped[str] = mapped_column(String(200), default="")
    title: Mapped[str] = mapped_column(String(500))
    url: Mapped[str] = mapped_column(String(2000))
    location_raw: Mapped[str] = mapped_column(String(500), default="")
    raw_text: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    content_changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    # Set when the listing disappears from its feed; cleared if it comes back.
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    company: Mapped[Company | None] = relationship()


class JobAnalysis(Base):
    """What was extracted from one version of a listing, and how it was judged.

    Extraction (an LLM call) is keyed by the content hash, so it runs once per
    version of a listing. The verdict and score are pure code, stamped with
    `rules_version`: changing a rule or a weight re-judges every job with no
    LLM calls.
    """

    __tablename__ = "job_analyses"
    __table_args__ = (
        UniqueConstraint("job_id", "content_hash", name="uq_job_analyses_job_id_content_hash"),
        _one_of("verdict", VERDICTS, nullable=True),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    content_hash: Mapped[str] = mapped_column(String(64))
    # The validated extraction as JSON; NULL when extraction failed (see extraction_error).
    extraction: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    extractor_version: Mapped[str] = mapped_column(String(50))
    extraction_error: Mapped[str | None] = mapped_column(Text)
    # Failed extractions of this version; the pipeline gives up after JOBS_EXTRACT_MAX_ATTEMPTS.
    extraction_attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    rules_version: Mapped[str | None] = mapped_column(String(50))
    verdict: Mapped[str | None] = mapped_column(String(20))  # NULL until judged
    # [{rule, outcome, detail, evidence}]: each rule that fired, with the sentence behind it.
    reasons: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, server_default=text("'[]'::jsonb")
    )
    score: Mapped[float | None] = mapped_column(Numeric(6, 4, asdecimal=False))
    # {component: {value, weight, contribution}}, so the UI can explain the rank.
    score_components: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), deferred=True)
    created_at: Mapped[datetime] = _created_at()
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    job: Mapped[Job] = relationship()


class Draft(Base):
    """A message the candidate may send. Created as needs_review; Recall never sends it.

    A database trigger (migration 0009) refuses an outreach_email unless the
    job's company has contact_policy = 'allowed', backing up the same check in code.
    """

    __tablename__ = "drafts"
    __table_args__ = (
        _one_of("kind", DRAFT_KINDS),
        _one_of("status", DRAFT_STATUSES),
        CheckConstraint("attempts >= 1", name="attempts"),
        Index("ix_drafts_job_id", "job_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(30))
    resume_variant_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("resume_variants.id", ondelete="SET NULL")
    )
    resume_reason: Mapped[str] = mapped_column(Text, default="")
    body: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="needs_review")
    # {passed, unsupported: [{claim, kind}]}
    claims_check: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    job: Mapped[Job] = relationship()


class Application(Base):
    """Progress on one job. Allowed transitions live in app/jobs/applications.py."""

    __tablename__ = "applications"
    __table_args__ = (
        _one_of("status", APPLICATION_STATUSES),
        Index("ix_applications_follow_up_on", "follow_up_on"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), unique=True)
    status: Mapped[str] = mapped_column(String(20), default="new")
    follow_up_on: Mapped[date | None] = mapped_column(Date)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = _created_at()
    status_changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    job: Mapped[Job] = relationship()
    events: Mapped[list["ApplicationEvent"]] = relationship(
        back_populates="application",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ApplicationEvent.id",
    )


class ApplicationEvent(Base):
    """Audit log: one row per status change, so the whole history has timestamps."""

    __tablename__ = "application_events"
    __table_args__ = (Index("ix_application_events_application_id", "application_id"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    application_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("applications.id", ondelete="CASCADE")
    )
    from_status: Mapped[str | None] = mapped_column(String(20))
    to_status: Mapped[str] = mapped_column(String(20))
    at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.clock_timestamp()
    )

    application: Mapped[Application] = relationship(back_populates="events")


class IngestionRun(Base):
    """One daily (or manual) pipeline run. Always ends in a terminal status.

    At most one run can be `running` (partial unique index), so a retried cron
    call cannot start a duplicate run.
    """

    __tablename__ = "ingestion_runs"
    __table_args__ = (
        _one_of("trigger", RUN_TRIGGERS),
        _one_of("status", RUN_STATUSES),
        Index(
            "uq_ingestion_runs_one_running",
            "status",
            unique=True,
            postgresql_where=text("status = 'running'"),
        ),
        Index("ix_ingestion_runs_started_at", "started_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    trigger: Mapped[str] = mapped_column(String(10))
    status: Mapped[str] = mapped_column(String(20), default="running")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # {source: {fetched, new, updated, unchanged, closed, errors}}
    counts: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    # [{source, error_type, message}]: messages never contain profile data.
    errors: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, default=list, server_default=text("'[]'::jsonb")
    )
