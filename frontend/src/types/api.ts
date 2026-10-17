/**
 * Types mirroring the backend's Pydantic models.
 * Keep in sync with backend/app/api/*.py and backend/app/agent/events.py.
 */

/** GET /api/health — mirrors app.api.health.HealthResponse */
/** GET /api/auth/status: whether this deployment needs the demo password. */
export interface AuthStatus {
  required: boolean;
  authenticated: boolean;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  database: "ok" | "unreachable";
  pgvector: string | null;
}

/** POST /api/chat body — mirrors app.api.chat.ChatRequest */
export interface ChatRequest {
  message: string;
  conversation_id?: string | null;
}

/** mirrors app.api.chat.ConversationSummary */
export interface ConversationSummary {
  id: string;
  title: string;
  updated_at: string;
}

export interface StoredToolCall {
  id: string;
  name: string;
  arguments: string;
}

/** mirrors app.api.chat.MessageOut */
export interface StoredMessage {
  id: string;
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  tool_calls: StoredToolCall[] | null;
  tool_call_id: string | null;
  tool_data: Record<string, unknown> | null;
  created_at: string;
}

/** mirrors app.api.chat.ConversationDetail */
export interface ConversationDetail extends ConversationSummary {
  messages: StoredMessage[];
}

// ---- Documents (RAG) — mirror app.api.documents ------------------------------

export type DocumentStatus = "processing" | "ready" | "failed";

/** mirrors app.api.documents.DocumentOut */
export interface DocumentInfo {
  id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  status: DocumentStatus;
  error: string | null;
  page_count: number | null;
  chunk_count: number;
  embedding_model: string;
  created_at: string;
}

// ---- Long-term memory ----------------------------------------------------------

export type MemoryKind = "fact" | "preference" | "goal" | "instruction";

export interface MemoryInfo {
  id: string;
  content: string;
  kind: MemoryKind;
  /** 1 (minor) to 5 (identity or standing instruction). */
  importance: number;
  /** How it was saved: by the agent's tool, by post-turn extraction, or by hand. */
  source: "tool" | "auto" | "manual";
  source_conversation_id: string | null;
  access_count: number;
  created_at: string;
  updated_at: string;
  last_accessed_at: string;
}

/** Each term of the ranking formula, all in 0-1. */
export interface ScoreBreakdown {
  score: number;
  /** Raw cosine similarity to the query. */
  similarity: number;
  /** Similarity rescaled above the relevance floor: the term the score uses. */
  relevance: number;
  recency: number;
  importance: number;
}

export interface ScoredMemoryInfo extends MemoryInfo {
  ranking: ScoreBreakdown;
}

export interface MemorySaved {
  action: "created" | "updated";
  previous_content: string | null;
  memory: MemoryInfo;
}

/** A memory injected into a turn's prompt. Core memories are always included, unscored. */
export interface RecalledMemory extends Partial<ScoreBreakdown> {
  id: string;
  content: string;
  kind: MemoryKind;
  importance_level: number;
  reason: "core" | "relevant";
}

/** query_database's `data` (mirrors the tool's output). */
export interface QueryResultData {
  /** The SQL that actually ran (normalized by the backend's guard). */
  sql: string;
  columns: string[];
  rows: unknown[][];
  row_count: number;
  /** More rows matched than were returned. */
  truncated: boolean;
  elapsed_ms: number;
  /** Where it ran: "postgres" or "snowflake". */
  backend?: string;
  /** The warehouse's own query ID (Snowflake: look it up in Query History). */
  query_id?: string | null;
}

/** One entry of search_documents' `data.results` (mirrors the tool's output). */
export interface Citation {
  n: number;
  document_id: string;
  filename: string;
  page: number | null;
  /** Innermost heading, for documents without pages. */
  section: string | null;
  /** Ready-made label from the backend, e.g. "a.pdf p.3" or "notes.md § Setup". */
  citation: string;
  chunk_index: number;
  similarity: number;
  snippet: string;
}

// ---- Streaming events (SSE) — mirror app.agent.events ------------------------

export type FinishReason = "stop" | "tool_calls" | "length" | "content_filter";

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
}

export interface ConversationStartedEvent {
  type: "conversation";
  conversation_id: string;
  title: string;
  run_id: string;
}

export interface MemoriesRecalledEvent {
  type: "memories";
  memories: RecalledMemory[];
}

export interface TokenEvent {
  type: "token";
  text: string;
}

export interface ToolCallEvent {
  type: "tool_call";
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface ToolResultEvent {
  type: "tool_result";
  id: string;
  name: string;
  /** Exactly what the model saw. */
  content: string;
  is_error: boolean;
  latency_ms: number;
  data: Record<string, unknown> | null;
}

export interface DoneEvent {
  type: "done";
  message_id: string;
  run_id: string;
  finish_reason: FinishReason;
  iterations: number;
  usage: Usage;
}

export interface ErrorEvent {
  type: "error";
  message: string;
  run_id: string | null;
}

/** Discriminated union: switch on `type` and TypeScript narrows the payload. */
export type AgentEvent =
  | ConversationStartedEvent
  | MemoriesRecalledEvent
  | TokenEvent
  | ToolCallEvent
  | ToolResultEvent
  | DoneEvent
  | ErrorEvent;
