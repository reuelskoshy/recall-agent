/**
 * Types mirroring the backend's Pydantic models.
 * Keep in sync with backend/app/api/*.py and backend/app/agent/events.py.
 */

/** GET /api/health — mirrors app.api.health.HealthResponse */
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
  | TokenEvent
  | ToolCallEvent
  | ToolResultEvent
  | DoneEvent
  | ErrorEvent;
