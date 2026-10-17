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
  created_at: string;
}

/** mirrors app.api.chat.ConversationDetail */
export interface ConversationDetail extends ConversationSummary {
  messages: StoredMessage[];
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
