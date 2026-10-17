import type {
  AgentEvent,
  ChatRequest,
  ConversationDetail,
  ConversationSummary,
  HealthResponse,
} from "../types/api";
import { readSSE } from "./sse";

/** Error carrying the HTTP status, so callers can tell 4xx from 5xx from network failures. */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/** Turn a non-OK response into an ApiError, using FastAPI's `detail` when present. */
async function toApiError(response: Response, path: string): Promise<ApiError> {
  let detail = `${path} failed with ${String(response.status)}`;
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") detail = body.detail;
  } catch {
    // Body was not JSON; keep the generic message.
  }
  return new ApiError(response.status, detail);
}

async function request<T>(path: string, init?: RequestInit, okStatuses: number[] = []): Promise<T> {
  const headers = new Headers(init?.headers);
  headers.set("Accept", "application/json");
  const response = await fetch(`/api${path}`, { ...init, headers });
  if (!response.ok && !okStatuses.includes(response.status)) {
    throw await toApiError(response, path);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

/**
 * POST /api/chat and yield typed events as they stream in.
 * Throws ApiError if the server rejects the request before streaming (404, 422, 503).
 */
async function* streamChat(body: ChatRequest, signal?: AbortSignal): AsyncGenerator<AgentEvent> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) throw await toApiError(response, "/chat");

  for await (const message of readSSE(response)) {
    const event = JSON.parse(message.data) as AgentEvent;
    if (event.type !== message.event) {
      throw new Error(`SSE event name "${message.event}" does not match payload "${event.type}"`);
    }
    yield event;
  }
}

/** Typed wrappers for every backend endpoint live here. */
export const api = {
  // /health returns a body on 503 too (status "degraded"), so it is not treated as an error.
  health: () => request<HealthResponse>("/health", undefined, [503]),
  listConversations: () => request<ConversationSummary[]>("/conversations"),
  getConversation: (id: string) => request<ConversationDetail>(`/conversations/${id}`),
  deleteConversation: (id: string) =>
    request<undefined>(`/conversations/${id}`, { method: "DELETE" }),
  streamChat,
};
