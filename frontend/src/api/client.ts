import type {
  AgentEvent,
  ChatRequest,
  ConversationDetail,
  ConversationSummary,
  DocumentInfo,
  AuthStatus,
  HealthResponse,
  MemoryInfo,
  MemoryKind,
  MemorySaved,
  ScoredMemoryInfo,
} from "../types/api";
import { readSSE } from "./sse";

// ---- Demo access token ---------------------------------------------------------
// A public deployment may require a shared password; the token it buys is kept
// in localStorage (wrapped in try/catch: storage can be unavailable) and sent as
// a bearer header, which works inside iframes where cookies may be blocked.

const TOKEN_KEY = "recall.demoToken";

export function getToken(): string | null {
  try {
    return window.localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

export function setToken(token: string | null): void {
  try {
    if (token) window.localStorage.setItem(TOKEN_KEY, token);
    else window.localStorage.removeItem(TOKEN_KEY);
  } catch {
    // Not persisted; the user will be asked again next visit.
  }
}

/** Called on any 401, so the app can show the password screen. */
let onUnauthorized: (() => void) | null = null;
export function setUnauthorizedHandler(handler: (() => void) | null): void {
  onUnauthorized = handler;
}

function withAuth(headers: Headers): Headers {
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  return headers;
}

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
  const headers = withAuth(new Headers(init?.headers));
  headers.set("Accept", "application/json");
  const response = await fetch(`/api${path}`, { ...init, headers });
  if (response.status === 401) onUnauthorized?.();
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
    headers: withAuth(
      new Headers({ "Content-Type": "application/json", Accept: "text/event-stream" }),
    ),
    body: JSON.stringify(body),
    signal,
  });
  if (response.status === 401) onUnauthorized?.();
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
  authStatus: () => request<AuthStatus>("/auth/status"),
  login: (password: string) =>
    request<{ token: string }>("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    }),
  // /health returns a body on 503 too (status "degraded"), so it is not treated as an error.
  health: () => request<HealthResponse>("/health", undefined, [503]),
  listConversations: () => request<ConversationSummary[]>("/conversations"),
  getConversation: (id: string) => request<ConversationDetail>(`/conversations/${id}`),
  deleteConversation: (id: string) =>
    request<undefined>(`/conversations/${id}`, { method: "DELETE" }),
  streamChat,
  listDocuments: () => request<DocumentInfo[]>("/documents"),
  /** multipart upload; the browser sets the Content-Type boundary itself. */
  uploadDocument: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<DocumentInfo>("/documents", { method: "POST", body });
  },
  listMemories: () => request<MemoryInfo[]>("/memories"),
  searchMemories: (q: string) =>
    request<ScoredMemoryInfo[]>(`/memories?${new URLSearchParams({ q }).toString()}`),
  createMemory: (body: { content: string; kind: MemoryKind; importance: number }) =>
    request<MemorySaved>("/memories", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  updateMemory: (
    id: string,
    body: Partial<{ content: string; kind: MemoryKind; importance: number }>,
  ) =>
    request<MemoryInfo>(`/memories/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  deleteMemory: (id: string) => request<undefined>(`/memories/${id}`, { method: "DELETE" }),
  deleteDocument: (id: string) => request<undefined>(`/documents/${id}`, { method: "DELETE" }),
};
