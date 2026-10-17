/**
 * Chat UI state as pure functions, so it can be unit-tested without React.
 *
 * An assistant turn is an ordered list of parts, e.g.
 *   [text "Let me calculate.", tool calculator, text "The answer is 42."]
 * built either live from stream events (`applyEvent`) or from stored history
 * (`messagesFromStored`).
 */

import type { AgentEvent, Citation, StoredMessage, Usage } from "../types/api";

export type MessageStatus = "streaming" | "done" | "stopped" | "error";

export interface ToolTrace {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  status: "running" | "ok" | "error";
  result?: string;
  latencyMs?: number;
  /** Structured tool output for the UI (e.g. search citations). */
  data?: Record<string, unknown> | null;
}

export type MessagePart = { kind: "text"; text: string } | { kind: "tool"; tool: ToolTrace };

export interface UIMessage {
  id: string;
  role: "user" | "assistant";
  parts: MessagePart[];
  status: MessageStatus;
  error?: string;
  usage?: Usage;
  iterations?: number;
}

/** Fold one stream event into the in-flight assistant message. */
export function applyEvent(message: UIMessage, event: AgentEvent): UIMessage {
  switch (event.type) {
    case "token": {
      const parts = [...message.parts];
      const last = parts.at(-1);
      if (last?.kind === "text") {
        parts[parts.length - 1] = { kind: "text", text: last.text + event.text };
      } else {
        parts.push({ kind: "text", text: event.text });
      }
      return { ...message, parts };
    }
    case "tool_call":
      return {
        ...message,
        parts: [
          ...message.parts,
          {
            kind: "tool",
            tool: { id: event.id, name: event.name, arguments: event.arguments, status: "running" },
          },
        ],
      };
    case "tool_result":
      return {
        ...message,
        parts: message.parts.map((part) =>
          part.kind === "tool" && part.tool.id === event.id
            ? {
                kind: "tool",
                tool: {
                  ...part.tool,
                  status: event.is_error ? "error" : "ok",
                  result: event.content,
                  latencyMs: event.latency_ms,
                  data: event.data,
                },
              }
            : part,
        ),
      };
    case "done":
      return { ...message, status: "done", usage: event.usage, iterations: event.iterations };
    case "error":
      return { ...message, status: "error", error: event.message };
    case "conversation":
      return message;
  }
}

/** Plain text of a message (used for user messages and copy). */
export function messageText(message: UIMessage): string {
  return message.parts
    .filter((p): p is { kind: "text"; text: string } => p.kind === "text")
    .map((p) => p.text)
    .join("");
}

function parseArguments(raw: string): Record<string, unknown> {
  try {
    const value: unknown = JSON.parse(raw);
    if (value && typeof value === "object" && !Array.isArray(value)) {
      return value as Record<string, unknown>;
    }
  } catch {
    // fall through
  }
  return { _raw: raw };
}

/**
 * Rebuild UI messages from stored history. Consecutive assistant/tool rows
 * after a user message belong to one assistant turn.
 */
export function messagesFromStored(stored: StoredMessage[]): UIMessage[] {
  const out: UIMessage[] = [];
  let turn: UIMessage | null = null;

  for (const row of stored) {
    if (row.role === "user") {
      turn = null;
      out.push({
        id: row.id,
        role: "user",
        parts: [{ kind: "text", text: row.content }],
        status: "done",
      });
      continue;
    }
    if (row.role === "assistant") {
      if (!turn) {
        turn = { id: row.id, role: "assistant", parts: [], status: "done" };
        out.push(turn);
      }
      if (row.content) turn.parts.push({ kind: "text", text: row.content });
      for (const call of row.tool_calls ?? []) {
        turn.parts.push({
          kind: "tool",
          tool: {
            id: call.id,
            name: call.name,
            arguments: parseArguments(call.arguments),
            status: "error",
            result: "No result was recorded.",
          },
        });
      }
      continue;
    }
    if (row.role === "tool" && turn) {
      const part = turn.parts.find((p) => p.kind === "tool" && p.tool.id === row.tool_call_id);
      if (part?.kind === "tool") {
        // Error results are prefixed by the backend registry.
        part.tool.status = row.content.startsWith("Error:") ? "error" : "ok";
        part.tool.result = row.content;
        part.tool.data = row.tool_data;
      }
    }
  }
  return out;
}

/** search_documents citations in a message, de-duplicated by label, in first-seen order. */
export function citationsOf(message: UIMessage): Citation[] {
  const seen = new Set<string>();
  const out: Citation[] = [];
  for (const part of message.parts) {
    if (part.kind !== "tool" || part.tool.name !== "search_documents") continue;
    const results = part.tool.data?.results;
    if (!Array.isArray(results)) continue;
    for (const result of results as Citation[]) {
      const key = result.citation;
      if (!seen.has(key)) {
        seen.add(key);
        out.push(result);
      }
    }
  }
  return out;
}

/** Label used in chips; the backend formats it (page, section, or filename). */
export function citationLabel(c: Pick<Citation, "citation">): string {
  return c.citation;
}
