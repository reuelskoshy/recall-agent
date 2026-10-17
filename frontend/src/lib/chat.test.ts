import { describe, expect, it } from "vitest";

import type { AgentEvent, StoredMessage } from "../types/api";
import { applyEvent, citationLabel, citationsOf, messagesFromStored, type UIMessage } from "./chat";

const empty: UIMessage = { id: "a", role: "assistant", parts: [], status: "streaming" };

function fold(events: AgentEvent[]): UIMessage {
  return events.reduce(applyEvent, empty);
}

describe("applyEvent", () => {
  it("interleaves text and tool parts in stream order", () => {
    const message = fold([
      { type: "token", text: "Let me " },
      { type: "token", text: "check." },
      { type: "tool_call", id: "c1", name: "calculator", arguments: { expression: "6*7" } },
      {
        type: "tool_result",
        id: "c1",
        name: "calculator",
        content: "6*7 = 42",
        is_error: false,
        latency_ms: 3,
        data: null,
      },
      { type: "token", text: "It's 42." },
      {
        type: "done",
        message_id: "m",
        run_id: "r",
        finish_reason: "stop",
        iterations: 2,
        usage: { prompt_tokens: 10, completion_tokens: 5 },
      },
    ]);

    expect(message.parts).toEqual([
      { kind: "text", text: "Let me check." },
      {
        kind: "tool",
        tool: {
          id: "c1",
          name: "calculator",
          arguments: { expression: "6*7" },
          status: "ok",
          result: "6*7 = 42",
          latencyMs: 3,
          data: null,
        },
      },
      { kind: "text", text: "It's 42." },
    ]);
    expect(message.status).toBe("done");
    expect(message.iterations).toBe(2);
  });

  it("marks failed tools and leaves other parallel tools running", () => {
    const message = fold([
      { type: "tool_call", id: "a", name: "calculator", arguments: {} },
      { type: "tool_call", id: "b", name: "get_current_datetime", arguments: {} },
      {
        type: "tool_result",
        id: "a",
        name: "calculator",
        content: "Error: bad",
        is_error: true,
        latency_ms: 1,
        data: null,
      },
    ]);

    const statuses = message.parts.map((p) => (p.kind === "tool" ? p.tool.status : null));
    expect(statuses).toEqual(["error", "running"]);
  });

  it("does not mutate the previous message", () => {
    const before = fold([{ type: "token", text: "a" }]);
    const snapshot = structuredClone(before);
    applyEvent(before, { type: "token", text: "b" });
    expect(before).toEqual(snapshot);
  });
});

describe("messagesFromStored", () => {
  const row = (partial: Partial<StoredMessage>): StoredMessage => ({
    id: crypto.randomUUID(),
    role: "user",
    content: "",
    tool_calls: null,
    tool_call_id: null,
    tool_data: null,
    created_at: "",
    ...partial,
  });

  it("groups assistant and tool rows of one turn into one message", () => {
    const messages = messagesFromStored([
      row({ role: "user", content: "6*7 and 2^3?" }),
      row({
        role: "assistant",
        content: "Checking.",
        tool_calls: [
          { id: "c1", name: "calculator", arguments: '{"expression":"6*7"}' },
          { id: "c2", name: "calculator", arguments: '{"expression":"2^3"}' },
        ],
      }),
      row({ role: "tool", content: "6*7 = 42", tool_call_id: "c1" }),
      row({ role: "tool", content: "Error: Use ** for powers", tool_call_id: "c2" }),
      row({ role: "assistant", content: "42, and use ** for powers." }),
      row({ role: "user", content: "thanks" }),
    ]);

    expect(messages.map((m) => m.role)).toEqual(["user", "assistant", "user"]);
    const assistant = messages[1];
    expect(assistant?.parts.map((p) => (p.kind === "tool" ? p.tool.status : p.text))).toEqual([
      "Checking.",
      "ok",
      "error",
      "42, and use ** for powers.",
    ]);
  });
});

describe("citationsOf", () => {
  const hit = (filename: string, page: number | null, n: number, section: string | null = null) => ({
    n,
    document_id: "d",
    filename,
    page,
    section,
    citation: page ? `${filename} p.${String(page)}` : section ? `${filename} § ${section}` : filename,
    chunk_index: n,
    similarity: 0.8,
    snippet: "",
  });

  it("collects search citations across tool calls, de-duplicated by file and page", () => {
    const message = fold([
      { type: "tool_call", id: "s1", name: "search_documents", arguments: {} },
      {
        type: "tool_result",
        id: "s1",
        name: "search_documents",
        content: "",
        is_error: false,
        latency_ms: 5,
        data: {
          results: [hit("a.pdf", 2, 1), hit("a.pdf", 2, 2), hit("b.md", null, 3, "Setup")],
        },
      },
      { type: "tool_call", id: "c1", name: "calculator", arguments: {} },
    ]);

    expect(citationsOf(message).map(citationLabel)).toEqual(["a.pdf p.2", "b.md § Setup"]);
  });
});
