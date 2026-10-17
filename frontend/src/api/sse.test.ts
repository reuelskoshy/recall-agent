import { describe, expect, it } from "vitest";

import { SSEParser } from "./sse";

describe("SSEParser", () => {
  it("parses named events with JSON data", () => {
    const parser = new SSEParser();
    const messages = parser.push('event: token\ndata: {"text":"Hi"}\n\n');
    expect(messages).toEqual([{ event: "token", data: '{"text":"Hi"}' }]);
  });

  it("reassembles a message split across arbitrary chunk boundaries", () => {
    const parser = new SSEParser();
    const wire = 'event: token\r\ndata: {"text":"a"}\r\n\r\nevent: done\r\ndata: {}\r\n\r\n';
    const out = [];
    for (const char of wire) out.push(...parser.push(char)); // worst case: one char per chunk
    expect(out).toEqual([
      { event: "token", data: '{"text":"a"}' },
      { event: "done", data: "{}" },
    ]);
  });

  it("ignores keep-alive comments and defaults the event name to 'message'", () => {
    const parser = new SSEParser();
    expect(parser.push(": ping\n\ndata: x\n\n")).toEqual([{ event: "message", data: "x" }]);
  });

  it("joins multi-line data with newlines", () => {
    const parser = new SSEParser();
    expect(parser.push("data: line1\ndata: line2\n\n")).toEqual([
      { event: "message", data: "line1\nline2" },
    ]);
  });

  it("does not emit until the blank line arrives", () => {
    const parser = new SSEParser();
    expect(parser.push("event: token\ndata: {}\n")).toEqual([]);
    expect(parser.push("\n")).toEqual([{ event: "token", data: "{}" }]);
  });
});
