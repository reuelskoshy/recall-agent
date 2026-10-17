/**
 * Minimal Server-Sent Events parser.
 *
 * The browser's EventSource only supports GET, but /api/chat is a POST, so we
 * read the response body as a stream and parse the SSE wire format ourselves.
 * Spec: https://html.spec.whatwg.org/multipage/server-sent-events.html
 */

export interface SSEMessage {
  event: string;
  data: string;
}

/** Incremental parser: feed it text chunks in any split, get complete messages out. */
export class SSEParser {
  private buffer = "";
  private event = "";
  private dataLines: string[] = [];

  /** Add a chunk of text and return every message it completed. */
  push(chunk: string): SSEMessage[] {
    this.buffer += chunk;
    // A trailing "\r" might be the first half of "\r\n": wait for the next chunk.
    const holdBack = this.buffer.endsWith("\r") ? 1 : 0;
    const ready = this.buffer.slice(0, this.buffer.length - holdBack).replace(/\r\n?/g, "\n");
    const lastNewline = ready.lastIndexOf("\n");
    if (lastNewline === -1) return [];

    this.buffer = ready.slice(lastNewline + 1) + this.buffer.slice(this.buffer.length - holdBack);
    const messages: SSEMessage[] = [];
    for (const line of ready.slice(0, lastNewline).split("\n")) {
      const message = this.processLine(line);
      if (message) messages.push(message);
    }
    return messages;
  }

  private processLine(line: string): SSEMessage | null {
    if (line === "") {
      // A blank line dispatches the message being built.
      if (this.dataLines.length === 0) {
        this.event = "";
        return null;
      }
      const message = { event: this.event || "message", data: this.dataLines.join("\n") };
      this.event = "";
      this.dataLines = [];
      return message;
    }
    if (line.startsWith(":")) return null; // comment, e.g. keep-alive ping

    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);

    if (field === "event") this.event = value;
    else if (field === "data") this.dataLines.push(value);
    // "id" and "retry" are not needed by this app.
    return null;
  }
}

/** Read a fetch Response body as a stream of SSE messages. */
export async function* readSSE(response: Response): AsyncGenerator<SSEMessage> {
  if (!response.body) throw new Error("Response has no body to stream");
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  const parser = new SSEParser();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      yield* parser.push(value);
    }
    yield* parser.push("\n\n"); // flush a final message the server did not terminate
  } finally {
    reader.releaseLock();
  }
}
