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

/** Thrown when a stream goes silent for longer than its idle timeout. */
export class StreamStalledError extends Error {
  constructor(idleMs: number) {
    super(`The server stopped responding (no data for ${String(Math.round(idleMs / 1000))}s).`);
    this.name = "StreamStalledError";
  }
}

/** Resolve with `read`, or call `onTimeout` and reject if it takes longer than `ms`. */
function within<T>(read: Promise<T>, ms: number | undefined, onTimeout: () => void): Promise<T> {
  if (!ms) return read;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      // Reject first: `onTimeout` may settle `read` (cancelling a reader resolves
      // its pending read as done), which would otherwise win the race and turn a
      // stall into what looks like a clean end of stream.
      reject(new StreamStalledError(ms));
      onTimeout();
    }, ms);
  });
  return Promise.race([read, timeout]).finally(() => {
    clearTimeout(timer);
  });
}

/**
 * Read a fetch Response body as a stream of SSE messages.
 *
 * With `idleTimeoutMs`, a connection that delivers nothing at all (not even the
 * server's keep-alive pings) for that long is cancelled and a
 * `StreamStalledError` is thrown, instead of waiting forever on a dead socket.
 */
export async function* readSSE(
  response: Response,
  idleTimeoutMs?: number,
): AsyncGenerator<SSEMessage> {
  if (!response.body) throw new Error("Response has no body to stream");
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  const parser = new SSEParser();
  try {
    for (;;) {
      const { done, value } = await within(reader.read(), idleTimeoutMs, () => {
        void reader.cancel();
      });
      if (done) break;
      yield* parser.push(value);
    }
    yield* parser.push("\n\n"); // flush a final message the server did not terminate
  } finally {
    reader.releaseLock();
  }
}
