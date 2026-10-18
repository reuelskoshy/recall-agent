import Markdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import { citationLabel, citationsOf, messageText, type UIMessage } from "../lib/chat";
import type { RecalledMemory } from "../types/api";
import { ToolTraceCard } from "./ToolTraceCard";

/** Which long-term memories shaped this answer, and why each was included. */
/** Links in answers (e.g. web citations) open in a new tab and pass no referrer. */
const MARKDOWN_COMPONENTS: Components = {
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noopener noreferrer nofollow">
      {children}
    </a>
  ),
};

function MemoriesUsed({ memories }: { memories: RecalledMemory[] }) {
  return (
    <details className="memories-used">
      <summary>
        Using {memories.length} memor{memories.length === 1 ? "y" : "ies"}
      </summary>
      <ul>
        {memories.map((m) => (
          <li key={m.id}>
            {m.content}{" "}
            <span className="hit-score">
              {m.reason === "core"
                ? "· always on"
                : `· score ${(m.score ?? 0).toFixed(2)} (sim ${(m.similarity ?? 0).toFixed(2)})`}
            </span>
          </li>
        ))}
      </ul>
    </details>
  );
}

function Typing() {
  return (
    <span className="typing" aria-label="Assistant is thinking">
      <span />
      <span />
      <span />
    </span>
  );
}

/** One chat message. Assistant turns render text and tool-trace parts in order. */
export function MessageBubble({ message }: { message: UIMessage }) {
  const { role, parts, status, error, usage, iterations, memories } = message;

  if (role === "user") {
    return (
      <div className="message message-user">
        <div className="message-body">
          <p>{messageText(message)}</p>
        </div>
      </div>
    );
  }

  const last = parts.at(-1);
  // Waiting on the model: nothing yet, or tools finished and the next LLM call is pending.
  const waiting =
    status === "streaming" && (!last || (last.kind === "tool" && last.tool.status !== "running"));
  const toolCount = parts.filter((p) => p.kind === "tool").length;
  const sources = status === "streaming" ? [] : citationsOf(message);

  return (
    <div className="message message-assistant">
      {memories && memories.length > 0 && <MemoriesUsed memories={memories} />}
      <div className="message-body">
        {parts.map((part, i) =>
          part.kind === "text" ? (
            <Markdown key={i} remarkPlugins={[remarkGfm]} components={MARKDOWN_COMPONENTS}>
              {part.text}
            </Markdown>
          ) : (
            <ToolTraceCard key={part.tool.id} tool={part.tool} />
          ),
        )}
        {waiting && <Typing />}
        {status === "streaming" && last?.kind === "text" && <span className="cursor" aria-hidden />}
      </div>
      {sources.length > 0 && (
        <div className="sources" aria-label="Sources">
          <span className="sources-label">Sources</span>
          {sources.map((c) => (
            <span key={c.citation} className="citation" title={c.snippet}>
              {citationLabel(c)}
            </span>
          ))}
        </div>
      )}
      {status === "error" && <div className="message-error">⚠ {error}</div>}
      {status === "stopped" && <div className="message-meta">Stopped</div>}
      {status === "done" && usage && (
        <div className="message-meta">
          {toolCount > 0 && `${String(toolCount)} tool call${toolCount > 1 ? "s" : ""} · `}
          {iterations !== undefined && iterations > 1 && `${String(iterations)} steps · `}
          {usage.prompt_tokens} in · {usage.completion_tokens} out tokens
        </div>
      )}
    </div>
  );
}
