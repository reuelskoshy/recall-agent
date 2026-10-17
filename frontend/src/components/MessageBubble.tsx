import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import type { UIMessage } from "../hooks/useChat";

/** One chat message. Assistant text is rendered as Markdown (raw HTML is not allowed). */
export function MessageBubble({ message }: { message: UIMessage }) {
  const { role, content, status, error, usage } = message;
  const isWaiting = status === "streaming" && content === "";

  return (
    <div className={`message message-${role}`}>
      <div className="message-body">
        {role === "assistant" ? (
          isWaiting ? (
            <span className="typing" aria-label="Assistant is typing">
              <span />
              <span />
              <span />
            </span>
          ) : (
            <Markdown remarkPlugins={[remarkGfm]}>{content}</Markdown>
          )
        ) : (
          <p>{content}</p>
        )}
        {status === "streaming" && content !== "" && <span className="cursor" aria-hidden />}
      </div>
      {status === "error" && <div className="message-error">⚠ {error}</div>}
      {status === "stopped" && <div className="message-meta">Stopped</div>}
      {status === "done" && usage && (
        <div className="message-meta">
          {usage.prompt_tokens} in · {usage.completion_tokens} out tokens
        </div>
      )}
    </div>
  );
}
