import { useEffect, useRef } from "react";

import type { UIMessage } from "../hooks/useChat";
import { Composer } from "./Composer";
import { MessageBubble } from "./MessageBubble";

interface ChatViewProps {
  messages: UIMessage[];
  isStreaming: boolean;
  loadError: string | null;
  onSend: (text: string) => void;
  onStop: () => void;
}

const SUGGESTIONS = [
  "Explain retrieval-augmented generation in two sentences.",
  "What makes a good tool description for an LLM agent?",
  "Write a SQL query that finds the top 5 customers by revenue.",
];

/** Message list (auto-scrolling) plus the composer. */
export function ChatView({ messages, isStreaming, loadError, onSend, onStop }: ChatViewProps) {
  const endRef = useRef<HTMLDivElement>(null);
  const lastContent = messages.at(-1)?.content;

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [messages.length, lastContent]);

  return (
    <section className="chat">
      <div className="chat-messages">
        {loadError && <p className="message-error">Could not load conversation: {loadError}</p>}
        {messages.length === 0 && !loadError && (
          <div className="chat-empty">
            <h2>What can I help with?</h2>
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  type="button"
                  className="suggestion"
                  onClick={() => {
                    onSend(s);
                  }}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}
        <div ref={endRef} />
      </div>
      <Composer isStreaming={isStreaming} onSend={onSend} onStop={onStop} />
    </section>
  );
}
