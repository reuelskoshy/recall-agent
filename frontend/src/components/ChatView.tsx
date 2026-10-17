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
  "How many vacation days do I get, and how many carry over?",
  "Which 3 customers brought in the most revenue in 2026?",
  "Who has unused leave above the carry-over limit this year?",
  "What time is it in Tokyo and in New York right now?",
];

/** Message list (auto-scrolling) plus the composer. */
export function ChatView({ messages, isStreaming, loadError, onSend, onStop }: ChatViewProps) {
  const endRef = useRef<HTMLDivElement>(null);
  // Re-scroll whenever the last message grows (new text or a new tool card).
  const lastMessage = messages.at(-1);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [messages.length, lastMessage]);

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
