import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "../api/client";
import { applyEvent, messagesFromStored, type UIMessage } from "../lib/chat";

export type { UIMessage } from "../lib/chat";

interface UseChatOptions {
  /** Called when the server confirms a turn's conversation (new or existing). */
  onConversationActivity?: (conversationId: string) => void;
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** Chat state for one conversation: messages, streaming, stop, load, and reset. */
export function useChat({ onConversationActivity }: UseChatOptions = {}) {
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Abort any in-flight stream when the component using this hook unmounts.
  useEffect(() => () => abortRef.current?.abort(), []);

  const updateMessage = useCallback((id: string, update: (m: UIMessage) => UIMessage) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? update(m) : m)));
  }, []);

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || abortRef.current) return;

      const controller = new AbortController();
      abortRef.current = controller;
      setIsStreaming(true);

      const assistantId = crypto.randomUUID();
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "user",
          parts: [{ kind: "text", text: trimmed }],
          status: "done",
        },
        { id: assistantId, role: "assistant", parts: [], status: "streaming" },
      ]);

      try {
        const events = api.streamChat(
          { message: trimmed, conversation_id: conversationId },
          controller.signal,
        );
        for await (const event of events) {
          if (event.type === "conversation") {
            setConversationId(event.conversation_id);
            onConversationActivity?.(event.conversation_id);
          }
          updateMessage(assistantId, (m) => applyEvent(m, event));
        }
      } catch (err) {
        const stopped = controller.signal.aborted;
        updateMessage(assistantId, (m) =>
          stopped ? { ...m, status: "stopped" } : { ...m, status: "error", error: errorMessage(err) },
        );
      } finally {
        abortRef.current = null;
        setIsStreaming(false);
      }
    },
    [conversationId, onConversationActivity, updateMessage],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setConversationId(null);
    setMessages([]);
    setLoadError(null);
  }, []);

  const load = useCallback(
    async (id: string) => {
      reset();
      setConversationId(id);
      try {
        const detail = await api.getConversation(id);
        setMessages(messagesFromStored(detail.messages));
      } catch (err) {
        setLoadError(errorMessage(err));
      }
    },
    [reset],
  );

  return { conversationId, messages, isStreaming, loadError, send, stop, reset, load };
}
