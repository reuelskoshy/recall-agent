import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, api } from "../api/client";
import type { Usage } from "../types/api";

export type MessageStatus = "streaming" | "done" | "stopped" | "error";

/** A message as the chat UI shows it (stored messages and in-flight ones alike). */
export interface UIMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: MessageStatus;
  error?: string;
  usage?: Usage | null;
}

interface UseChatOptions {
  /** Called when the server confirms a turn's conversation (new or existing). */
  onConversationActivity?: (conversationId: string) => void;
}

function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return String(err);
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

  const updateMessage = useCallback((id: string, patch: (m: UIMessage) => Partial<UIMessage>) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch(m) } : m)));
  }, []);

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || abortRef.current) return;

      const controller = new AbortController();
      abortRef.current = controller;
      setIsStreaming(true);

      const userId = crypto.randomUUID();
      const assistantId = crypto.randomUUID();
      setMessages((prev) => [
        ...prev,
        { id: userId, role: "user", content: trimmed, status: "done" },
        { id: assistantId, role: "assistant", content: "", status: "streaming" },
      ]);

      try {
        const events = api.streamChat(
          { message: trimmed, conversation_id: conversationId },
          controller.signal,
        );
        for await (const event of events) {
          switch (event.type) {
            case "conversation":
              setConversationId(event.conversation_id);
              onConversationActivity?.(event.conversation_id);
              break;
            case "token":
              updateMessage(assistantId, (m) => ({ content: m.content + event.text }));
              break;
            case "done":
              updateMessage(assistantId, () => ({ status: "done", usage: event.usage }));
              break;
            case "error":
              updateMessage(assistantId, () => ({ status: "error", error: event.message }));
              break;
          }
        }
      } catch (err) {
        if (controller.signal.aborted) {
          updateMessage(assistantId, () => ({ status: "stopped" }));
        } else {
          updateMessage(assistantId, () => ({ status: "error", error: errorMessage(err) }));
        }
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
        setMessages(
          detail.messages
            .filter((m) => (m.role === "user" || m.role === "assistant") && m.content)
            .map((m) => ({
              id: m.id,
              role: m.role as "user" | "assistant",
              content: m.content,
              status: "done",
            })),
        );
      } catch (err) {
        setLoadError(errorMessage(err));
      }
    },
    [reset],
  );

  return { conversationId, messages, isStreaming, loadError, send, stop, reset, load };
}
