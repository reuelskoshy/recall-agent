import { useCallback, useEffect, useState } from "react";

import { api } from "../api/client";
import type { ConversationSummary } from "../types/api";

/** The sidebar's conversation list, with refresh and delete. */
export function useConversations() {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setConversations(await api.listConversations());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  const remove = useCallback(async (id: string) => {
    await api.deleteConversation(id);
    setConversations((prev) => prev.filter((c) => c.id !== id));
  }, []);

  // Initial load. State is only set in the promise callbacks, after the fetch resolves.
  useEffect(() => {
    let cancelled = false;
    api
      .listConversations()
      .then((data) => {
        if (!cancelled) setConversations(data);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { conversations, error, refresh, remove };
}
