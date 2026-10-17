import { useCallback, useEffect, useState } from "react";

import { api } from "../api/client";
import type { MemoryInfo, MemoryKind, MemorySaved, ScoredMemoryInfo } from "../types/api";

const SEARCH_DEBOUNCE_MS = 300;

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** All memories, ranked search over them, and add / edit / delete. */
export function useMemories() {
  const [memories, setMemories] = useState<MemoryInfo[]>([]);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<ScoredMemoryInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setMemories(await api.listMemories());
      setError(null);
    } catch (err) {
      setError(message(err));
    }
  }, []);

  // Initial load; state is set only in the promise callbacks.
  useEffect(() => {
    let cancelled = false;
    api
      .listMemories()
      .then((data) => {
        if (!cancelled) setMemories(data);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(message(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Debounced ranked search; an empty query shows the full list instead.
  useEffect(() => {
    const q = query.trim();
    let cancelled = false;
    const timer = window.setTimeout(() => {
      if (!q) {
        setResults(null);
        return;
      }
      api
        .searchMemories(q)
        .then((data) => {
          if (!cancelled) setResults(data);
        })
        .catch((err: unknown) => {
          if (!cancelled) setError(message(err));
        });
    }, SEARCH_DEBOUNCE_MS);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [query, memories]);

  const create = useCallback(
    async (body: {
      content: string;
      kind: MemoryKind;
      importance: number;
    }): Promise<MemorySaved> => {
      const saved = await api.createMemory(body);
      await refresh();
      return saved;
    },
    [refresh],
  );

  const update = useCallback(
    async (
      id: string,
      body: Partial<{ content: string; kind: MemoryKind; importance: number }>,
    ) => {
      await api.updateMemory(id, body);
      await refresh();
    },
    [refresh],
  );

  const remove = useCallback(async (id: string) => {
    await api.deleteMemory(id);
    setMemories((prev) => prev.filter((m) => m.id !== id));
  }, []);

  return {
    memories,
    query,
    setQuery,
    results,
    error,
    refresh,
    create,
    update,
    remove,
  };
}
