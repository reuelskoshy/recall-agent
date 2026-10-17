import { useCallback, useEffect, useState } from "react";

import { api } from "../api/client";
import type { DocumentInfo } from "../types/api";

const POLL_MS = 1500;

export interface UploadState {
  name: string;
  error?: string;
}

/** Document list with upload, delete, and polling while anything is still processing. */
export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [uploads, setUploads] = useState<UploadState[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setDocuments(await api.listDocuments());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  // Initial load; state is set only in the promise callbacks.
  useEffect(() => {
    let cancelled = false;
    api
      .listDocuments()
      .then((data) => {
        if (!cancelled) setDocuments(data);
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Poll only while ingestion is running somewhere.
  const processing = documents.some((d) => d.status === "processing");
  useEffect(() => {
    if (!processing) return;
    const timer = window.setInterval(() => void refresh(), POLL_MS);
    return () => {
      window.clearInterval(timer);
    };
  }, [processing, refresh]);

  const upload = useCallback(
    async (files: File[]) => {
      for (const file of files) {
        setUploads((prev) => [...prev, { name: file.name }]);
        try {
          await api.uploadDocument(file);
          setUploads((prev) => prev.filter((u) => u.name !== file.name));
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          setUploads((prev) =>
            prev.map((u) => (u.name === file.name ? { ...u, error: message } : u)),
          );
        }
        await refresh();
      }
    },
    [refresh],
  );

  const dismissUpload = useCallback((name: string) => {
    setUploads((prev) => prev.filter((u) => u.name !== name));
  }, []);

  const remove = useCallback(async (id: string) => {
    await api.deleteDocument(id);
    setDocuments((prev) => prev.filter((d) => d.id !== id));
  }, []);

  return { documents, uploads, error, upload, dismissUpload, remove };
}
