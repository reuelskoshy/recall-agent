import { useEffect, useState, type ReactNode, type SubmitEvent } from "react";

import { api, setToken, setUnauthorizedHandler } from "../api/client";

type GateState = "checking" | "locked" | "open";

/**
 * Shows a password screen when the deployment requires the demo password,
 * otherwise renders the app. Any later 401 (e.g. the password was changed)
 * locks the app again.
 */
export function AccessGate({ children }: { children: ReactNode }) {
  const [state, setState] = useState<GateState>("checking");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setUnauthorizedHandler(() => {
      setToken(null);
      setState("locked");
    });
    api
      .authStatus()
      .then((status) => {
        if (!cancelled) setState(status.authenticated ? "open" : "locked");
      })
      .catch(() => {
        // Backend unreachable: render the app, whose health badge explains the problem.
        if (!cancelled) setState("open");
      });
    return () => {
      cancelled = true;
      setUnauthorizedHandler(null);
    };
  }, []);

  const submit = async (e: SubmitEvent<HTMLFormElement>) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const { token } = await api.login(password);
      setToken(token);
      setPassword("");
      setState("open");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  if (state === "checking") return null;
  if (state === "open") return <>{children}</>;

  return (
    <main className="gate">
      <form className="gate-card" onSubmit={(e) => void submit(e)}>
        <h1>Recall</h1>
        <p className="page-intro">
          An AI agent with tools, RAG over documents, read-only SQL, and long-term memory. This
          demo is password-protected to keep LLM costs in check.
        </p>
        <input
          type="password"
          className="memory-search"
          placeholder="Demo password"
          aria-label="Demo password"
          autoComplete="current-password"
          value={password}
          autoFocus
          onChange={(e) => {
            setPassword(e.target.value);
          }}
        />
        <button type="submit" className="btn btn-primary" disabled={busy || !password}>
          {busy ? "Checking…" : "Enter"}
        </button>
        {error && <p className="message-error">{error}</p>}
      </form>
    </main>
  );
}
