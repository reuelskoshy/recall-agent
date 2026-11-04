import { useEffect, useState } from "react";

import { api } from "../api/client";
import { summarizeHealth, UNREACHABLE, type HealthSummary } from "../lib/health";

export type HealthState = { kind: "loading" } | { kind: "loaded"; summary: HealthSummary };

/** Poll less often when healthy, more often while something is wrong (to show recovery). */
const HEALTHY_POLL_MS = 60_000;
const UNHEALTHY_POLL_MS = 10_000;

/** Backend readiness, polled while the tab is visible. */
export function useHealth(): HealthState {
  const [state, setState] = useState<HealthState>({ kind: "loading" });

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const check = async () => {
      let summary: HealthSummary;
      try {
        summary = summarizeHealth(await api.health());
      } catch {
        summary = UNREACHABLE;
      }
      if (cancelled) return;
      setState({ kind: "loaded", summary });
      timer = setTimeout(
        () => void poll(),
        summary.tone === "ok" ? HEALTHY_POLL_MS : UNHEALTHY_POLL_MS,
      );
    };

    // A hidden tab skips its check and looks again when it comes back.
    const poll = async () => {
      if (document.visibilityState === "visible") await check();
      else if (!cancelled) timer = setTimeout(() => void poll(), UNHEALTHY_POLL_MS);
    };

    const onVisible = () => {
      if (document.visibilityState !== "visible") return;
      clearTimeout(timer);
      void check();
    };

    void check();
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      cancelled = true;
      clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, []);

  return state;
}
