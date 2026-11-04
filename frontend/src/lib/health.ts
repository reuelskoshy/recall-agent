import type { ComponentHealth, HealthResponse } from "../types/api";

export type HealthTone = "ok" | "warn" | "bad";

export interface HealthSummary {
  tone: HealthTone;
  /** Short text for the header badge. */
  label: string;
  /** One line per impaired dependency, for the banner. Empty when all is well. */
  problems: string[];
}

const COMPONENT_LABEL: Record<string, string> = {
  database: "Database",
  llm: "Language model",
  embeddings: "Embeddings",
  web_search: "Web search",
};

/** What each impaired dependency means for the user, not just that it is impaired. */
const IMPACT: Record<string, string> = {
  database: "nothing can be loaded or saved",
  llm: "chat answers may fail",
  embeddings: "document and memory search is unavailable",
  web_search: "web search is unavailable",
};

function describeComponent(c: ComponentHealth): string {
  const name = COMPONENT_LABEL[c.name] ?? c.name;
  const impact = IMPACT[c.name];
  const detail = c.detail ? ` (${c.detail})` : "";
  return impact ? `${name} ${c.status}${detail}: ${impact}.` : `${name} ${c.status}${detail}.`;
}

/** Turn a readiness response into what the UI shows. "disabled" is a choice, not a problem. */
export function summarizeHealth(health: HealthResponse): HealthSummary {
  const problems = health.components
    .filter((c) => c.status === "degraded" || c.status === "down")
    .map(describeComponent);
  if (health.status === "down") return { tone: "bad", label: "Database down", problems };
  if (health.status === "degraded") return { tone: "warn", label: "Degraded", problems };
  return { tone: "ok", label: "All systems ok", problems };
}

/** The badge/banner state when the backend itself cannot be reached. */
export const UNREACHABLE: HealthSummary = {
  tone: "bad",
  label: "Backend unreachable",
  problems: ["The server is not responding. Retrying automatically…"],
};
