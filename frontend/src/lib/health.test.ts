import { describe, expect, it } from "vitest";

import type { ComponentHealth, HealthResponse } from "../types/api";
import { summarizeHealth } from "./health";

function component(
  name: string,
  status: ComponentHealth["status"],
  detail: string | null = null,
): ComponentHealth {
  return { name, status, detail, latency_ms: null };
}

function health(status: HealthResponse["status"], components: ComponentHealth[]): HealthResponse {
  return { status, database: status === "down" ? "unreachable" : "ok", pgvector: null, components };
}

describe("summarizeHealth", () => {
  it("is quiet when everything is ok, and treats disabled as a choice", () => {
    const summary = summarizeHealth(
      health("ok", [component("database", "ok"), component("web_search", "disabled")]),
    );
    expect(summary).toEqual({ tone: "ok", label: "All systems ok", problems: [] });
  });

  it("explains a degraded dependency and what it means for the user", () => {
    const summary = summarizeHealth(
      health("degraded", [
        component("database", "ok"),
        component("llm", "down", "k2 failing; requests paused for 25s"),
      ]),
    );
    expect(summary.tone).toBe("warn");
    expect(summary.problems).toEqual([
      "Language model down (k2 failing; requests paused for 25s): chat answers may fail.",
    ]);
  });

  it("is bad when the database is down", () => {
    const summary = summarizeHealth(health("down", [component("database", "down", "unreachable")]));
    expect(summary.tone).toBe("bad");
    expect(summary.label).toBe("Database down");
    expect(summary.problems[0]).toContain("nothing can be loaded or saved");
  });

  it("falls back to the raw name for components it does not know", () => {
    const summary = summarizeHealth(health("degraded", [component("cache", "degraded")]));
    expect(summary.problems).toEqual(["cache degraded."]);
  });
});
