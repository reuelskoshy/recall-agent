import type { ToolTrace } from "../lib/chat";

const STATUS_LABEL: Record<ToolTrace["status"], string> = {
  running: "Running",
  ok: "Succeeded",
  error: "Failed",
};

/** One-line preview of the arguments, e.g. `expression: "6*7"`. */
function previewArguments(args: Record<string, unknown>): string {
  const text = Object.entries(args)
    .map(([key, value]) => `${key}: ${JSON.stringify(value)}`)
    .join(", ");
  return text.length > 60 ? `${text.slice(0, 59)}…` : text;
}

/** Collapsible card showing one tool call: arguments in, result out, latency. */
export function ToolTraceCard({ tool }: { tool: ToolTrace }) {
  return (
    <details className={`tool-card tool-${tool.status}`}>
      <summary>
        <span className="tool-status" role="img" aria-label={STATUS_LABEL[tool.status]}>
          {tool.status === "running" ? <span className="spinner" /> : tool.status === "ok" ? "✓" : "✕"}
        </span>
        <code className="tool-name">{tool.name}</code>
        <span className="tool-args">{previewArguments(tool.arguments)}</span>
        {tool.latencyMs !== undefined && <span className="tool-latency">{tool.latencyMs} ms</span>}
      </summary>
      <div className="tool-body">
        <div className="tool-section-label">Arguments</div>
        <pre>{JSON.stringify(tool.arguments, null, 2)}</pre>
        <div className="tool-section-label">Result (sent to the model)</div>
        <pre>{tool.result ?? "…"}</pre>
      </div>
    </details>
  );
}
