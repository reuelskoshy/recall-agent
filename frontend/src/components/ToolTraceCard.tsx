import { citationLabel, type ToolTrace } from "../lib/chat";
import type { Citation, QueryResultData } from "../types/api";

const BACKEND_LABEL: Record<string, string> = { postgres: "PostgreSQL", snowflake: "Snowflake" };

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

function searchResults(tool: ToolTrace): Citation[] | null {
  const results = tool.data?.results;
  return tool.name === "search_documents" && Array.isArray(results)
    ? (results as Citation[])
    : null;
}

function queryResult(tool: ToolTrace): QueryResultData | null {
  const data = tool.data;
  return tool.name === "query_database" && data && Array.isArray(data.rows)
    ? (data as unknown as QueryResultData)
    : null;
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return "NULL";
  if (typeof value === "string") return value;
  return typeof value === "number" || typeof value === "boolean"
    ? String(value)
    : JSON.stringify(value);
}

/** The SQL that ran and its rows as a scrollable table. */
function QueryResultView({ result }: { result: QueryResultData }) {
  return (
    <>
      <div className="tool-section-label">SQL</div>
      <pre className="sql">{result.sql}</pre>
      <div className="tool-section-label">
        {result.row_count}
        {result.truncated ? "+" : ""} rows · {result.elapsed_ms} ms
        {result.backend && ` · ${BACKEND_LABEL[result.backend] ?? result.backend}`}
        {result.query_id && ` · query ${result.query_id}`}
      </div>
      {result.rows.length > 0 && (
        <div className="result-table-wrap">
          <table className="result-table">
            <thead>
              <tr>
                {result.columns.map((c, i) => (
                  <th key={i}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.rows.map((row, r) => (
                <tr key={r}>
                  {row.map((cell, c) => (
                    <td key={c} className={typeof cell === "number" ? "num" : undefined}>
                      {formatCell(cell)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {result.truncated && (
        <div className="hit-score">Capped at {result.row_count} rows; more matched.</div>
      )}
    </>
  );
}

/** Collapsible card showing one tool call: arguments in, result out, latency. */
export function ToolTraceCard({ tool }: { tool: ToolTrace }) {
  const results = searchResults(tool);
  const query = queryResult(tool);

  return (
    <details className={`tool-card tool-${tool.status}`}>
      <summary>
        <span className="tool-status" role="img" aria-label={STATUS_LABEL[tool.status]}>
          {tool.status === "running" ? <span className="spinner" /> : tool.status === "ok" ? "✓" : "✕"}
        </span>
        <code className="tool-name">{tool.name}</code>
        <span className="tool-args">{previewArguments(tool.arguments)}</span>
        {results && <span className="tool-latency">{results.length} hits</span>}
        {query && (
          <span className="tool-latency">
            {query.row_count}
            {query.truncated ? "+" : ""} rows
          </span>
        )}
        {tool.latencyMs !== undefined && <span className="tool-latency">{tool.latencyMs} ms</span>}
      </summary>
      <div className="tool-body">
        {query ? (
          <QueryResultView result={query} />
        ) : (
          <>
            <div className="tool-section-label">Arguments</div>
            <pre>{JSON.stringify(tool.arguments, null, 2)}</pre>
          </>
        )}
        {results && results.length > 0 && (
          <>
            <div className="tool-section-label">Retrieved passages</div>
            <ol className="hits">
              {results.map((hit) => (
                <li key={`${hit.document_id}-${String(hit.chunk_index)}`}>
                  <div className="hit-head">
                    <span className="citation">{citationLabel(hit)}</span>
                    <span className="hit-score">similarity {hit.similarity.toFixed(2)}</span>
                  </div>
                  <div className="hit-snippet">{hit.snippet}…</div>
                </li>
              ))}
            </ol>
          </>
        )}
        <div className="tool-section-label">Result (sent to the model)</div>
        <pre>{tool.result ?? "…"}</pre>
      </div>
    </details>
  );
}
