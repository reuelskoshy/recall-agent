import { useState, type SubmitEvent } from "react";

import { useMemories } from "../hooks/useMemories";
import type { MemoryInfo, MemoryKind, ScoreBreakdown, ScoredMemoryInfo } from "../types/api";

const KINDS: MemoryKind[] = ["fact", "preference", "goal", "instruction"];
const SOURCE_LABEL: Record<MemoryInfo["source"], string> = {
  tool: "saved by agent",
  auto: "auto-extracted",
  manual: "added by you",
};
/** Mirrors the backend default MEMORY_CORE_MIN_IMPORTANCE. */
const CORE_MIN_IMPORTANCE = 4;

function isCore(m: MemoryInfo): boolean {
  return m.kind === "instruction" || m.importance >= CORE_MIN_IMPORTANCE;
}

function timeAgo(iso: string): string {
  const seconds = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${String(minutes)} min ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${String(hours)} h ago`;
  return `${String(Math.floor(hours / 24))} d ago`;
}

function Bar({ label, value, title }: { label: string; value: number; title?: string }) {
  return (
    <span className="score-bar" title={title ?? `${label}: ${value.toFixed(3)}`}>
      <span className="score-bar-label">{label}</span>
      <span className="score-bar-track">
        <span className="score-bar-fill" style={{ width: `${String(Math.round(value * 100))}%` }} />
      </span>
      <span className="score-bar-value">{value.toFixed(2)}</span>
    </span>
  );
}

function Ranking({ ranking }: { ranking: ScoreBreakdown }) {
  return (
    <div className="ranking">
      <Bar label="score" value={ranking.score} />
      <Bar
        label="relevance"
        value={ranking.relevance}
        title={`cosine similarity ${ranking.similarity.toFixed(3)}, rescaled above the floor`}
      />
      <Bar label="recency" value={ranking.recency} />
      <Bar label="importance" value={ranking.importance} />
    </div>
  );
}

interface RowProps {
  memory: MemoryInfo | ScoredMemoryInfo;
  onSave: (content: string, kind: MemoryKind, importance: number) => Promise<void>;
  onDelete: () => void;
}

function MemoryRow({ memory, onSave, onDelete }: RowProps) {
  const [editing, setEditing] = useState(false);
  const [content, setContent] = useState(memory.content);
  const [kind, setKind] = useState<MemoryKind>(memory.kind);
  const [importance, setImportance] = useState(memory.importance);
  const [error, setError] = useState<string | null>(null);

  const save = async () => {
    try {
      await onSave(content, kind, importance);
      setEditing(false);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <li className="doc-row memory-row">
      <div className="doc-main">
        {editing ? (
          <div className="memory-edit">
            <textarea
              value={content}
              rows={2}
              maxLength={500}
              aria-label="Memory text"
              onChange={(e) => {
                setContent(e.target.value);
              }}
            />
            <div className="memory-form-row">
              <KindSelect value={kind} onChange={setKind} />
              <ImportanceSelect value={importance} onChange={setImportance} />
              <button type="button" className="btn btn-primary" onClick={() => void save()}>
                Save
              </button>
              <button
                type="button"
                className="btn"
                onClick={() => {
                  setEditing(false);
                  setContent(memory.content);
                }}
              >
                Cancel
              </button>
            </div>
            {error && <span className="doc-error">{error}</span>}
          </div>
        ) : (
          <span className="memory-content">{memory.content}</span>
        )}
        <span className="doc-meta">
          <span className={`badge kind-${memory.kind}`}>{memory.kind}</span>
          {isCore(memory) && <span className="badge badge-ok">always on</span>} importance{" "}
          {memory.importance}/5 · {SOURCE_LABEL[memory.source]} · used {memory.access_count}× · last
          used {timeAgo(memory.last_accessed_at)}
        </span>
        {"ranking" in memory && <Ranking ranking={memory.ranking} />}
      </div>
      {!editing && (
        <button
          type="button"
          className="link-button memory-edit-button"
          onClick={() => {
            setEditing(true);
          }}
        >
          Edit
        </button>
      )}
      <button
        type="button"
        className="sidebar-delete doc-delete"
        aria-label={`Forget "${memory.content}"`}
        onClick={onDelete}
      >
        ×
      </button>
    </li>
  );
}

function KindSelect({ value, onChange }: { value: MemoryKind; onChange: (k: MemoryKind) => void }) {
  return (
    <select
      aria-label="Kind"
      value={value}
      onChange={(e) => {
        onChange(e.target.value as MemoryKind);
      }}
    >
      {KINDS.map((k) => (
        <option key={k} value={k}>
          {k}
        </option>
      ))}
    </select>
  );
}

function ImportanceSelect({ value, onChange }: { value: number; onChange: (n: number) => void }) {
  return (
    <select
      aria-label="Importance"
      value={value}
      onChange={(e) => {
        onChange(Number(e.target.value));
      }}
    >
      {[1, 2, 3, 4, 5].map((n) => (
        <option key={n} value={n}>
          importance {n}
        </option>
      ))}
    </select>
  );
}

/** Everything Recall remembers about the user: inspect, search, add, correct, forget. */
export function MemoryPage() {
  const { memories, query, setQuery, results, error, create, update, remove } = useMemories();
  const [content, setContent] = useState("");
  const [kind, setKind] = useState<MemoryKind>("fact");
  const [importance, setImportance] = useState(3);
  const [notice, setNotice] = useState<string | null>(null);

  const add = async (e: SubmitEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!content.trim()) return;
    try {
      const saved = await create({ content, kind, importance });
      setNotice(
        saved.action === "created"
          ? "Saved."
          : `Merged into an existing memory (was: "${saved.previous_content ?? ""}").`,
      );
      setContent("");
    } catch (err) {
      setNotice(err instanceof Error ? err.message : String(err));
    }
  };

  const shown = results ?? memories;

  return (
    <section className="page">
      <h2>Memory</h2>
      <p className="page-intro">
        Facts about you that Recall keeps across conversations: saved when you ask, or extracted
        automatically after each answer. Instructions and importance 4–5 memories are used in every
        chat; others are ranked per message by{" "}
        <code>0.7·relevance + 0.2·recency + 0.1·importance</code> (relevance is similarity rescaled
        above a 0.5 floor; recency halves every 30 days since last use).
      </p>

      <form className="memory-form" onSubmit={(e) => void add(e)}>
        <textarea
          placeholder="e.g. I prefer metric units"
          value={content}
          rows={2}
          maxLength={500}
          aria-label="New memory"
          onChange={(e) => {
            setContent(e.target.value);
          }}
        />
        <div className="memory-form-row">
          <KindSelect value={kind} onChange={setKind} />
          <ImportanceSelect value={importance} onChange={setImportance} />
          <button type="submit" className="btn btn-primary" disabled={!content.trim()}>
            Remember
          </button>
          {notice && <span className="doc-meta">{notice}</span>}
        </div>
      </form>

      <input
        type="search"
        className="memory-search"
        placeholder="Search memories (shows how each one would rank)"
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
        }}
      />

      {error && <p className="message-error">Could not load memories: {error}</p>}

      {shown.length === 0 && !error ? (
        <p className="page-empty">
          {results ? "No memory is relevant to that search." : "Nothing remembered yet."}
        </p>
      ) : (
        <ul className="doc-list">
          {shown.map((m) => (
            <MemoryRow
              key={`${m.id}-${m.updated_at}`}
              memory={m}
              onSave={(c, k, i) => update(m.id, { content: c, kind: k, importance: i })}
              onDelete={() => {
                if (window.confirm(`Forget "${m.content}"?`)) void remove(m.id);
              }}
            />
          ))}
        </ul>
      )}
    </section>
  );
}
