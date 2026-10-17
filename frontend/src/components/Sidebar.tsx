import type { ConversationSummary } from "../types/api";

interface SidebarProps {
  conversations: ConversationSummary[];
  activeId: string | null;
  error: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
}

/** Conversation list with "new chat" and per-conversation delete. */
export function Sidebar({ conversations, activeId, error, onSelect, onNew, onDelete }: SidebarProps) {
  return (
    <nav className="sidebar" aria-label="Conversations">
      <button type="button" className="btn btn-primary sidebar-new" onClick={onNew}>
        + New chat
      </button>
      {error && <p className="sidebar-error">Could not load conversations: {error}</p>}
      <ul className="sidebar-list">
        {conversations.map((c) => (
          <li key={c.id} className={c.id === activeId ? "active" : undefined}>
            <button
              type="button"
              className="sidebar-item"
              title={c.title}
              onClick={() => {
                onSelect(c.id);
              }}
            >
              {c.title}
            </button>
            <button
              type="button"
              className="sidebar-delete"
              aria-label={`Delete "${c.title}"`}
              onClick={() => {
                if (window.confirm(`Delete "${c.title}"?`)) onDelete(c.id);
              }}
            >
              ×
            </button>
          </li>
        ))}
      </ul>
      {conversations.length === 0 && !error && <p className="sidebar-empty">No conversations yet.</p>}
    </nav>
  );
}
