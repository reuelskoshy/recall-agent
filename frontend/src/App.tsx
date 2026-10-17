import { useCallback, useState } from "react";

import { ChatView } from "./components/ChatView";
import { HealthBadge } from "./components/HealthBadge";
import { Sidebar } from "./components/Sidebar";
import { useChat } from "./hooks/useChat";
import { useConversations } from "./hooks/useConversations";
import { DocumentsPage } from "./pages/DocumentsPage";

type View = "chat" | "documents";

export default function App() {
  const { conversations, error: listError, refresh, remove } = useConversations();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [view, setView] = useState<View>("chat");

  // A turn updates the conversation's title/recency, so re-sort the sidebar.
  const onConversationActivity = useCallback(() => {
    void refresh();
  }, [refresh]);
  const chat = useChat({ onConversationActivity });

  const select = (id: string) => {
    setSidebarOpen(false);
    setView("chat");
    if (id !== chat.conversationId) void chat.load(id);
  };

  const startNew = () => {
    setSidebarOpen(false);
    setView("chat");
    chat.reset();
  };

  const deleteConversation = async (id: string) => {
    await remove(id);
    if (id === chat.conversationId) chat.reset();
  };

  return (
    <div className="app">
      <header className="app-header">
        <button
          type="button"
          className="btn menu-toggle"
          aria-label="Toggle conversations"
          aria-expanded={sidebarOpen}
          onClick={() => {
            setSidebarOpen((open) => !open);
          }}
        >
          ☰
        </button>
        <h1>Recall</h1>
        <nav className="tabs" aria-label="Views">
          {(["chat", "documents"] as const).map((v) => (
            <button
              key={v}
              type="button"
              className={`tab ${view === v ? "active" : ""}`}
              aria-current={view === v ? "page" : undefined}
              onClick={() => {
                setView(v);
              }}
            >
              {v === "chat" ? "Chat" : "Documents"}
            </button>
          ))}
        </nav>
        <HealthBadge />
      </header>
      <div className={`app-body ${sidebarOpen ? "sidebar-open" : ""}`}>
        <Sidebar
          conversations={conversations}
          activeId={chat.conversationId}
          error={listError}
          onSelect={select}
          onNew={startNew}
          onDelete={(id) => void deleteConversation(id)}
        />
        {view === "chat" ? (
          <ChatView
            messages={chat.messages}
            isStreaming={chat.isStreaming}
            loadError={chat.loadError}
            onSend={(text) => void chat.send(text)}
            onStop={chat.stop}
          />
        ) : (
          <DocumentsPage />
        )}
      </div>
    </div>
  );
}
