import { useCallback, useState } from "react";

import { ChatView } from "./components/ChatView";
import { HealthBadge } from "./components/HealthBadge";
import { Sidebar } from "./components/Sidebar";
import { useChat } from "./hooks/useChat";
import { useConversations } from "./hooks/useConversations";

export default function App() {
  const { conversations, error: listError, refresh, remove } = useConversations();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // A turn updates the conversation's title/recency, so re-sort the sidebar.
  const onConversationActivity = useCallback(() => {
    void refresh();
  }, [refresh]);
  const chat = useChat({ onConversationActivity });

  const select = (id: string) => {
    setSidebarOpen(false);
    if (id !== chat.conversationId) void chat.load(id);
  };

  const startNew = () => {
    setSidebarOpen(false);
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
        <ChatView
          messages={chat.messages}
          isStreaming={chat.isStreaming}
          loadError={chat.loadError}
          onSend={(text) => void chat.send(text)}
          onStop={chat.stop}
        />
      </div>
    </div>
  );
}
