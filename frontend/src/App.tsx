import { HealthBadge } from "./components/HealthBadge";

export default function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>Recall</h1>
        <HealthBadge />
      </header>
      <main className="app-main">
        <p>An AI agent with tools, long-term memory, and RAG. Chat arrives in Phase 1.</p>
      </main>
    </div>
  );
}
