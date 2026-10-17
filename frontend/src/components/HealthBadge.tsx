import { useHealth } from "../hooks/useHealth";

/** Small status pill showing whether the backend, Postgres, and pgvector are up. */
export function HealthBadge() {
  const health = useHealth();

  if (health.kind === "loading") {
    return <span className="badge">Checking backend…</span>;
  }
  if (health.kind === "error") {
    return <span className="badge badge-bad">Backend unreachable</span>;
  }

  const { status, database, pgvector } = health.data;
  return (
    <span className={`badge ${status === "ok" ? "badge-ok" : "badge-bad"}`}>
      API {status} · DB {database} · pgvector {pgvector ?? "missing"}
    </span>
  );
}
