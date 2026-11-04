import type { HealthState } from "../hooks/useHealth";

const TONE_CLASS = { ok: "badge-ok", warn: "badge-warn", bad: "badge-bad" } as const;

/** Small status pill in the header; hover for what is wrong. */
export function HealthBadge({ health }: { health: HealthState }) {
  if (health.kind === "loading") {
    return <span className="badge">Checking backend…</span>;
  }
  const { tone, label, problems } = health.summary;
  return (
    <span className={`badge ${TONE_CLASS[tone]}`} title={problems.join("\n") || undefined}>
      {label}
    </span>
  );
}

/** Full-width notice under the header while something is impaired (visible on mobile too). */
export function HealthBanner({ health }: { health: HealthState }) {
  if (health.kind === "loading" || health.summary.tone === "ok") return null;
  const { tone, problems } = health.summary;
  return (
    <div className={`health-banner health-banner-${tone}`} role="status">
      {problems.map((problem) => (
        <p key={problem}>{problem}</p>
      ))}
    </div>
  );
}
