/**
 * Types mirroring the backend's Pydantic response models.
 * Keep in sync with backend/app/api/*.py (see README: "Shared types").
 */

/** GET /api/health — mirrors app.api.health.HealthResponse */
export interface HealthResponse {
  status: "ok" | "degraded";
  database: "ok" | "unreachable";
  pgvector: string | null;
}
