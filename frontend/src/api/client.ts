import type { HealthResponse } from "../types/api";

/** Error carrying the HTTP status, so callers can tell 4xx from 5xx from network failures. */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`/api${path}`, { headers: { Accept: "application/json" } });
  if (!response.ok && response.status !== 503) {
    throw new ApiError(response.status, `${path} failed: ${String(response.status)}`);
  }
  return (await response.json()) as T;
}

/** Typed wrappers for every backend endpoint live here. */
export const api = {
  // /health returns a body on 503 too (status "degraded"), so it is not treated as an error.
  health: () => getJson<HealthResponse>("/health"),
};
