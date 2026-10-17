#!/bin/sh
# Container entrypoint for the production image.
# Every step is idempotent, so restarts and redeploys are safe.
set -e

echo "==> Migrating database"
alembic upgrade head

echo "==> Read-only SQL role and sample dataset"
python -m app.datasets.seed

echo "==> Sample documents"
# A failed ingestion must not keep the app down; it is retried on the next start.
python -m app.rag.seed_samples /app/samples || echo "Sample ingestion failed; continuing"

echo "==> Starting API on port ${PORT:-7860}"
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-7860}" --timeout-graceful-shutdown 10
