# syntax=docker/dockerfile:1
# Production image: one container serving the API and the built frontend.
# Deployed to Google Cloud Run (deploy/DEPLOY.md); also runs anywhere with
#   docker build -t recall . && docker run -p 7860:7860 --env-file .env recall
# Local development uses docker-compose.yml instead (hot reload, Vite dev server).

# ---- 1. Build the React frontend -------------------------------------------------
FROM node:22-alpine AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- 2. Python runtime -----------------------------------------------------------
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.7.12 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/hf-cache \
    STATIC_DIR=/app/static \
    ENVIRONMENT=production \
    PORT=7860

# Run as an unprivileged user (UID 1000), not root.
RUN useradd --create-home --uid 1000 user
WORKDIR /app

# Runtime dependencies only (no pytest/ruff/mypy). Cached until the lockfile changes.
COPY backend/pyproject.toml backend/uv.lock backend/.python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    for attempt in 1 2 3; do \
      uv sync --frozen --no-install-project --no-dev --python-preference only-system && break; \
      if [ "$attempt" = 3 ]; then exit 1; fi; \
      echo "uv sync failed (attempt $attempt), retrying"; sleep 5; \
    done

# Bake the embedding model into the image: no download at startup, no network
# dependency on the Hugging Face Hub when the container (re)starts.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY backend/ ./
COPY samples/ ./samples/
COPY deploy/start.sh ./start.sh
COPY --from=frontend /build/dist ./static
RUN chown -R user:user /app

USER user
EXPOSE 7860
CMD ["sh", "/app/start.sh"]
