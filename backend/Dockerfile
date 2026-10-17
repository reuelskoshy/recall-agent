FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.7.12 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install dependencies first so this layer is cached until the lockfile changes.
# The uv download cache is a BuildKit cache mount: it survives failed builds, so
# on a flaky connection each retry resumes instead of re-downloading ~1.4 GB.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    for attempt in 1 2 3 4 5; do \
      uv sync --frozen --no-install-project --python-preference only-system && break; \
      if [ "$attempt" = 5 ]; then exit 1; fi; \
      echo "uv sync failed (attempt $attempt), retrying"; sleep 5; \
    done

COPY . .

EXPOSE 8000

# Apply migrations, then start the API.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
