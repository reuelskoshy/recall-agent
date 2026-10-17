---
title: Recall
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: AI agent with tools, RAG, read-only SQL, and long-term memory
---

# Recall

An AI agent with a hand-built tool-calling loop, RAG over uploaded documents with
citations, read-only natural-language SQL (Postgres or Snowflake), and ranked
long-term memory. FastAPI + PostgreSQL/pgvector + React.

> The block above is Hugging Face Spaces configuration. Full documentation
> (architecture diagram, design decisions, demo script) is coming in Phase 7.

## Run locally

```bash
cp .env.example .env   # then set LLM_API_KEY
docker compose up --build
```

Open http://localhost:5173.

## Deploy (Hugging Face Spaces + Neon)

See [deploy/DEPLOY.md](deploy/DEPLOY.md).
