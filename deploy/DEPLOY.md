# Deploying Recall: Hugging Face Spaces + Neon ($0)

One container (API + built frontend + embedding model) runs on a free Hugging
Face Docker Space; PostgreSQL with pgvector runs on Neon's free tier.

```
browser ──HTTPS──▶ HF Space (Docker, 16 GB RAM CPU)        Neon (Postgres 16 + pgvector)
                    FastAPI + React build + bge-small  ──TLS──▶ app tables, memories,
                    port 7860, demo password gate              acme dataset, recall_reader
                         │
                         └──HTTPS──▶ LLM API (IFM K2-Horizon)
```

On every start the container runs `deploy/start.sh`: migrations, the read-only
role + Acme dataset, the sample handbook, then the API. All steps are
idempotent, so restarts and redeploys are safe.

## 1. Database (Neon)

1. Sign up at https://neon.tech and create a project (Postgres 16 or 17).
   Spaces run in the US, so a US East region keeps queries fast.
2. **Connect** → turn **Connection pooling OFF** (use the direct endpoint, whose
   host does not contain `-pooler`). Recall uses prepared statements, which
   PgBouncer's transaction mode breaks.
3. Copy the connection string. It looks like
   `postgresql://neondb_owner:...@ep-xxx.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require`.
   Paste it as-is: Recall converts it for asyncpg.

The first start creates the `vector` extension, all tables, the `acme` dataset,
and the `recall_reader` login role (SELECT on `acme` only).

## 2. Space (Hugging Face)

1. https://huggingface.co/new-space → name `recall`, SDK **Docker** (Blank
   template), hardware **CPU basic (free)**, visibility **Public** (the demo
   password protects it).
2. Space **Settings → Variables and secrets**:

   | Name | Kind | Value |
   |---|---|---|
   | `DATABASE_URL` | Secret | the Neon connection string |
   | `LLM_API_KEY` | Secret | your LLM key |
   | `DEMO_PASSWORD` | Secret | the password you give recruiters |
   | `READONLY_DB_PASSWORD` | Secret | a long random string (see below) |
   | `LLM_BASE_URL` | Variable | `https://api.ifm.ai/v1` |
   | `LLM_CHAT_MODEL` | Variable | `IFM/K2-Horizon-375B-A23B` |
   | `LLM_REASONING_FIELD` | Variable | `reasoning` |
   | `TRUSTED_PROXY_HOPS` | Variable | `1` |
   | `RATE_LIMIT_CHAT_PER_IP` | Variable | `20` (per 10 minutes) |
   | `DAILY_CHAT_LIMIT` | Variable | `200` (all visitors together) |

   Generate the reader password with
   `python -c "import secrets; print(secrets.token_urlsafe(24))"`
   (Neon rejects weak passwords for roles created over SQL).

3. Deploy the code. Either:

   **a. Through GitHub (recommended: deploys only after CI passes).**
   Push the repo to GitHub, then in the GitHub repo **Settings → Secrets and
   variables → Actions** add the secret `HF_TOKEN` (a Hugging Face **write**
   token from https://huggingface.co/settings/tokens) and the variable
   `HF_SPACE` (`<your-username>/recall`). Every push to `main` now runs lint,
   type checks, and tests; if they pass, the `deploy` job mirrors the repo to
   the Space, which rebuilds.

   **b. Directly** (use the write token as the password):

   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/recall
   git push space main
   ```

   The Space builds the root `Dockerfile` (about 10 minutes the first time) and
   starts. Logs: the Space's **Logs** tab.

4. Open `https://<your-username>-recall.hf.space` (the direct URL; share this
   one rather than the huggingface.co page, which wraps the app in an iframe).

## Operating notes

- **Cost control.** Every chat turn costs one or more LLM calls (plus one for
  memory extraction). The per-IP window and daily cap bound the worst case;
  set `MEMORY_AUTO_EXTRACT=false` to halve calls. Limits are in memory and
  reset when the Space restarts.
- **Sleep.** Free Spaces sleep after ~48 h without traffic; the next visit
  wakes it in about a minute. Data lives in Neon, so nothing is lost.
- **Shared state.** The demo has one shared memory and conversation list (no
  user accounts): visitors can see what others told it. Fine for a guided
  demo; per-user isolation is listed as future work.
- **Snowflake.** The Space uses the Postgres dataset (`SQL_BACKEND=postgres`).
  Running against Snowflake would need the key provided as a secret instead
  of a file.
- **Changing the password** (`DEMO_PASSWORD`) logs everyone out immediately.

## Run the production image anywhere

```bash
docker build -t recall .
docker run -p 7860:7860 --env-file .env -e DATABASE_URL=... recall
```
