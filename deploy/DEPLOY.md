# Deploying Recall: Render + Neon (free tier)

The live demo runs as one container (API + built frontend + embedding model) on a
free Render web service; PostgreSQL with pgvector runs on Neon's free tier.
Render deploys each push to `main` after GitHub Actions CI passes.

```
git push ─▶ GitHub Actions (lint, types, tests) ─▶ Render builds the root Dockerfile
                                                         │
browser ──HTTPS──▶ Render (Virginia, 512 MB, sleeps when idle) ──TLS──▶ Neon (us-east-1)
                   FastAPI + React build + bge-small, demo password gate    app tables, memories,
                         │                                                   acme dataset, recall_reader
                         └──HTTPS──▶ LLM API (IFM K2-Horizon)
```

On every start the container runs `deploy/start.sh`: migrations, the read-only
role + Acme dataset, the sample handbook, then the API. All steps are
idempotent, so restarts and redeploys are safe. The image peaks around 445 MB
of RAM (measured under a 512 MB cap), so it fits the free instance.

## 1. Database (Neon)

1. Sign up at https://neon.tech and create a project (Postgres 16 or 17) in
   **AWS US East 1**, next to the Render region.
2. **Connect** → turn **Connection pooling OFF** (use the direct endpoint, whose
   host does not contain `-pooler`). Recall uses prepared statements, which
   PgBouncer's transaction mode breaks.
3. Copy the connection string. Paste it as-is: Recall converts it for asyncpg.

The first start creates the `vector` extension, all tables, the `acme` dataset,
and the `recall_reader` login role (SELECT on `acme` only). Neon's owner role is
not a superuser, so the role is created without, and then checked for, any
privileged attributes.

## 2. Render

1. https://render.com → sign up with GitHub → **New → Web Service** → pick the
   repository (if it is missing, grant Render access to it in GitHub →
   Settings → Applications → Render → Configure).
2. Language **Docker**, branch `main`, region **Virginia**, instance **Free**.
3. **Environment** (only these; do not copy a local `.env`, whose Snowflake
   and Docker settings do not apply here):

   | Key | Value |
   |---|---|
   | `DATABASE_URL` | the Neon connection string |
   | `LLM_API_KEY` | your LLM key |
   | `DEMO_PASSWORD` | the password you give visitors |
   | `READONLY_DB_PASSWORD` | `python -c "import secrets; print(secrets.token_urlsafe(24))"` |
   | `LLM_BASE_URL` | `https://api.ifm.ai/v1` |
   | `LLM_CHAT_MODEL` | `IFM/K2-Horizon-375B-A23B` |
   | `LLM_REASONING_FIELD` | `reasoning` |
   | `TRUSTED_PROXY_HOPS` | `1` |
   | `DAILY_CHAT_LIMIT` | `200` |
   | `WEB_SEARCH_API_KEY` | optional: a Tavily key (enables `web_search`) |

4. **Advanced** → Health Check Path `/api/health`, Auto-Deploy **After CI
   Checks Pass**.
5. Create the service. The first build takes 10–20 minutes; the log ends with
   `Application startup complete`.

## Operating notes

- **Sleep.** Free instances sleep after 15 minutes without traffic; the next
  visit wakes it in about a minute. Open the link yourself before sharing it.
  Data lives in Neon, so nothing is lost.
- **Cost control.** Every chat turn costs one or more LLM calls (plus one for
  memory extraction). The per-IP window and daily cap bound that, and
  `MEMORY_AUTO_EXTRACT=false` halves it. Limits are in memory and reset when
  the instance restarts.
- **Shared state.** The demo has one shared memory and conversation list (no
  user accounts): visitors can see what others told it.
- **Changing a secret** (LLM key, demo password, Neon password): edit it under
  Environment and redeploy. A new `DEMO_PASSWORD` logs everyone out.
- **Snowflake.** The deployment uses the Postgres dataset (`SQL_BACKEND`
  unset). Running against Snowflake would need the private key as a secret
  file instead of a path in the repo.

## Alternative: Google Cloud Run

`ci.yml` also contains a Cloud Run deploy job (build → Artifact Registry →
`gcloud run deploy`, authenticated with Workload Identity Federation, so no
keys are stored in GitHub). It is skipped unless the repository variables
`GCP_PROJECT_ID`, `GCP_REGION`, `GCP_WORKLOAD_IDENTITY_PROVIDER` and
`GCP_SERVICE_ACCOUNT` exist. `deploy/gcp-setup.sh`, run once in Cloud Shell,
creates everything it needs (image repository, service accounts, Secret
Manager secrets, the GitHub trust) and prints those four values.

## Run the production image anywhere

```bash
docker build -t recall .
docker run -p 7860:7860 --env-file .env -e DATABASE_URL=... recall
```
