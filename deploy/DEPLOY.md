# Deploying Recall: Google Cloud Run + Neon (free tier)

One container (API + built frontend + embedding model) runs on Cloud Run;
PostgreSQL with pgvector runs on Neon's free tier. GitHub Actions deploys every
push to `main` once lint, type checks, and tests pass.

```
git push ─▶ GitHub Actions: tests ─▶ build image ─▶ Artifact Registry ─▶ Cloud Run
                                                                            │
browser ──HTTPS──▶ Cloud Run (us-east4, 1 vCPU / 2 GiB, scales to zero) ──TLS──▶ Neon (us-east-1)
                   FastAPI + React build + bge-small, demo password gate     app tables, memories,
                         │                                                    acme dataset, recall_reader
                         └──HTTPS──▶ LLM API (IFM K2-Horizon)
```

On every start the container runs `deploy/start.sh`: migrations, the read-only
role + Acme dataset, the sample handbook, then the API. All steps are
idempotent, so restarts and redeploys are safe.

## 1. Database (Neon)

1. Sign up at https://neon.tech and create a project (Postgres 16 or 17) in
   **AWS US East 1**, next to the Cloud Run region.
2. **Connect** → turn **Connection pooling OFF** (use the direct endpoint, whose
   host does not contain `-pooler`). Recall uses prepared statements, which
   PgBouncer's transaction mode breaks.
3. Copy the connection string (`postgresql://neondb_owner:...@ep-xxx.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require`).
   Paste it as-is later: Recall converts it for asyncpg.

The first start creates the `vector` extension, all tables, the `acme` dataset,
and the `recall_reader` login role (SELECT on `acme` only).

## 2. Google Cloud project

1. https://console.cloud.google.com → create a project (e.g. `recall-demo`) and
   note its **project ID**.
2. **Billing** → link a billing account (a card is required, but the usage below
   stays inside the free tier). Then **Billing → Budgets & alerts** → create a
   budget of e.g. $5 with email alerts, so any surprise is visible early.
3. Open **Cloud Shell** (the `>_` icon, top right), upload `deploy/gcp-setup.sh`
   (⋮ → Upload), and run:

   ```bash
   bash gcp-setup.sh <project-id> <github-owner>/recall-agent
   ```

   It enables the APIs, creates the image repository, two service accounts,
   the secrets (it prompts for the Neon string, the LLM key, and the demo
   password without echoing them, and generates the read-only DB password),
   and a Workload Identity Federation pool that trusts only your repository.
   It ends by printing four values.

## 3. Connect GitHub

In the GitHub repo: **Settings → Secrets and variables → Actions → Variables**
→ add the four printed values as repository variables:
`GCP_PROJECT_ID`, `GCP_REGION`, `GCP_WORKLOAD_IDENTITY_PROVIDER`,
`GCP_SERVICE_ACCOUNT`. No secrets are stored in GitHub: Actions exchanges a
short-lived OIDC token for Google credentials on each run.

Optional variables override the defaults: `LLM_BASE_URL`, `LLM_CHAT_MODEL`,
`LLM_REASONING_FIELD`, `DAILY_CHAT_LIMIT`.

## 4. Deploy

Push to `main` (or re-run the latest workflow on the **Actions** tab). The
`deploy` job builds the root `Dockerfile`, pushes it, runs `gcloud run deploy`,
and smoke-tests `/api/health`; the job page links the live URL
(`https://recall-xxxxx.us-east4.run.app`). Logs: Cloud Console → Cloud Run →
`recall` → **Logs**.

## Operating notes

- **Cost control.** `--max-instances 1` caps compute; with no traffic the
  service scales to zero. CPU stays allocated while an instance is up
  (`--no-cpu-throttling`) so background memory extraction finishes after the
  answer is streamed; the instance stops ~15 minutes after the last request.
  Every chat turn costs one or more LLM calls (plus one for memory extraction);
  the per-IP window and daily cap bound that, and `MEMORY_AUTO_EXTRACT=false`
  halves it. Limits are in memory and reset when the instance restarts.
- **Cold start.** The first visit after idle boots the container (~20–40 s:
  loading PyTorch and the embedding model).
- **Shared state.** The demo has one shared memory and conversation list (no
  user accounts): visitors can see what others told it. Fine for a guided
  demo; per-user isolation is listed as future work.
- **Changing a secret** (e.g. a new LLM key or demo password): re-run
  `gcp-setup.sh` and enter only that value (Enter keeps the others), then
  redeploy. A new `DEMO_PASSWORD` logs everyone out.
- **Snowflake.** The deployment uses the Postgres dataset (`SQL_BACKEND=postgres`).
  Running against Snowflake would need the private key in Secret Manager
  instead of a file.

## Run the production image anywhere

```bash
docker build -t recall .
docker run -p 7860:7860 --env-file .env -e DATABASE_URL=... recall
```
