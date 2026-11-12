# Recall Jobs: Phase 0 plan

Status: **approved 2026-11-11** (all five recommendations in section 5).
Phase 1 (schema) done. Phase 2 (source connectors) and Phase 3 (extraction
and eligibility) done: see the phase notes at the end. Everything under "How Recall works today" was read from
the code at commit `38dfd46`, not assumed.

---

## 1. How Recall works today

### 1.1 Tools: definition, registration, execution

- **Interface** (`backend/app/tools/base.py`). A tool subclasses
  `Tool[InputT]` and sets `name`, `description` (what the model reads),
  `input_model` (a Pydantic model, converted to JSON Schema with titles
  stripped) and optionally `timeout_s`. It implements
  `async run(args, ctx) -> ToolResult`. An optional `enabled(settings)`
  classmethod hides a tool when its config is missing (`web_search` without a
  key).
- **Result**: `ToolResult(content, data, is_error)`. `content` goes to the
  model. `data` goes to the UI only (citations, tables), is stored on the
  `messages.tool_data` column, and is never sent to the LLM.
- **Expected failures** raise `ToolError(msg)`. The registry turns it into
  `ToolResult(is_error=True)` so the model can retry or explain.
- **Context**: `ToolContext(conversation_id, session_factory, embedder,
  settings)`. Tools get a session *factory*, because tools in one round run
  concurrently and an `AsyncSession` must not be shared between tasks.
- **Registration** (`backend/app/tools/registry.py`). `@register_tool` on the
  class. `discover_tools()` imports every module in `app/tools/`, so **adding a
  tool means adding one module**. `build_registry()` keeps the enabled ones;
  `get_tool_registry()` is a process-wide `lru_cache`.
- **Execution**: `ToolRegistry.execute()` validates arguments (validation
  errors go back to the model as text), applies `asyncio.timeout(tool.timeout_s
  or TOOL_TIMEOUT_S)`, converts `ToolError`, timeouts and unexpected exceptions
  into error results (unexpected ones are `logger.exception`ed), and truncates
  `content` to `TOOL_MAX_RESULT_CHARS` (4000).

### 1.2 Runs, failures and timeouts

- `run_turn()` in `backend/app/agent/runner.py` commits the user message and
  an `AgentRun` (`status="running"`) before calling the model.
- Every LLM call and tool call becomes an `AgentStep` (kind, name, arguments,
  `result_summary` clipped to 500 chars, latency, tokens).
- **Failure contract**: the stream ends with exactly one `Done` or `Error`,
  and the run ends `completed | failed | timeout`, `cancelled` (client left,
  written from a separate task) or `interrupted` (set by the reaper).
  `_finish_failed()` rolls back the half-finished round so history never holds
  a tool call without its results.
- **Deadlines**: one wall-clock deadline per turn (`AGENT_TURN_TIMEOUT_S`,
  120 s) wraps every model stream (`_before_deadline`, per `__anext__`) and
  every tool round (`asyncio.timeout_at`). Memory recall has its own 5 s
  budget and degrades to "no memories" instead of failing the turn.
- **LLM resilience** (`backend/app/llm/resilience.py`): SDK retries on
  429/5xx, then a consecutive-failure circuit breaker, with an optional
  fallback model used only before the first token is streamed.
- **Database resilience** (`backend/app/db/session.py`): connects retried with
  backoff on transient errors only (`OSError`, starting up, too many
  connections), bounded connect/checkout/statement timeouts, `pool_pre_ping`,
  recycling for Neon.
- **Health**: `/api/health/live` (process up, used by Render) and
  `/api/health/ready` (per-dependency status, 503 when degraded).

### 1.3 Background work

There is no job queue and no scheduler. Three patterns exist:

| Work | Mechanism | How it ends cleanly |
|---|---|---|
| Document ingestion | FastAPI `BackgroundTasks` after the upload response | `asyncio.timeout(INGEST_TIMEOUT_S)`, never raises, writes `ready`/`failed` on the row |
| Memory extraction | `asyncio.create_task`, reference kept in `_pending` | Catches everything, logs, drops; drained on shutdown |
| Reaper | `asyncio.create_task(run_reaper())` in the lifespan | Every 60 s fails `running` runs / `processing` documents older than their timeout + 60 s grace. Age-based, so it is safe with several instances |

### 1.4 Migrations, tests, evals, CI

- **Migrations**: Alembic, async `env.py`, `target_metadata` from
  `app.db.models` (one models file). Files are
  `alembic/versions/YYYYMMDD_NNNN_<slug>.py`, eight so far. Production runs
  `alembic upgrade head` on every container start (`deploy/start.sh`).
- **Tests**: pytest against a real `<db>_test` Postgres, created and migrated
  once per session (`tests/conftest.py`). Fake LLM (`app/llm/fake.py`, scripted
  turns) and `FakeEmbedder`; web search disabled; `MEMORY_AUTO_EXTRACT=false`.
  The `db` fixture truncates `conversations, documents, memories CASCADE` after
  each test. API tests use `httpx.AsyncClient` over `ASGITransport` and
  `parse_sse()`. Failure-mode tests live in `test_failure_modes.py`,
  `test_db_resilience.py`, `test_llm_resilience.py`. 322 pytest + 17 Vitest.
- **Evals** (`backend/evals/`): `run_eval.py` sends `cases.json` questions to
  a **running server** over the public SSE API; `scoring.py` checks answers
  deterministically (required/forbidden facts, tool routing, retrieval rank).
  Results go to the gitignored `evals/results/`.
- **CI** (`.github/workflows/ci.yml`): backend (ruff check + format, mypy
  strict, `alembic upgrade head`, pytest against `pgvector/pgvector:pg16`) and
  frontend (ESLint, Vitest, `tsc` + build). Render auto-deploys "after CI
  checks pass". The Cloud Run job in the same file is skipped (no GCP vars).

---

## 2. Where the brief does not match the code

These need a decision before Phase 1. Each has a recommendation.

### 2.1 "Behind the existing access control" is a *shared* demo password (blocks Hard Rule 5)

`app/core/security.py` is a demo gate: one `DEMO_PASSWORD`, shared with the
people you send the link to. `deploy/DEPLOY.md` says so explicitly: *"The demo
has one shared memory and conversation list: visitors can see what others told
it."* Anyone with the demo password could read `/api/runs/{id}` (tool
arguments and results), `/api/conversations`, `/api/memories`, and would get
any jobs tool the agent offers. Your profile and applications would leak to
every recruiter you sent the link to.

Also: **memory auto-extraction** would turn a chat like "I graduated in 2025,
draft outreach for job 12" into shared long-term memories, shown to visitors.

**Recommendation: a separate private deployment, feature-flagged.**
- `JOBS_ENABLED=false` by default. When false: no jobs routes are mounted, the
  jobs tools' `enabled()` returns false, no scheduler endpoint exists, and the
  UI hides the tabs (it reads a flag from `/api/auth/status`). The public demo
  stays exactly as it is.
- A second Render free service from the same repo/image, with
  `JOBS_ENABLED=true`, its own strong `DEMO_PASSWORD` known only to you, and
  **its own Neon database** (Neon free allows several databases in a
  project). Physical separation is simpler to defend in an interview than
  per-row ACLs on a system that has no users.
- Alternative (not recommended): an "owner" token scope inside the one
  deployment. That needs caller identity in `ToolContext`, per-turn tool
  filtering, owner-only filtering of runs/conversations/memories. That is a lot
  of new surface, and one missed endpoint leaks everything.

### 2.2 Logs already contain user content

- `app/memory/extraction.py:205` logs every extracted memory's text at INFO.
  `:169` logs 200 chars of raw model output.
- `registry.execute` uses `logger.exception` on unexpected tool errors. The
  exception message can contain argument values.
- The runner's `_finish_failed` stores `f"{type(exc).__name__}: {exc}"` in
  `agent_runs.error`, which `/api/runs` returns.

For Hard Rule 5 the jobs code will **log IDs, counts and error types only**,
never profile values, listing text or draft bodies. I'll add a test that runs
the pipeline with a sentinel string in the profile and asserts it never shows
up in captured logs (`caplog`) or in any error string. I'll also propose
dropping the memory-content INFO log (one line, a change to existing
behaviour, so flagged here rather than done silently).

### 2.3 Hard Rule 3 vs Phase 5 conflict on facts

Phase 5 wants "one specific detail about the company taken from its listing or
site". Hard Rule 3 says every company name, technology and number must exist in
the profile. The company's own name and the listing's detail never will.

**Proposal**: the claims check runs against two **allowed fact sources**,
tagged separately:
1. the profile (every claim about *me*), and
2. the single job listing being drafted for (only the company name, product
   and technologies the listing names).

A claim passes only if it is found in one of them. A technology that appears
only in the listing must not be claimed as *my* skill. The deterministic
version: technologies in the draft must be in my skills, **or** appear in a
sentence that does not contain first-person markers (I, my, I've). That is
crude, so I will be honest about the limit in the README and in interviews:
the check catches invented numbers, dates, employers and tech; it cannot
catch an invented *qualitative* claim ("I led the team") with no checkable
token. Mitigation: the prompt is built only from profile facts, and you review
every draft (Hard Rule 1).

How "technology" is detected deterministically: a curated lexicon file
(`app/jobs/tech_lexicon.txt`, ~300 terms, e.g. "PostgreSQL", "React",
"Kubernetes") plus your skill names. No NER model, no new dependency.

### 2.4 The daily run vs Render free tier

Render free instances **sleep after 15 minutes without traffic** and cap RAM
at 512 MB (the image peaks around 445 MB). So:
- An in-process scheduler (APScheduler or an asyncio loop) does not fire while
  the instance is asleep. Not viable.
- Render Cron Jobs are a paid service type (from memory; to verify against
  Render's current pricing page before Phase 7, as is Neon's free-tier
  database limit in 2.1).
- **Recommendation**: a scheduled **GitHub Actions workflow** (free) that
  calls `POST /api/jobs/runs` with a separate `JOBS_CRON_TOKEN` (not the UI
  password). The endpoint returns `202` at once and runs the pipeline in a
  background task, like document ingestion. The call itself also wakes the
  instance.
- A long run could be killed when the instance sleeps or restarts. Design for
  it: each job is processed and committed individually, a run has a hard
  deadline (`JOBS_RUN_TIMEOUT_S`), the reaper marks orphaned runs
  `interrupted`, and the next run picks up unprocessed jobs. Nothing is lost,
  only delayed.
- Concurrency: a partial unique index allows only one `running` ingestion run,
  so a cron retry during a run gets `409`, not a duplicate run.
- LLM calls happen once per **new or edited** listing (by content hash), not
  per run. Per-run caps (`JOBS_MAX_NEW_PER_RUN`) bound the LLM bill.

### 2.5 Robots.txt and LinkedIn are not handled anywhere today

- `app/web/fetch.py` does not read robots.txt. Its SSRF guard is solid, but it
  is a *security* check, not a *permission* check.
- Greenhouse, Lever, Ashby and HN (Algolia) are documented public APIs meant
  for programmatic access, so robots.txt does not govern them. I'll still
  verify each provider's docs and terms in Phase 2 (as you asked).
- For any **page** fetch done by the pipeline (the company-site detail in
  Phase 5), I'll add a robots check with the stdlib `urllib.robotparser`
  (no dependency), cached per host, fail-closed if robots.txt is unreachable.
- The generic "search" connector uses Tavily, which can return LinkedIn,
  Indeed or Glassdoor URLs. A **domain denylist** (`linkedin.com`,
  `lnkd.in`, plus any site whose robots.txt disallows us) is applied to search
  results **before** they are stored or fetched. A test enforces that a
  LinkedIn hit never becomes a job row.

### 2.6 Other smaller mismatches

| Brief says | Reality | Plan |
|---|---|---|
| Ask "what follow-ups are due" | `query_database` connects as `recall_reader`, SELECT on the `acme` schema only | Good: jobs tables stay invisible to text-to-SQL. Dedicated read tools instead (Phase 8) |
| Semantic similarity with "existing embeddings" | bge-small (384-d) truncates at 512 tokens; Recall adds a retrieval **query prefix** | Embed a compact summary (title + extracted requirements + technologies), not the raw listing. Profile embedded as a document, no prefix, so the comparison is symmetric |
| Resume variants "file reference" | Uploaded files already live in `documents` (bytes kept, deduped by hash) | `resume_variants.document_id` FK to `documents`. Upload through the existing flow |
| Eval "like the existing harness" | Existing harness drives a live server through chat | Jobs eval is offline: fixtures → extraction (real LLM) → rules (code) → compare to labels. Reuses `scoring.py`'s deterministic style, not its HTTP client |
| Test cleanup | `db` fixture truncates three tables | Add the jobs tables to the truncate list |
| "339 tests" | 322 pytest + 17 Vitest = 339 | Matches |

---

## 3. Proposed data model (Phase 1)

All tables in the `public` schema (not `acme`, so the read-only SQL role can't
see them). Enumerations are `String` + `CHECK` constraints, like
`memories.importance`, not Postgres `ENUM` types (easier to migrate).

```
candidate_profile        1 row (CHECK id = 1)
  graduation_year, degree, degree_field, years_experience (numeric),
  allowed_locations text[], remote_ok, role_families text[],
  employment_types text[], min_salary + currency, summary,
  embedding vector(384), updated_at

profile_entries          education / experience / project / link / achievement
  kind, title, organization, location, start_date, end_date,
  description, technologies text[], url, sort_order

profile_skills           name (unique, case-insensitive), category, years

resume_variants          name, target_role_family, document_id → documents,
                         notes, is_default

companies                name (unique on lower(name)), website,
                         ats (greenhouse|lever|ashby|none), board_token,
                         watch (bool), contact_policy
                         (allowed | application_form_only | do_not_contact),
                         notes
  NEW companies default to application_form_only (fail closed): a company
  found by search is never contactable until you say so.

jobs                     source, external_id, UNIQUE(source, external_id)
                         company_id → companies, title, url, location_raw,
                         raw_text, content_hash (sha256 of normalised text),
                         posted_at, first_seen_at, last_seen_at,
                         content_changed_at, closed_at (gone from the feed)

job_analyses             job_id, content_hash  UNIQUE(job_id, content_hash)
                         extraction JSONB (validated schema),
                         extractor_version, rules_version,
                         verdict (eligible | flagged | rejected | error),
                         reasons JSONB [{rule, outcome, evidence_sentence}],
                         score, score_components JSONB, embedding vector(384)

drafts                   job_id, kind (outreach_email | application_answers),
                         resume_variant_id, resume_reason, body,
                         status (needs_review | approved | discarded),
                         claims_check JSONB {passed, unsupported[...]},
                         attempts, created_at, updated_at

applications             job_id UNIQUE, status, follow_up_on (date), notes,
                         status_changed_at
application_events       application_id, from_status, to_status, at
                         (the timestamps the brief asks for, as an audit log)

ingestion_runs           started_at, finished_at, status (running | completed
                         | partial | failed | interrupted), trigger (cron |
                         manual), counts JSONB {source: {fetched, new,
                         updated, closed, errors}}, errors JSONB
  Partial unique index: only one row WHERE status = 'running'.
```

**Design choices to defend:**

- **Dedup** is `UNIQUE(source, external_id)` with `INSERT ... ON CONFLICT DO
  UPDATE`. That makes re-runs idempotent at the database level, not by
  "check then insert" (which races). The content hash detects edits: a changed
  hash creates a new analysis row; an unchanged one just bumps `last_seen_at`.
- **Extraction and judgment are versioned separately.** The LLM extraction is
  expensive and keyed by content hash. The verdict and score are cheap, pure
  code. When a rule or weight changes, I bump `rules_version` and re-evaluate
  every job **without a single LLM call**.
- **Status state machine** lives in code (a transition table) with a `CHECK`
  on allowed values in the DB. Transitions: `new → shortlisted → drafted →
  applied → interview → offer`; `rejected` and `withdrawn` from any
  non-terminal state. I'd add `skipped` ("not interested") from `new` and
  `shortlisted`, so dismissing a role is not mislabelled as a rejection.
- **Do-not-contact** is enforced at one choke point (`create_draft()` refuses
  `outreach_email` when the policy is `do_not_contact`, and also when it is
  `application_form_only`). Defence in depth: a Postgres trigger on `drafts`
  that raises on the same condition, so even a bug or a hand-written SQL insert
  cannot create one. Both are tested.

---

## 4. Proposed module layout

```
backend/app/jobs/
  __init__.py
  schemas.py          Pydantic: ExtractedJob, verdict/reason types, score breakdown
  profile.py          load profile, fact index for the claims check, profile embedding
  sources/
    base.py           Connector ABC: name, fetch() -> list[RawListing]; timeouts, retries
    greenhouse.py
    lever.py
    ashby.py
    hn_hiring.py      HN "Who is hiring" via the Algolia HN Search API
    web_search.py     reuses app/web/search.py providers + domain denylist
  robots.py           urllib.robotparser, per-host cache, fail closed
  ingest.py           run orchestration: per-source isolation, upsert, counts, deadline
  extraction.py       LLM -> ExtractedJob (lenient parse like memory/extraction.py)
  rules.py            eligibility rules (pure functions, each returns rule, outcome, evidence)
  scoring.py          weighted components, weights from Settings
  claims.py           deterministic claims check
  drafting.py         resume choice, draft generation, check/regenerate loop, policy gate
  applications.py     state machine
  tech_lexicon.txt

backend/app/db/models.py          + jobs tables (keeping the one-models-file pattern)
backend/app/api/jobs.py           jobs, analyses, drafts, applications, companies, profile
backend/app/api/job_runs.py       POST /jobs/runs (cron token), GET run history
backend/app/tools/job_tools.py    Phase 8 tools, enabled() only when JOBS_ENABLED
backend/app/core/reaper.py        + ingestion_runs
backend/alembic/versions/2026111x_0009_jobs_*.py
backend/tests/jobs/...            + fixtures/ (recorded feeds, committed, synthetic)
backend/evals/jobs/               run_jobs_eval.py, sample.json (synthetic, committed)
backend/evals/jobs/private/       gitignored: your labelled real listings
frontend/src/pages/{Jobs,Drafts,Tracker,Digest}Page.tsx + hooks
.github/workflows/jobs-cron.yml   daily POST to the private deployment
```

**Dependencies**: none expected. httpx (HTTP), stdlib `urllib.robotparser`,
`hashlib`, existing LLM layer, embedder and Tavily provider cover everything.
I'll ask before adding any.

---

## 5. Decisions I need from you

1. **Isolation (2.1)**: separate private deployment + `JOBS_ENABLED` flag
   (recommended), or an owner scope inside the public demo?
2. **Claims check sources (2.3)**: profile + the target listing, with the
   first-person rule for technologies. OK?
3. **Scheduling (2.4)**: GitHub Actions cron → protected endpoint → background
   run. OK?
4. **`skipped` status**: add it to the state machine?
5. **Log hygiene (2.2)**: may I remove the memory-content INFO log in
   `app/memory/extraction.py` as part of this work?

Once these are settled, Phase 1 starts with a short plan and the file list.

---

## Phase 2 notes: source connectors (2026-11-12)

Provider documentation and terms were read live on 2026-11-12, and each
endpoint was probed. Findings that differ from the docs or from assumptions:

| Source | Endpoint | Notes |
|---|---|---|
| Greenhouse | `boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true` | No auth for GET. `content` is HTML-entity-encoded. Pay ranges only on the single-job endpoint (salary comes from text in Phase 3). Unknown board: 404 |
| Lever | `api.lever.co/v0/postings/{site}?mode=json` | EU accounts use `api.eu.lever.co`: write the token as `eu:{site}`. `createdAt` (epoch ms) is in live responses but not the README. robots.txt asks for `Crawl-delay: 1`; we make one request per board per day |
| Ashby | `api.ashbyhq.com/posting-api/job-board/{name}` | `jobs[].id` is in live responses but not the docs (falls back to the `jobUrl` path). `isListed: false` postings are skipped. Unknown board: 404 with a plain-text body |
| HN | `hn.algolia.com/api/v1/search_by_date`, `/search` | Documented limit: 10,000 requests/hour per IP. `whoishiring` also posts "Who wants to be hired?", so the thread is chosen by title. `hitsPerPage` is capped at 1000 |
| Search | Tavily via `app/web/search.py` | Terms and AUP allow it; the AUP's employment clause targets screening candidates, not a candidate searching listings |

None of the three ATS APIs documents rate limits or usage terms for GET.

**Design.** A feed (one board, the HN thread, one query) is the unit of
isolation: each has its own deadline, and its failure is one entry in
`ingestion_runs.errors`. HTTP retries cover transient failures only
(connection errors, timeouts, 429, 5xx), with at most `JOBS_HTTP_MAX_ATTEMPTS`
attempts, exponential backoff and `Retry-After` capped at 30 s. 4xx and
malformed bodies fail at once. Board feeds are complete, so postings that
disappear get `closed_at` (and reopen if they come back). HN and search are
not complete, so nothing is closed from them. Search hits pass the domain
denylist (before any request) and then robots.txt (`urllib.robotparser`,
fetched through the SSRF-safe transport, fail closed). The reaper marks
orphaned runs `interrupted`. Fixtures in `backend/tests/fixtures/jobs/` were
recorded from the live APIs and trimmed.

---

## Phase 3 notes: extraction and eligibility (2026-11-12)

The candidate is looking for jobs **in India**, so location matching is
country-aware.

**Extraction** (`app/jobs/extraction.py`, `schemas.py`) runs one LLM call per
listing *version*: there is one `job_analyses` row per (job, content hash), so
an unchanged listing is never sent to the model again.
- Each run takes the newest open jobs, at most `JOBS_EXTRACT_MAX_PER_RUN`, with
  `JOBS_EXTRACT_CONCURRENCY` at a time and each under `JOBS_EXTRACT_TIMEOUT_S`.
- The prompt covers Indian listing conventions: LPA, lakhs and crore salaries,
  "2025 batch" and "passouts" graduation years, and "freshers" for no experience.
- Parsing is lenient. The schema turns odd values into safe defaults, so one
  bad field never discards the whole extraction.
- The model also decides `is_job_listing`. In live testing, HN "Who is hiring"
  threads contained job seekers' posts, which were otherwise judged "eligible".

**Failure handling.**
- Unparseable output, a timeout, a non-transient LLM error, or any other
  per-listing exception counts one attempt (`extraction_attempts`, migration
  0010). After `JOBS_EXTRACT_MAX_ATTEMPTS` the listing stays at verdict `error`.
- A *transient* LLM failure (an outage, or the circuit breaker is open) counts
  no attempt and stops new extractions for the run. The rest are deferred, and
  the run is recorded `partial` with an `LLMUnavailable` error.
- Fetching and analysis share the run deadline.

**Rules** (`app/jobs/rules.py`) are pure functions, run against every open job
on every run with no LLM calls, so profile edits apply at once. Each stores
its rule, outcome, detail and the listing's own sentence.

| Rule | Outcome |
|---|---|
| Not a job listing | reject |
| Required graduation year later than yours | reject (an earlier year only flags) |
| Role family not in your role types | reject |
| Location outside your allowed list | reject. A remote job passes if its region includes an allowed country (India matches APAC and Asia), or if it has no restriction |
| In a sponsorship-only place (`sponsorship_locations`, e.g. United States) | pass if the listing offers visa sponsorship (with a verified quote); reject if it says it does not ("no sponsorship", "must be authorized to work in the US", "US citizens only"); flag if it does not say |
| No location given | flag |
| Minimum experience above yours | flag, with the size of the gap |
| Internship with no graduation-year requirement | flag |
| Senior duties on a junior title (SDE-1, trainee, fresher...) | flag |
| Profile field empty | not checked |

**Evidence is verified.** A rule may only *reject* if the model's quote is
actually in the listing (the comparison ignores case, whitespace and smart
punctuation). If the quote is missing or invented, the rule only *flags*, so
a hallucination can never silently drop a job.

**Live check** (K2-Horizon, 7 India-related posts from the November 2026 HN
thread):
- 1–32 s per listing.
- Two job-seeker posts were rejected with verified quotes. A third was flagged
  because its quote did not match the text.
- A US-only role was rejected on location.
- Two remote roles open to India were judged eligible.

**Visa sponsorship (added 2026-11-12).** US jobs are acceptable only with visa
sponsorship. This is a profile preference, `sponsorship_locations` (migration
0011), not a hard-coded country. The extraction schema gained
`visa_sponsorship`, and the extractor version was bumped. Listings extracted by
an older prompt are extracted again, within the per-run cap. In a live check on
HN posts that mention visas:
- A New York role saying "Visa sponsorship is not available." was rejected with
  that sentence.
- Roles in Zurich, Amsterdam and London that offer sponsorship were rejected,
  because those places are not in the list.
- One reply in six was not valid JSON: that counts as one attempt and is
  retried on the next run.

## Phase 4 notes: scoring and ranking (2026-11-12)

Every open job judged **eligible** or **flagged** gets a score in [0, 1]
(`app/jobs/scoring.py`). Rejected jobs have no score; a job that becomes
rejected loses it. Scoring runs on every analysis pass, after judging, with no
LLM calls.

`score = Σ wᵢ·vᵢ / Σ wᵢ` over the components that can be measured:

| Component | Value | Default weight | Why this weight |
|---|---|---|---|
| semantic | cosine(listing, profile) embeddings, mapped from [0.55, 0.80] to [0, 1] | 0.30 | Fit, the fuzzy view |
| technology | (matched + 1) / (listed + 2), against profile skills and the technologies of jobs and projects | 0.25 | Fit, the exact view; smoothing stops "SQL, Python" scoring 1.0 |
| experience | 1 when you meet the minimum, linear to 0 at a 3-year gap | 0.20 | The main barrier for an early-career candidate |
| freshness | 0.5 ^ (age / 14 days), from the posting date or first seen | 0.10 | Early applications do better, but fit matters more |
| location | 1 in an allowed place or allowed remote, 0.5 in a sponsorship-only place | 0.10 | The rules already removed disallowed places |
| salary | min(1, top of range / your minimum), annual and same currency only | 0.05 | Most Indian listings give no salary |

**Missing data is neutral.** A component that cannot be measured (no salary,
no technologies, no stated minimum, a failed embedding) is left out and the
other weights are rescaled. Each stored component has `value`, the effective
`weight`, `contribution` and a `detail` sentence, so the UI can explain a rank.
All weights, the half-life and the similarity range are `JOBS_SCORE_*` settings.

**Embeddings.** Each listing version is embedded once (bge-small, stored on
`job_analyses.embedding`). The profile is re-embedded every run, since it can
change at any time. If the model fails, jobs are still scored, without the
semantic component, and `embedding_failed` is counted.

**Calibration.** The similarity range comes from 100 live listings against the
real profile: cosines ran from 0.54 (key account management) to 0.78 (SDE2, API
Platform), median 0.64.

**Known limits.**
- `technologies` mixes required and nice-to-have tools, since the extractor does
  not separate them. Splitting them would need a prompt change and re-extraction.
- Board posting dates can be months old for listings that are still open, so
  freshness can bury an old but good fit. Lower `JOBS_SCORE_W_FRESHNESS` if
  that happens too often.
