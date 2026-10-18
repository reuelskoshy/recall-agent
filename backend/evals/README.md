# Evals

End-to-end evaluation of Recall: real questions sent to a running server through
the public API (the same SSE stream the browser uses), scored automatically.

```bash
cd backend
uv run python -m evals.run_eval                                   # local: http://localhost:8000
uv run python -m evals.run_eval --base-url https://recall-s9qe.onrender.com --password "$DEMO_PASSWORD"
uv run python -m evals.run_eval --only rag-carryover,sql-headcount --keep
```

Each case runs in a fresh conversation. Afterwards the run deletes the
conversations and memories it created (`--keep` leaves them). Reports go to
`evals/results/<timestamp>.json` and `.md`; `--min-pass-rate 0.8` makes the run
exit with status 1 below that rate, for use as a CI gate.

## Results

Two consecutive runs on 2026-10-18 against a local server (model IFM
K2-Horizon, dataset `acme-v1`, web search not configured so 1 case skipped):

| Metric | Run 1 | Run 2 |
|---|---|---|
| Pass rate (14 cases) | 100% | 100% |
| Answer correctness | 100% | 100% |
| Tool routing accuracy | 100% | 100% |
| Retrieval hit rate | 100% (MRR 1.00) | 100% (MRR 0.93) |
| Median latency per question | 2.9 s | 3.3 s |

LLM output varies between runs (in run 2 the combined RAG+SQL question found
its passage at rank 2), so re-run before quoting numbers for a changed prompt
or model. A small suite like this catches regressions; it does not prove
general accuracy.

## What is measured

| Metric | Meaning |
|---|---|
| **Retrieval hit rate** | `search_documents` returned the handbook section that holds the answer (MRR: mean of 1/rank) |
| **Tool routing accuracy** | the agent called every tool the question needs (e.g. both `search_documents` and `query_database`) |
| **Answer correctness** | every required fact is in the answer and no known-wrong one is |
| **Pass rate** | all three hold for the case |

Answers are checked with deterministic rules, not an LLM judge: the same answer
always gets the same score, scoring costs nothing, and a failure names the
missing fact. Matching ignores case, Markdown, currency signs, and digit
grouping (`€92,121.43` matches `92121.43`), and a number must match a whole
number (`10` does not match `100`). See `scoring.py` and its tests in
`tests/test_eval_scoring.py`.

## The cases (`cases.json`)

| Category | What it checks |
|---|---|
| `rag` (6) | facts from the sample handbook, including an exact token (`SEC-LOST-01`) that keyword search must catch |
| `refusal` (1) | a question the handbook does not cover: the agent must say so instead of inventing a policy |
| `sql` (4) | counts, averages, and top-1 queries over the Acme dataset (expected values computed from the deterministic `acme-v1` data) |
| `rag+sql` (1) | needs both: the carry-over limit from the handbook, then a query against leave balances |
| `tool` (1) | arithmetic must go through the calculator |
| `memory` (1) | a fact stated in one conversation is used in a new one |
| `web` (1) | web search; skipped when the server has no search API key |

To add a case, append an object to `cases.json`: `question`, `expected_tools`,
optional `expected_sources` (substrings of a citation, e.g. `"2.1 Annual leave"`),
and `must_include` as groups of alternatives (`[["10"], ["31 march", "march 31"]]`
means "10" AND one of the two dates).
