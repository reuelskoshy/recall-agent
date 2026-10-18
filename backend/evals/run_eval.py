"""Run the eval suite against a running Recall server (local or deployed).

    uv run python -m evals.run_eval                                   # http://localhost:8000
    uv run python -m evals.run_eval --base-url https://recall-s9qe.onrender.com --password ...
    uv run python -m evals.run_eval --only rag-carryover,sql-headcount

It talks to the public API exactly like the browser does (POST /api/chat, SSE),
so it measures the real system: prompt, model, tools, retrieval, and memory.
Each question runs in a fresh conversation. Afterwards the conversations and
memories the run created are deleted (unless --keep), so the demo stays clean.

Metrics:
- retrieval hit rate: the expected handbook section was among the passages
  search_documents returned (plus MRR, the mean of 1/rank);
- tool routing accuracy: the agent called every expected tool;
- answer correctness: every required fact is in the answer, no forbidden one is;
- pass rate: all of the above for a case.

Writes evals/results/<timestamp>.json and .md, and exits with status 1 when the
pass rate is below --min-pass-rate (to gate CI).
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evals.scoring import (
    CaseResult,
    EvalCase,
    EvalSuite,
    TurnTrace,
    score_case,
    skipped_case,
    summarize,
)

EVALS_DIR = Path(__file__).resolve().parent
SAMPLE_DOCUMENT = EVALS_DIR.parents[1] / "samples" / "acme-employee-handbook.md"
MAX_RATE_LIMIT_RETRIES = 3


class EvalClient:
    """The few API calls the eval needs, with auth and rate-limit handling."""

    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.http = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_s)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.http.aclose()

    async def login(self, password: str | None) -> None:
        """Exchange the demo password for a token when the server requires one."""
        status = (await self.http.get("/api/auth/status")).json()
        if not status["required"]:
            return
        if not password:
            sys.exit("This server needs the demo password: pass --password or set EVAL_PASSWORD.")
        response = await self.http.post("/api/auth/login", json={"password": password})
        if response.status_code != 200:
            sys.exit(f"Login failed: HTTP {response.status_code} {response.text}")
        self.http.headers["Authorization"] = f"Bearer {response.json()['token']}"

    async def get(self, path: str) -> Any:
        """GET a JSON resource."""
        response = await self.http.get(path)
        response.raise_for_status()
        return response.json()

    async def delete(self, path: str) -> None:
        """DELETE, ignoring resources that are already gone."""
        response = await self.http.delete(path)
        if response.status_code not in (204, 404):
            response.raise_for_status()

    async def ensure_sample_document(self) -> str:
        """Upload the handbook if it is not there, and wait until it is searchable."""
        documents = await self.get("/api/documents")
        document = next((d for d in documents if d["filename"] == SAMPLE_DOCUMENT.name), None)
        if document is None:
            files = {"file": (SAMPLE_DOCUMENT.name, SAMPLE_DOCUMENT.read_bytes(), "text/markdown")}
            response = await self.http.post("/api/documents", files=files)
            response.raise_for_status()
            document = response.json()
        for _ in range(60):
            if document["status"] != "processing":
                break
            await asyncio.sleep(2)
            document = await self.get(f"/api/documents/{document['id']}")
        status: str = document["status"]
        return status

    async def chat(self, message: str) -> TurnTrace:
        """Send one message in a new conversation and collect the turn from the SSE stream."""
        trace = TurnTrace()
        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            started = time.perf_counter()
            async with self.http.stream(
                "POST",
                "/api/chat",
                json={"message": message},
                headers={"Accept": "text/event-stream"},
            ) as response:
                if response.status_code == 429 and attempt < MAX_RATE_LIMIT_RETRIES:
                    wait = int(response.headers.get("Retry-After", "60"))
                    print(f"    rate limited; waiting {wait}s", flush=True)
                    await asyncio.sleep(wait)
                    continue
                if response.status_code != 200:
                    trace.error = f"HTTP {response.status_code}: {(await response.aread())[:200]!r}"
                    return trace
                parts: list[str] = []
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        _apply_event(trace, json.loads(line[5:]), parts)
                trace.answer = "".join(parts).strip()
                trace.latency_s = time.perf_counter() - started
                return trace
        return trace


def _apply_event(trace: TurnTrace, event: dict[str, Any], parts: list[str]) -> None:
    """Fold one SSE event into the trace."""
    kind = event["type"]
    if kind == "conversation":
        trace.conversation_id = event["conversation_id"]
    elif kind == "token":
        parts.append(event["text"])
    elif kind == "tool_call":
        trace.tools_called.append(event["name"])
    elif kind == "tool_result":
        trace.tool_errors += bool(event["is_error"])
        data = event.get("data") or {}
        if event["name"] == "search_documents":
            trace.citations.append([r["citation"] for r in data.get("results", [])])
    elif kind == "done":
        trace.iterations = event["iterations"]
        trace.prompt_tokens = event["usage"]["prompt_tokens"]
        trace.completion_tokens = event["usage"]["completion_tokens"]
    elif kind == "error":
        trace.error = event.get("message", "error event")


async def run_suite(
    client: EvalClient, cases: list[EvalCase], tools: set[str]
) -> tuple[list[CaseResult], list[str]]:
    """Run every case in order; returns the results and the conversations created."""
    results: list[CaseResult] = []
    conversations: list[str] = []
    for n, case in enumerate(cases, 1):
        print(f"[{n}/{len(cases)}] {case.id}: {case.question}", flush=True)
        if case.requires_tool and case.requires_tool not in tools:
            result = skipped_case(case, f"{case.requires_tool} is not enabled on this server")
        else:
            for message in case.setup:
                setup = await client.chat(message)
                if setup.conversation_id:
                    conversations.append(setup.conversation_id)
            if case.setup:
                await asyncio.sleep(3)  # let background memory extraction finish
            trace = await client.chat(case.question)
            if trace.conversation_id:
                conversations.append(trace.conversation_id)
            result = score_case(case, trace)
        results.append(result)
        print(f"    {_verdict(result)}", flush=True)
    return results, conversations


def _verdict(r: CaseResult) -> str:
    if r.skipped:
        return f"SKIP  {r.skipped}"
    details = [f"tools={','.join(r.tools_called) or '-'}", f"{r.latency_s:.1f}s"]
    if r.retrieval_hit is not None:
        details.append(
            f"retrieval={'rank ' + str(r.retrieval_rank) if r.retrieval_hit else 'MISS'}"
        )
    if not r.routing_ok:
        details.append("WRONG TOOLS")
    if r.missing:
        details.append(f"missing: {'; '.join(r.missing)}")
    if r.forbidden_found:
        details.append(f"forbidden: {'; '.join(r.forbidden_found)}")
    if r.error:
        details.append(f"error: {r.error}")
    return f"{'PASS' if r.passed else 'FAIL'}  " + "  ".join(details)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def render_markdown(
    meta: dict[str, Any], summary: dict[str, Any], results: list[CaseResult]
) -> str:
    """A human-readable report."""
    lines = [
        f"# Recall eval: {meta['timestamp']}",
        "",
        f"Server `{meta['base_url']}` · model `{meta['model'] or 'unknown'}` · "
        f"dataset `{meta['dataset_version']}`",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Pass rate | {_pct(summary['pass_rate'])} ({summary['cases_run']} cases) |",
        f"| Answer correctness | {_pct(summary['answer_correctness'])} |",
        f"| Tool routing accuracy | {_pct(summary['tool_routing_accuracy'])} |",
        f"| Retrieval hit rate | {_pct(summary['retrieval_hit_rate'])} |",
        f"| Retrieval MRR | {summary['retrieval_mrr'] or 'n/a'} |",
        f"| Median latency | {summary['latency_s']['median']} s |",
        "",
        "| Case | Category | Result | Tools | Retrieval | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        verdict = "skipped" if r.skipped else ("pass" if r.passed else "**fail**")
        retrieval = (
            ""
            if r.retrieval_hit is None
            else (f"rank {r.retrieval_rank}" if r.retrieval_hit else "miss")
        )
        notes = r.skipped or "; ".join(
            filter(None, [r.error, *(f"missing {m}" for m in r.missing), *r.forbidden_found])
        )
        lines.append(
            f"| {r.id} | {r.category} | {verdict} | {', '.join(r.tools_called)} | {retrieval} | "
            f"{notes.replace('|', '/')} |"
        )
    return "\n".join(lines) + "\n"


def write_reports(out_dir: Path, stem: str, report_json: str, report_md: str) -> None:
    """Save the JSON and Markdown reports."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(report_json, encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(report_md, encoding="utf-8")


async def main(args: argparse.Namespace) -> int:
    """Run the suite and write the reports; returns the process exit code."""
    suite = EvalSuite.model_validate_json(
        await asyncio.to_thread(Path(args.cases).read_text, encoding="utf-8")
    )
    cases = suite.cases
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c.id in wanted]
        if not cases:
            sys.exit(f"No cases match --only {args.only}")

    client = EvalClient(args.base_url, args.timeout)
    try:
        await client.login(args.password)
        health = await client.get("/api/health")
        tools = {t["name"] for t in await client.get("/api/tools")}
        print(f"Server {args.base_url}: {health['status']}; tools: {', '.join(sorted(tools))}")
        if any(c.expected_sources for c in cases):
            status = await client.ensure_sample_document()
            print(f"Sample handbook: {status}")
        memories_before = {m["id"] for m in await client.get("/api/memories")}

        started = time.perf_counter()
        results, conversations = await run_suite(client, cases, tools)
        elapsed = time.perf_counter() - started
        runs = await client.get("/api/runs?limit=1")  # before cleanup deletes them

        if not args.keep:
            await asyncio.sleep(3)  # background extraction of the last turn
            new_memories = {m["id"] for m in await client.get("/api/memories")} - memories_before
            for conversation_id in conversations:
                await client.delete(f"/api/conversations/{conversation_id}")
            for memory_id in new_memories:
                await client.delete(f"/api/memories/{memory_id}")
            print(
                f"Cleaned up {len(conversations)} conversations and {len(new_memories)} memories."
            )
    finally:
        await client.close()

    summary = summarize(results)
    model = runs[0]["model"] if runs else None
    meta = {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "base_url": args.base_url,
        "model": model,
        "dataset_version": suite.dataset_version,
        "duration_s": round(elapsed, 1),
    }
    out_dir = Path(args.out)
    stem = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    report = {"meta": meta, "summary": summary, "results": [r.as_dict() for r in results]}
    await asyncio.to_thread(
        write_reports,
        out_dir,
        stem,
        json.dumps(report, indent=2),
        render_markdown(meta, summary, results),
    )

    print()
    print(
        f"Pass rate            {_pct(summary['pass_rate'])}  ({summary['cases_run']} cases, "
        f"{summary['cases_skipped']} skipped)"
    )
    print(f"Answer correctness   {_pct(summary['answer_correctness'])}")
    print(f"Tool routing         {_pct(summary['tool_routing_accuracy'])}")
    hit_rate, mrr = _pct(summary["retrieval_hit_rate"]), summary["retrieval_mrr"]
    print(f"Retrieval hit rate   {hit_rate}  (MRR {mrr})")
    print(f"Median latency       {summary['latency_s']['median']} s;  total {meta['duration_s']} s")
    print(f"Report: {out_dir / stem}.md")

    pass_rate = summary["pass_rate"] or 0.0
    if pass_rate < args.min_pass_rate:
        print(f"FAILED: pass rate {pass_rate:.0%} is below {args.min_pass_rate:.0%}")
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description="Evaluate Recall end to end.")
    parser.add_argument("--base-url", default=os.getenv("EVAL_BASE_URL", "http://localhost:8000"))
    parser.add_argument(
        "--password",
        default=os.getenv("EVAL_PASSWORD"),
        help="Demo password, if the server has one (or set EVAL_PASSWORD).",
    )
    parser.add_argument("--cases", default=str(EVALS_DIR / "cases.json"))
    parser.add_argument("--only", help="Comma-separated case ids to run.")
    parser.add_argument("--out", default=str(EVALS_DIR / "results"))
    parser.add_argument("--keep", action="store_true", help="Keep the conversations and memories.")
    parser.add_argument("--timeout", type=float, default=240.0, help="Seconds per request.")
    parser.add_argument("--min-pass-rate", type=float, default=0.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
