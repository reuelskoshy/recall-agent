"""Scoring for eval cases: pure functions, so the rules are unit-tested and reproducible.

Answers are checked deterministically (required facts present, wrong ones absent)
rather than by an LLM judge: the same answer always gets the same score, a run
costs no extra LLM calls, and a failure points at the exact missing fact.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

# "92,121.43" -> "92121.43", "INR 25,000" -> "inr 25000"; a comma *between digits* only.
THOUSANDS = re.compile(r"(?<=[0-9])[,'](?=[0-9]{3}(?![0-9]))")
CURRENCY = re.compile(
    "[" + "".join(map(chr, (0x20AC, 0x24, 0xA3, 0x20B9))) + "]"
)  # euro $ pound rupee
# Hyphen, en/em dashes, minus sign -> "-".
DASHES = str.maketrans(dict.fromkeys([*range(0x2010, 0x2016), 0x2212], "-"))
MARKDOWN = re.compile(r"[*_`]")


class EvalCase(BaseModel):
    """One question with its expectations (see evals/cases.json)."""

    id: str
    category: str
    question: str
    setup: list[str] = Field(
        default_factory=list, description="Messages sent first, in a separate conversation."
    )
    expected_tools: list[str] = Field(default_factory=list)
    expected_sources: list[str] = Field(
        default_factory=list, description="Substrings of a search_documents citation."
    )
    must_include: list[list[str]] = Field(
        default_factory=list, description="All groups must match; any alternative in a group."
    )
    must_not_include: list[str] = Field(default_factory=list)
    requires_tool: str | None = Field(
        default=None, description="Skip the case when the server does not offer this tool."
    )


class EvalSuite(BaseModel):
    """The cases file."""

    dataset_version: str
    notes: str = ""
    cases: list[EvalCase]


@dataclass
class TurnTrace:
    """What the agent did for one question, collected from the SSE stream."""

    answer: str = ""
    tools_called: list[str] = field(default_factory=list)
    citations: list[list[str]] = field(default_factory=list)
    """One list per search_documents call: the citations it returned, best first."""
    tool_errors: int = 0
    error: str | None = None
    latency_s: float = 0.0
    iterations: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    conversation_id: str | None = None


@dataclass
class CaseResult:
    """The scores for one case."""

    id: str
    category: str
    question: str
    answer: str
    correct: bool
    missing: list[str]
    forbidden_found: list[str]
    routing_ok: bool
    tools_called: list[str]
    retrieval_hit: bool | None
    """None when the case has no expected source."""
    retrieval_rank: int | None
    """1-based rank of the first expected source in the best search call."""
    latency_s: float
    iterations: int
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None
    skipped: str | None = None

    @property
    def passed(self) -> bool:
        """Correct answer, expected tools, and (where checked) the right passage retrieved."""
        return (
            self.skipped is None
            and self.error is None
            and self.correct
            and self.routing_ok
            and self.retrieval_hit is not False
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form, including `passed`."""
        return {**self.__dict__, "passed": self.passed}


def normalize(text: str) -> str:
    """Lowercase, unify dashes, and drop Markdown emphasis, currency signs, and digit grouping."""
    text = " ".join(text.lower().translate(DASHES).split())  # also unifies Unicode spaces
    text = MARKDOWN.sub("", text)
    text = CURRENCY.sub("", text)
    text = THOUSANDS.sub("", text)
    return " ".join(text.split())


def _contains(haystack: str, needle: str) -> bool:
    """Substring match; a numeric needle must not be part of a longer number."""
    needle = normalize(needle)
    if re.fullmatch(r"[\d.]+", needle):
        return re.search(rf"(?<![\d.]){re.escape(needle)}(?![\d]|\.\d)", haystack) is not None
    return needle in haystack


def check_answer(
    answer: str, must_include: list[list[str]], must_not_include: list[str]
) -> tuple[list[str], list[str]]:
    """(missing groups, forbidden phrases found). Both empty means correct."""
    text = normalize(answer)
    missing = [
        " | ".join(group)
        for group in must_include
        if not any(_contains(text, option) for option in group)
    ]
    forbidden = [phrase for phrase in must_not_include if _contains(text, phrase)]
    return missing, forbidden


def retrieval_rank(citations: list[list[str]], expected_sources: list[str]) -> int | None:
    """Best 1-based rank of an expected source across all search calls, or None if never found."""
    best: int | None = None
    wanted = [s.lower() for s in expected_sources]
    for call in citations:
        for rank, citation in enumerate(call, 1):
            if any(w in citation.lower() for w in wanted):
                best = rank if best is None else min(best, rank)
                break
    return best


def score_case(case: EvalCase, trace: TurnTrace) -> CaseResult:
    """Score one traced turn against its case."""
    missing, forbidden = check_answer(trace.answer, case.must_include, case.must_not_include)
    rank = retrieval_rank(trace.citations, case.expected_sources) if case.expected_sources else None
    return CaseResult(
        id=case.id,
        category=case.category,
        question=case.question,
        answer=trace.answer,
        correct=not missing and not forbidden and trace.error is None,
        missing=missing,
        forbidden_found=forbidden,
        routing_ok=set(case.expected_tools) <= set(trace.tools_called),
        tools_called=trace.tools_called,
        retrieval_hit=(rank is not None) if case.expected_sources else None,
        retrieval_rank=rank,
        latency_s=round(trace.latency_s, 2),
        iterations=trace.iterations,
        prompt_tokens=trace.prompt_tokens,
        completion_tokens=trace.completion_tokens,
        error=trace.error,
    )


def skipped_case(case: EvalCase, reason: str) -> CaseResult:
    """A case that was not run (e.g. its tool is not enabled on this server)."""
    return CaseResult(
        id=case.id,
        category=case.category,
        question=case.question,
        answer="",
        correct=False,
        missing=[],
        forbidden_found=[],
        routing_ok=False,
        tools_called=[],
        retrieval_hit=None,
        retrieval_rank=None,
        latency_s=0.0,
        iterations=0,
        prompt_tokens=0,
        completion_tokens=0,
        skipped=reason,
    )


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    """Headline metrics over the cases that ran."""
    ran = [r for r in results if r.skipped is None]
    retrieval = [r for r in ran if r.retrieval_hit is not None]
    ranks = [r.retrieval_rank for r in retrieval]
    categories: dict[str, list[CaseResult]] = {}
    for r in ran:
        categories.setdefault(r.category, []).append(r)
    latencies = sorted(r.latency_s for r in ran)
    return {
        "cases_run": len(ran),
        "cases_skipped": len(results) - len(ran),
        "pass_rate": _rate([r.passed for r in ran]),
        "answer_correctness": _rate([r.correct for r in ran]),
        "tool_routing_accuracy": _rate([r.routing_ok for r in ran]),
        "retrieval_hit_rate": _rate([r.retrieval_hit is True for r in retrieval]),
        "retrieval_mrr": (round(sum(1 / k for k in ranks if k) / len(ranks), 3) if ranks else None),
        "by_category": {
            name: {"passed": sum(r.passed for r in rs), "total": len(rs)}
            for name, rs in sorted(categories.items())
        },
        "latency_s": {
            "median": latencies[len(latencies) // 2] if latencies else None,
            "max": latencies[-1] if latencies else None,
        },
        "tokens": {
            "prompt": sum(r.prompt_tokens for r in ran),
            "completion": sum(r.completion_tokens for r in ran),
        },
    }
