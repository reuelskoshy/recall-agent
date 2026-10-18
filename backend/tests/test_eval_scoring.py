"""The eval's scoring rules (evals/scoring.py) and the cases file."""

from pathlib import Path
from typing import Any

import pytest

from evals.run_eval import _apply_event
from evals.scoring import (
    EvalCase,
    EvalSuite,
    TurnTrace,
    check_answer,
    normalize,
    retrieval_rank,
    score_case,
    skipped_case,
    summarize,
)

CASES_FILE = Path(__file__).resolve().parents[1] / "evals" / "cases.json"


def test_normalize_drops_grouping_currency_and_markdown() -> None:
    assert normalize("**€92,121.43** and INR 25,000 — ok") == "92121.43 and inr 25000 - ok"
    # A comma that is not digit grouping stays.
    assert normalize("Berlin, Lisbon") == "berlin, lisbon"


@pytest.mark.parametrize(
    ("answer", "needle", "found"),
    [
        ("The average is €92,121.43.", "92121.43", True),
        ("You can carry over 10 days.", "10", True),
        ("You can carry over 100 days.", "10", False),  # not part of a longer number
        ("Version 2.10 shipped.", "2.1", False),
        ("It was 26.", "26", True),
        ("It was 26.5.", "26", False),
        ("Report it within one hour.", "one hour", True),
    ],
)
def test_numbers_match_whole_numbers_only(answer: str, needle: str, found: bool) -> None:
    missing, _ = check_answer(answer, [[needle]], [])
    assert (not missing) is found


def test_groups_are_and_alternatives_are_or() -> None:
    must = [["10"], ["31 march", "march 31"]]

    assert check_answer("Up to 10 days; the rest lapse on March 31.", must, []) == ([], [])
    missing, _ = check_answer("Up to 10 days.", must, [])
    assert missing == ["31 march | march 31"]


def test_forbidden_phrases_fail_the_answer() -> None:
    _, forbidden = check_answer("It is INR 1,000 in metro cities.", [["1000"]], ["1000 in metro"])
    assert forbidden == ["1000 in metro"]


def test_retrieval_rank_takes_the_best_search_call() -> None:
    calls = [
        ["handbook.md § 3. Remote work", "handbook.md § 2.1 Annual leave"],
        ["handbook.md § 2.1 Annual leave"],
    ]
    assert retrieval_rank(calls, ["2.1 Annual leave"]) == 1
    assert retrieval_rank(calls[:1], ["2.1 annual leave"]) == 2
    assert retrieval_rank(calls, ["8. Notice period"]) is None
    assert retrieval_rank([], ["2.1 Annual leave"]) is None


CASE = EvalCase(
    id="c",
    category="rag",
    question="q",
    expected_tools=["search_documents"],
    expected_sources=["2.1 Annual leave"],
    must_include=[["10"]],
)


def test_a_case_passes_only_when_answer_tools_and_retrieval_are_right() -> None:
    good = TurnTrace(
        answer="10 days",
        tools_called=["search_documents"],
        citations=[["h.md § 2.1 Annual leave"]],
    )
    assert score_case(CASE, good).passed

    wrong_tool = TurnTrace(answer="10 days", tools_called=["query_database"])
    result = score_case(CASE, wrong_tool)
    assert not result.passed and not result.routing_ok and result.retrieval_hit is False

    missed = TurnTrace(
        answer="10 days", tools_called=["search_documents"], citations=[["h.md § 3. Remote work"]]
    )
    assert not score_case(CASE, missed).passed

    errored = TurnTrace(
        answer="10",
        tools_called=["search_documents"],
        citations=[["§ 2.1 Annual leave"]],
        error="boom",
    )
    assert not score_case(CASE, errored).passed


def test_summary_metrics() -> None:
    hit = score_case(
        CASE,
        TurnTrace(
            answer="10",
            tools_called=["search_documents"],
            citations=[["x", "§ 2.1 Annual leave"]],
            latency_s=2.0,
        ),
    )
    miss = score_case(CASE, TurnTrace(answer="nope", tools_called=[], latency_s=4.0))
    skipped = skipped_case(CASE, "web_search is not enabled")

    summary = summarize([hit, miss, skipped])

    assert summary["cases_run"] == 2 and summary["cases_skipped"] == 1
    assert summary["pass_rate"] == 0.5
    assert summary["retrieval_hit_rate"] == 0.5
    assert summary["retrieval_mrr"] == 0.25  # (1/2 + 0) / 2
    assert summary["by_category"] == {"rag": {"passed": 1, "total": 2}}


def test_sse_events_fold_into_a_trace() -> None:
    trace, parts = TurnTrace(), list[str]()
    events: list[dict[str, Any]] = [
        {"type": "conversation", "conversation_id": "c1", "run_id": "r1"},
        {"type": "tool_call", "id": "t", "name": "search_documents", "arguments": {}},
        {
            "type": "tool_result",
            "id": "t",
            "name": "search_documents",
            "content": "...",
            "is_error": False,
            "latency_ms": 5,
            "data": {"results": [{"citation": "h.md § 2.1 Annual leave"}]},
        },
        {"type": "token", "text": "Ten "},
        {"type": "token", "text": "days."},
        {
            "type": "done",
            "iterations": 2,
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        },
    ]
    for event in events:
        _apply_event(trace, event, parts)

    assert trace.conversation_id == "c1"
    assert trace.tools_called == ["search_documents"]
    assert trace.citations == [["h.md § 2.1 Annual leave"]]
    assert "".join(parts) == "Ten days."
    assert (trace.iterations, trace.prompt_tokens) == (2, 100)


def test_cases_file_is_valid_and_ids_are_unique() -> None:
    suite = EvalSuite.model_validate_json(CASES_FILE.read_text(encoding="utf-8"))

    assert len(suite.cases) >= 10
    assert len({c.id for c in suite.cases}) == len(suite.cases)
    for case in suite.cases:
        assert case.must_include, f"{case.id} checks nothing in the answer"
        assert all(group for group in case.must_include)
