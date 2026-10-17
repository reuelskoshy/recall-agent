"""Unit tests for short-term history helpers and title generation."""

from app.agent.runner import TITLE_MAX_CHARS, make_title
from app.llm.base import ChatMessage, ToolCall
from app.memory.short_term import sanitize_tool_pairs

CALL = ToolCall("c1", "calculator", '{"expression": "2+2"}')


def test_drops_leading_tool_results_cut_off_from_their_call() -> None:
    history = [
        ChatMessage(role="tool", content="4", tool_call_id="c1"),
        ChatMessage(role="assistant", content="It's 4."),
        ChatMessage(role="user", content="thanks"),
    ]

    assert sanitize_tool_pairs(history) == history[1:]


def test_keeps_complete_tool_exchange() -> None:
    history = [
        ChatMessage(role="user", content="2+2?"),
        ChatMessage(role="assistant", tool_calls=(CALL,)),
        ChatMessage(role="tool", content="4", tool_call_id="c1"),
        ChatMessage(role="assistant", content="4"),
    ]

    assert sanitize_tool_pairs(history) == history


def test_drops_call_missing_a_result_but_keeps_its_text() -> None:
    second = ToolCall("c2", "get_current_datetime", "{}")
    history = [
        ChatMessage(role="user", content="q"),
        ChatMessage(role="assistant", content="Let me check.", tool_calls=(CALL, second)),
        ChatMessage(role="tool", content="4", tool_call_id="c1"),  # c2's result is missing
        ChatMessage(role="user", content="hello?"),
    ]

    assert sanitize_tool_pairs(history) == [
        ChatMessage(role="user", content="q"),
        ChatMessage(role="assistant", content="Let me check."),
        ChatMessage(role="user", content="hello?"),
    ]


def test_drops_call_with_no_results_and_no_text() -> None:
    history = [
        ChatMessage(role="user", content="q"),
        ChatMessage(role="assistant", tool_calls=(CALL,)),
    ]

    assert sanitize_tool_pairs(history) == history[:1]


def test_make_title_collapses_whitespace_and_truncates() -> None:
    assert make_title("  hello \n  world ") == "hello world"
    long_title = make_title("word " * 50)
    assert len(long_title) <= TITLE_MAX_CHARS
    assert long_title.endswith("…")
