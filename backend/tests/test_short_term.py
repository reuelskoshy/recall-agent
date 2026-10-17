"""Unit tests for short-term history helpers and title generation."""

from app.agent.runner import TITLE_MAX_CHARS, make_title
from app.llm.base import ChatMessage, ToolCall
from app.memory.short_term import trim_orphan_tool_results


def test_trim_drops_tool_results_cut_off_from_their_call() -> None:
    history = [
        ChatMessage(role="tool", content="4", tool_call_id="c1"),
        ChatMessage(role="tool", content="5", tool_call_id="c2"),
        ChatMessage(role="assistant", content="It's 4."),
        ChatMessage(role="user", content="thanks"),
    ]

    assert trim_orphan_tool_results(history) == history[2:]


def test_trim_keeps_tool_results_that_follow_their_call() -> None:
    history = [
        ChatMessage(role="user", content="2+2?"),
        ChatMessage(role="assistant", tool_calls=(ToolCall("c1", "calculator", "{}"),)),
        ChatMessage(role="tool", content="4", tool_call_id="c1"),
    ]

    assert trim_orphan_tool_results(history) == history


def test_make_title_collapses_whitespace_and_truncates() -> None:
    assert make_title("  hello \n  world ") == "hello world"
    long_title = make_title("word " * 50)
    assert len(long_title) <= TITLE_MAX_CHARS
    assert long_title.endswith("…")
