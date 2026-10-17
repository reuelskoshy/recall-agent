"""Unit tests for the tool registry, the Tool base class, and the datetime tool."""

import asyncio
import uuid
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.db.session import SessionLocal
from app.llm.base import ToolCall
from app.tools.base import Tool, ToolContext, ToolError, ToolResult, json_schema_for
from app.tools.datetime_tool import CurrentDatetimeTool
from app.tools.registry import ToolRegistry, build_registry

CTX = ToolContext(conversation_id=uuid.uuid4(), session_factory=SessionLocal)


class EchoInput(BaseModel):
    text: str = Field(description="Text to echo", max_length=20)
    times: int = Field(default=1, ge=1, le=3)


class EchoTool(Tool[EchoInput]):
    name = "echo"
    description = "Echo text."
    input_model = EchoInput

    async def run(self, args: EchoInput, ctx: ToolContext) -> ToolResult:
        if args.text == "fail":
            raise ToolError("asked to fail")
        if args.text == "crash":
            raise RuntimeError("secret internal detail")
        if args.text == "slow":
            await asyncio.sleep(1)
        return ToolResult(content=args.text * args.times)


def registry_with_echo(**kwargs: float) -> ToolRegistry:
    registry = ToolRegistry(**kwargs)  # type: ignore[arg-type]
    registry.register(EchoTool())
    return registry


async def run(registry: ToolRegistry, arguments: str, name: str = "echo") -> ToolResult:
    return await registry.execute(ToolCall(id="c1", name=name, arguments=arguments), CTX)


def test_schema_is_generated_from_the_pydantic_model_without_titles() -> None:
    schema = json_schema_for(EchoInput)

    assert schema == {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to echo", "maxLength": 20},
            "times": {"type": "integer", "default": 1, "minimum": 1, "maximum": 3},
        },
        "required": ["text"],
    }


def test_strip_titles_keeps_a_property_named_title() -> None:
    class Doc(BaseModel):
        title: str

    assert "title" in json_schema_for(Doc)["properties"]


def test_register_rejects_duplicates_and_invalid_names() -> None:
    registry = registry_with_echo()
    with pytest.raises(ValueError, match="Duplicate"):
        registry.register(EchoTool())

    class BadName(EchoTool):
        name = "has spaces"

    with pytest.raises(ValueError, match="Invalid tool name"):
        registry.register(BadName())


def test_build_registry_discovers_tools_in_the_package() -> None:
    registry = build_registry(Settings())

    assert {"calculator", "get_current_datetime"} <= set(registry.names)
    assert all(spec.parameters["type"] == "object" for spec in registry.specs())


async def test_execute_success() -> None:
    result = await run(registry_with_echo(), '{"text": "hi", "times": 2}')

    assert result == ToolResult(content="hihi")


@pytest.mark.parametrize(
    ("arguments", "name", "expected"),
    [
        ("{}", "nope", "Error: Unknown tool 'nope'. Available tools: echo."),
        ("{not json", "echo", "Error: Invalid arguments for echo: "),
        ('{"times": 2}', "echo", "Error: Invalid arguments for echo: text: Field required"),
        ('{"text": "x", "times": 9}', "echo", "times: Input should be less than or equal to 3"),
        ('{"text": "fail"}', "echo", "Error: asked to fail"),
    ],
)
async def test_execute_turns_failures_into_error_results(
    arguments: str, name: str, expected: str
) -> None:
    result = await run(registry_with_echo(), arguments, name)

    assert result.is_error
    assert expected in result.content


async def test_crash_is_reported_without_leaking_internal_details() -> None:
    result = await run(registry_with_echo(), '{"text": "crash"}')

    assert result.is_error
    assert result.content == "Error: echo failed unexpectedly (RuntimeError)."


async def test_slow_tool_times_out() -> None:
    result = await run(registry_with_echo(default_timeout_s=0.05), '{"text": "slow"}')

    assert result.is_error
    assert "timed out after 0.05s" in result.content


async def test_long_results_are_truncated() -> None:
    registry = registry_with_echo(max_result_chars=5)

    result = await run(registry, '{"text": "abcdefghij"}')

    assert result.content.startswith("abcde\n[truncated: 5 more characters]")


async def test_datetime_tool_converts_to_requested_timezone() -> None:
    fixed = datetime(2026, 10, 17, 10, 30, tzinfo=UTC)
    tool = CurrentDatetimeTool(clock=lambda: fixed)

    result = await tool.run(tool.input_model(timezone="Asia/Kolkata"), CTX)

    assert result.content == "Saturday, 17 October 2026, 16:00:00 IST (UTC+05:30, Asia/Kolkata)"


async def test_datetime_tool_rejects_unknown_timezone() -> None:
    tool = CurrentDatetimeTool()

    with pytest.raises(ToolError, match="Unknown timezone"):
        await tool.run(tool.input_model(timezone="Mars/Olympus"), CTX)
