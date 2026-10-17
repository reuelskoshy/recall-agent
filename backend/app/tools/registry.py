"""Tool registry: discovery, lookup, and safe execution.

Adding a tool = adding one module to `app/tools/` with a `@register_tool` class.
"""

import asyncio
import importlib
import logging
import pkgutil
from functools import lru_cache
from typing import Any

from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.llm.base import ToolCall, ToolSpec
from app.tools.base import TOOL_NAME_PATTERN, Tool, ToolContext, ToolError, ToolResult

logger = logging.getLogger(__name__)

_TOOL_CLASSES: list[type[Tool[Any]]] = []


def register_tool(cls: type[Tool[Any]]) -> type[Tool[Any]]:
    """Class decorator: make a tool available to `build_registry()`."""
    _TOOL_CLASSES.append(cls)
    return cls


def discover_tools(package: str = "app.tools") -> list[type[Tool[Any]]]:
    """Import every module in `package` so their `@register_tool` decorators run."""
    pkg = importlib.import_module(package)
    for module in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"{package}.{module.name}")
    return list(_TOOL_CLASSES)


class ToolRegistry:
    """Holds tool instances and executes calls without ever raising.

    Every failure (unknown tool, bad JSON, invalid arguments, timeout, crash)
    becomes an error `ToolResult` that is sent back to the model, which can then
    retry with corrected arguments or explain the problem to the user.
    """

    def __init__(self, default_timeout_s: float = 15.0, max_result_chars: int = 4000) -> None:
        self._tools: dict[str, Tool[Any]] = {}
        self.default_timeout_s = default_timeout_s
        self.max_result_chars = max_result_chars

    def register(self, tool: Tool[Any]) -> None:
        """Add a tool instance. Names must be unique and provider-compatible."""
        if not TOOL_NAME_PATTERN.match(tool.name):
            raise ValueError(f"Invalid tool name {tool.name!r}")
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool[Any] | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        """Registered tool names, sorted."""
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        """Specs for every tool, in a stable order (stable prompts cache better)."""
        return [self._tools[name].spec() for name in self.names]

    async def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        """Validate arguments, run the tool with a timeout, and normalize the result."""
        tool = self._tools.get(call.name)
        if tool is None:
            return _error(f"Unknown tool {call.name!r}. Available tools: {', '.join(self.names)}.")

        try:
            args = tool.input_model.model_validate_json(call.arguments or "{}")
        except ValidationError as exc:
            return _error(f"Invalid arguments for {call.name}: {_format_validation(exc)}")

        timeout = tool.timeout_s or self.default_timeout_s
        try:
            async with asyncio.timeout(timeout):
                result = await tool.run(args, ctx)
        except TimeoutError:
            return _error(f"{call.name} timed out after {timeout:g}s.")
        except ToolError as exc:
            return _error(str(exc))
        except Exception as exc:
            logger.exception("Tool %s crashed", call.name)
            return _error(f"{call.name} failed unexpectedly ({type(exc).__name__}).")

        return self._truncate(result)

    def _truncate(self, result: ToolResult) -> ToolResult:
        if len(result.content) <= self.max_result_chars:
            return result
        clipped = result.content[: self.max_result_chars]
        note = f"\n[truncated: {len(result.content) - self.max_result_chars} more characters]"
        return ToolResult(content=clipped + note, data=result.data, is_error=result.is_error)


def _error(message: str) -> ToolResult:
    return ToolResult(content=f"Error: {message}", is_error=True)


def _format_validation(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors(include_url=False):
        location = ".".join(str(p) for p in err["loc"]) or "arguments"
        parts.append(f"{location}: {err['msg']}")
    return "; ".join(parts)


def build_registry(settings: Settings) -> ToolRegistry:
    """Discover all tools and register one instance of each."""
    registry = ToolRegistry(
        default_timeout_s=settings.tool_timeout_s,
        max_result_chars=settings.tool_max_result_chars,
    )
    for cls in discover_tools():
        registry.register(cls())
    return registry


@lru_cache
def get_tool_registry() -> ToolRegistry:
    """FastAPI dependency: the process-wide registry."""
    return build_registry(get_settings())
