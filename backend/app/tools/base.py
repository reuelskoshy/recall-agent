"""The Tool interface.

A tool is: a name, a description (the model reads it to decide when to call the
tool), a Pydantic input model (converted to the JSON Schema the model fills in),
and an async `run()` method.
"""

import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.llm.base import ToolSpec
from app.rag.embeddings import EmbeddingProvider

InputT = TypeVar("InputT", bound=BaseModel)

# OpenAI and Anthropic both require tool names to match this.
TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class ToolError(Exception):
    """Raise from `run()` for expected failures; the message is shown to the model as-is."""


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What a tool returns.

    `content` is the text the model reads. `data` is optional structured output
    for the UI (e.g. citations), never sent to the model.
    """

    content: str
    data: dict[str, Any] | None = None
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Per-turn context handed to every tool.

    Tools get a session *factory*, not a session: tools in one turn run
    concurrently, and an AsyncSession must not be shared between tasks.
    Shared services (embedder, settings) are injected here rather than imported
    as globals, so tests and other deployments can swap them.
    """

    conversation_id: uuid.UUID
    session_factory: async_sessionmaker[AsyncSession]
    embedder: EmbeddingProvider
    settings: Settings


class Tool(ABC, Generic[InputT]):
    """Base class for tools. Subclass it, set the class attributes, implement `run()`."""

    name: ClassVar[str]
    description: ClassVar[str]
    input_model: type[InputT]
    timeout_s: ClassVar[float | None] = None
    """Overrides the registry's default timeout for this tool."""

    @abstractmethod
    async def run(self, args: InputT, ctx: ToolContext) -> ToolResult:
        """Execute the tool with validated arguments."""

    def spec(self) -> ToolSpec:
        """The provider-neutral description sent to the model."""
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=json_schema_for(self.input_model),
        )


def json_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON Schema with the noise removed.

    Pydantic adds a `title` to every field; models do not need them and they
    cost prompt tokens on every call, so they are stripped.
    """
    schema: dict[str, Any] = _strip_titles(model.model_json_schema())
    return schema


def _strip_titles(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            key: _strip_titles(value)
            for key, value in node.items()
            # Only drop schema titles (strings); a *property* named "title" is a dict.
            if not (key == "title" and isinstance(value, str))
        }
    if isinstance(node, list):
        return [_strip_titles(item) for item in node]
    return node
