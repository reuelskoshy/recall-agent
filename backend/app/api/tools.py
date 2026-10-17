"""GET /api/tools: what the agent can do (also shown in the UI)."""

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.tools.registry import ToolRegistry, get_tool_registry

router = APIRouter(tags=["tools"])


class ToolOut(BaseModel):
    """A registered tool, exactly as described to the model."""

    name: str
    description: str
    parameters: dict[str, Any]


@router.get("/tools", response_model=list[ToolOut])
async def list_tools(
    registry: Annotated[ToolRegistry, Depends(get_tool_registry)],
) -> list[ToolOut]:
    """List every registered tool with its JSON Schema."""
    return [
        ToolOut(name=s.name, description=s.description, parameters=s.parameters)
        for s in registry.specs()
    ]
