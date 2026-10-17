"""get_current_datetime: the model has no clock, so it must ask for the time."""

from collections.abc import Callable
from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolContext, ToolError, ToolResult
from app.tools.registry import register_tool


class DatetimeInput(BaseModel):
    """Arguments for get_current_datetime."""

    timezone: str = Field(
        default="UTC",
        description="IANA timezone name, e.g. 'UTC', 'Asia/Kolkata', 'America/New_York'.",
        max_length=64,
    )


@register_tool
class CurrentDatetimeTool(Tool[DatetimeInput]):
    """Current date and time in a timezone."""

    name = "get_current_datetime"
    description = (
        "Get the current date, time, and weekday in a given timezone. Use this "
        "whenever the answer depends on today's date or the current time."
    )
    input_model = DatetimeInput

    def __init__(self, clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        # Injectable clock keeps tests deterministic.
        self._clock = clock

    async def run(self, args: DatetimeInput, ctx: ToolContext) -> ToolResult:
        """Return the current time converted to the requested timezone."""
        try:
            zone = ZoneInfo(args.timezone)
        except (ZoneInfoNotFoundError, ValueError):
            raise ToolError(
                f"Unknown timezone {args.timezone!r}. Use an IANA name like 'Asia/Kolkata'."
            ) from None
        now = self._clock().astimezone(zone)
        offset = now.strftime("%z")
        human = now.strftime("%A, %d %B %Y, %H:%M:%S")
        return ToolResult(
            content=f"{human} {now.tzname()} (UTC{offset[:3]}:{offset[3:]}, {args.timezone})",
            data={"iso": now.isoformat(), "timezone": args.timezone},
        )
