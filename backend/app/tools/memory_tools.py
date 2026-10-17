"""save_memory and recall_memories: the agent's explicit long-term memory tools.

Most memories are also captured automatically after each turn (see
app/memory/extraction.py), and the most relevant ones are injected into the
system prompt. These tools cover the explicit cases: "remember that ...", and
looking further back than the injected top few.
"""

from pydantic import BaseModel, Field

from app.memory.long_term import (
    MemoryKind,
    MemoryRejectedError,
    RankWeights,
    edit_memory,
    find_similar,
    recall_memories,
    save_memory,
)
from app.tools.base import Tool, ToolContext, ToolError, ToolResult
from app.tools.registry import register_tool

REPLACE_MATCH_SIMILARITY = 0.8
"""How close `replaces` must be to a stored memory's text to identify it."""
CONFLICT_SIMILARITY = 0.75
"""Same-topic statements are surfaced after a save so the model can fix outdated
ones. Measured with bge-small on "The user ..." facts: same topic 0.80-0.82
('lives in Berlin' vs 'lives in Lisbon'), different topics 0.59-0.74."""
SUBJECT = "the user"
"""Every memory starts with this subject. Mixed subjects ('Ada is vegetarian' vs
'The user is vegetarian') cost ~0.2 similarity and break dedup and conflict checks."""


class SaveMemoryInput(BaseModel):
    """Arguments for save_memory."""

    content: str = Field(
        description="One atomic fact, starting with 'The user', e.g. 'The user prefers "
        "answers in bullet points.' Save name, job, and location as separate memories.",
        min_length=3,
        max_length=500,
    )
    kind: MemoryKind = Field(
        default="fact",
        description="fact (about the user), preference, goal, or instruction "
        "(a standing rule for how to answer).",
    )
    importance: int = Field(
        default=3,
        ge=1,
        le=5,
        description="5 = identity or standing instruction; 4 = matters in many conversations "
        "(diet, allergies, role); 3 = typical preference or fact; 1 = minor. "
        "Importance 4-5 memories are shown in every future conversation.",
    )
    replaces: str | None = Field(
        default=None,
        max_length=500,
        description="If this corrects or supersedes a memory you have seen (e.g. the user "
        "moved city), that memory's text. It is then updated instead of duplicated.",
    )


@register_tool
class SaveMemoryTool(Tool[SaveMemoryInput]):
    """Persist a fact about the user across conversations."""

    name = "save_memory"
    description = (
        "Save a durable fact about the user to long-term memory, so it is remembered in "
        "future conversations. Use it when the user asks you to remember something, or "
        "shares a lasting preference, personal fact, goal, or instruction. Do not save "
        "secrets (passwords, keys), one-off requests, or general knowledge."
    )
    input_model = SaveMemoryInput

    async def run(self, args: SaveMemoryInput, ctx: ToolContext) -> ToolResult:
        """Save, replace, or merge into a near-duplicate; then flag possible conflicts."""
        if not args.content.strip().lower().startswith(SUBJECT):
            raise ToolError(
                "Write the memory as one fact starting with 'The user', e.g. "
                "'The user is vegetarian.' (not the user's name), then call save_memory again."
            )
        async with ctx.session_factory() as session:
            try:
                action: str
                previous: str | None
                target = None
                if args.replaces:
                    matches = await find_similar(
                        session, ctx.embedder, args.replaces, REPLACE_MATCH_SIMILARITY, limit=1
                    )
                    target = matches[0][0] if matches else None
                existing = (
                    await find_similar(
                        session,
                        ctx.embedder,
                        args.content,
                        ctx.settings.memory_dedup_similarity,
                        limit=1,
                        exclude=[target.id],
                    )
                    if target is not None
                    else []
                )
                if target is not None and existing:
                    # The corrected fact is already stored: just drop the outdated one.
                    action, previous, memory = "replaced", target.content, existing[0][0]
                    await session.delete(target)
                    await session.commit()
                elif target is not None:
                    action, previous = "replaced", target.content
                    memory = await edit_memory(
                        session, ctx.embedder, target, args.content, args.kind, args.importance
                    )
                else:
                    outcome = await save_memory(
                        session,
                        ctx.embedder,
                        args.content,
                        args.kind,
                        args.importance,
                        source="tool",
                        dedup_similarity=ctx.settings.memory_dedup_similarity,
                        conversation_id=ctx.conversation_id,
                    )
                    action, previous, memory = (
                        outcome.action,
                        outcome.previous_content,
                        outcome.memory,
                    )
            except MemoryRejectedError as exc:
                raise ToolError(str(exc)) from exc
            related = await find_similar(
                session, ctx.embedder, memory.content, CONFLICT_SIMILARITY, exclude=[memory.id]
            )

        if action == "replaced":
            content = f"Replaced the outdated memory {previous!r}."
        elif action == "updated":
            content = f"Updated an existing memory (was: {previous!r})."
        else:
            content = "Saved to long-term memory."
        if args.replaces and target is None:
            content += " (No stored memory matched `replaces`, so it was saved as new.)"
        if related:
            listed = "\n".join(f"- {m.content}" for m, _ in related)
            content += (
                "\nRelated memories on the same topic:\n"
                f"{listed}\nIf one of these is now outdated, call save_memory again with the "
                "corrected fact and `replaces` set to that memory's text. Otherwise ignore this."
            )
        return ToolResult(
            content=content,
            data={
                "action": action,
                "id": str(memory.id),
                "content": memory.content,
                "kind": memory.kind,
                "importance_level": memory.importance,
                "previous_content": previous,
                "related": [m.content for m, _ in related],
            },
        )


class RecallMemoriesInput(BaseModel):
    """Arguments for recall_memories."""

    query: str = Field(
        description="What to look for in memory, e.g. 'dietary preferences'.",
        min_length=2,
        max_length=300,
    )
    limit: int = Field(default=5, ge=1, le=10, description="Maximum memories to return.")


@register_tool
class RecallMemoriesTool(Tool[RecallMemoriesInput]):
    """Search long-term memory about the user."""

    name = "recall_memories"
    description = (
        "Search long-term memory for facts the user shared in earlier conversations "
        "(preferences, personal details, goals, instructions). The most relevant memories "
        "are already in your instructions; use this to look for something more specific."
    )
    input_model = RecallMemoriesInput

    async def run(self, args: RecallMemoriesInput, ctx: ToolContext) -> ToolResult:
        """Return ranked memories with their score breakdown."""
        settings = ctx.settings
        async with ctx.session_factory() as session:
            found = await recall_memories(
                session,
                ctx.embedder,
                args.query,
                RankWeights.from_settings(settings),
                limit=args.limit,
                min_similarity=settings.memory_min_similarity,
            )
            await session.commit()  # persist the access bump
        if not found:
            return ToolResult(
                content="No relevant memories found.",
                data={"query": args.query, "memories": []},
            )
        lines = [
            f"- ({s.memory.kind}, saved {s.memory.created_at:%Y-%m-%d}) {s.memory.content}"
            for s in found
        ]
        return ToolResult(
            content="Relevant memories about the user:\n" + "\n".join(lines),
            data={"query": args.query, "memories": [s.to_dict() for s in found]},
        )
