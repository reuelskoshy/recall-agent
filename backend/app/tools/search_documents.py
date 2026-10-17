"""search_documents: RAG over the user's uploaded files, with citations."""

from pydantic import BaseModel, Field

from app.rag.retrieval import RetrievedChunk, search_chunks
from app.tools.base import Tool, ToolContext, ToolResult
from app.tools.registry import register_tool

NO_RESULTS = (
    "No relevant passages were found in the uploaded documents. Tell the user the "
    "documents do not appear to cover this, rather than answering from memory."
)
SNIPPET_CHARS = 240


class SearchDocumentsInput(BaseModel):
    """Arguments for search_documents."""

    query: str = Field(
        description="What to look for, phrased as a specific question or keywords. "
        "Rephrase the user's question for search if needed.",
        min_length=2,
        max_length=500,
    )
    top_k: int | None = Field(
        default=None, ge=1, le=10, description="How many passages to return (default 5)."
    )


def format_for_model(results: list[RetrievedChunk]) -> str:
    """Numbered passages with source labels the model can cite verbatim."""
    blocks = [
        f"[{i}] Source: {r.citation} (relevance {r.similarity:.2f})\n{r.content}"
        for i, r in enumerate(results, 1)
    ]
    return (
        "Answer using only these passages. Cite each fact with its source in square "
        "brackets, e.g. [" + results[0].citation + "].\n\n" + "\n\n---\n\n".join(blocks)
    )


@register_tool
class SearchDocumentsTool(Tool[SearchDocumentsInput]):
    """Hybrid (vector + keyword) search over uploaded documents."""

    name = "search_documents"
    description = (
        "Search the user's uploaded documents (PDF, TXT, Markdown) and return the most "
        "relevant passages with their source file and page. Use this for any question "
        "that the uploaded documents might answer."
    )
    input_model = SearchDocumentsInput

    async def run(self, args: SearchDocumentsInput, ctx: ToolContext) -> ToolResult:
        """Search and return passages (or an explicit 'nothing relevant' signal)."""
        settings = ctx.settings
        async with ctx.session_factory() as session:
            results = await search_chunks(
                session,
                ctx.embedder,
                args.query,
                top_k=args.top_k or settings.rag_top_k,
                min_similarity=settings.rag_min_similarity,
                candidates=settings.rag_candidates,
            )
        if not results:
            return ToolResult(content=NO_RESULTS, data={"query": args.query, "results": []})
        return ToolResult(
            content=format_for_model(results),
            data={
                "query": args.query,
                "results": [
                    {
                        "n": i,
                        "document_id": str(r.document_id),
                        "filename": r.filename,
                        "page": r.page,
                        "section": r.section,
                        "citation": r.citation,
                        "chunk_index": r.chunk_index,
                        "similarity": round(r.similarity, 3),
                        "snippet": r.content[:SNIPPET_CHARS],
                    }
                    for i, r in enumerate(results, 1)
                ],
            },
        )
