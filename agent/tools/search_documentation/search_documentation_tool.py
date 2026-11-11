"""search-documentation-<scope>: semantic search in the documentation of one product."""

from typing import TYPE_CHECKING, Any

import strands

from agent.tools import tool_result, tool_spec
from shared import embeddings, kb_index, logger

if TYPE_CHECKING:
    from agent.models.chat import ChatContext
    from agent.toolkits import ToolServices
    from shared.scope_catalog import Scope

_LOGGER = logger.get_logger(__name__)

TOOL_NAME_PREFIX = "search-documentation-"
PROMPT_NAME = "tool_search_documentation"


def build(
    *, scope: "Scope", chat_ctx: "ChatContext", services: "ToolServices"
) -> tool_spec.ToolSpec:
    """Builds the search tool of one product.

    The model gets the id and the text of the chunks; the sources are kept
    apart so that the answer can cite them.

    Args:
        scope: Product whose documentation is searched.
        chat_ctx: Context of the chat.
        services: Services of the tools; uses the retriever.
    """

    @strands.tool(
        context=True,
        name=f"{TOOL_NAME_PREFIX}{scope.scope_id}",
        description=(
            f"Searches the {scope.label} documentation, FAQs and support guides to find the cause "
            f"or the resolution of a user's {scope.label} issue."
        ),
    )
    def search_documentation(
        tool_context: strands.ToolContext, question: str
    ) -> dict[str, Any]:
        """Search the documentation of the product.

        Write the question in a clear and concise way, with the key terms of the issue.

        Args:
            tool_context: The tool context provided by Strands.
            question: The search question, based on the conversation with the user.
        """
        tool_use_id = tool_context.tool_use["toolUseId"]
        if not question.strip():
            return tool_result.error(
                "The question cannot be empty.", tool_use_id=tool_use_id
            )

        try:
            hits = services.retriever.search(question, scope_id=scope.scope_id)
        except (
            kb_index.IndexNotFoundError,
            kb_index.InvalidIndexError,
            kb_index.IndexMismatchError,
            embeddings.EmbeddingError,
        ):
            _LOGGER.exception(
                "Knowledge base search failed (chat %s)", chat_ctx.chat_id
            )
            return tool_result.error(
                "The documentation is temporarily unavailable, please try again later.",
                tool_use_id=tool_use_id,
            )

        payload = {
            "retrieved_chunks": [
                {"id": hit.chunk.chunk_id, "text": hit.chunk.text} for hit in hits
            ],
            "sources": [
                {
                    "chunk_id": hit.chunk.chunk_id,
                    "title": hit.chunk.title,
                    "section": hit.chunk.section,
                    "source": hit.chunk.source,
                    "score": hit.score,
                }
                for hit in hits
            ],
        }
        _LOGGER.info(
            "Search on %s returned %s chunks (chat %s)",
            scope.scope_id,
            len(hits),
            chat_ctx.chat_id,
        )
        return tool_result.success({"json": payload}, tool_use_id=tool_use_id)

    return tool_spec.ToolSpec(tool=search_documentation, prompt_name=PROMPT_NAME)
