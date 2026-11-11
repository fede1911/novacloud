"""Tests of the documentation search tool."""

from unittest import mock

import pytest

from agent.tools.search_documentation import search_documentation_tool
from shared import embeddings, kb_index, kb_search

pytestmark = pytest.mark.unit


def _hit(chunk_id: str, *, section: str, score: float) -> kb_search.SearchHit:
    chunk = kb_index.KbChunk(
        chunk_id=chunk_id,
        doc_id="novadrive-spazio",
        scope_id="novadrive",
        title="Spazio e cestino",
        section=section,
        source="https://help.example/spazio",
        chunk_num=0,
        text=f"Spazio e cestino > {section}\nTesto.",
    )
    return kb_search.SearchHit(chunk=chunk, score=score)


def _search_tool(catalog, chat_ctx, retriever: mock.Mock):
    services = mock.Mock(retriever=retriever)
    return search_documentation_tool.build(
        scope=catalog.get("novadrive"), chat_ctx=chat_ctx, services=services
    ).tool


class TestSearchDocumentationTool:
    """The tool searches the documentation of its product and never makes the agent fail."""

    def test_hits_become_chunks_for_the_model_and_separate_sources(
        self, catalog, chat_ctx, tool_context
    ):
        """Test: the retriever finds one chunk.
        Expected: id and text in retrieved_chunks; title, section, URL and score in sources.
        """
        # Arrange
        retriever = mock.Mock(
            search=mock.Mock(return_value=[_hit("c1", section="Cestino", score=0.52)])
        )

        # Act
        result = _search_tool(catalog, chat_ctx, retriever)(
            tool_context=tool_context, question="recuperare file eliminati"
        )

        # Assert
        retriever.search.assert_called_once_with(
            "recuperare file eliminati", scope_id="novadrive"
        )
        assert result["content"][0]["json"] == {
            "retrieved_chunks": [
                {"id": "c1", "text": "Spazio e cestino > Cestino\nTesto."}
            ],
            "sources": [
                {
                    "chunk_id": "c1",
                    "title": "Spazio e cestino",
                    "section": "Cestino",
                    "source": "https://help.example/spazio",
                    "score": 0.52,
                }
            ],
        }

    @pytest.mark.parametrize(
        "failure",
        [
            kb_index.IndexNotFoundError("no index"),
            kb_index.IndexMismatchError("other model"),
            embeddings.EmbeddingError("throttled"),
        ],
    )
    def test_search_failure_becomes_error_result(
        self, catalog, chat_ctx, tool_context, failure
    ):
        """Test: the index is missing or incompatible, or the query cannot be embedded.
        Expected: an error result the model can explain, no exception for the agent.
        """
        # Arrange
        retriever = mock.Mock(search=mock.Mock(side_effect=failure))

        # Act
        result = _search_tool(catalog, chat_ctx, retriever)(
            tool_context=tool_context, question="recuperare file eliminati"
        )

        # Assert
        assert result["status"] == "error"
        assert "temporarily unavailable" in result["content"][0]["text"]

    def test_empty_question_is_rejected_without_searching(
        self, catalog, chat_ctx, tool_context
    ):
        """Test: blank question.
        Expected: error result, the retriever is not called.
        """
        # Arrange
        retriever = mock.Mock()

        # Act
        result = _search_tool(catalog, chat_ctx, retriever)(
            tool_context=tool_context, question="   "
        )

        # Assert
        assert result["status"] == "error"
        retriever.search.assert_not_called()
