"""Search on the knowledge base index: vector similarity, filtered by product and by a relevance threshold.

The read side of shared/kb_index.py, used by the search tool of the agent,
by the test search of the ingestion CLI and by the retrieval evaluation.
"""

import dataclasses
import pathlib
import threading

import numpy as np

from shared import embeddings, kb_index, logger

_LOGGER = logger.get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class SearchHit:
    """A chunk found by the search, with its cosine similarity to the query."""

    chunk: kb_index.KbChunk
    score: float


class Retriever:
    """Searches the chunks of one product, loading the index at the first search.

    The index is loaded lazily, so the agent starts even before the ingestion:
    the search then fails with IndexNotFoundError, which the tool turns into
    an error result for the model. Before loading, the index metadata are
    checked against the embedder of the queries. Once loaded, the index stays
    in memory: restart the process to use an index rebuilt in the meantime.
    """

    def __init__(
        self,
        *,
        index_dir: pathlib.Path,
        embedder: embeddings.Embedder,
        top_k: int,
        min_score: float,
    ):
        """Initializes the retriever.

        Args:
            index_dir: Folder of the index built by the ingestion.
            embedder: Embeds the queries; must match the model of the index.
            top_k: Most chunks returned by a search.
            min_score: Chunks under this cosine similarity are not relevant.
        """
        self._index_dir = index_dir
        self._embedder = embedder
        self._top_k = top_k
        self._min_score = min_score
        self._index: kb_index.KbIndex | None = None
        self._lock = threading.Lock()

    def search(self, query: str, *, scope_id: str) -> list[SearchHit]:
        """Returns the most relevant chunks of a product, best first.

        Raises:
            IndexNotFoundError: If the index has not been built.
            IndexMismatchError: If the index was built with another embedding model.
            InvalidIndexError: If the index files are not consistent.
            EmbeddingError: If the query cannot be embedded.
        """
        index = self._get_index()
        scope_rows = np.array(
            [
                row
                for row, chunk in enumerate(index.chunks)
                if chunk.scope_id == scope_id
            ],
            dtype=int,
        )
        if scope_rows.size == 0:
            _LOGGER.warning("No chunks for scope %s in the index", scope_id)
            return []

        # Vectors are L2-normalized: the dot product is the cosine similarity
        scores = index.vectors[scope_rows] @ self._embedder.embed_query(query)
        best_positions = np.argsort(-scores)[: self._top_k]
        hits = [
            SearchHit(
                chunk=index.chunks[scope_rows[position]],
                score=round(float(scores[position]), 4),
            )
            for position in best_positions
            if scores[position] >= self._min_score
        ]
        _LOGGER.info(
            "Search on scope %s: %s chunks over the threshold", scope_id, len(hits)
        )
        return hits

    def _get_index(self) -> kb_index.KbIndex:
        with self._lock:
            if self._index is None:
                kb_index.check_compatibility(self._index_dir, embedder=self._embedder)
                self._index = kb_index.KbIndex.load(self._index_dir)
                _LOGGER.info(
                    "Loaded knowledge base index with %s chunks",
                    len(self._index.chunks),
                )
            return self._index
