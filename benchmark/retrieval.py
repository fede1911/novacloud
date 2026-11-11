"""Evaluation of the retrieval alone: does the search find the expected section, and in which position?

Expected sections are written as "<doc_id>#<section>", not as chunk ids, so the
dataset survives changes of the chunking. Cases without expected sections are
negatives: the search should return nothing over the threshold.
"""

import dataclasses
import pathlib
import statistics
from typing import TYPE_CHECKING

import pydantic

if TYPE_CHECKING:
    from shared.kb_index import KbChunk
    from shared.kb_search import Retriever, SearchHit


class RetrievalCase(pydantic.BaseModel):
    """One query with the sections that answer it (none for a negative case)."""

    case_id: str = pydantic.Field(min_length=1)
    scope_id: str
    query: str = pydantic.Field(min_length=1)
    relevant: list[str]


class RetrievalDataset(pydantic.BaseModel):
    """A named list of retrieval cases."""

    dataset_name: str
    cases: list[RetrievalCase] = pydantic.Field(min_length=1)

    @classmethod
    def from_file(cls, path: pathlib.Path) -> "RetrievalDataset":
        """Loads and validates a retrieval dataset.

        Raises:
            pydantic.ValidationError: If the file does not match the schema.
        """
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


@dataclasses.dataclass(frozen=True)
class RankedHit:
    """A search result reduced to what the evaluation needs."""

    reference: str
    score: float


@dataclasses.dataclass(frozen=True)
class ThresholdScore:
    """Retrieval quality with one relevance threshold.

    Attributes:
        min_score: The threshold.
        recall_at_1: Share of positive cases with an expected section first.
        recall_at_k: Share of positive cases with an expected section among the results.
        mrr: Mean of 1 / position of the first expected section (0 when missing).
        negative_rejection: Share of negative cases with no result over the threshold.
        Every score is None when the dataset has no case of its kind.
    """

    min_score: float
    recall_at_1: float | None
    recall_at_k: float | None
    mrr: float | None
    negative_rejection: float | None


def reference(chunk: "KbChunk") -> str:
    """Returns the stable reference of a chunk, "<doc_id>#<section>"."""
    return f"{chunk.doc_id}#{chunk.section}"


def rank_of_first_relevant(hits: list[RankedHit], relevant: list[str]) -> int | None:
    """Returns the 1-based position of the first expected section, None if missing."""
    for position, hit in enumerate(hits, start=1):
        if hit.reference in relevant:
            return position
    return None


def score_threshold(
    results: list[tuple[RetrievalCase, list[RankedHit]]],
    *,
    min_score: float,
    top_k: int,
) -> ThresholdScore:
    """Scores the results of every case as if the search had used min_score and top_k.

    Args:
        results: Every case with its hits, best first, searched with no threshold.
        min_score: Threshold to apply.
        top_k: Most results kept per search.
    """
    ranks = []
    rejections = []
    for case, hits in results:
        kept = [hit for hit in hits if hit.score >= min_score][:top_k]
        if case.relevant:
            ranks.append(rank_of_first_relevant(kept, case.relevant))
        else:
            rejections.append(1.0 if not kept else 0.0)
    return ThresholdScore(
        min_score=min_score,
        recall_at_1=_mean([1.0 if rank == 1 else 0.0 for rank in ranks]),
        recall_at_k=_mean([0.0 if rank is None else 1.0 for rank in ranks]),
        mrr=_mean([1 / rank if rank else 0.0 for rank in ranks]),
        negative_rejection=_mean(rejections),
    )


def search_all(
    dataset: RetrievalDataset, *, retriever: "Retriever"
) -> list[tuple[RetrievalCase, list[RankedHit]]]:
    """Runs every query once; the retriever must use no threshold, so every threshold can be scored after."""
    return [
        (case, _ranked(retriever.search(case.query, scope_id=case.scope_id)))
        for case in dataset.cases
    ]


def _ranked(hits: list["SearchHit"]) -> list[RankedHit]:
    return [RankedHit(reference=reference(hit.chunk), score=hit.score) for hit in hits]


def _mean(values: list[float]) -> float | None:
    """Rounded mean, None for no values."""
    return round(statistics.fmean(values), 3) if values else None
