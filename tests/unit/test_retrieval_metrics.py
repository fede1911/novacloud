"""Tests of the retrieval evaluation metrics."""

import pytest

from benchmark import retrieval

pytestmark = pytest.mark.unit

REFUNDS = "novabill-fatture#Rimborsi"
PLANS = "novabill-piani#Piani disponibili"


def _case(relevant: list[str]) -> retrieval.RetrievalCase:
    return retrieval.RetrievalCase(
        case_id="c", scope_id="novabill", query="q", relevant=relevant
    )


def _hits(*pairs: tuple[str, float]) -> list[retrieval.RankedHit]:
    return [retrieval.RankedHit(reference=ref, score=score) for ref, score in pairs]


class TestScoreThreshold:
    """Recall, MRR and rejection of the negatives, with the threshold applied after the search."""

    def test_expected_section_second_counts_for_recall_at_k_only(self):
        """Test: the expected section is second, both hits over the threshold.
        Expected: recall@1 0, recall@k 1, MRR 0.5.
        """
        # Arrange
        results = [(_case([REFUNDS]), _hits((PLANS, 0.40), (REFUNDS, 0.39)))]

        # Act
        score = retrieval.score_threshold(results, min_score=0.2, top_k=5)

        # Assert
        assert (score.recall_at_1, score.recall_at_k, score.mrr) == (0.0, 1.0, 0.5)

    def test_threshold_removes_the_expected_section(self):
        """Test: the expected section is under the threshold.
        Expected: counted as missing.
        """
        # Arrange
        results = [(_case([REFUNDS]), _hits((REFUNDS, 0.15)))]

        # Act
        score = retrieval.score_threshold(results, min_score=0.2, top_k=5)

        # Assert
        assert (score.recall_at_k, score.mrr) == (0.0, 0.0)

    @pytest.mark.parametrize(
        ("min_score", "expected_rejection"), [(0.3, 1.0), (0.2, 0.0)]
    )
    def test_negative_case_is_rejected_only_when_nothing_passes(
        self, min_score, expected_rejection
    ):
        """Test: negative case whose best hit scores 0.25.
        Expected: rejected with threshold 0.3, not rejected with 0.2.
        """
        # Arrange
        results = [(_case([]), _hits((PLANS, 0.25)))]

        # Act
        score = retrieval.score_threshold(results, min_score=min_score, top_k=5)

        # Assert
        assert score.negative_rejection == expected_rejection
