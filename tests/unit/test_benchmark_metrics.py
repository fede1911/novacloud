"""Tests of the deterministic metrics of the benchmark."""

import pytest

from benchmark import metrics

pytestmark = pytest.mark.unit


class TestScoreRouting:
    """Routing outcome and Jaccard of the triage choice."""

    @pytest.mark.parametrize(
        ("gold_scopes", "predicted_scopes", "expected_outcome", "expected_jaccard"),
        [
            (["novaid"], ["novaid"], "exact", 1.0),
            (["novaid"], ["novaid", "novadrive"], "over_routing", 0.5),
            (["novaid", "novadrive"], ["novadrive"], "under_routing", 0.5),
            (["novabill"], ["novaid"], "wrong", 0.0),
            ([], [], "exact", 1.0),
        ],
    )
    def test_routing_outcome(
        self, gold_scopes, predicted_scopes, expected_outcome, expected_jaccard
    ):
        """Test: gold and predicted scopes.
        Expected: the outcome and Jaccard of the table.
        """
        # Act
        score = metrics.score_routing(
            gold_scopes=gold_scopes, predicted_scopes=predicted_scopes
        )

        # Assert
        assert (score.outcome, score.jaccard) == (expected_outcome, expected_jaccard)


class TestOpenedTicket:
    """A ticket counts as opened only with a successful open_ticket call."""

    @pytest.mark.parametrize(
        ("tool_calls", "expected"),
        [
            ([("open_ticket_novabill", "success")], True),
            ([("open_ticket_novabill", "error")], False),
            ([("search-documentation-novabill", "success")], False),
        ],
    )
    def test_opened_ticket(self, tool_calls, expected):
        """Test: tool calls of a conversation.
        Expected: True only for a successful ticket tool call.
        """
        # Act / Assert
        assert metrics.opened_ticket(tool_calls) is expected
