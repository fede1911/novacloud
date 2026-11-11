"""Tests of the projection of the tool calls of other agents."""

import pytest

from agent import history_projection
from agent.tools.set_scope import set_scope_tool

pytestmark = pytest.mark.unit


def _triage_history() -> list[dict]:
    return [
        {"role": "user", "content": [{"text": "non mi sincronizza il pc"}]},
        {
            "role": "assistant",
            "content": [
                {"reasoningContent": {"reasoningText": {"text": "..."}}},
                {
                    "toolUse": {
                        "toolUseId": "t1",
                        "name": "set_scope",
                        "input": {"scope_ids": ["novadrive"]},
                    }
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "t1", "status": "success", "content": []}}
            ],
        },
    ]


class TestProjectForeignToolCalls:
    """Foreign tool calls become text, registered ones stay structured."""

    def test_foreign_calls_become_text(self):
        """Test: history of the triage seen by the scope agent.
        Expected: set_scope call and result as text, reasoning dropped, same number of messages.
        """
        # Act
        projected, converted = history_projection.project_foreign_tool_calls(
            _triage_history(), {"search-documentation-novadrive"}
        )

        # Assert
        assert converted == 3
        assert projected[1]["content"] == [
            {"text": "[Conversazione affidata a: novadrive]"}
        ]
        assert projected[2]["content"] == [
            {"text": f"[Passaggio eseguito] {set_scope_tool.HANDOVER_RESULT_TEXT}"}
        ]

    def test_registered_calls_stay_structured(self):
        """Test: history seen by the triage itself.
        Expected: only the reasoning block is converted.
        """
        # Act
        projected, converted = history_projection.project_foreign_tool_calls(
            _triage_history(), {"set_scope"}
        )

        # Assert
        assert converted == 1
        assert "toolUse" in projected[1]["content"][0]

    def test_projection_is_idempotent(self):
        """Test: projection applied twice.
        Expected: the second pass converts nothing.
        """
        # Arrange
        projected, _ = history_projection.project_foreign_tool_calls(
            _triage_history(), set()
        )

        # Act
        _, converted_again = history_projection.project_foreign_tool_calls(
            projected, set()
        )

        # Assert
        assert converted_again == 0
