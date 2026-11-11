"""Tests of the handover tools: set_scope (triage) and request_rerouting (scope agent)."""

import pytest

from agent.tools.request_rerouting import request_rerouting_tool
from agent.tools.set_scope import set_scope_tool

pytestmark = pytest.mark.unit

ALLOWED_SCOPE_IDS = ["novaid", "novadrive", "novabill"]


class TestSetScopeTool:
    """set_scope validates the scopes and leaves them in the agent state."""

    @pytest.fixture
    def set_scope(self, chat_ctx):
        return set_scope_tool.build(
            chat_ctx=chat_ctx, allowed_scope_ids=ALLOWED_SCOPE_IDS
        ).tool

    @pytest.mark.parametrize(
        ("scope_ids", "expected_error_start"),
        [
            ([], "scope_ids must contain at least one scope"),
            (
                ["novaid", "novadrive", "novabill"],
                "scope_ids can contain at most 2 scopes",
            ),
            (["novaphone"], "Unknown scopes: ['novaphone']"),
        ],
    )
    def test_invalid_scopes_return_error_without_stopping(
        self, set_scope, tool_context, scope_ids, expected_error_start
    ):
        """Test: invalid scope_ids.
        Expected: error result for the model, no scopes in the state, event loop not stopped.
        """
        # Act
        result = set_scope(tool_context=tool_context, scope_ids=scope_ids)

        # Assert
        assert result["status"] == "error"
        assert result["content"][0]["text"].startswith(expected_error_start)
        assert (
            tool_context.agent.state.get(set_scope_tool.SELECTED_SCOPE_IDS_STATE_KEY)
            is None
        )
        assert tool_context.invocation_state == {}

    def test_valid_scopes_are_stored_deduplicated_and_stop_loop(
        self, set_scope, tool_context
    ):
        """Test: valid scopes with a duplicate.
        Expected: scopes in the state without duplicates, in order, and event loop stopped.
        """
        # Act
        result = set_scope(
            tool_context=tool_context, scope_ids=["novadrive", "novaid", "novadrive"]
        )

        # Assert
        assert result["status"] == "success"
        assert tool_context.agent.state.get(
            set_scope_tool.SELECTED_SCOPE_IDS_STATE_KEY
        ) == ["novadrive", "novaid"]
        assert tool_context.invocation_state["request_state"]["stop_event_loop"] is True


class TestRequestReroutingTool:
    """request_rerouting hands the turn back to the triage at most once per turn."""

    @pytest.fixture
    def request_rerouting(self, chat_ctx):
        return request_rerouting_tool.build(chat_ctx=chat_ctx).tool

    def test_allowed_rerouting_is_requested_and_stops_loop(
        self, request_rerouting, tool_context
    ):
        """Test: rerouting allowed in the turn.
        Expected: request in the state and event loop stopped.
        """
        # Arrange
        tool_context.agent.state.set(
            request_rerouting_tool.REROUTING_ALLOWED_STATE_KEY, True
        )

        # Act
        request_rerouting(tool_context=tool_context, reason="billing question")

        # Assert
        assert (
            tool_context.agent.state.get(
                request_rerouting_tool.REROUTING_REQUESTED_STATE_KEY
            )
            is True
        )
        assert tool_context.invocation_state["request_state"]["stop_event_loop"] is True

    def test_not_allowed_rerouting_tells_model_to_answer(
        self, request_rerouting, tool_context
    ):
        """Test: the turn was already rerouted.
        Expected: no request, loop not stopped, message that tells the model to answer on its own.
        """
        # Arrange
        tool_context.agent.state.set(
            request_rerouting_tool.REROUTING_ALLOWED_STATE_KEY, False
        )

        # Act
        result = request_rerouting(tool_context=tool_context, reason="billing question")

        # Assert
        assert (
            result["content"][0]["text"]
            == request_rerouting_tool.REROUTING_NOT_AVAILABLE_TEXT
        )
        assert (
            tool_context.agent.state.get(
                request_rerouting_tool.REROUTING_REQUESTED_STATE_KEY
            )
            is None
        )
        assert tool_context.invocation_state == {}
