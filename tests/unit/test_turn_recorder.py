"""Tests of the recorder of the turn events."""

import types

import pytest

from agent import turn_events

pytestmark = pytest.mark.unit


def _tool_use_message() -> dict:
    return {
        "role": "assistant",
        "content": [
            {"text": "Cerco nella documentazione."},
            {
                "toolUse": {
                    "toolUseId": "t1",
                    "name": "search-documentation-novaid",
                    "input": {"question": "password"},
                }
            },
        ],
    }


def _tool_result_message() -> dict:
    return {
        "role": "user",
        "content": [
            {"toolResult": {"toolUseId": "t1", "status": "success", "content": []}}
        ],
    }


class TestTurnRecorder:
    """The recorder builds text deltas, tool calls, answer and token usage."""

    def test_text_chunk_becomes_text_delta(self):
        """Test: text chunk of the stream.
        Expected: one TextDelta with the text.
        """
        # Act
        events = turn_events.TurnRecorder().observe({"data": "Ciao"})

        # Assert
        assert events == [turn_events.TextDelta(text="Ciao")]

    def test_tool_call_and_result_are_recorded(self):
        """Test: assistant message with a tool call, then its result.
        Expected: ToolStarted event and one ToolUsed with the status of the result.
        """
        # Arrange
        recorder = turn_events.TurnRecorder()

        # Act
        started = recorder.observe({"message": _tool_use_message()})
        recorder.observe({"message": _tool_result_message()})

        # Assert
        assert started == [
            turn_events.ToolStarted(tool_name="search-documentation-novaid")
        ]
        assert [(tool.tool_name, tool.status) for tool in recorder.tools_used] == [
            ("search-documentation-novaid", "success")
        ]

    def test_discarded_text_is_not_in_answer(self):
        """Test: text written before a discard, then more text.
        Expected: discard reports text, the answer has only the text after it.
        """
        # Arrange
        recorder = turn_events.TurnRecorder()
        recorder.observe({"message": _tool_use_message()})

        # Act
        discarded = recorder.discard_text()
        recorder.observe(
            {"message": {"role": "assistant", "content": [{"text": "Risposta finale"}]}}
        )

        # Assert
        assert discarded is True
        assert recorder.answer() == "Risposta finale"

    def test_result_event_adds_invocation_usage(self):
        """Test: result event of an agent invocation.
        Expected: its input and output tokens are counted.
        """
        # Arrange
        recorder = turn_events.TurnRecorder()
        invocation = types.SimpleNamespace(
            usage={"inputTokens": 120, "outputTokens": 30}
        )
        result = types.SimpleNamespace(
            metrics=types.SimpleNamespace(latest_agent_invocation=invocation)
        )

        # Act
        recorder.observe({"result": result})

        # Assert
        assert (recorder.input_tokens, recorder.output_tokens) == (120, 30)
