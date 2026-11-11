"""Tests of the terminal printer of the chat CLI."""

import pytest

from agent import cli, turn_events

pytestmark = pytest.mark.unit


def _completed(agent_type: str) -> turn_events.TurnCompleted:
    return turn_events.TurnCompleted(
        result=turn_events.TurnResult(
            chat_id="chat-1",
            answer="",
            agent_type=agent_type,
            scope_ids=[],
            tools_used=[],
            handovers=[],
            input_tokens=0,
            output_tokens=0,
        )
    )


@pytest.fixture
def output() -> list[str]:
    return []


@pytest.fixture
def printer(output) -> cli.TurnPrinter:
    return cli.TurnPrinter(write=output.append)


class TestTurnPrinter:
    """The text of the triage is shown only if it is the answer of the turn."""

    def test_triage_preamble_is_never_printed(self, printer, output):
        """Test: the triage writes a preamble, then hands the turn over to a specialist that answers.
        Expected: the specialist answer is printed, the preamble is not.
        """
        # Act
        printer.handle(
            turn_events.TextDelta(text="Ti aiuto subito!", agent_type="triage")
        )
        printer.handle(
            turn_events.Handover(
                from_agent="triage",
                to_agent="scope",
                scope_ids=["novabill"],
                discarded_text=True,
            )
        )
        printer.handle(
            turn_events.TextDelta(
                text="Ecco la regola dei rimborsi.", agent_type="scope"
            )
        )
        printer.handle(_completed("scope"))

        # Assert
        printed = "".join(output)
        assert "Ecco la regola dei rimborsi." in printed
        assert "Ti aiuto subito!" not in printed

    def test_triage_answer_is_printed_when_the_turn_ends(self, printer, output):
        """Test: the triage asks a clarification and does not hand the turn over.
        Expected: its text is printed, after the turn ends.
        """
        # Act
        printer.handle(
            turn_events.TextDelta(text="Quale prodotto?", agent_type="triage")
        )
        printed_before_end = "".join(output)
        printer.handle(_completed("triage"))

        # Assert
        assert "Quale prodotto?" not in printed_before_end
        assert "Quale prodotto?" in "".join(output)
