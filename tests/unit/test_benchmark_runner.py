"""Tests of the benchmark runner: routing and escalation computed from the turns of a scenario."""

import pytest

from agent import turn_events
from agent.models import chat as chat_models
from benchmark import dataset, runner

pytestmark = pytest.mark.unit


def _turn(
    *,
    scope_ids: list[str],
    handovers: list[turn_events.Handover],
    tools: list[tuple[str, str]],
) -> turn_events.TurnResult:
    return turn_events.TurnResult(
        chat_id="bench-chat",
        answer="risposta",
        agent_type="scope",
        scope_ids=scope_ids,
        tools_used=[
            chat_models.ToolUsed(
                tool_name=name, input={}, status=status, total_seconds=0.1
            )
            for name, status in tools
        ],
        handovers=handovers,
        input_tokens=10,
        output_tokens=5,
    )


def _to_scope(scope_ids: list[str]) -> turn_events.Handover:
    return turn_events.Handover(
        from_agent="triage", to_agent="scope", scope_ids=scope_ids, discarded_text=False
    )


class _ScriptedOrchestrator:
    """Answers every turn with the next prepared TurnResult."""

    def __init__(self, results: list[turn_events.TurnResult]):
        self._results = results

    async def run_turn(self, chat_ctx, prompt):
        yield turn_events.TurnCompleted(result=self._results.pop(0))


def _scenario(**fields) -> dataset.Scenario:
    return dataset.Scenario(scenario_id="case", title="case", **fields)


async def _run(
    scenario: dataset.Scenario, results: list[turn_events.TurnResult]
) -> runner.CaseResult:
    benchmark_runner = runner.BenchmarkRunner(
        orchestrator=_ScriptedOrchestrator(results),
        judge=None,
        run_id="test-run",
    )
    return await benchmark_runner.run_case(scenario)


class TestBenchmarkRunner:
    """The runner plays the turns and scores routing and escalation."""

    async def test_routing_is_the_first_choice_of_the_first_turn(self):
        """Test: the triage picks NovaBill at turn 1; at turn 2 a rerouting moves the chat to NovaID.
        Expected: routing scored on NovaBill only, the first choice, and exact for gold NovaBill.
        """
        # Arrange
        scenario = _scenario(
            turns=["fattura", "e la password?"], gold_scopes=["novabill"]
        )
        turn_1 = _turn(
            scope_ids=["novabill"], handovers=[_to_scope(["novabill"])], tools=[]
        )
        rerouting = turn_events.Handover(
            from_agent="scope", to_agent="triage", scope_ids=[], discarded_text=False
        )
        turn_2 = _turn(
            scope_ids=["novaid"], handovers=[rerouting, _to_scope(["novaid"])], tools=[]
        )

        # Act
        result = await _run(scenario, [turn_1, turn_2])

        # Assert
        assert (result.routing["predicted_scopes"], result.routing["outcome"]) == (
            ["novabill"],
            "exact",
        )

    async def test_triage_clarification_counts_as_no_choice(self):
        """Test: the triage asks a clarification at turn 1, so there is no handover.
        Expected: no scope predicted, exact routing for a scenario that expects a clarification.
        """
        # Arrange
        scenario = _scenario(turns=["ho un problema"], gold_scopes=[])
        clarification = _turn(scope_ids=[], handovers=[], tools=[])

        # Act
        result = await _run(scenario, [clarification])

        # Assert
        assert (result.routing["predicted_scopes"], result.routing["outcome"]) == (
            [],
            "exact",
        )

    async def test_scopes_fixed_by_the_caller_skip_routing(self):
        """Test: scenario with scopes fixed by the caller, so the triage never runs.
        Expected: no routing score.
        """
        # Arrange
        scenario = _scenario(
            turns=["cestino"], scope_ids=["novadrive"], gold_scopes=["novadrive"]
        )

        # Act
        result = await _run(
            scenario, [_turn(scope_ids=["novadrive"], handovers=[], tools=[])]
        )

        # Assert
        assert result.routing is None

    @pytest.mark.parametrize(
        ("ticket_status", "should_open_ticket", "expected_correct"),
        [
            ("success", True, True),
            ("error", True, False),
            ("success", False, False),
        ],
    )
    async def test_escalation_is_correct_only_when_ticket_matches_expectation(
        self, ticket_status, should_open_ticket, expected_correct
    ):
        """Test: a ticket call with a given outcome, for a scenario that does or does not expect one.
        Expected: escalation correct only when a ticket was opened exactly when expected.
        """
        # Arrange
        scenario = _scenario(
            turns=["addebito doppio"], should_open_ticket=should_open_ticket
        )
        turn = _turn(
            scope_ids=["novabill"],
            handovers=[],
            tools=[("open_ticket_novabill", ticket_status)],
        )

        # Act
        result = await _run(scenario, [turn])

        # Assert
        assert result.escalation_correct is expected_correct
