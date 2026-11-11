"""Runs the scenarios of a dataset through the orchestrator and scores them."""

import dataclasses
import time
from typing import TYPE_CHECKING

from agent import turn_events
from agent.models import chat as chat_models
from benchmark import judges, metrics
from shared import logger

if TYPE_CHECKING:
    from agent.orchestrator import AgentOrchestrator
    from benchmark.dataset import Scenario

_LOGGER = logger.get_logger(__name__)
BENCHMARK_USER = "benchmark"


@dataclasses.dataclass
class TurnRecord:
    """One scripted turn of a case."""

    user_message: str
    answer: str
    agent_type: str
    scope_ids: list[str]
    handovers: list[dict]
    tools_used: list[dict]
    latency_seconds: float
    input_tokens: int
    output_tokens: int


@dataclasses.dataclass
class CaseResult:
    """Outcome of one scenario: transcript, deterministic metrics and judge scores."""

    scenario_id: str
    title: str
    chat_id: str
    turns: list[TurnRecord]
    routing: dict | None
    opened_ticket: bool
    escalation_correct: bool
    expected_tools_hit_rate: float | None
    quality: dict[str, dict]
    error: str | None


class BenchmarkRunner:
    """Plays the scripted turns of every scenario on a new chat and scores the conversation."""

    def __init__(
        self,
        *,
        orchestrator: "AgentOrchestrator",
        judge: judges.LlmJudge | None,
        run_id: str,
    ):
        """Initializes the runner.

        Args:
            orchestrator: Answers the turns, as in production.
            judge: Scores the answers; None to compute only the deterministic metrics.
            run_id: Id of the run, part of the chat ids.
        """
        self._orchestrator = orchestrator
        self._judge = judge
        self._run_id = run_id

    async def run_case(self, scenario: "Scenario") -> CaseResult:
        """Runs one scenario; an error of the agent is recorded in the result, not raised."""
        chat_ctx = chat_models.ChatContext(
            chat_id=f"bench-{self._run_id}-{scenario.scenario_id}",
            user=BENCHMARK_USER,
            scope_ids=scenario.scope_ids,
        )
        turns: list[TurnRecord] = []
        try:
            for user_message in scenario.turns:
                turns.append(await self._run_turn(chat_ctx, user_message))
        except Exception as e:  # noqa: BLE001 - one failing case must not stop the benchmark
            _LOGGER.exception("Scenario %s failed", scenario.scenario_id)
            return self._result(
                scenario,
                chat_ctx.chat_id,
                turns,
                quality={},
                error=f"{type(e).__name__}: {e}",
            )

        quality = await self._judge_case(scenario, turns)
        return self._result(
            scenario, chat_ctx.chat_id, turns, quality=quality, error=None
        )

    async def _run_turn(
        self, chat_ctx: chat_models.ChatContext, user_message: str
    ) -> TurnRecord:
        started_at = time.monotonic()
        result: turn_events.TurnResult | None = None
        async for event in self._orchestrator.run_turn(chat_ctx, user_message):
            if isinstance(event, turn_events.TurnCompleted):
                result = event.result
        if result is None:
            raise RuntimeError("The turn ended without a TurnCompleted event")
        return TurnRecord(
            user_message=user_message,
            answer=result.answer,
            agent_type=result.agent_type,
            scope_ids=result.scope_ids,
            handovers=[dataclasses.asdict(handover) for handover in result.handovers],
            tools_used=[tool.model_dump() for tool in result.tools_used],
            latency_seconds=round(time.monotonic() - started_at, 2),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    async def _judge_case(
        self, scenario: "Scenario", turns: list[TurnRecord]
    ) -> dict[str, dict]:
        """Scores the conversation on the judge metrics that the scenario can support.

        accuracy needs the ground truth; completeness needs the ground truth or the required behaviours.
        """
        if self._judge is None:
            return {}
        metric_names = [
            metric
            for metric in judges.JUDGE_METRICS
            if scenario.ground_truth
            or (metric == "completeness" and scenario.must_include)
        ]
        exchanges = [(turn.user_message, turn.answer) for turn in turns]
        quality: dict[str, dict] = {}
        for metric in metric_names:
            try:
                judgement = await self._judge.judge(
                    metric,
                    ground_truth=scenario.ground_truth,
                    must_include=scenario.must_include,
                    must_not_include=scenario.must_not_include,
                    exchanges=exchanges,
                )
            except judges.JudgementError as e:
                _LOGGER.warning(
                    "Judge %s of scenario %s returned no valid judgement: %s",
                    metric,
                    scenario.scenario_id,
                    e,
                )
                quality[metric] = {"score": None, "reasoning": str(e)}
            else:
                quality[metric] = judgement.model_dump()
        return quality

    @staticmethod
    def _result(
        scenario: "Scenario",
        chat_id: str,
        turns: list[TurnRecord],
        *,
        quality: dict[str, dict],
        error: str | None,
    ) -> CaseResult:
        """Computes the deterministic metrics of a case."""
        tool_calls = [
            (tool["tool_name"], tool["status"])
            for turn in turns
            for tool in turn.tools_used
        ]
        routing = None
        if scenario.gold_scopes is not None and not scenario.scope_ids and turns:
            # Scopes chosen by the triage at the first turn: the first handover to a scope agent
            first_handovers = [
                handover
                for handover in turns[0].handovers
                if handover["to_agent"] == "scope"
            ]
            predicted_scopes = (
                first_handovers[0]["scope_ids"] if first_handovers else []
            )
            routing = dataclasses.asdict(
                metrics.score_routing(
                    gold_scopes=scenario.gold_scopes, predicted_scopes=predicted_scopes
                )
            )
            routing["predicted_scopes"] = predicted_scopes
        ticket = metrics.opened_ticket(tool_calls)
        return CaseResult(
            scenario_id=scenario.scenario_id,
            title=scenario.title,
            chat_id=chat_id,
            turns=turns,
            routing=routing,
            opened_ticket=ticket,
            escalation_correct=ticket == scenario.should_open_ticket,
            expected_tools_hit_rate=metrics.expected_tools_hit_rate(
                expected_tools=scenario.expected_tools,
                used_tool_names=[name for name, _ in tool_calls],
            ),
            quality=quality,
            error=error,
        )
