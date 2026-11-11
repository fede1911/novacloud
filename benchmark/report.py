"""Aggregates and writes the results of a benchmark run."""

import dataclasses
import json
import pathlib
from typing import TYPE_CHECKING

from benchmark import judges, metrics

if TYPE_CHECKING:
    from benchmark.runner import CaseResult


def summarize(results: list["CaseResult"]) -> dict:
    """Returns the overall metrics of a run."""
    routed = [result.routing for result in results if result.routing is not None]
    return {
        "cases": len(results),
        "failed_cases": [result.scenario_id for result in results if result.error],
        "routing": {
            "cases": len(routed),
            "exact_match_rate": metrics.mean_or_none(
                [float(routing["exact_match"]) for routing in routed]
            ),
            "mean_jaccard": metrics.mean_or_none(
                [routing["jaccard"] for routing in routed]
            ),
            "outcomes": {
                outcome: sum(1 for routing in routed if routing["outcome"] == outcome)
                for outcome in ("exact", "over_routing", "under_routing", "wrong")
            },
        },
        "escalation_accuracy": metrics.mean_or_none(
            [float(result.escalation_correct) for result in results]
        ),
        "expected_tools_hit_rate": metrics.mean_or_none(
            [result.expected_tools_hit_rate for result in results]
        ),
        "quality": {
            metric: metrics.mean_or_none(
                [result.quality.get(metric, {}).get("score") for result in results]
            )
            for metric in judges.JUDGE_METRICS
        },
        "mean_latency_seconds_per_turn": metrics.mean_or_none(
            [turn.latency_seconds for result in results for turn in result.turns]
        ),
        "total_input_tokens": sum(
            turn.input_tokens for result in results for turn in result.turns
        ),
        "total_output_tokens": sum(
            turn.output_tokens for result in results for turn in result.turns
        ),
    }


def write_run(
    run_dir: pathlib.Path,
    *,
    dataset_name: str,
    results: list["CaseResult"],
    summary: dict,
) -> None:
    """Writes one JSON per case and the summary of the run."""
    cases_dir = run_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        (cases_dir / f"{result.scenario_id}.json").write_text(
            json.dumps(dataclasses.asdict(result), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {"dataset_name": dataset_name, **summary}, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )


def format_case_line(result: "CaseResult") -> str:
    """One line per case for the console."""
    if result.error:
        return f"✗ {result.scenario_id:<32} ERROR {result.error}"
    routing = (
        f"routing={result.routing['outcome']}{result.routing['predicted_scopes']}"
        if result.routing
        else "routing=-"
    )
    scores = (
        " ".join(
            f"{metric}={values.get('score')}"
            for metric, values in result.quality.items()
        )
        or "judges=-"
    )
    escalation = "ok" if result.escalation_correct else "KO"
    return (
        f"• {result.scenario_id:<32} {routing:<38} escalation={escalation:<3} {scores}"
    )
