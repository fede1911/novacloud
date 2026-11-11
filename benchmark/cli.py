"""Runs a benchmark dataset against the agent.

Usage:
    python -m benchmark.cli                                  # default dataset, with judges
    python -m benchmark.cli --no-judge                       # only deterministic metrics
    python -m benchmark.cli --scenario novaid_password_reset # a single case
"""

import argparse
import asyncio
import datetime as dt
import json
import pathlib

from agent import agent_config, runtime
from benchmark import dataset as dataset_module
from benchmark import judges, report, runner
from shared import kb_index, logger

DATASETS_DIR = pathlib.Path(__file__).parent / "datasets"
DEFAULT_DATASET = DATASETS_DIR / "novacloud_v1.json"
JUDGE_PROMPTS_PATH = pathlib.Path(__file__).parent / "judge_prompts.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark of the blueprint agent")
    parser.add_argument(
        "--dataset",
        type=pathlib.Path,
        default=DEFAULT_DATASET,
        help="Dataset JSON file",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Run only these scenario ids (repeatable)",
    )
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM judges")
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    config = agent_config.AgentConfig.from_environment()
    logger.configure_logging(
        logs_dir=config.settings.logs_dir, level=config.settings.log_level
    )
    try:
        app = runtime.build_runtime(config)
    except kb_index.IndexMismatchError as e:
        raise SystemExit(str(e)) from e

    benchmark_dataset = dataset_module.Dataset.from_file(args.dataset)
    scenarios = [
        scenario
        for scenario in benchmark_dataset.scenarios
        if not args.scenario or scenario.scenario_id in args.scenario
    ]
    if not scenarios:
        raise SystemExit(f"No scenario matches {args.scenario}")

    judge = None
    if not args.no_judge:
        judge = judges.LlmJudge(
            models=app.model_catalog.for_role("judge"),
            prompts_path=JUDGE_PROMPTS_PATH,
        )

    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    benchmark_runner = runner.BenchmarkRunner(
        orchestrator=app.orchestrator, judge=judge, run_id=run_id
    )
    results = []
    for scenario in scenarios:
        result = await benchmark_runner.run_case(scenario)
        print(report.format_case_line(result), flush=True)
        results.append(result)

    summary = report.summarize(results)
    run_dir = (
        config.settings.benchmark_runs_dir
        / f"{run_id}__{benchmark_dataset.dataset_name}"
    )
    report.write_run(
        run_dir,
        dataset_name=benchmark_dataset.dataset_name,
        results=results,
        summary=summary,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Report: {run_dir}")


def main() -> None:
    """Runs the benchmark."""
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
