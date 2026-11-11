"""Evaluates the retrieval on a dataset of queries with expected sections, for several thresholds.

Usage:
    python -m benchmark.retrieval_cli
    python -m benchmark.retrieval_cli --dataset benchmark/datasets/retrieval_v1.json

It calls only the embedding model (one call per query), no LLM.
"""

import argparse
import dataclasses
import datetime as dt
import json
import pathlib

from benchmark import retrieval
from shared import embeddings, kb_config, kb_index, kb_search, logger, settings

DEFAULT_DATASET = pathlib.Path(__file__).parent / "datasets" / "retrieval_v1.json"
# Thresholds scored on the same results: the configured one is marked in the table
THRESHOLDS = (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieval evaluation of the knowledge base"
    )
    parser.add_argument(
        "--dataset",
        type=pathlib.Path,
        default=DEFAULT_DATASET,
        help="Retrieval dataset JSON file",
    )
    return parser.parse_args()


def main() -> None:
    """Runs the queries once and prints the scores of every threshold."""
    args = _parse_args()
    app_settings = settings.Settings.from_environment()
    logger.configure_logging(
        logs_dir=app_settings.logs_dir, level=app_settings.log_level
    )
    config = kb_config.KnowledgeBaseConfig.from_file(app_settings.kb_config_path)
    dataset = retrieval.RetrievalDataset.from_file(args.dataset)

    embedder = embeddings.create_embedder(config.embedder)
    try:
        kb_index.check_compatibility(app_settings.index_dir, embedder=embedder)
    except (kb_index.IndexNotFoundError, kb_index.IndexMismatchError) as e:
        raise SystemExit(str(e)) from e
    # No threshold here: every threshold is applied afterwards to the same results
    retriever = kb_search.Retriever(
        index_dir=app_settings.index_dir,
        embedder=embedder,
        top_k=config.retrieval.top_k,
        min_score=0.0,
    )
    results = retrieval.search_all(dataset, retriever=retriever)
    scores = [
        retrieval.score_threshold(
            results, min_score=threshold, top_k=config.retrieval.top_k
        )
        for threshold in THRESHOLDS
    ]

    print(
        f"{len(dataset.cases)} cases, top_k={config.retrieval.top_k}, configured threshold {config.retrieval.min_score}\n"
    )
    print("thresh  recall@1  recall@k  MRR    negatives rejected")
    for score in scores:
        marker = (
            "  ← configured" if score.min_score == config.retrieval.min_score else ""
        )
        cells = [
            "-" if value is None else f"{value:.2f}"
            for value in (
                score.recall_at_1,
                score.recall_at_k,
                score.mrr,
                score.negative_rejection,
            )
        ]
        print(
            f"{score.min_score:5.2f}   {cells[0]:>6}    {cells[1]:>6}  {cells[2]:>5}  {cells[3]:>6}{marker}"
        )

    misses = [
        (case.case_id, case.query, [hit.reference for hit in hits[:3]])
        for case, hits in results
        if case.relevant and retrieval.rank_of_first_relevant(hits, case.relevant) != 1
    ]
    if misses:
        print("\nCases where the expected section is not first:")
        for case_id, query, top_refs in misses:
            print(f"  {case_id} «{query}» → {top_refs}")

    run_dir = (
        app_settings.benchmark_runs_dir
        / f"{dt.datetime.now(dt.UTC):%Y%m%d_%H%M%S}__{dataset.dataset_name}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "dataset_name": dataset.dataset_name,
                "top_k": config.retrieval.top_k,
                "configured_min_score": config.retrieval.min_score,
                "thresholds": [dataclasses.asdict(score) for score in scores],
                "cases": [
                    {
                        "case_id": case.case_id,
                        "relevant": case.relevant,
                        "hits": [dataclasses.asdict(hit) for hit in hits],
                    }
                    for case, hits in results
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nReport: {run_dir}")


if __name__ == "__main__":
    main()
