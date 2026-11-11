"""Builds the knowledge base index from data/kb, or searches the existing one.

Usage:
    python -m ingestion.cli                                              # rebuild the index
    python -m ingestion.cli --query "mi ridate i soldi?" --scope novabill   # search, no rebuild
"""

import argparse
import dataclasses
import json

from ingestion import pipeline
from shared import embeddings, kb_config, kb_index, kb_search, logger, settings
from shared import scope_catalog as scope_catalog_module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the knowledge base index")
    parser.add_argument(
        "--query",
        help="Search this text in the existing index instead of rebuilding it",
    )
    parser.add_argument("--scope", help="Product of the --query search")
    return parser.parse_args()


def main() -> None:
    """Rebuilds the index, or with --query searches the existing index of one product."""
    args = _parse_args()
    app_settings = settings.Settings.from_environment()
    logger.configure_logging(
        logs_dir=app_settings.logs_dir, level=app_settings.log_level
    )
    catalog = scope_catalog_module.ScopeCatalog.from_file(app_settings.scopes_path)
    config = kb_config.KnowledgeBaseConfig.from_file(app_settings.kb_config_path)
    embedder = embeddings.create_embedder(config.embedder)

    if args.query:
        _search(
            args.query,
            scope_id=args.scope,
            app_settings=app_settings,
            catalog=catalog,
            config=config,
            embedder=embedder,
        )
        return

    summary = pipeline.run(
        source_dir=app_settings.kb_source_dir,
        index_dir=app_settings.index_dir,
        scope_ids=catalog.scope_ids(),
        embedder=embedder,
        max_chars=config.chunking.max_chars,
        overlap_chars=config.chunking.overlap_chars,
    )
    print(json.dumps(dataclasses.asdict(summary), indent=2, ensure_ascii=False))


def _search(
    query: str,
    *,
    scope_id: str | None,
    app_settings: settings.Settings,
    catalog: scope_catalog_module.ScopeCatalog,
    config: kb_config.KnowledgeBaseConfig,
    embedder: embeddings.Embedder,
) -> None:
    """Searches the existing index as the search tool of the agent does, and prints the results."""
    if scope_id not in catalog.scope_ids():
        raise SystemExit(f"--query needs --scope, one of {catalog.scope_ids()}")
    try:
        kb_index.check_compatibility(app_settings.index_dir, embedder=embedder)
    except (kb_index.IndexNotFoundError, kb_index.IndexMismatchError) as e:
        raise SystemExit(f"{e}") from e

    retriever = kb_search.Retriever(
        index_dir=app_settings.index_dir,
        embedder=embedder,
        top_k=config.retrieval.top_k,
        min_score=config.retrieval.min_score,
    )
    hits = retriever.search(query, scope_id=scope_id)
    print(
        f"«{query}» in {scope_id}, top {config.retrieval.top_k} over the threshold {config.retrieval.min_score}:"
    )
    if not hits:
        print("  no chunk over the threshold: the search tool would return nothing")
    for hit in hits:
        print(f"  {hit.score:.3f}  {hit.chunk.chunk_id}  {hit.chunk.section}")


if __name__ == "__main__":
    main()
