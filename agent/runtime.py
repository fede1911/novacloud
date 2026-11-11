"""Composition root: wires configuration, stores, tools and agents into an orchestrator."""

import dataclasses

from agent import agent_builder, agent_config, orchestrator, prompt_loader, toolkits
from agent.models import model_config
from agent.stores import chat_store, ticket_store
from shared import embeddings, kb_config, kb_index, kb_search, logger
from shared import scope_catalog as scope_catalog_module

_LOGGER = logger.get_logger(__name__)


@dataclasses.dataclass(frozen=True)
class Runtime:
    """The wired components, for the CLI and the benchmark."""

    config: agent_config.AgentConfig
    orchestrator: orchestrator.AgentOrchestrator
    chat_store: chat_store.ChatStore
    scope_catalog: scope_catalog_module.ScopeCatalog
    model_catalog: model_config.ModelCatalog


def build_runtime(config: agent_config.AgentConfig) -> Runtime:
    """Builds every component of the agent from the configuration.

    Args:
        config: Configuration of the agent.

    Returns:
        The wired runtime.

    Raises:
        IndexMismatchError: If the index was built with another embedding model than the configured one.
    """
    settings = config.settings
    catalog = scope_catalog_module.ScopeCatalog.from_file(settings.scopes_path)
    models = model_config.ModelCatalog.from_file(settings.models_path)
    knowledge_base = kb_config.KnowledgeBaseConfig.from_file(settings.kb_config_path)
    retrieval = knowledge_base.retrieval
    query_embedder = embeddings.create_embedder(knowledge_base.embedder)
    _check_index(settings.index_dir, embedder=query_embedder)

    services = toolkits.ToolServices(
        retriever=kb_search.Retriever(
            index_dir=settings.index_dir,
            embedder=query_embedder,
            top_k=retrieval.top_k,
            min_score=retrieval.min_score,
        ),
        ticket_store=ticket_store.TicketStore(settings.tickets_dir),
        prompt_loader=prompt_loader.PromptLoader(settings.prompts_dir),
        scope_catalog=catalog,
        systems_status_path=settings.systems_status_path,
    )
    store = chat_store.ChatStore(settings.chats_dir)
    agent_orchestrator = orchestrator.AgentOrchestrator(
        agent_factory=agent_builder.AgentBuilder(config=config, services=services),
        model_catalog=models,
        chat_store=store,
    )
    return Runtime(
        config=config,
        orchestrator=agent_orchestrator,
        chat_store=store,
        scope_catalog=catalog,
        model_catalog=models,
    )


def _check_index(index_dir, *, embedder: embeddings.Embedder) -> None:
    """Stops at startup if the index was built with another model; only warns if it does not exist yet."""
    try:
        metadata = kb_index.check_compatibility(index_dir, embedder=embedder)
    except kb_index.IndexNotFoundError:
        _LOGGER.warning(
            "Knowledge base index not built yet: the search tools will fail until `make ingest`"
        )
        return
    _LOGGER.info(
        "Index of %s chunks built at %s with %s, compatible with the configuration",
        metadata.chunks,
        metadata.built_at,
        metadata.embedding_model,
    )
