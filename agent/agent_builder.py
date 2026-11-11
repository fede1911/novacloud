"""Builds the Strands agents of a chat: model with fallbacks, tools, system prompt, session."""

import dataclasses
from typing import TYPE_CHECKING

import strands
from strands.agent import conversation_manager
from strands.session import file_session_manager
from strands.types import content

from agent import history_projection, model_factory, prompt_loader, toolkits
from agent.hooks import empty_response_retry, model_retry
from shared import logger

if TYPE_CHECKING:
    from agent.agent_config import AgentConfig
    from agent.models.chat import ChatContext
    from agent.models.model_config import ModelEntry

_LOGGER = logger.get_logger(__name__)

SCOPE_AGENT_NAME = "scope-agent"
TRIAGE_AGENT_NAME = "triage-agent"
# Every agent has the same id: the triage and the scope agent write the same history
SHARED_AGENT_ID = "0"
SCOPE_PROMPT_NAME = "scope_agent_system_prompt"
TRIAGE_PROMPT_NAME = "triage_system_prompt"
SCOPE_INFOS_PROMPT_PREFIX = "scope_additional_infos_"
# Placeholder of the prompts replaced with the product catalog
SCOPE_CATALOG_PLACEHOLDER = "{{scope_catalog}}"

SystemPromptContent = str | list[content.SystemContentBlock]


@dataclasses.dataclass(frozen=True)
class BuiltAgent:
    """An agent with the names of the prompts of its system prompt."""

    agent: strands.Agent
    prompt_names: list[str]


class AgentBuilder:
    """Creates the triage agent and the scope agents on the session of the chat."""

    def __init__(self, *, config: "AgentConfig", services: toolkits.ToolServices):
        """Initializes the builder.

        Args:
            config: Configuration of the agent.
            services: Services of the tools.
        """
        self._config = config
        self._services = services

    def build_scope_agent(
        self,
        *,
        chat_ctx: "ChatContext",
        models: list["ModelEntry"],
        rerouting_enabled: bool,
    ) -> BuiltAgent:
        """Builds the agent that answers on the scopes of chat_ctx.

        Args:
            chat_ctx: Context of the chat, with the scopes of the agent.
            models: LLMs of the agent, first the default one.
            rerouting_enabled: Whether the agent gets the request_rerouting tool.
        """
        toolkit = toolkits.scope_toolkit(
            chat_ctx=chat_ctx,
            services=self._services,
            rerouting_enabled=rerouting_enabled,
        )
        system_prompt, prompt_names = compose_scope_system_prompt(
            scope_ids=chat_ctx.scope_ids,
            loader=self._services.prompt_loader,
            catalog_text=self._services.scope_catalog.render_for_prompt(),
            toolkit=toolkit,
        )
        agent = self._create_agent(
            chat_ctx=chat_ctx,
            models=models,
            toolkit=toolkit,
            system_prompt=system_prompt,
            name=SCOPE_AGENT_NAME,
        )
        return BuiltAgent(agent=agent, prompt_names=prompt_names)

    def build_triage_agent(
        self, *, chat_ctx: "ChatContext", models: list["ModelEntry"]
    ) -> BuiltAgent:
        """Builds the triage agent, which chooses the scopes of the chat with set_scope.

        Args:
            chat_ctx: Context of the chat, without scopes.
            models: LLMs of the agent, first the default one.
        """
        toolkit = toolkits.triage_toolkit(chat_ctx=chat_ctx, services=self._services)
        triage_prompt = self._services.prompt_loader.get_prompt(
            TRIAGE_PROMPT_NAME
        ).replace(
            SCOPE_CATALOG_PLACEHOLDER, self._services.scope_catalog.render_for_prompt()
        )
        agent = self._create_agent(
            chat_ctx=chat_ctx,
            models=models,
            toolkit=toolkit,
            system_prompt=triage_prompt,
            name=TRIAGE_AGENT_NAME,
        )
        return BuiltAgent(agent=agent, prompt_names=[TRIAGE_PROMPT_NAME])

    def _create_agent(
        self,
        *,
        chat_ctx: "ChatContext",
        models: list["ModelEntry"],
        toolkit: toolkits.Toolkit,
        system_prompt: str,
        name: str,
    ) -> strands.Agent:
        """Creates the Strands agent on the session of the chat, with the components shared by every agent."""
        session_manager = file_session_manager.FileSessionManager(
            session_id=f"chat-{chat_ctx.chat_id}",
            storage_dir=str(self._config.settings.sessions_dir),
        )
        prompt_caching = any(model.prompt_caching for model in models)

        # The Strands retry strategy is disabled: the fallback hook handles the provider errors
        agent = strands.Agent(
            agent_id=SHARED_AGENT_ID,
            name=name,
            model=model_factory.create_model(models, model_index=0),
            tools=toolkit.tools(),
            system_prompt=_with_cache_point(
                system_prompt, prompt_caching=prompt_caching
            ),
            conversation_manager=conversation_manager.SlidingWindowConversationManager(
                window_size=self._config.context_window_size
            ),
            session_manager=session_manager,
            callback_handler=None,
            retry_strategy=None,
            hooks=[
                model_retry.ModelFallbackRetry(models=models),
                empty_response_retry.EmptyResponseRetry(),
            ],
        )

        # The history is loaded from the session when the agent is created
        projected_messages, converted_blocks = (
            history_projection.project_foreign_tool_calls(
                agent.messages, set(agent.tool_names)
            )
        )
        if converted_blocks:
            agent.messages[:] = projected_messages
            _LOGGER.info(
                "Projected %s blocks of other agents to text in chat %s",
                converted_blocks,
                chat_ctx.chat_id,
            )

        _LOGGER.info(
            "Built %s for chat %s with tools %s",
            name,
            chat_ctx.chat_id,
            agent.tool_names,
        )
        return agent


def compose_scope_system_prompt(
    *,
    scope_ids: list[str],
    loader: "prompt_loader.PromptLoader",
    catalog_text: str,
    toolkit: toolkits.Toolkit,
) -> tuple[str, list[str]]:
    """Composes the system prompt of a scope agent.

    Base prompt (with the catalog of every product), one section per active
    product, then the prompts added by the tools.

    Args:
        scope_ids: Active scopes of the agent.
        loader: Loads the prompts.
        catalog_text: Catalog of the products, replacing the placeholder of the base prompt.
        toolkit: Tools of the agent.

    Returns:
        The system prompt and the names of the prompts it is made of.
    """
    sections = [
        loader.get_prompt(SCOPE_PROMPT_NAME).replace(
            SCOPE_CATALOG_PLACEHOLDER, catalog_text
        )
    ]
    prompt_names = [SCOPE_PROMPT_NAME]
    for scope_id in scope_ids:
        prompt_name = f"{SCOPE_INFOS_PROMPT_PREFIX}{scope_id}"
        try:
            scope_infos = loader.get_prompt(prompt_name)
        except prompt_loader.PromptNotFoundError:
            # The section stays, so the product is still listed as active
            _LOGGER.warning(
                "Prompt %s not found: scope %s has no additional infos",
                prompt_name,
                scope_id,
            )
            scope_infos = ""
        else:
            prompt_names.append(prompt_name)
        sections.append(f"## Prodotto attivo: {scope_id}\n{scope_infos}".rstrip())
    for tool_prompt_name in toolkit.prompt_names():
        sections.append(loader.get_prompt(tool_prompt_name))
        prompt_names.append(tool_prompt_name)
    return "\n\n".join(sections), prompt_names


def _with_cache_point(
    system_prompt: str, *, prompt_caching: bool
) -> SystemPromptContent:
    """Returns the system prompt, followed by a cache point when prompt caching is enabled."""
    if not prompt_caching:
        return system_prompt
    return [
        content.SystemContentBlock(text=system_prompt),
        content.SystemContentBlock(cachePoint={"type": "default"}),
    ]
