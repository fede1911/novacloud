"""Toolkits: the tools each agent gets, assembled from the product catalog.

Every product declares its tools in config/scopes.json; this module maps
each tool name to the function that builds it. Nothing is registered at
import time: the mapping below is the whole list.
"""

import dataclasses
import pathlib
from typing import TYPE_CHECKING, Any

from agent.tools import tool_spec
from agent.tools.open_ticket import open_ticket_tool
from agent.tools.request_rerouting import request_rerouting_tool
from agent.tools.search_documentation import search_documentation_tool
from agent.tools.set_scope import set_scope_tool
from agent.tools.systems_health import systems_health_tool
from shared import logger

if TYPE_CHECKING:
    from agent.models.chat import ChatContext
    from agent.prompt_loader import PromptLoader
    from agent.stores.ticket_store import TicketStore
    from shared.kb_search import Retriever
    from shared.scope_catalog import ScopeCatalog

_LOGGER = logger.get_logger(__name__)

# Builders of the tools a product can declare in config/scopes.json
PRODUCT_TOOL_BUILDERS = {
    "search_documentation": search_documentation_tool.build,
    "systems_health": systems_health_tool.build,
    "open_ticket": open_ticket_tool.build,
}


@dataclasses.dataclass(frozen=True)
class ToolServices:
    """Services the tools need, created once by the composition root (agent/runtime.py)."""

    retriever: "Retriever"
    ticket_store: "TicketStore"
    prompt_loader: "PromptLoader"
    scope_catalog: "ScopeCatalog"
    systems_status_path: pathlib.Path


@dataclasses.dataclass(frozen=True)
class Toolkit:
    """The tools of one agent, with the prompts they add to its system prompt."""

    specs: list[tool_spec.ToolSpec]

    def tools(self) -> list[Any]:
        """Returns the Strands tools."""
        return [spec.tool for spec in self.specs]

    def prompt_names(self) -> list[str]:
        """Returns the prompts added by the tools, once each, in order of appearance."""
        names: list[str] = []
        for spec in self.specs:
            if spec.prompt_name and spec.prompt_name not in names:
                names.append(spec.prompt_name)
        return names


def scope_toolkit(
    *, chat_ctx: "ChatContext", services: ToolServices, rerouting_enabled: bool
) -> Toolkit:
    """Returns the tools of the scope agent of the scopes of chat_ctx.

    Args:
        chat_ctx: Context of the chat, with the scopes of the agent.
        services: Services of the tools.
        rerouting_enabled: Whether the agent can give the conversation back to the
            triage: only when the triage chose the scopes, not when the caller fixed them.
    """
    specs = []
    for scope_id in chat_ctx.scope_ids:
        scope = services.scope_catalog.get(scope_id)
        specs.extend(
            PRODUCT_TOOL_BUILDERS[tool_name](
                scope=scope, chat_ctx=chat_ctx, services=services
            )
            for tool_name in scope.tools
        )
    if rerouting_enabled:
        specs.append(request_rerouting_tool.build(chat_ctx=chat_ctx))
    _LOGGER.info("Toolkit of scopes %s: %s tools", chat_ctx.scope_ids, len(specs))
    return Toolkit(specs=specs)


def triage_toolkit(*, chat_ctx: "ChatContext", services: ToolServices) -> Toolkit:
    """Returns the tools of the triage: only set_scope, on the scopes of the catalog."""
    return Toolkit(
        specs=[
            set_scope_tool.build(
                chat_ctx=chat_ctx, allowed_scope_ids=services.scope_catalog.scope_ids()
            )
        ]
    )
