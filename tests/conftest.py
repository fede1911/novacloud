"""Shared fixtures of the blueprint tests."""

import types
from unittest import mock

import pytest
from strands.agent import state as agent_state

from agent.models import chat as chat_models
from shared import scope_catalog


@pytest.fixture
def chat_ctx() -> chat_models.ChatContext:
    """Chat without scopes fixed by the caller."""
    return chat_models.ChatContext(chat_id="chat-1", user="user-1", scope_ids=[])


@pytest.fixture
def catalog() -> scope_catalog.ScopeCatalog:
    """Catalog with the three demo scopes."""
    return scope_catalog.ScopeCatalog(
        [
            scope_catalog.Scope(
                scope_id="novaid",
                label="NovaID",
                description="account",
                tools=["search_documentation", "systems_health", "open_ticket"],
            ),
            scope_catalog.Scope(
                scope_id="novadrive",
                label="NovaDrive",
                description="storage",
                tools=["search_documentation", "systems_health", "open_ticket"],
            ),
            scope_catalog.Scope(
                scope_id="novabill",
                label="NovaBill",
                description="billing",
                tools=["search_documentation"],
            ),
        ]
    )


@pytest.fixture
def tool_context() -> types.SimpleNamespace:
    """Stand-in of strands.ToolContext with a real AgentState."""
    return types.SimpleNamespace(
        tool_use={"toolUseId": "tool-1"},
        agent=types.SimpleNamespace(state=agent_state.AgentState()),
        invocation_state={},
    )


@pytest.fixture
def prompt_loader() -> mock.Mock:
    """PromptLoader that returns "<prompt name> text" for every prompt."""
    loader = mock.Mock()
    loader.get_prompt.side_effect = lambda prompt_name: f"{prompt_name} text"
    return loader
