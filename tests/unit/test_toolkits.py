"""Tests of the toolkits of the scope agent and of the triage."""

import pathlib
from unittest import mock

import pytest

from agent import toolkits

pytestmark = pytest.mark.unit


@pytest.fixture
def services(catalog, prompt_loader) -> toolkits.ToolServices:
    return toolkits.ToolServices(
        retriever=mock.Mock(),
        ticket_store=mock.Mock(),
        prompt_loader=prompt_loader,
        scope_catalog=catalog,
        systems_status_path=pathlib.Path("status.json"),
    )


def _tool_names(toolkit: toolkits.Toolkit) -> list[str]:
    return [tool.tool_name for tool in toolkit.tools()]


class TestScopeToolkit:
    """The scope agent gets the tools its products declare, and request_rerouting only when enabled."""

    def test_declared_tools_are_built_for_every_product(self, chat_ctx, services):
        """Test: two active products; NovaBill declares only the search tool.
        Expected: the declared tools of each product, in order, without request_rerouting.
        """
        # Arrange
        two_scopes_ctx = chat_ctx.model_copy(
            update={"scope_ids": ["novaid", "novabill"]}
        )

        # Act
        toolkit = toolkits.scope_toolkit(
            chat_ctx=two_scopes_ctx, services=services, rerouting_enabled=False
        )

        # Assert
        assert _tool_names(toolkit) == [
            "search-documentation-novaid",
            "system_health_check_novaid",
            "open_ticket_novaid",
            "search-documentation-novabill",
        ]

    def test_rerouting_enabled_adds_request_rerouting(self, chat_ctx, services):
        """Test: rerouting enabled (scopes chosen by the triage).
        Expected: request_rerouting is the last tool.
        """
        # Arrange
        one_scope_ctx = chat_ctx.model_copy(update={"scope_ids": ["novadrive"]})

        # Act
        toolkit = toolkits.scope_toolkit(
            chat_ctx=one_scope_ctx, services=services, rerouting_enabled=True
        )

        # Assert
        assert _tool_names(toolkit)[-1] == "request_rerouting"

    def test_shared_tool_prompts_are_listed_once(self, chat_ctx, services):
        """Test: two products whose search and ticket tools share their prompts.
        Expected: each prompt listed once, in order of appearance.
        """
        # Arrange
        two_scopes_ctx = chat_ctx.model_copy(
            update={"scope_ids": ["novaid", "novadrive"]}
        )

        # Act
        toolkit = toolkits.scope_toolkit(
            chat_ctx=two_scopes_ctx, services=services, rerouting_enabled=False
        )

        # Assert
        assert toolkit.prompt_names() == [
            "tool_search_documentation",
            "tool_open_ticket",
        ]


class TestTriageToolkit:
    """The triage gets only set_scope."""

    def test_triage_has_only_set_scope(self, chat_ctx, services):
        """Test: triage toolkit.
        Expected: the only tool is set_scope.
        """
        # Act
        toolkit = toolkits.triage_toolkit(chat_ctx=chat_ctx, services=services)

        # Assert
        assert _tool_names(toolkit) == ["set_scope"]
