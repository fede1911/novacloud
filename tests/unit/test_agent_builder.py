"""Tests of the composition of the scope agent system prompt."""

from unittest import mock

import pytest

from agent import agent_builder, prompt_loader, toolkits
from agent.tools import tool_spec

pytestmark = pytest.mark.unit


@pytest.fixture
def loader() -> mock.Mock:
    prompts = {
        "scope_agent_system_prompt": "Prodotti:\n{{scope_catalog}}",
        "scope_additional_infos_novaid": "Info NovaID",
        "tool_search_documentation": "Come cercare",
    }

    def get_prompt(prompt_name: str) -> str:
        if prompt_name not in prompts:
            raise prompt_loader.PromptNotFoundError(prompt_name)
        return prompts[prompt_name]

    return mock.Mock(get_prompt=mock.Mock(side_effect=get_prompt))


class TestComposeScopeSystemPrompt:
    """The scope prompt lists every product, the infos of the active ones and the tool prompts."""

    def test_prompt_has_catalog_product_sections_and_tool_prompts(self, loader):
        """Test: two active products, one without additional infos, and one tool with a prompt.
        Expected: catalog rendered, one section per product, the tool prompt last.
        """
        # Arrange
        toolkit = toolkits.Toolkit(
            specs=[
                tool_spec.ToolSpec(
                    tool=mock.Mock(), prompt_name="tool_search_documentation"
                )
            ]
        )

        # Act
        system_prompt, prompt_names = agent_builder.compose_scope_system_prompt(
            scope_ids=["novaid", "novadrive"],
            loader=loader,
            catalog_text="- novaid\n- novadrive\n- novabill",
            toolkit=toolkit,
        )

        # Assert
        assert system_prompt == (
            "Prodotti:\n- novaid\n- novadrive\n- novabill\n\n"
            "## Prodotto attivo: novaid\nInfo NovaID\n\n"
            "## Prodotto attivo: novadrive\n\n"
            "Come cercare"
        )
        assert prompt_names == [
            "scope_agent_system_prompt",
            "scope_additional_infos_novaid",
            "tool_search_documentation",
        ]
