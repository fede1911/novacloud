"""Catalog of the scopes (products) the agent supports, from config/scopes.json.

It is the single source of truth of the products: the prompts list them,
set_scope validates the model choice against them, and every product
declares the tools of its scope agent.
"""

import json
import pathlib
from typing import Literal

import pydantic

# Tools a product can declare; agent.toolkits builds them
ProductToolName = Literal["search_documentation", "systems_health", "open_ticket"]


class Scope(pydantic.BaseModel):
    """One product the agent can answer on.

    Attributes:
        scope_id: Identifier of the product, used in tool names and prompt names.
        label: Commercial name of the product.
        description: What the product covers, shown to the models.
        tools: Tools of the scope agent for this product.
    """

    scope_id: str = pydantic.Field(pattern=r"^[a-z0-9-]+$")
    label: str
    description: str
    tools: list[ProductToolName] = pydantic.Field(min_length=1)


class ScopeCatalog:
    """The supported scopes, in the order of the configuration file."""

    def __init__(self, scopes: list[Scope]):
        """Initializes the catalog.

        Args:
            scopes: The supported scopes.
        """
        self._scopes = scopes

    @classmethod
    def from_file(cls, path: pathlib.Path) -> "ScopeCatalog":
        """Loads the catalog from a JSON list of scopes."""
        raw_scopes = json.loads(path.read_text(encoding="utf-8"))
        return cls([Scope.model_validate(raw_scope) for raw_scope in raw_scopes])

    def scope_ids(self) -> list[str]:
        """Returns the ids of the supported scopes."""
        return [scope.scope_id for scope in self._scopes]

    def get(self, scope_id: str) -> Scope:
        """Returns the scope with the given id.

        Raises:
            KeyError: If the scope is not in the catalog.
        """
        for scope in self._scopes:
            if scope.scope_id == scope_id:
                return scope
        raise KeyError(f"Unknown scope '{scope_id}'")

    def render_for_prompt(self) -> str:
        """Returns the scopes as a Markdown list for the triage prompt."""
        return "\n".join(
            f"- `{scope.scope_id}` ({scope.label}): {scope.description}"
            for scope in self._scopes
        )
