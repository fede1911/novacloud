"""Models of each agent role, from config/models.json.

Every role (triage, scope, judge) has an ordered list of models: the first
is used by default, the next ones are the fallbacks tried when a model is
unavailable or throttled.
"""

import pathlib
from typing import Any, Literal

import pydantic

from agent import model_factory

ModelRole = Literal["triage", "scope", "judge"]


class ModelEntry(pydantic.BaseModel):
    """One model of a role.

    Attributes:
        provider: Name of the provider, a key of model_factory.PROVIDERS (e.g. "bedrock").
        model_id: Model identifier understood by the provider.
        temperature: Sampling temperature, or None to not send it.
        prompt_caching: Whether prompt caching is requested; ignored by the
            providers that do not support it.
        params: Provider-specific settings (e.g. region_name for Bedrock).
    """

    provider: str
    model_id: str = pydantic.Field(min_length=1)
    temperature: float | None = pydantic.Field(ge=0)
    prompt_caching: bool = False
    params: dict[str, Any] = pydantic.Field(default_factory=dict)


class RoleModels(pydantic.BaseModel):
    """The models of one role, in order of preference."""

    description: str
    models: list[ModelEntry] = pydantic.Field(min_length=1)


class ModelCatalog(pydantic.RootModel[dict[ModelRole, RoleModels]]):
    """Content of config/models.json: one entry per role."""

    @classmethod
    def from_file(cls, path: pathlib.Path) -> "ModelCatalog":
        """Loads and validates the catalog.

        Raises:
            pydantic.ValidationError: If the file does not match the schema.
            ValueError: If a role is missing or a model has an unknown provider or missing params.
        """
        catalog = cls.model_validate_json(path.read_text(encoding="utf-8"))
        missing_roles = sorted(set(ModelRole.__args__) - set(catalog.root))
        if missing_roles:
            raise ValueError(f"{path} has no models for the roles {missing_roles}")
        for role_models in catalog.root.values():
            for model in role_models.models:
                model_factory.validate_model_entry(model)
        return catalog

    def for_role(self, role: ModelRole) -> list[ModelEntry]:
        """Returns the models of a role, first the default one."""
        return self.root[role].models
