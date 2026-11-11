"""Creates the providers and the Strands models of the configured LLMs."""

from typing import TYPE_CHECKING

from agent.model_providers import bedrock_provider, litellm_provider

if TYPE_CHECKING:
    from strands.models import Model

    from agent.model_providers.base import ModelProvider
    from agent.models.model_config import ModelEntry

# Providers by the name used in config/models.json: add a provider here
PROVIDERS: dict[str, type["ModelProvider"]] = {
    "bedrock": bedrock_provider.BedrockProvider,
    "litellm": litellm_provider.LiteLLMProvider,
}


def get_provider_class(provider_name: str) -> type["ModelProvider"]:
    """Returns the provider class of a provider name.

    Raises:
        ValueError: If the provider is unknown.
    """
    if provider_name not in PROVIDERS:
        raise ValueError(
            f"Unknown model provider '{provider_name}'. Available: {sorted(PROVIDERS)}"
        )
    return PROVIDERS[provider_name]


def validate_model_entry(model: "ModelEntry") -> None:
    """Checks that the provider of a model exists and its required params are set.

    Raises:
        ValueError: If the provider is unknown or a required param is missing.
    """
    provider_class = get_provider_class(model.provider)
    missing_params = [
        key for key in provider_class.REQUIRED_PARAMS if key not in model.params
    ]
    if missing_params:
        raise ValueError(
            f"Model '{model.model_id}' of provider '{model.provider}' is missing required params: {missing_params}"
        )


def create_provider(models: list["ModelEntry"], *, model_index: int) -> "ModelProvider":
    """Builds the provider of the model at the given position of the fallback list."""
    model = models[model_index]
    return get_provider_class(model.provider)(model)


def create_model(models: list["ModelEntry"], *, model_index: int) -> "Model":
    """Builds the Strands model at the given position of the fallback list."""
    return create_provider(models, model_index=model_index).create_model()
