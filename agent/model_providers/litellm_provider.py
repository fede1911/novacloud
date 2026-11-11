"""LiteLLM provider: any backend supported by LiteLLM (OpenAI, Databricks, Ollama, ...).

LiteLLM is not in requirements.txt: install it (pip install "strands-agents[litellm]")
only to use this provider. Endpoint and credentials are read by LiteLLM from
the backend environment variables (e.g. OPENAI_API_KEY).
"""

from typing import TYPE_CHECKING

from agent.model_providers import base

if TYPE_CHECKING:
    from strands.models import Model


class LiteLLMProvider(base.ModelProvider):
    """Builds a LiteLLMModel; the model_id prefix selects the backend (e.g. "ollama/llama3.1").

    The params are passed as model arguments (e.g. max_tokens); the model
    temperature overrides any temperature param. Prompt caching is ignored.
    """

    def create_model(self) -> "Model":
        """Creates the LiteLLMModel of the entry."""
        # Imported lazily: LiteLLM is optional and heavy
        from strands.models import litellm as strands_litellm

        params = {**self._model.params}
        if self._model.temperature is None:
            params.pop("temperature", None)
        else:
            params["temperature"] = self._model.temperature
        return strands_litellm.LiteLLMModel(
            model_id=self._model.model_id, params=params
        )

    def is_retryable(self, exception: BaseException) -> bool:
        """Retries on rate limits, unavailability and connection errors."""
        import litellm

        retryable_errors = (
            litellm.RateLimitError,
            litellm.ServiceUnavailableError,
            litellm.BadGatewayError,
            litellm.InternalServerError,
            litellm.APIConnectionError,
            litellm.Timeout,
        )
        return isinstance(exception, retryable_errors)
