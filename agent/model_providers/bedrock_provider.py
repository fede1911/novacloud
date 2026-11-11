"""Amazon Bedrock provider: credentials come from the standard AWS chain (e.g. AWS_PROFILE)."""

from strands.models import BedrockModel

from agent.model_providers import base


class BedrockProvider(base.ModelProvider):
    """Builds a BedrockModel, with optional tools caching.

    Required params: region_name.
    Optional params: streaming (default true). Turn it off for the models that
    do not support tool use while streaming, such as the Mistral ones: they
    need it for structured output.
    """

    REQUIRED_PARAMS = ("region_name",)
    CACHE_TOOLS_TYPE = "default"
    DEFAULT_STREAMING = True
    RETRYABLE_ERRORS = (
        "ServiceUnavailable",
        "ModelThrottledException",
        "ThrottlingException",
    )

    @property
    def description(self) -> str:
        """Model and region, used in logs."""
        return f"{super().description}@{self._model.params['region_name']}"

    def create_model(self) -> BedrockModel:
        """Creates the BedrockModel of the entry."""
        return BedrockModel(
            model_id=self._model.model_id,
            region_name=self._model.params["region_name"],
            temperature=self._model.temperature,
            streaming=self._model.params.get("streaming", self.DEFAULT_STREAMING),
            cache_tools=self.CACHE_TOOLS_TYPE if self._model.prompt_caching else None,
        )

    def is_retryable(self, exception: BaseException) -> bool:
        """Retries on unavailability and throttling of the model."""
        return any(
            error in f"{type(exception).__name__} {exception}"
            for error in self.RETRYABLE_ERRORS
        )
