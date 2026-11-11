"""Base abstraction of the LLM providers.

A provider is an adapter that builds the Strands model of one LLM backend
(Bedrock, any LiteLLM backend, ...) from a ModelEntry of the agent params.
"""

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strands.models import Model

    from agent.models.model_config import ModelEntry


class ModelProvider(abc.ABC):
    """Builds the Strands model of a ModelEntry and classifies its errors.

    Attributes:
        REQUIRED_PARAMS: Keys that the model params must contain for this provider.
    """

    REQUIRED_PARAMS: tuple[str, ...] = ()

    def __init__(self, model: "ModelEntry"):
        """Initializes the provider.

        Args:
            model: The model to build.
        """
        self._model = model

    @property
    def description(self) -> str:
        """Human readable identification of the model, used in logs."""
        return f"{self._model.provider}:{self._model.model_id}"

    @abc.abstractmethod
    def create_model(self) -> "Model":
        """Creates the Strands model instance."""

    @abc.abstractmethod
    def is_retryable(self, exception: BaseException) -> bool:
        """Tells whether an invocation error justifies switching to the next model.

        Args:
            exception: The exception raised by the model invocation.

        Returns:
            True if the error is a transient provider failure (unavailable, throttled, ...).
        """
