"""Hook that falls back to the next model of the role on transient provider errors."""

from typing import TYPE_CHECKING

from strands import hooks

from agent import model_factory
from shared import logger

if TYPE_CHECKING:
    from agent.models.model_config import ModelEntry

_LOGGER = logger.get_logger(__name__)


class ModelFallbackRetry(hooks.HookProvider):
    """Switches the agent to the next configured model when an invocation fails with a retryable error.

    Attributes:
        models: The models to try, in order of preference.
        model_index: Position in models of the model currently in use.
    """

    def __init__(self, models: list["ModelEntry"]):
        """Initializes the hook on the first model.

        Args:
            models: The models of the role, first the default one.
        """
        self.models = models
        self.model_index = 0

    def register_hooks(self, registry: hooks.HookRegistry) -> None:
        """Registers the fallback after every model call."""
        registry.add_callback(hooks.AfterModelCallEvent, self.switch_model)

    async def switch_model(self, event: hooks.AfterModelCallEvent) -> None:
        """Switches to the next model and asks for a retry if the call failed with a retryable error.

        When the failed model was the last one, the agent goes back to the first
        model without retrying, so the error reaches the caller.
        """
        if not event.exception:
            return

        failed_provider = model_factory.create_provider(
            self.models, model_index=self.model_index
        )
        if not failed_provider.is_retryable(event.exception):
            return

        if self.model_index >= len(self.models) - 1:
            self.model_index = 0
            event.agent.model = model_factory.create_model(
                self.models, model_index=self.model_index
            )
            _LOGGER.error(
                "Model %s failed and no fallback is left: %s",
                failed_provider.description,
                event.exception,
            )
            return

        self.model_index += 1
        next_provider = model_factory.create_provider(
            self.models, model_index=self.model_index
        )
        _LOGGER.warning(
            "Model %s failed (%s): switching to %s",
            failed_provider.description,
            event.exception,
            next_provider.description,
        )
        event.agent.model = model_factory.create_model(
            self.models, model_index=self.model_index
        )
        event.retry = True
