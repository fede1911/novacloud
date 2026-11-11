"""Configuration of the agent, read from the environment.

The LLMs are not part of this configuration: they are in config/models.json,
one list of models per agent role (see agent.models.model_config).
"""

import dataclasses
import os

from shared import settings as shared_settings

CONTEXT_WINDOW_ENV_VAR = "BLUEPRINT_CONTEXT_WINDOW_SIZE"
# Messages kept in the conversation window of every agent
DEFAULT_CONTEXT_WINDOW_SIZE = 20


@dataclasses.dataclass(frozen=True)
class AgentConfig:
    """Configuration of the agent.

    Attributes:
        settings: Paths of the blueprint.
        context_window_size: Messages kept in the conversation window.
    """

    settings: shared_settings.Settings
    context_window_size: int

    @classmethod
    def from_environment(cls) -> "AgentConfig":
        """Builds the configuration from the environment variables."""
        return cls(
            settings=shared_settings.Settings.from_environment(),
            context_window_size=int(
                os.getenv(CONTEXT_WINDOW_ENV_VAR) or DEFAULT_CONTEXT_WINDOW_SIZE
            ),
        )
