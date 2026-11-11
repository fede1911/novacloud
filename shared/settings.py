"""Paths and settings shared by the agent, the ingestion and the benchmark.

Everything is read from environment variables and from files under the
project folder: the environment is read only here and in AgentConfig.
"""

import dataclasses
import os
import pathlib

# Root of the blueprint, overridable to run it from another location
HOME_ENV_VAR = "BLUEPRINT_HOME"
# Folder with the generated state (index, sessions, chats, tickets, logs, runs)
RUNTIME_DIR_ENV_VAR = "BLUEPRINT_RUNTIME_DIR"
LOG_LEVEL_ENV_VAR = "BLUEPRINT_LOG_LEVEL"
DEFAULT_LOG_LEVEL = "INFO"


@dataclasses.dataclass(frozen=True)
class Settings:
    """Locations of the configuration, the prompts, the data and the runtime state.

    Attributes:
        home_dir: Root folder of the blueprint.
        runtime_dir: Folder with the state generated at runtime.
        log_level: Level of the file log.
    """

    home_dir: pathlib.Path
    runtime_dir: pathlib.Path
    log_level: str

    @classmethod
    def from_environment(cls) -> "Settings":
        """Builds the settings from the environment, falling back to the blueprint folder."""
        home_dir = pathlib.Path(
            os.getenv(HOME_ENV_VAR) or pathlib.Path(__file__).resolve().parents[1]
        )
        runtime_dir = pathlib.Path(os.getenv(RUNTIME_DIR_ENV_VAR) or home_dir / "var")
        return cls(
            home_dir=home_dir,
            runtime_dir=runtime_dir,
            log_level=os.getenv(LOG_LEVEL_ENV_VAR) or DEFAULT_LOG_LEVEL,
        )

    @property
    def config_dir(self) -> pathlib.Path:
        """Folder with the configuration files."""
        return self.home_dir / "config"

    @property
    def models_path(self) -> pathlib.Path:
        """LLMs of every agent role, with their fallbacks."""
        return self.config_dir / "models.json"

    @property
    def scopes_path(self) -> pathlib.Path:
        """Catalog of the supported products."""
        return self.config_dir / "scopes.json"

    @property
    def kb_config_path(self) -> pathlib.Path:
        """Embedders, chunking and retrieval of the knowledge base."""
        return self.config_dir / "knowledge_base.json"

    @property
    def prompts_dir(self) -> pathlib.Path:
        """Folder with the system prompts, one Markdown file per prompt."""
        return self.home_dir / "prompts"

    @property
    def data_dir(self) -> pathlib.Path:
        """Folder with the source documents and the mocked external systems."""
        return self.home_dir / "data"

    @property
    def kb_source_dir(self) -> pathlib.Path:
        """Folder with the knowledge base documents, one subfolder per scope."""
        return self.data_dir / "kb"

    @property
    def systems_status_path(self) -> pathlib.Path:
        """Mocked status page of the services."""
        return self.data_dir / "systems_status.json"

    @property
    def index_dir(self) -> pathlib.Path:
        """Folder with the knowledge base index built by the ingestion."""
        return self.runtime_dir / "index"

    @property
    def sessions_dir(self) -> pathlib.Path:
        """Folder with the Strands sessions, one per chat."""
        return self.runtime_dir / "sessions"

    @property
    def chats_dir(self) -> pathlib.Path:
        """Folder with the chat histories, one JSON file per chat."""
        return self.runtime_dir / "chats"

    @property
    def tickets_dir(self) -> pathlib.Path:
        """Folder with the opened support tickets."""
        return self.runtime_dir / "tickets"

    @property
    def logs_dir(self) -> pathlib.Path:
        """Folder with the application log."""
        return self.runtime_dir / "logs"

    @property
    def benchmark_runs_dir(self) -> pathlib.Path:
        """Folder with the benchmark reports."""
        return self.runtime_dir / "benchmark_runs"
