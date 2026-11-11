"""Loads the system prompts from Markdown files, one file per prompt."""

import pathlib

from shared import logger

_LOGGER = logger.get_logger(__name__)
PROMPT_SUFFIX = ".md"


class PromptNotFoundError(LookupError):
    """Raised when a prompt file does not exist."""


class PromptLoader:
    """Reads the prompt <name>.md of the prompts folder."""

    def __init__(self, prompts_dir: pathlib.Path):
        """Initializes the loader.

        Args:
            prompts_dir: Folder with one Markdown file per prompt.
        """
        self._prompts_dir = prompts_dir

    def get_prompt(self, prompt_name: str) -> str:
        """Returns the text of a prompt.

        Args:
            prompt_name: Name of the prompt, without extension.

        Returns:
            The prompt text, stripped.

        Raises:
            PromptNotFoundError: If the prompt file does not exist.
        """
        prompt_path = self._prompts_dir / f"{prompt_name}{PROMPT_SUFFIX}"
        try:
            text = prompt_path.read_text(encoding="utf-8")
        except FileNotFoundError as e:
            raise PromptNotFoundError(
                f"Prompt '{prompt_name}' not found in {self._prompts_dir}"
            ) from e
        _LOGGER.debug("Loaded prompt %s", prompt_name)
        return text.strip()
