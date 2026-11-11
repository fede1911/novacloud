"""What a tool builder returns: the Strands tool and the prompt it adds to the system prompt."""

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class ToolSpec:
    """A tool ready for an agent.

    Attributes:
        tool: The function decorated with @strands.tool.
        prompt_name: Prompt appended to the system prompt of the agent, or None.
            Tools with the same prompt (e.g. the search tools of two products) add it once.
    """

    tool: Any
    prompt_name: str | None
