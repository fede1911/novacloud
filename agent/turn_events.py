"""Events streamed by the orchestrator during a turn, and the recorder that builds them.

The recorder lives for one turn and turns the raw Strands stream events into
typed turn events, so the callers (CLI, benchmark) never parse Strands events.
"""

import dataclasses
import time
from typing import Any

from agent.models import chat as chat_models


@dataclasses.dataclass(frozen=True)
class TextDelta:
    """A chunk of text written by the current agent."""

    text: str


@dataclasses.dataclass(frozen=True)
class ToolStarted:
    """The current agent called a tool."""

    tool_name: str


@dataclasses.dataclass(frozen=True)
class Handover:
    """The turn passed from one agent to the other.

    Attributes:
        from_agent: Agent that handed the turn over.
        to_agent: Agent that takes the turn.
        scope_ids: Scopes of the agent that takes the turn (none for the triage).
        discarded_text: Whether the text written before the handover was discarded:
            it is not the answer, a frontend has to remove it.
    """

    from_agent: chat_models.AgentType
    to_agent: chat_models.AgentType
    scope_ids: list[str]
    discarded_text: bool


@dataclasses.dataclass(frozen=True)
class TurnResult:
    """Outcome of a turn.

    Attributes:
        chat_id: Chat of the turn.
        answer: Text of the answer saved in the history.
        agent_type: Agent that wrote the answer.
        scope_ids: Scopes of the agent that wrote the answer.
        tools_used: Tool calls of the turn, of every agent.
        handovers: Handovers of the turn, in order.
        input_tokens: Input tokens of every model call of the turn.
        output_tokens: Output tokens of every model call of the turn.
    """

    chat_id: str
    answer: str
    agent_type: chat_models.AgentType
    scope_ids: list[str]
    tools_used: list[chat_models.ToolUsed]
    handovers: list[Handover]
    input_tokens: int
    output_tokens: int


@dataclasses.dataclass(frozen=True)
class TurnCompleted:
    """Last event of a turn."""

    result: TurnResult


TurnEvent = TextDelta | ToolStarted | Handover | TurnCompleted


class TurnRecorder:
    """Records text, tool calls and token usage of one turn from the Strands stream events."""

    def __init__(self) -> None:
        """Initializes an empty recording."""
        self._texts: list[str] = []
        self._pending_tools: dict[str, tuple[str, dict, float]] = {}
        self.tools_used: list[chat_models.ToolUsed] = []
        self.handovers: list[Handover] = []
        self.input_tokens = 0
        self.output_tokens = 0

    def observe(self, event: dict[str, Any]) -> list[TurnEvent]:
        """Records a Strands stream event and returns the turn events it produces."""
        if event.get("data"):
            return [TextDelta(text=event["data"])]

        if "message" in event:
            return self._observe_message(event["message"])

        if "result" in event:
            invocation = event["result"].metrics.latest_agent_invocation
            if invocation is not None:
                self.input_tokens += invocation.usage.get("inputTokens", 0)
                self.output_tokens += invocation.usage.get("outputTokens", 0)

        return []

    def discard_text(self) -> bool:
        """Discards the text written so far, returning whether there was any."""
        had_text = any(text.strip() for text in self._texts)
        self._texts = []
        return had_text

    def answer(self) -> str:
        """Returns the text written since the last discard."""
        return "\n".join(text.strip() for text in self._texts if text.strip())

    def _observe_message(self, message: dict[str, Any]) -> list[TurnEvent]:
        # Complete messages only: the text of a retried model call is never in them
        turn_events: list[TurnEvent] = []
        for block in message.get("content", []):
            if message["role"] == "assistant" and block.get("text"):
                self._texts.append(block["text"])
            elif message["role"] == "assistant" and "toolUse" in block:
                tool_use = block["toolUse"]
                self._pending_tools[tool_use["toolUseId"]] = (
                    tool_use["name"],
                    tool_use.get("input") or {},
                    time.monotonic(),
                )
                turn_events.append(ToolStarted(tool_name=tool_use["name"]))
            elif message["role"] == "user" and "toolResult" in block:
                self._record_tool_result(block["toolResult"])
        return turn_events

    def _record_tool_result(self, tool_result: dict[str, Any]) -> None:
        pending = self._pending_tools.pop(tool_result["toolUseId"], None)
        if pending is None:
            return
        tool_name, tool_input, started_at = pending
        self.tools_used.append(
            chat_models.ToolUsed(
                tool_name=tool_name,
                input=tool_input,
                status=tool_result.get("status", "unknown"),
                total_seconds=round(time.monotonic() - started_at, 3),
            )
        )
