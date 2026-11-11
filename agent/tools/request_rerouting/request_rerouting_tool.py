"""request_rerouting: the scope agent gives the conversation back to the triage."""

from typing import TYPE_CHECKING, Any

import strands

from agent.tools import tool_result, tool_spec
from agent.tools.set_scope import set_scope_tool
from shared import logger

if TYPE_CHECKING:
    from agent.models.chat import ChatContext

_LOGGER = logger.get_logger(__name__)

TOOL_NAME = "request_rerouting"
# Key of the agent state where the tool leaves the request for the orchestrator
REROUTING_REQUESTED_STATE_KEY = "rerouting_requested"
# Key of the agent state where the orchestrator tells whether the turn can still be rerouted
REROUTING_ALLOWED_STATE_KEY = "rerouting_allowed"
# Result when the turn was already rerouted: the scope agent has to answer on its own
REROUTING_NOT_AVAILABLE_TEXT = (
    "In questo turno la conversazione è già stata passata una volta: rispondi con i tuoi strumenti "
    "oppure spiega che l'argomento non è di tua competenza."
)


def build(*, chat_ctx: "ChatContext") -> tool_spec.ToolSpec:
    """Builds the request_rerouting tool of the scope agent.

    A call stops the event loop of the scope agent and leaves the request in
    the agent state: the orchestrator then builds the triage, which chooses
    the scopes again. Only one rerouting per turn is allowed: after it the
    call returns a message without stopping the loop.

    Args:
        chat_ctx: Context of the chat.
    """

    @strands.tool(
        context=True,
        name=TOOL_NAME,
        description=(
            "Gives the conversation back to the triage, which routes it to the right NovaCloud product. "
            "Use it when the user's request is about a NovaCloud product that is not among the active ones."
        ),
    )
    def request_rerouting(
        tool_context: strands.ToolContext, reason: str
    ) -> dict[str, Any]:
        """Hand the conversation back to the triage.

        Args:
            tool_context: The tool context provided by Strands.
            reason: Why the request is not about the active products, in one short sentence.
        """
        tool_use_id = tool_context.tool_use["toolUseId"]
        state = tool_context.agent.state

        if not state.get(REROUTING_ALLOWED_STATE_KEY):
            _LOGGER.warning(
                "Rerouting not allowed in chat %s, reason: %s", chat_ctx.chat_id, reason
            )
            return tool_result.success(
                REROUTING_NOT_AVAILABLE_TEXT, tool_use_id=tool_use_id
            )

        state.set(REROUTING_REQUESTED_STATE_KEY, True)
        tool_context.invocation_state.setdefault("request_state", {})[
            "stop_event_loop"
        ] = True
        _LOGGER.info(
            "Rerouting requested in chat %s, reason: %s", chat_ctx.chat_id, reason
        )
        return tool_result.success(
            set_scope_tool.HANDOVER_RESULT_TEXT, tool_use_id=tool_use_id
        )

    return tool_spec.ToolSpec(tool=request_rerouting, prompt_name=None)
