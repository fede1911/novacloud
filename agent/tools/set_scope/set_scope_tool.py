"""set_scope: the triage hands the conversation over to the scope agent of the chosen products."""

from typing import TYPE_CHECKING, Any

import strands

from agent.tools import tool_result, tool_spec
from shared import logger

if TYPE_CHECKING:
    from agent.models.chat import ChatContext

_LOGGER = logger.get_logger(__name__)

TOOL_NAME = "set_scope"
# Key of the agent state where the tool leaves the chosen scopes for the orchestrator
SELECTED_SCOPE_IDS_STATE_KEY = "selected_scope_ids"
# Most scopes the triage can choose at once, as stated in the triage prompt
MAX_SCOPE_IDS = 2
# Result of a handover: the next agent reads it as the last message before answering
HANDOVER_RESULT_TEXT = "Il passaggio è avvenuto. Ora rispondi tu alla domanda più recente dell'utente, usando i tuoi strumenti."


def build(
    *, chat_ctx: "ChatContext", allowed_scope_ids: list[str]
) -> tool_spec.ToolSpec:
    """Builds the set_scope tool of the triage.

    A valid call stops the event loop of the triage and leaves the scopes in
    the agent state: the orchestrator then builds the scope agent, which
    answers the pending question in the same turn. An invalid call returns an
    error without stopping the loop, so the model can correct it.

    Args:
        chat_ctx: Context of the chat.
        allowed_scope_ids: Scopes the triage can route to.
    """

    @strands.tool(
        context=True,
        name=TOOL_NAME,
        description="Routes the conversation to the specialist of the given NovaCloud products, who answers the pending request.",
    )
    def set_scope(
        tool_context: strands.ToolContext, scope_ids: list[str]
    ) -> dict[str, Any]:
        """Route the conversation to the specialist of the chosen products.

        Args:
            tool_context: The tool context provided by Strands.
            scope_ids: Identifiers of the products the request needs, from 1 to 2.
        """
        tool_use_id = tool_context.tool_use["toolUseId"]
        validation_error = validate_scope_ids(
            scope_ids, allowed_scope_ids=allowed_scope_ids
        )
        if validation_error is not None:
            _LOGGER.warning(
                "Invalid set_scope call in chat %s: %s",
                chat_ctx.chat_id,
                validation_error,
            )
            return tool_result.error(validation_error, tool_use_id=tool_use_id)

        # Duplicates are dropped, keeping the order chosen by the model
        selected_scope_ids = list(dict.fromkeys(scope_ids))
        tool_context.agent.state.set(SELECTED_SCOPE_IDS_STATE_KEY, selected_scope_ids)
        tool_context.invocation_state.setdefault("request_state", {})[
            "stop_event_loop"
        ] = True
        _LOGGER.info(
            "Triage of chat %s routed to %s", chat_ctx.chat_id, selected_scope_ids
        )
        return tool_result.success(HANDOVER_RESULT_TEXT, tool_use_id=tool_use_id)

    return tool_spec.ToolSpec(tool=set_scope, prompt_name=None)


def validate_scope_ids(
    scope_ids: list[str], *, allowed_scope_ids: list[str]
) -> str | None:
    """Returns the error message for the model, or None if the scopes are valid."""
    if not scope_ids:
        return "scope_ids must contain at least one scope."

    if len(set(scope_ids)) > MAX_SCOPE_IDS:
        return f"scope_ids can contain at most {MAX_SCOPE_IDS} scopes: ask a clarification question instead."

    unknown_scope_ids = [
        scope_id for scope_id in scope_ids if scope_id not in allowed_scope_ids
    ]
    if unknown_scope_ids:
        return f"Unknown scopes: {unknown_scope_ids}. Use only these identifiers: {allowed_scope_ids}."

    return None
