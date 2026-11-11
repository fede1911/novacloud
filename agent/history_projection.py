"""Projection of the tool calls of other agents in the history loaded from the session.

The triage and the scope agents share one session history. Each agent sees
the toolUse/toolResult blocks of tools it does not have as text blocks, so
that the model neither calls them again nor reads the documentation of
another product. The projection is in memory only: the session stays raw.

The reasoningContent blocks are dropped too: their content is signed for the
model that wrote it, and another model rejects it.
"""

from typing import Any

from agent.tools.request_rerouting import request_rerouting_tool
from agent.tools.search_documentation import search_documentation_tool
from agent.tools.set_scope import set_scope_tool

ContentBlock = dict[str, Any]
Message = dict[str, Any]

REASONING_PLACEHOLDER = "[Ragionamento non mostrato]"


def project_foreign_tool_calls(
    messages: list[Message], registered_tool_names: set[str]
) -> tuple[list[Message], int]:
    """Turns the tool calls of the tools not registered in the agent into text blocks.

    Number and order of the messages do not change, only their blocks: the
    session manager appends the new messages by their position. The
    projection is idempotent; the calls of the registered tools stay structured.

    Args:
        messages: Messages of the agent, as loaded from the session.
        registered_tool_names: Names of the tools of the agent.

    Returns:
        The projected messages and the number of converted blocks.
    """
    tool_uses_by_id = {
        block["toolUse"]["toolUseId"]: block["toolUse"]
        for message in messages
        for block in message["content"]
        if "toolUse" in block
    }

    projected_messages = []
    converted_blocks = 0
    for message in messages:
        content = []
        for block in message["content"]:
            if "reasoningContent" in block:
                converted_blocks += 1
                continue
            projected_block = _project_block(
                block, tool_uses_by_id, registered_tool_names
            )
            if projected_block is not block:
                converted_blocks += 1
            content.append(projected_block)
        if not content:
            # A message cannot be empty, and the number of messages must not change
            content = [{"text": REASONING_PLACEHOLDER}]
        projected_messages.append({**message, "content": content})

    return projected_messages, converted_blocks


def _project_block(
    block: ContentBlock,
    tool_uses_by_id: dict[str, dict[str, Any]],
    registered_tool_names: set[str],
) -> ContentBlock:
    """Returns the block as is, or its text version if it is a call of a foreign tool."""
    if "toolUse" in block:
        tool_use = block["toolUse"]
        if tool_use["name"] in registered_tool_names:
            return block
        return {"text": _tool_use_text(tool_use["name"], tool_use.get("input") or {})}

    if "toolResult" in block:
        # A result whose call is not in the history (e.g. trimmed by the window) is foreign too
        tool_use = tool_uses_by_id.get(block["toolResult"]["toolUseId"])
        tool_name = tool_use["name"] if tool_use else None
        if tool_name in registered_tool_names:
            return block
        return {"text": _tool_result_text(tool_name)}

    return block


def _tool_use_text(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Placeholder of a foreign toolUse block."""
    if tool_name.startswith(search_documentation_tool.TOOL_NAME_PREFIX):
        scope_id = tool_name.removeprefix(search_documentation_tool.TOOL_NAME_PREFIX)
        return f"[Ricerca nella documentazione {scope_id}: «{tool_input.get('question', '')}»]"
    if tool_name == set_scope_tool.TOOL_NAME:
        return (
            f"[Conversazione affidata a: {', '.join(tool_input.get('scope_ids', []))}]"
        )
    if tool_name == request_rerouting_tool.TOOL_NAME:
        return f"[Argomento fuori dal prodotto attivo: {tool_input.get('reason', '')}]"
    # The input of the other tools is left out: it can hold personal data (e.g. an email)
    return f"[Strumento usato: {tool_name}]"


def _tool_result_text(tool_name: str | None) -> str:
    """Placeholder of a foreign toolResult block.

    The handover results keep the instruction to answer the pending question:
    it is the last message the next agent reads, and without it the model
    tends to answer without calling its tools.
    """
    if tool_name and tool_name.startswith(search_documentation_tool.TOOL_NAME_PREFIX):
        return "[Risultati non mostrati: documentazione di un altro prodotto]"
    if tool_name == set_scope_tool.TOOL_NAME:
        return f"[Passaggio eseguito] {set_scope_tool.HANDOVER_RESULT_TEXT}"
    if tool_name == request_rerouting_tool.TOOL_NAME:
        return f"[Ritorno allo smistamento] {set_scope_tool.HANDOVER_RESULT_TEXT}"
    return "[Risultato non mostrato]"
