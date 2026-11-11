"""Results of the tools in the Strands format.

See https://strandsagents.com/latest/documentation/docs/user-guide/concepts/tools/custom-tools/
"""

from typing import Any

ToolContent = str | dict[str, Any] | list[dict[str, Any]]


def success(content: ToolContent, *, tool_use_id: str) -> dict[str, Any]:
    """Returns a successful tool result."""
    return _result("success", content, tool_use_id=tool_use_id)


def error(content: ToolContent, *, tool_use_id: str) -> dict[str, Any]:
    """Returns a failed tool result: the model reads the content and can correct the call."""
    return _result("error", content, tool_use_id=tool_use_id)


def _result(status: str, content: ToolContent, *, tool_use_id: str) -> dict[str, Any]:
    if isinstance(content, str):
        normalized = [{"text": content}]
    elif isinstance(content, dict):
        normalized = [content]
    else:
        normalized = content
    return {"toolUseId": tool_use_id, "status": status, "content": normalized}
