"""system_health_check_<scope>: status of the services of one product, from a simulated status page."""

import json
import pathlib
from typing import TYPE_CHECKING, Any, Literal

import pydantic
import strands

from agent.tools import tool_result, tool_spec
from shared import logger

if TYPE_CHECKING:
    from agent.models.chat import ChatContext
    from agent.toolkits import ToolServices
    from shared.scope_catalog import Scope

_LOGGER = logger.get_logger(__name__)

TOOL_NAME_PREFIX = "system_health_check_"

ResourceStatus = Literal[
    "operational", "degraded", "maintenance", "downtime", "not_monitored"
]


class Resource(pydantic.BaseModel):
    """Status of one resource of a service."""

    resource_name: str
    status: ResourceStatus
    note: str | None = None


def build(
    *, scope: "Scope", chat_ctx: "ChatContext", services: "ToolServices"
) -> tool_spec.ToolSpec:
    """Builds the health check tool of one product.

    The status page is re-read at every call: edit data/systems_status.json
    during a demo to simulate an incident.

    Args:
        scope: Product whose services are checked.
        chat_ctx: Context of the chat.
        services: Services of the tools; uses the path of the status page.
    """

    @strands.tool(
        context=True,
        name=f"{TOOL_NAME_PREFIX}{scope.scope_id}",
        description=f"Checks the health status of the {scope.label} services: ongoing incidents and maintenance.",
    )
    def systems_health(tool_context: strands.ToolContext) -> dict[str, Any]:
        """Check if there are incidents or maintenance on the service the user is asking about.

        Explain the result in plain words, without technical details. Status values:
        operational, degraded, maintenance, downtime, not_monitored.

        Args:
            tool_context: The tool context provided by Strands.
        """
        tool_use_id = tool_context.tool_use["toolUseId"]
        try:
            resources = read_status(
                services.systems_status_path, scope_id=scope.scope_id
            )
        except (OSError, ValueError):
            _LOGGER.exception("Status page unreadable (chat %s)", chat_ctx.chat_id)
            return tool_result.error(
                "The status page is not available: the status of the service is unknown.",
                tool_use_id=tool_use_id,
            )

        payload = {
            "resource_status": [
                resource.model_dump(exclude_none=True) for resource in resources
            ]
        }
        return tool_result.success({"json": payload}, tool_use_id=tool_use_id)

    return tool_spec.ToolSpec(tool=systems_health, prompt_name=None)


def read_status(status_path: pathlib.Path, *, scope_id: str) -> list[Resource]:
    """Returns the resources of a product from the status page.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the file is not valid.
    """
    status_page = json.loads(status_path.read_text(encoding="utf-8"))
    return [Resource.model_validate(raw) for raw in status_page.get(scope_id, [])]
