"""open_ticket_<scope>: opens a support ticket for the team of one product."""

import re
from typing import TYPE_CHECKING, Any

import pydantic
import strands

from agent.tools import tool_result, tool_spec
from shared import logger

if TYPE_CHECKING:
    from agent.models.chat import ChatContext
    from agent.toolkits import ToolServices
    from shared.scope_catalog import Scope

_LOGGER = logger.get_logger(__name__)

TOOL_NAME_PREFIX = "open_ticket_"
PROMPT_NAME = "tool_open_ticket"
VALIDATION_ERROR_TEXT = (
    "Input parameter validation failed. Check the invalid parameters in the json below."
)
_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
MIN_DESCRIPTION_LENGTH = 20


class TicketRequest(pydantic.BaseModel):
    """Data of a ticket, validated before it is opened: the model reads the errors and asks the user."""

    requester_name: str = pydantic.Field(min_length=2)
    requester_email: str
    subject: str = pydantic.Field(min_length=5)
    description: str = pydantic.Field(min_length=MIN_DESCRIPTION_LENGTH)

    @pydantic.field_validator("requester_name")
    @classmethod
    def _name_without_digits(cls, value: str) -> str:
        if any(char.isdigit() for char in value):
            raise ValueError("The name can only contain letters.")
        return value

    @pydantic.field_validator("requester_email")
    @classmethod
    def _valid_email(cls, value: str) -> str:
        if not _EMAIL_PATTERN.match(value):
            raise ValueError("Invalid email address.")
        return value


def build(
    *, scope: "Scope", chat_ctx: "ChatContext", services: "ToolServices"
) -> tool_spec.ToolSpec:
    """Builds the tool that opens a ticket for the support team of one product.

    Args:
        scope: Product of the support team that receives the ticket.
        chat_ctx: Context of the chat.
        services: Services of the tools; uses the ticket store.
    """

    @strands.tool(
        context=True,
        name=f"{TOOL_NAME_PREFIX}{scope.scope_id}",
        description=f"Opens a support ticket for the {scope.label} team, when the documentation does not solve the issue.",
    )
    def open_ticket(
        tool_context: strands.ToolContext,
        requester_name: str,
        requester_email: str,
        subject: str,
        description: str,
    ) -> dict[str, Any]:
        """Open a support ticket. Ask the user every field you do not know: never invent them.

        Args:
            tool_context: The tool context provided by Strands.
            requester_name: First and last name of the user.
            requester_email: Email address where the user wants to be contacted.
            subject: Short title of the issue.
            description: The issue, what the user already tried and any error message.
        """
        tool_use_id = tool_context.tool_use["toolUseId"]
        try:
            request = TicketRequest(
                requester_name=requester_name,
                requester_email=requester_email,
                subject=subject,
                description=description,
            )
        except pydantic.ValidationError as e:
            errors = {str(error["loc"][0]): error["msg"] for error in e.errors()}
            _LOGGER.info(
                "Ticket validation failed on fields %s (chat %s)",
                sorted(errors),
                chat_ctx.chat_id,
            )
            return tool_result.error(
                [{"text": VALIDATION_ERROR_TEXT}, {"json": {"errors": errors}}],
                tool_use_id=tool_use_id,
            )

        ticket = services.ticket_store.create(
            chat_id=chat_ctx.chat_id, scope_id=scope.scope_id, **request.model_dump()
        )
        _LOGGER.info(
            "Ticket %s opened for %s (chat %s)",
            ticket.ticket_id,
            scope.scope_id,
            chat_ctx.chat_id,
        )
        return tool_result.success(
            {"json": {"ticket_id": ticket.ticket_id, "status": "open"}},
            tool_use_id=tool_use_id,
        )

    return tool_spec.ToolSpec(tool=open_ticket, prompt_name=PROMPT_NAME)
