"""Tests of the open ticket tool."""

from unittest import mock

import pytest

from agent.tools.open_ticket import open_ticket_tool

pytestmark = pytest.mark.unit

VALID_FIELDS = {
    "requester_name": "Giulia Bianchi",
    "requester_email": "giulia@example.com",
    "subject": "Addebito doppio",
    "description": "Due addebiti del piano Plus il 3 marzo, entrambi contabilizzati.",
}


@pytest.fixture
def ticket_store() -> mock.Mock:
    return mock.Mock(create=mock.Mock(return_value=mock.Mock(ticket_id="NC-1234ABCD")))


@pytest.fixture
def open_ticket(chat_ctx, ticket_store, catalog):
    services = mock.Mock(ticket_store=ticket_store)
    return open_ticket_tool.build(
        scope=catalog.get("novabill"), chat_ctx=chat_ctx, services=services
    ).tool


class TestOpenTicketTool:
    """The tool validates the fields before storing the ticket."""

    def test_valid_fields_open_ticket(self, open_ticket, tool_context, ticket_store):
        """Test: valid fields.
        Expected: ticket stored for the scope and its id returned.
        """
        # Act
        result = open_ticket(tool_context=tool_context, **VALID_FIELDS)

        # Assert
        assert result["content"][0]["json"] == {
            "ticket_id": "NC-1234ABCD",
            "status": "open",
        }
        ticket_store.create.assert_called_once_with(
            chat_id="chat-1", scope_id="novabill", **VALID_FIELDS
        )

    def test_invalid_email_returns_field_error(
        self, open_ticket, tool_context, ticket_store
    ):
        """Test: invalid email.
        Expected: error result naming the field, no ticket stored.
        """
        # Act
        result = open_ticket(
            tool_context=tool_context,
            **{**VALID_FIELDS, "requester_email": "giulia-at-example"},
        )

        # Assert
        assert result["status"] == "error"
        assert list(result["content"][1]["json"]["errors"]) == ["requester_email"]
        ticket_store.create.assert_not_called()
