"""Support tickets on JSON files: swap this store to open them on a real helpdesk."""

import datetime as dt
import pathlib
import uuid

import pydantic


class Ticket(pydantic.BaseModel):
    """A support ticket opened by the agent."""

    ticket_id: str
    chat_id: str
    scope_id: str
    requester_name: str
    requester_email: str
    subject: str
    description: str
    created_at: dt.datetime


class TicketStore:
    """Writes every ticket as <tickets_dir>/<ticket_id>.json."""

    def __init__(self, tickets_dir: pathlib.Path):
        """Initializes the store.

        Args:
            tickets_dir: Folder of the ticket files, created if missing.
        """
        self._tickets_dir = tickets_dir
        self._tickets_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        *,
        chat_id: str,
        scope_id: str,
        requester_name: str,
        requester_email: str,
        subject: str,
        description: str,
    ) -> Ticket:
        """Creates and stores a ticket.

        Returns:
            The stored ticket, with its new id.
        """
        ticket = Ticket(
            ticket_id=f"NC-{uuid.uuid4().hex[:8].upper()}",
            chat_id=chat_id,
            scope_id=scope_id,
            requester_name=requester_name,
            requester_email=requester_email,
            subject=subject,
            description=description,
            created_at=dt.datetime.now(dt.UTC),
        )
        path = self._tickets_dir / f"{ticket.ticket_id}.json"
        path.write_text(ticket.model_dump_json(indent=2), encoding="utf-8")
        return ticket
