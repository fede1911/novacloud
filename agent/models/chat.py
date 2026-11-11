"""Models of a chat: context of the request, state read by the orchestrator, stored history."""

import datetime as dt
from typing import Literal

import pydantic

# Agent that answers a turn: the triage chooses the scopes, the scope agent answers on them.
AgentType = Literal["triage", "scope"]
TRIAGE_AGENT: AgentType = "triage"
SCOPE_AGENT: AgentType = "scope"

# Who chose the active scopes: the caller (e.g. the --scope flag of the CLI) or the triage
ScopeSource = Literal["payload", "triage"]


class ChatContext(pydantic.BaseModel):
    """Invariant coordinates of a chat request.

    Attributes:
        chat_id: Identifier of the chat.
        user: Identifier of the user (never logged).
        scope_ids: Scopes fixed by the caller; empty to let the triage choose them.
    """

    chat_id: str = pydantic.Field(min_length=1)
    user: str
    scope_ids: list[str]


class ToolUsed(pydantic.BaseModel):
    """One tool call of a turn, as recorded in the history."""

    tool_name: str
    input: dict
    status: str
    total_seconds: float


class HistoryItem(pydantic.BaseModel):
    """One message of the chat history.

    Attributes:
        role: "user" or "assistant".
        message: Text of the message.
        message_datetime: When the message was written.
        agent_type: Agent that wrote the assistant message, None for the user ones.
        scope_ids: Scopes of the agent that wrote the assistant message.
        tools_used: Tool calls made to write the assistant message.
    """

    role: Literal["user", "assistant"]
    message: str
    message_datetime: dt.datetime
    agent_type: AgentType | None = None
    scope_ids: list[str] = pydantic.Field(default_factory=list)
    tools_used: list[ToolUsed] = pydantic.Field(default_factory=list)


class ActiveScope(pydantic.BaseModel):
    """Scopes of the chat and who chose them."""

    scope_ids: list[str]
    source: ScopeSource
    updated_at: dt.datetime


class ChatState(pydantic.BaseModel):
    """State of the chat read by the orchestrator at the start of a turn."""

    phase: AgentType | None = None
    active_scope: ActiveScope | None = None


class ModelsUsed(pydantic.BaseModel):
    """Models configured for an agent of the chat: the first one, then its fallbacks."""

    agent_type: AgentType
    model_ids: list[str]


class ChatRecord(ChatState):
    """Stored chat: its state, history and the configuration it used.

    Attributes:
        chat_id: Identifier of the chat.
        user: Identifier of the user.
        created_at: Creation date of the chat.
        history: Messages of the chat, oldest first.
        active_scope_history: Every change of active_scope, oldest first.
        models_used: Models of the agents of the chat, one entry per change.
        prompts: Names of the prompts used by the chat.
    """

    chat_id: str
    user: str
    created_at: dt.datetime
    history: list[HistoryItem] = pydantic.Field(default_factory=list)
    active_scope_history: list[ActiveScope] = pydantic.Field(default_factory=list)
    models_used: list[ModelsUsed] = pydantic.Field(default_factory=list)
    prompts: list[str] = pydantic.Field(default_factory=list)

    def to_state(self) -> ChatState:
        """Returns only the state read by the orchestrator."""
        return ChatState(
            phase=self.phase,
            active_scope=self.active_scope,
        )
