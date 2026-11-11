"""Orchestrator of the agents of a chat.

A chat without scopes from the caller starts with the triage agent, which
chooses the scopes with the set_scope tool; the scope agent of those scopes
then answers in the same turn. If the question is not about its scopes, the
scope agent gives the conversation back to the triage with request_rerouting,
at most once per turn. Chats with the scopes fixed by the caller use only the
scope agent.
"""

import datetime as dt
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol

from agent import turn_events
from agent.models import chat as chat_models
from agent.tools.request_rerouting import request_rerouting_tool
from agent.tools.set_scope import set_scope_tool
from shared import logger

if TYPE_CHECKING:
    import strands
    from strands.agent.state import AgentState

    from agent.agent_builder import BuiltAgent
    from agent.models.model_config import ModelCatalog, ModelEntry
    from agent.stores.chat_store import ChatStore

_LOGGER = logger.get_logger(__name__)

# Most handovers in one turn: triage -> scope -> triage (the one rerouting) -> scope.
# The rerouting rule already stops the loop there; the cap guards against a broken invariant.
MAX_HANDOVERS_PER_TURN = 3


class AgentFactory(Protocol):
    """Builds the agents of a chat (implemented by AgentBuilder)."""

    def build_scope_agent(
        self,
        *,
        chat_ctx: chat_models.ChatContext,
        models: list["ModelEntry"],
        rerouting_enabled: bool,
    ) -> "BuiltAgent":
        """Builds the scope agent of the scopes of chat_ctx."""

    def build_triage_agent(
        self, *, chat_ctx: chat_models.ChatContext, models: list["ModelEntry"]
    ) -> "BuiltAgent":
        """Builds the triage agent."""


class AgentOrchestrator:
    """Chooses the agent of each turn and hands the turn over between the triage and the scope agent.

    The agents of a chat share one session and run one after the other, never
    two at the same time. The orchestrator keeps no agent in memory: every turn
    and every handover builds the agent from the session, so a turn depends only
    on what is stored and any instance can serve it.
    """

    def __init__(
        self,
        *,
        agent_factory: AgentFactory,
        model_catalog: "ModelCatalog",
        chat_store: "ChatStore",
    ):
        """Initializes the orchestrator.

        Args:
            agent_factory: Builds the agents.
            model_catalog: Models of each agent role.
            chat_store: Stores state and history of the chats.
        """
        self._agent_factory = agent_factory
        self._model_catalog = model_catalog
        self._chat_store = chat_store

    async def run_turn(
        self, chat_ctx: chat_models.ChatContext, prompt: str
    ) -> AsyncIterator[turn_events.TurnEvent]:
        """Answers a user message with the agent of the chat phase and saves the answer.

        Args:
            chat_ctx: Context of the chat.
            prompt: The user message.

        Yields:
            The events of the turn; the last one is TurnCompleted.
        """
        chat_id = chat_ctx.chat_id
        self._chat_store.init_chat(chat_id=chat_id, user=chat_ctx.user)
        state = self._chat_store.get_state(chat_id)
        # The scopes of the caller win over the stored ones
        if chat_ctx.scope_ids and (
            state.active_scope is None
            or state.active_scope.scope_ids != chat_ctx.scope_ids
        ):
            self._chat_store.update_active_scope(
                chat_id, scope_ids=chat_ctx.scope_ids, source="payload"
            )
            state = self._chat_store.get_state(chat_id)

        agent_type, scope_ids = self.current_agent(chat_ctx, state)
        if state.phase != agent_type:
            self._chat_store.update_phase(chat_id, agent_type)

        self._chat_store.add_message(
            chat_id,
            chat_models.HistoryItem(
                role="user", message=prompt, message_datetime=_now()
            ),
        )
        recorder = turn_events.TurnRecorder()

        # The turn is a loop: stream the current agent, then hand the turn over while an agent asks for it
        rerouted = False
        agent = self._start_agent(
            chat_ctx, agent_type, scope_ids, rerouting_allowed=True
        )
        pending_prompt: str | None = prompt
        for handovers_done in range(MAX_HANDOVERS_PER_TURN + 1):
            async for event in self._stream(agent, pending_prompt, recorder):
                yield event
            # The next agent answers the question already in the session
            pending_prompt = None

            handover = self.requested_handover(agent_type, agent.state)
            if handover is None:
                break
            if handovers_done == MAX_HANDOVERS_PER_TURN:
                # Checked before handing over: the turn ends with the agent that just ran
                _LOGGER.error(
                    "Chat %s asked for more than %s handovers in one turn: stopped",
                    chat_id,
                    MAX_HANDOVERS_PER_TURN,
                )
                break
            next_type, next_scope_ids = handover
            if next_type == chat_models.TRIAGE_AGENT:
                # Only one rerouting per turn: the next scope agent cannot ask for another
                rerouted = True
            else:
                self._chat_store.update_active_scope(
                    chat_id, scope_ids=next_scope_ids, source="triage"
                )
            yield self._hand_over(
                chat_id,
                recorder,
                from_agent=agent_type,
                to_agent=next_type,
                scope_ids=next_scope_ids,
            )
            agent_type, scope_ids = next_type, next_scope_ids
            agent = self._start_agent(
                chat_ctx, agent_type, scope_ids, rerouting_allowed=not rerouted
            )

        answer = recorder.answer()
        self._chat_store.add_message(
            chat_id,
            chat_models.HistoryItem(
                role="assistant",
                message=answer,
                message_datetime=_now(),
                agent_type=agent_type,
                scope_ids=scope_ids,
                tools_used=recorder.tools_used,
            ),
        )
        yield turn_events.TurnCompleted(
            result=turn_events.TurnResult(
                chat_id=chat_id,
                answer=answer,
                agent_type=agent_type,
                scope_ids=scope_ids,
                tools_used=recorder.tools_used,
                handovers=recorder.handovers,
                input_tokens=recorder.input_tokens,
                output_tokens=recorder.output_tokens,
            )
        )

    @staticmethod
    def current_agent(
        chat_ctx: chat_models.ChatContext, state: chat_models.ChatState
    ) -> tuple[chat_models.AgentType, list[str]]:
        """Returns the agent that answers the user message and its scopes.

        Args:
            chat_ctx: Context of the chat.
            state: State of the chat stored in the history.

        Returns:
            The agent type and the scopes of the agent (none for the triage).
        """
        # Scopes fixed by the caller: they cannot change, no triage
        if chat_ctx.scope_ids:
            return chat_models.SCOPE_AGENT, chat_ctx.scope_ids

        if state.phase == chat_models.SCOPE_AGENT and state.active_scope is not None:
            return chat_models.SCOPE_AGENT, state.active_scope.scope_ids

        return chat_models.TRIAGE_AGENT, []

    @staticmethod
    def requested_handover(
        agent_type: chat_models.AgentType, agent_state: "AgentState"
    ) -> tuple[chat_models.AgentType, list[str]] | None:
        """Returns the agent the current one hands the turn over to, with its scopes, or None.

        Args:
            agent_type: Type of the agent that just streamed.
            agent_state: Its state, where the handover tools leave their requests.
        """
        if agent_type == chat_models.SCOPE_AGENT and agent_state.get(
            request_rerouting_tool.REROUTING_REQUESTED_STATE_KEY
        ):
            return chat_models.TRIAGE_AGENT, []
        selected_scope_ids = agent_state.get(
            set_scope_tool.SELECTED_SCOPE_IDS_STATE_KEY
        )
        if agent_type == chat_models.TRIAGE_AGENT and selected_scope_ids:
            return chat_models.SCOPE_AGENT, selected_scope_ids
        return None

    @staticmethod
    async def _stream(
        agent: "strands.Agent", prompt: str | None, recorder: turn_events.TurnRecorder
    ) -> AsyncIterator[turn_events.TurnEvent]:
        """Streams an agent; with no prompt the agent answers the question already in the history."""
        async for event in agent.stream_async(prompt):
            for turn_event in recorder.observe(event):
                yield turn_event

    def _hand_over(
        self,
        chat_id: str,
        recorder: turn_events.TurnRecorder,
        *,
        from_agent: chat_models.AgentType,
        to_agent: chat_models.AgentType,
        scope_ids: list[str],
    ) -> turn_events.Handover:
        """Moves the chat to the next agent; the text written before the handover is not the answer."""
        self._chat_store.update_phase(chat_id, to_agent)
        discarded_text = recorder.discard_text()
        if discarded_text:
            _LOGGER.warning(
                "The %s agent of chat %s wrote text before handing over: discarded",
                from_agent,
                chat_id,
            )
        handover = turn_events.Handover(
            from_agent=from_agent,
            to_agent=to_agent,
            scope_ids=scope_ids,
            discarded_text=discarded_text,
        )
        recorder.handovers.append(handover)
        _LOGGER.info(
            "Chat %s handed over from %s to %s %s",
            chat_id,
            from_agent,
            to_agent,
            scope_ids,
        )
        return handover

    def _start_agent(
        self,
        chat_ctx: chat_models.ChatContext,
        agent_type: chat_models.AgentType,
        scope_ids: list[str],
        *,
        rerouting_allowed: bool,
    ) -> "strands.Agent":
        """Builds the agent of the given type and scopes from the session, ready for this turn.

        Args:
            chat_ctx: Context of the chat.
            agent_type: Type of the agent.
            scope_ids: Scopes of the agent (none for the triage).
            rerouting_allowed: Whether the agent can still hand the turn back to the triage.
        """
        agent = self._build_agent(chat_ctx, agent_type, scope_ids)
        # The agent state is restored from the session: the handover requests are valid only in their turn
        agent.state.delete(set_scope_tool.SELECTED_SCOPE_IDS_STATE_KEY)
        agent.state.delete(request_rerouting_tool.REROUTING_REQUESTED_STATE_KEY)
        agent.state.set(
            request_rerouting_tool.REROUTING_ALLOWED_STATE_KEY, rerouting_allowed
        )
        return agent

    def _build_agent(
        self,
        chat_ctx: chat_models.ChatContext,
        agent_type: chat_models.AgentType,
        scope_ids: list[str],
    ) -> "strands.Agent":
        """Builds the agent with the models of its role, recording models and prompts in the chat."""
        agent_chat_ctx = chat_ctx.model_copy(update={"scope_ids": scope_ids})
        # The agent types are also the roles of config/models.json
        models = self._model_catalog.for_role(agent_type)
        if agent_type == chat_models.TRIAGE_AGENT:
            built = self._agent_factory.build_triage_agent(
                chat_ctx=agent_chat_ctx, models=models
            )
        else:
            # With the scopes fixed by the caller the chat never goes back to the triage
            built = self._agent_factory.build_scope_agent(
                chat_ctx=agent_chat_ctx,
                models=models,
                rerouting_enabled=not chat_ctx.scope_ids,
            )

        self._chat_store.record_models(
            chat_ctx.chat_id,
            chat_models.ModelsUsed(
                agent_type=agent_type, model_ids=[model.model_id for model in models]
            ),
        )
        self._chat_store.record_prompts(chat_ctx.chat_id, built.prompt_names)
        return built.agent


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)
