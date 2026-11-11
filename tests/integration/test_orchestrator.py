"""Tests of the orchestrator with scripted agents and a chat store on a temporary folder."""

from unittest import mock

import pytest
from strands.agent import state as agent_state

from agent import agent_builder, orchestrator, turn_events
from agent.models import model_config
from agent.stores import chat_store
from agent.tools.request_rerouting import request_rerouting_tool
from agent.tools.set_scope import set_scope_tool

pytestmark = pytest.mark.integration


class _ScriptedAgent:
    """Stand-in of strands.Agent: writes a text and sets some state keys when streamed."""

    def __init__(self, *, text: str, state_updates: dict):
        self.state = agent_state.AgentState()
        self.prompts: list[str | None] = []
        self._text = text
        self._state_updates = state_updates

    async def stream_async(self, prompt: str | None):
        self.prompts.append(prompt)
        for key, value in self._state_updates.items():
            self.state.set(key, value)
        if self._text:
            yield {"data": self._text}
            yield {"message": {"role": "assistant", "content": [{"text": self._text}]}}


class _ScriptedFactory:
    """Gives the scripted agents in order and records how the scope agents were built."""

    def __init__(
        self, *, triage_agents: list[_ScriptedAgent], scope_agents: list[_ScriptedAgent]
    ):
        self._triage_agents = triage_agents
        self._scope_agents = scope_agents
        self.scope_builds: list[tuple[list[str], bool]] = []

    def build_triage_agent(self, *, chat_ctx, models):
        return agent_builder.BuiltAgent(
            agent=self._triage_agents.pop(0), prompt_names=["triage_system_prompt"]
        )

    def build_scope_agent(self, *, chat_ctx, models, rerouting_enabled):
        self.scope_builds.append((chat_ctx.scope_ids, rerouting_enabled))
        return agent_builder.BuiltAgent(
            agent=self._scope_agents.pop(0), prompt_names=["scope_agent_system_prompt"]
        )


@pytest.fixture
def store(tmp_path) -> chat_store.ChatStore:
    return chat_store.ChatStore(tmp_path / "chats")


@pytest.fixture
def model_catalog() -> mock.Mock:
    models = [
        model_config.ModelEntry(provider="bedrock", model_id="model", temperature=0.0)
    ]
    return mock.Mock(for_role=mock.Mock(return_value=models))


def _orchestrator(
    factory: _ScriptedFactory, model_catalog, store
) -> orchestrator.AgentOrchestrator:
    return orchestrator.AgentOrchestrator(
        agent_factory=factory, model_catalog=model_catalog, chat_store=store
    )


async def _collect(
    agent_orchestrator, chat_ctx, prompt: str
) -> list[turn_events.TurnEvent]:
    return [event async for event in agent_orchestrator.run_turn(chat_ctx, prompt)]


class TestAgentOrchestrator:
    """The orchestrator hands the turn over between triage and scope agent."""

    async def test_triage_choice_is_answered_by_scope_agent_in_same_turn(
        self, chat_ctx, model_catalog, store
    ):
        """Test: the triage calls set_scope.
        Expected: the scope agent continues the turn without a new prompt, its text is the answer,
        the chat moves to the scope phase with scopes chosen by the triage.
        """
        # Arrange
        triage = _ScriptedAgent(
            text="",
            state_updates={set_scope_tool.SELECTED_SCOPE_IDS_STATE_KEY: ["novadrive"]},
        )
        scope_agent = _ScriptedAgent(
            text="Riprendi la sincronizzazione.", state_updates={}
        )
        factory = _ScriptedFactory(triage_agents=[triage], scope_agents=[scope_agent])

        # Act
        events = await _collect(
            _orchestrator(factory, model_catalog, store), chat_ctx, "non sincronizza"
        )

        # Assert
        result = events[-1].result
        state = store.get_state(chat_ctx.chat_id)
        assert (result.agent_type, result.scope_ids, result.answer) == (
            "scope",
            ["novadrive"],
            "Riprendi la sincronizzazione.",
        )
        assert scope_agent.prompts == [None]
        assert factory.scope_builds == [(["novadrive"], True)]
        assert (
            state.phase,
            state.active_scope.scope_ids,
            state.active_scope.source,
        ) == ("scope", ["novadrive"], "triage")

    async def test_caller_scopes_skip_triage_and_rerouting(
        self, chat_ctx, model_catalog, store
    ):
        """Test: scopes fixed by the caller.
        Expected: no triage, scope agent built without request_rerouting.
        """
        # Arrange
        fixed_ctx = chat_ctx.model_copy(update={"scope_ids": ["novabill"]})
        factory = _ScriptedFactory(
            triage_agents=[],
            scope_agents=[_ScriptedAgent(text="Ecco la fattura.", state_updates={})],
        )

        # Act
        events = await _collect(
            _orchestrator(factory, model_catalog, store), fixed_ctx, "fattura"
        )

        # Assert
        assert events[-1].result.agent_type == "scope"
        assert factory.scope_builds == [(["novabill"], False)]

    async def test_triage_clarification_keeps_triage_phase(
        self, chat_ctx, model_catalog, store
    ):
        """Test: the triage asks a clarification instead of choosing.
        Expected: answer of the triage, chat still in the triage phase without scopes.
        """
        # Arrange
        factory = _ScriptedFactory(
            triage_agents=[_ScriptedAgent(text="Quale prodotto?", state_updates={})],
            scope_agents=[],
        )

        # Act
        events = await _collect(
            _orchestrator(factory, model_catalog, store), chat_ctx, "ho un problema"
        )

        # Assert
        state = store.get_state(chat_ctx.chat_id)
        assert (events[-1].result.agent_type, events[-1].result.answer) == (
            "triage",
            "Quale prodotto?",
        )
        assert (state.phase, state.active_scope) == ("triage", None)

    async def test_text_before_handover_is_discarded(
        self, chat_ctx, model_catalog, store
    ):
        """Test: the triage writes a preamble before set_scope.
        Expected: handover event flags the discarded text, the answer has only the scope agent text.
        """
        # Arrange
        triage = _ScriptedAgent(
            text="Ti passo allo specialista.",
            state_updates={set_scope_tool.SELECTED_SCOPE_IDS_STATE_KEY: ["novaid"]},
        )
        factory = _ScriptedFactory(
            triage_agents=[triage],
            scope_agents=[
                _ScriptedAgent(text="Vai su Password dimenticata.", state_updates={})
            ],
        )

        # Act
        events = await _collect(
            _orchestrator(factory, model_catalog, store), chat_ctx, "password"
        )

        # Assert
        handovers = [
            event for event in events if isinstance(event, turn_events.Handover)
        ]
        assert handovers == [
            turn_events.Handover(
                from_agent="triage",
                to_agent="scope",
                scope_ids=["novaid"],
                discarded_text=True,
            )
        ]
        assert events[-1].result.answer == "Vai su Password dimenticata."

    async def test_rerouting_in_a_later_turn_goes_through_triage_once(
        self, chat_ctx, model_catalog, store
    ):
        """Test: second turn off topic for the scope agent, which asks for a rerouting.
        Expected: triage then new scope agent in the same turn, built without further rerouting allowed.
        """
        # Arrange
        first_triage = _ScriptedAgent(
            text="",
            state_updates={set_scope_tool.SELECTED_SCOPE_IDS_STATE_KEY: ["novadrive"]},
        )
        drive_agent = _ScriptedAgent(text="Crea il link.", state_updates={})
        # Every turn builds its agent from the session: turn 2 gets a new NovaDrive agent
        drive_agent_turn_2 = _ScriptedAgent(
            text="",
            state_updates={request_rerouting_tool.REROUTING_REQUESTED_STATE_KEY: True},
        )
        second_triage = _ScriptedAgent(
            text="",
            state_updates={set_scope_tool.SELECTED_SCOPE_IDS_STATE_KEY: ["novabill"]},
        )
        bill_agent = _ScriptedAgent(text="Scarica la fattura.", state_updates={})
        factory = _ScriptedFactory(
            triage_agents=[first_triage, second_triage],
            scope_agents=[drive_agent, drive_agent_turn_2, bill_agent],
        )
        agent_orchestrator = _orchestrator(factory, model_catalog, store)
        await _collect(agent_orchestrator, chat_ctx, "link condivisione")

        # Act
        events = await _collect(agent_orchestrator, chat_ctx, "e la fattura?")

        # Assert
        handovers = [
            (event.from_agent, event.to_agent, event.scope_ids)
            for event in events
            if isinstance(event, turn_events.Handover)
        ]
        assert handovers == [("scope", "triage", []), ("triage", "scope", ["novabill"])]
        assert events[-1].result.answer == "Scarica la fattura."
        assert (
            bill_agent.state.get(request_rerouting_tool.REROUTING_ALLOWED_STATE_KEY)
            is False
        )

    async def test_rerouting_right_after_triage_is_handled(
        self, chat_ctx, model_catalog, store
    ):
        """Test: in the first turn the triage picks NovaDrive, whose agent asks for a rerouting at once.
        Expected: triage again, then the NovaBill agent answers in the same turn, without further rerouting.
        """
        # Arrange
        first_triage = _ScriptedAgent(
            text="",
            state_updates={set_scope_tool.SELECTED_SCOPE_IDS_STATE_KEY: ["novadrive"]},
        )
        drive_agent = _ScriptedAgent(
            text="",
            state_updates={request_rerouting_tool.REROUTING_REQUESTED_STATE_KEY: True},
        )
        second_triage = _ScriptedAgent(
            text="",
            state_updates={set_scope_tool.SELECTED_SCOPE_IDS_STATE_KEY: ["novabill"]},
        )
        bill_agent = _ScriptedAgent(text="Scarica la fattura.", state_updates={})
        factory = _ScriptedFactory(
            triage_agents=[first_triage, second_triage],
            scope_agents=[drive_agent, bill_agent],
        )

        # Act
        events = await _collect(
            _orchestrator(factory, model_catalog, store), chat_ctx, "fattura del drive"
        )

        # Assert
        handovers = [
            (event.from_agent, event.to_agent, event.scope_ids)
            for event in events
            if isinstance(event, turn_events.Handover)
        ]
        assert handovers == [
            ("triage", "scope", ["novadrive"]),
            ("scope", "triage", []),
            ("triage", "scope", ["novabill"]),
        ]
        assert events[-1].result.answer == "Scarica la fattura."
        assert (
            bill_agent.state.get(request_rerouting_tool.REROUTING_ALLOWED_STATE_KEY)
            is False
        )

    async def test_handover_cap_stops_without_building_an_unused_agent(
        self, chat_ctx, model_catalog, store
    ):
        """Test: agents that keep asking for handovers, as if the one-rerouting rule were broken.
        Expected: the turn stops after the third handover, with the answer of the last agent that ran.
        """

        # Arrange
        def triage(scope_ids):
            return _ScriptedAgent(
                text="",
                state_updates={set_scope_tool.SELECTED_SCOPE_IDS_STATE_KEY: scope_ids},
            )

        def rerouting_scope_agent(text):
            return _ScriptedAgent(
                text=text,
                state_updates={
                    request_rerouting_tool.REROUTING_REQUESTED_STATE_KEY: True
                },
            )

        factory = _ScriptedFactory(
            triage_agents=[
                triage(["novadrive"]),
                triage(["novabill"]),
                triage(["novaid"]),
            ],
            scope_agents=[
                rerouting_scope_agent(""),
                rerouting_scope_agent("Risposta NovaBill."),
            ],
        )

        # Act
        events = await _collect(
            _orchestrator(factory, model_catalog, store), chat_ctx, "domanda"
        )

        # Assert
        handovers = [
            event for event in events if isinstance(event, turn_events.Handover)
        ]
        assert len(handovers) == 3
        assert (events[-1].result.agent_type, events[-1].result.scope_ids) == (
            "scope",
            ["novabill"],
        )
        assert store.get_state(chat_ctx.chat_id).phase == "scope"
