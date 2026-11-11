# 0001 · Explicit orchestrator

**Context.** A single agent with every tool of every product would have a huge prompt and would
pick tools poorly. Specialists are needed, and something to coordinate them.

**Decision.** A triage agent picks the scopes with `set_scope`; a specialist can hand the turn back
with `request_rerouting`. The tools write the request into the agent state and stop the agent;
`AgentOrchestrator` reads the state and performs the handover, always within the same turn. The turn
is a loop: it runs the current agent and, while an agent asks for a handover (`requested_handover`),
builds the other one and lets it continue. At most one rerouting per turn, hence at most three
handovers: triage → specialist → triage → specialist.

**Rejected alternatives.**
- Strands Swarm/Graph: less code, but handovers that are less visible and less controllable (limits,
  text to discard, state on disk).
- "Agents as tools" (the triage calls the specialist as a tool): the triage would stay in the middle
  of every answer, with more latency and tokens.

**Consequences.** The state machine is explicit and testable without an LLM. In exchange there is an
orchestration module to maintain (about 330 lines including docstrings), and every handover costs
one more model call.
