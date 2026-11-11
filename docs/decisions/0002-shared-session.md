# 0002 · Shared session and history projection

**Context.** When the turn moves from one agent to another, the new agent must know the
conversation. But the history contains calls to tools it does not have, and reasoning signed by
another model.

**Decision.** All the agents of a chat use the same Strands session (`agent_id="0"`). When an agent
is built, `project_foreign_tool_calls` turns the calls to foreign tools into text and removes the
reasoning blocks, in memory only. The results of the handovers keep the instruction to answer the
pending question. Only one agent runs at a time.

**Rejected alternatives.**
- One session per agent with a summary at each handover: loses detail and costs one more call.
- Passing the raw history: the model tries to call tools it does not have, and a different model
  rejects signed reasoning.

**Consequences.** The handover is invisible to the user and costs nothing extra. The projection must
know the handover and search tools (`set_scope`, `request_rerouting`, search).
