# 0003 · Two memories

**Context.** The model needs the complete messages, tool calls included. The application needs a
queryable state: phase, chosen scopes and who chose them, tools used, models.

**Decision.** The Strands session (`FileSessionManager`) serves the model. `ChatStore` records the
application state and a readable history. The orchestrator decides only on the latter.

**Rejected alternatives.** Deriving the state from the session messages: fragile, because it depends
on the message format of the framework and on the context window that trims them.

**Consequences.** Data is partly duplicated, but each memory evolves on its own: the framework can be
replaced without losing analytics and audit. Both stores are file based, so they are meant for a
single process ([production](../production.md)).
