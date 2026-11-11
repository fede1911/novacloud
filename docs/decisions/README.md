# Architecture decisions

| # | Decision |
|---|---|
| [0001](0001-explicit-orchestrator.md) | Hand-written orchestrator, handovers through tools and state |
| [0002](0002-shared-session.md) | One session shared by the agents, with history projection |
| [0003](0003-two-memories.md) | Model session kept apart from the application chat history |
| [0004](0004-composition-root-and-toolkits.md) | Composition root, tools as functions, toolkits declared in config |
| [0005](0005-models-per-role.md) | Models per role with fallbacks; Haiku for the triage, Sonnet for the specialists, Mistral as judge |
| [0006](0006-rag.md) | Titan, model metadata and compatibility check, single-step vector search |
| [0007](0007-evaluation.md) | Benchmark on the real orchestrator, deterministic metrics, structured judge, retrieval evaluation |
