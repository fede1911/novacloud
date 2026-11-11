# Towards production

The blueprint runs locally with file-based stores. In production the architecture stays the same:
the implementations behind the same interfaces change, and what is missing today is added.

## What to replace

| Component | Today | In production |
|---|---|---|
| interface | CLI (`agent/cli.py`) | streaming HTTP API (SSE) that consumes the same `TurnEvent`s |
| execution | local process | containers (ECS/Fargate) or a managed agent runtime; the orchestrator is already stateless between turns, so any instance can serve a request |
| sessions | `FileSessionManager`, `var/sessions/` | the Strands `S3SessionManager` |
| chat history | `ChatStore` on JSON | a document database or DynamoDB, with a lock per chat |
| tickets | `TicketStore` on JSON | the helpdesk API |
| status page | `data/systems_status.json` | the status page API |
| index | `chunks.json`, `vectors.npy` and `index_meta.json` in `var/index/` | a managed vector store (OpenSearch, pgvector) with the same model metadata on the collection; scheduled, incremental ingestion; blue/green reindexing when the model changes |
| configuration | `config/*.json`, environment variables | parameter store and secrets manager; versioned prompts and models |
| reranking | none | a managed reranker, when the documents grow |

Thanks to the composition root ([0004](decisions/0004-composition-root-and-toolkits.md)) every
replacement touches one store or service and `agent/runtime.py`, not the orchestrator.

## What to add

- **Security.**
  - Authenticated requests, with the real user in the `ChatContext`.
  - Guardrails on input and output against prompt injection and off-policy content.
  - Mask or leave out the personal data in the tool inputs saved in the history: today the email of
    the tickets ends up there.
- **Observability.**
  - OpenTelemetry tracing per turn, agent and tool.
  - Tokens and cost per turn and per role.
  - Metrics on handovers, reroutings, fallbacks and empty responses.
  - Structured logs with the `chat_id`.
- **Reliability.**
  - Tool timeouts and a cap on the cycles of an agent.
  - Safe concurrency on the same chat.
  - Quotas and rate limits per user.
- **Continuous quality.**
  - A larger benchmark dataset, with adversarial cases; several judges, or a stricter judge
    calibrated on a hand-labelled sample.
  - The benchmark run before every release of prompts or models, with minimum thresholds.
  - User feedback stored next to the chat history.
