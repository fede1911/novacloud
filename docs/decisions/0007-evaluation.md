# 0007 · Evaluation

**Context.** In a multi-agent system the bugs are often in the orchestration: missed handovers,
skipped searches, tickets opened when not needed. Calling the model in isolation does not reveal them.

**Decision.**
- The benchmark runs scripted conversations on the real orchestrator.
- Deterministic metrics where possible: routing, escalation, expected tools.
- An LLM judge for accuracy and completeness, from another family than the agents (Mistral), with a
  structured answer validated by Pydantic.
- A separate evaluation of the retrieval alone, without LLMs: for every query the expected section
  (`document#section`, stable even when the chunking changes), with recall@1, recall@k, MRR and the
  rejection of negative cases, computed for several thresholds on the same results.

**Evidence.** The benchmark found three problems that the unit tests did not see:
- a missed rerouting, because the specialist did not know the other products;
- an instruction lost in the history projection;
- a wrong reasoning by Haiku ([0005](0005-models-per-role.md)).

The retrieval evaluation showed that the initial threshold (0.2) let off-domain questions through.
It also showed that questions about another product cannot be separated by a threshold: that is the
job of the triage and of the rerouting.

**Rejected alternatives.**
- A user simulated by an LLM: more realistic, but not reproducible.
- Parsing the score from text: fragile.

**Consequences.**
- The judge is independent of the evaluated models ([0005](0005-models-per-role.md)), but less
  strict than Sonnet on blatant errors.
- 10 scenarios are enough for a demo, not for decisions.
- The benchmark does not run in CI because it needs AWS credentials.
