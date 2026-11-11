# 0005 · Models per role

**Context.** Triage, specialists and judge have different needs: speed, reasoning, stability.
Providers have outages and rate limits.

**Decision.**
- `config/models.json` gives every role an ordered list of models. `ModelFallbackRetry` moves to the
  next one **only** on transient errors.
- The triage uses Haiku 4.5, with a fallback on the same model in another region.
- Specialists use Sonnet 4.5, with a fallback on Haiku.
- The benchmark judge uses Mistral Pixtral Large, with a fallback on Mistral Large: another family,
  so it does not reward the style of the Claude models it evaluates. On Bedrock, Mistral models do
  not support tools while streaming, and structured output needs them: this is why the Bedrock
  provider accepts `"streaming": false`.

**Rejected alternatives.**
- Haiku everywhere: cheaper, but it got the reasoning on the conditions in the documents wrong.
- Sonnet for the triage too: with Haiku the routing was already exact, so it would only add latency.
- Sonnet as the judge: stricter, but it tends to reward its own style.
- Mistral Large as the main judge: it misses subtle errors, and such a judge inflates the scores.
