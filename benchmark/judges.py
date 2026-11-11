"""LLM-as-judge metrics: one model call per metric, answered as a validated Pydantic object."""

import pathlib
from typing import TYPE_CHECKING

import pydantic
import strands
import yaml
from strands.types import exceptions as strands_exceptions

from agent import model_factory
from agent.hooks import model_retry
from shared import logger

if TYPE_CHECKING:
    from agent.models.model_config import ModelEntry

_LOGGER = logger.get_logger(__name__)

JUDGE_METRICS = ("accuracy", "completeness")


class Judgement(pydantic.BaseModel):
    """Verdict of the judge on one metric."""

    score: int = pydantic.Field(
        ge=1, le=10, description="Voto da 1 (pessimo) a 10 (perfetto)."
    )
    reasoning: str = pydantic.Field(
        description="Motivazione del voto, in massimo 3 frasi."
    )


class JudgementError(RuntimeError):
    """Raised when the judge does not return a valid judgement."""


def format_bullets(items: list[str]) -> str:
    """Formats a list as Markdown bullets, or "(nessuno)" when empty."""
    return "\n".join(f"- {item}" for item in items) if items else "(nessuno)"


def format_transcript(exchanges: list[tuple[str, str]]) -> str:
    """Formats the (user message, assistant answer) pairs of a conversation."""
    return "\n\n".join(
        f"Turno {position}\nUTENTE: {user_message}\nASSISTENTE: {answer}"
        for position, (user_message, answer) in enumerate(exchanges, start=1)
    )


class LlmJudge:
    """Scores a conversation on the judge metrics with a dedicated model."""

    def __init__(self, *, models: list["ModelEntry"], prompts_path: pathlib.Path):
        """Initializes the judge.

        Args:
            models: Models of the "judge" role, first the default one.
            prompts_path: YAML file with the system prompt and one prompt per metric.
        """
        self._models = models
        self._prompts = yaml.safe_load(prompts_path.read_text(encoding="utf-8"))

    async def judge(
        self,
        metric: str,
        *,
        ground_truth: list[str],
        must_include: list[str],
        must_not_include: list[str],
        exchanges: list[tuple[str, str]],
    ) -> Judgement:
        """Scores a conversation on one metric.

        Raises:
            JudgementError: If the judge does not return a valid judgement.
        """
        prompt = self._prompts[metric].format(
            ground_truth=format_bullets(ground_truth),
            must_include=format_bullets(must_include),
            must_not_include=format_bullets(must_not_include),
            question=exchanges[0][0],
            transcript=format_transcript(exchanges),
        )
        # A new agent per call: the judgements must not see each other
        judge_agent = strands.Agent(
            model=model_factory.create_model(self._models, model_index=0),
            system_prompt=self._prompts["system"],
            callback_handler=None,
            retry_strategy=None,
            hooks=[model_retry.ModelFallbackRetry(models=self._models)],
        )
        try:
            result = await judge_agent.invoke_async(
                prompt, structured_output_model=Judgement
            )
        except strands_exceptions.StructuredOutputException as e:
            raise JudgementError(
                f"The judge did not return a valid {metric} judgement"
            ) from e
        _LOGGER.info("Judge %s: %s", metric, result.structured_output.score)
        return result.structured_output
