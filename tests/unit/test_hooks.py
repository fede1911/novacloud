"""Tests of the model hooks: fallback on provider errors and retry on empty responses."""

import types

import pytest

from agent import model_factory
from agent.hooks import empty_response_retry, model_retry
from agent.model_providers import base
from agent.models import model_config

pytestmark = pytest.mark.unit


class _FakeProvider(base.ModelProvider):
    """Provider whose model is its model id; only TimeoutError is retryable."""

    def create_model(self):
        return f"model:{self._model.model_id}"

    def is_retryable(self, exception: BaseException) -> bool:
        return isinstance(exception, TimeoutError)


@pytest.fixture(autouse=True)
def _fake_provider(monkeypatch):
    monkeypatch.setitem(model_factory.PROVIDERS, "fake", _FakeProvider)


def _models(*model_ids: str) -> list[model_config.ModelEntry]:
    return [
        model_config.ModelEntry(provider="fake", model_id=model_id, temperature=None)
        for model_id in model_ids
    ]


def _after_model_call(
    *, exception: BaseException | None, stop_message: dict | None
) -> types.SimpleNamespace:
    stop_response = (
        None
        if stop_message is None
        else types.SimpleNamespace(message=stop_message, stop_reason="end_turn")
    )
    return types.SimpleNamespace(
        exception=exception,
        stop_response=stop_response,
        agent=types.SimpleNamespace(model="model:first"),
        retry=False,
    )


class TestModelFallbackRetry:
    """The hook switches to the next model only on retryable errors."""

    async def test_retryable_error_switches_to_next_model(self):
        """Test: retryable error on the first of two models.
        Expected: second model set on the agent and retry requested.
        """
        # Arrange
        hook = model_retry.ModelFallbackRetry(models=_models("first", "second"))
        event = _after_model_call(
            exception=TimeoutError("unavailable"), stop_message=None
        )

        # Act
        await hook.switch_model(event)

        # Assert
        assert (event.agent.model, event.retry, hook.model_index) == (
            "model:second",
            True,
            1,
        )

    async def test_non_retryable_error_keeps_model(self):
        """Test: non retryable error.
        Expected: same model, no retry.
        """
        # Arrange
        hook = model_retry.ModelFallbackRetry(models=_models("first", "second"))
        event = _after_model_call(
            exception=ValueError("bad request"), stop_message=None
        )

        # Act
        await hook.switch_model(event)

        # Assert
        assert (event.agent.model, event.retry, hook.model_index) == (
            "model:first",
            False,
            0,
        )

    async def test_error_on_last_model_restores_first_without_retry(self):
        """Test: retryable error on the last model.
        Expected: back to the first model for the next request, no retry.
        """
        # Arrange
        hook = model_retry.ModelFallbackRetry(models=_models("first", "second"))
        hook.model_index = 1
        event = _after_model_call(
            exception=TimeoutError("unavailable"), stop_message=None
        )

        # Act
        await hook.switch_model(event)

        # Assert
        assert (event.agent.model, event.retry, hook.model_index) == (
            "model:first",
            False,
            0,
        )


class TestEmptyResponseRetry:
    """The hook retries empty responses a limited number of times."""

    async def test_empty_response_is_retried(self):
        """Test: response with only blank text.
        Expected: retry requested.
        """
        # Arrange
        hook = empty_response_retry.EmptyResponseRetry()
        event = _after_model_call(
            exception=None, stop_message={"content": [{"text": "  "}]}
        )

        # Act
        await hook.retry_on_empty_response(event)

        # Assert
        assert event.retry is True

    async def test_gives_up_after_max_retries(self):
        """Test: one more empty response than the retry limit.
        Expected: the last one is not retried.
        """
        # Arrange
        hook = empty_response_retry.EmptyResponseRetry()
        first, second, third = (
            _after_model_call(exception=None, stop_message={"content": []})
            for _ in range(3)
        )

        # Act
        await hook.retry_on_empty_response(first)
        await hook.retry_on_empty_response(second)
        await hook.retry_on_empty_response(third)

        # Assert
        assert [first.retry, second.retry, third.retry] == [True, True, False]
