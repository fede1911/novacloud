"""Tests of the Bedrock provider."""

import pytest

from agent.model_providers import bedrock_provider
from agent.models import model_config

pytestmark = pytest.mark.unit


def _entry(params: dict) -> model_config.ModelEntry:
    return model_config.ModelEntry(
        provider="bedrock", model_id="mistral.model", temperature=0.0, params=params
    )


class TestBedrockProvider:
    """Streaming is on by default and can be turned off per model."""

    @pytest.mark.parametrize(
        ("params", "expected_streaming"),
        [
            ({"region_name": "eu-west-1"}, True),
            ({"region_name": "eu-west-1", "streaming": False}, False),
        ],
    )
    def test_streaming_follows_the_model_params(self, params, expected_streaming):
        """Test: model with and without the streaming param.
        Expected: streaming on by default, off when the param is false.
        """
        # Act
        model = bedrock_provider.BedrockProvider(_entry(params)).create_model()

        # Assert
        assert model.get_config()["streaming"] is expected_streaming
