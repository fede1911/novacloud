"""Tests of the Bedrock Titan embedder."""

import io
import json
from unittest import mock

import pytest
from botocore import exceptions as botocore_exceptions

from shared import embeddings

pytestmark = pytest.mark.unit

TITAN_MODEL_ID = "amazon.titan-embed-text-v2:0"


def _titan_with_client(client: mock.Mock) -> embeddings.BedrockTitanEmbedder:
    embedder = embeddings.BedrockTitanEmbedder(
        model_id=TITAN_MODEL_ID,
        region_name="eu-west-1",
        dimensions=256,
        max_concurrency=4,
    )
    embedder._client = client
    return embedder


def _titan_response(vector: list[float]) -> dict:
    return {
        "body": io.BytesIO(
            json.dumps({"embedding": vector, "inputTextTokenCount": 3}).encode()
        )
    }


class TestBedrockTitanEmbedder:
    """The embedder calls the Bedrock runtime API."""

    def test_query_is_embedded_with_requested_dimensions(self):
        """Test: embedding of a query.
        Expected: one invoke_model call with text, dimensions and normalization; the vector returned.
        """
        # Arrange
        client = mock.Mock(
            invoke_model=mock.Mock(return_value=_titan_response([0.6, 0.8]))
        )

        # Act
        vector = _titan_with_client(client).embed_query("cartella non sincronizzata")

        # Assert
        assert vector.tolist() == [0.6, 0.8]
        request = client.invoke_model.call_args.kwargs
        assert request["modelId"] == TITAN_MODEL_ID
        assert json.loads(request["body"]) == {
            "inputText": "cartella non sincronizzata",
            "dimensions": 256,
            "normalize": True,
        }

    def test_api_error_becomes_embedding_error(self):
        """Test: the Bedrock call fails.
        Expected: EmbeddingError, with the original error as cause.
        """
        # Arrange
        error = botocore_exceptions.ClientError(
            {"Error": {"Code": "ThrottlingException"}}, "InvokeModel"
        )
        client = mock.Mock(invoke_model=mock.Mock(side_effect=error))

        # Act / Assert
        with pytest.raises(embeddings.EmbeddingError):
            _titan_with_client(client).embed_query("ciao")

    def test_documents_keep_their_order_with_parallel_calls(self):
        """Test: five documents embedded by four parallel calls.
        Expected: one vector per document, in the order of the documents.
        """

        # Arrange
        def invoke_model(**request):
            text = json.loads(request["body"])["inputText"]
            return _titan_response([float(text)])

        client = mock.Mock(invoke_model=mock.Mock(side_effect=invoke_model))

        # Act
        vectors = _titan_with_client(client).embed_documents(["1", "2", "3", "4", "5"])

        # Assert
        assert vectors[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_client_retries_with_adaptive_backoff(self, monkeypatch):
        """Test: creation of the Bedrock client.
        Expected: adaptive retry mode with the configured attempts.
        """
        # Arrange
        create_client = mock.Mock()
        monkeypatch.setattr(embeddings.boto3, "client", create_client)
        embedder = embeddings.BedrockTitanEmbedder(
            model_id=TITAN_MODEL_ID,
            region_name="eu-west-1",
            dimensions=256,
            max_concurrency=4,
        )

        # Act
        embedder._get_client()

        # Assert
        retries = create_client.call_args.kwargs["config"].retries
        assert retries == {"max_attempts": embeddings.MAX_ATTEMPTS, "mode": "adaptive"}
