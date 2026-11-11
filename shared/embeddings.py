"""Embedder shared by the ingestion (documents) and the search tool (queries).

Documents and queries must be embedded by the same model: the ingestion
records model and dimensions in the index metadata, and the agent checks
them against its own embedder at startup (see kb_index.check_compatibility).
"""

import json
import threading
from concurrent import futures
from typing import TYPE_CHECKING, Any, Protocol

import boto3
import numpy as np
from botocore import config as botocore_config
from botocore import exceptions as botocore_exceptions

from shared import logger

if TYPE_CHECKING:
    from shared.kb_config import EmbedderConfig

_LOGGER = logger.get_logger(__name__)

# Attempts of every call, with the adaptive retry mode of botocore: it backs off on
# throttling and slows down the client when the service keeps throttling
MAX_ATTEMPTS = 6
RETRY_MODE = "adaptive"


class EmbeddingError(RuntimeError):
    """Raised when a text cannot be embedded (e.g. the embedding API fails)."""


class Embedder(Protocol):
    """Turns texts into L2-normalized vectors, comparable with a dot product.

    Attributes:
        name: Readable identification of model and settings, used in logs.
        model_id: Model that computes the vectors, recorded in the index metadata.
        dimensions: Size of the vectors, recorded in the index metadata.
    """

    name: str
    model_id: str
    dimensions: int

    def embed_documents(self, documents: list[str]) -> np.ndarray:
        """Returns one vector per document, as rows of a matrix."""

    def embed_query(self, query: str) -> np.ndarray:
        """Returns the vector of a search query."""


class BedrockTitanEmbedder:
    """Semantic embedder: Amazon Titan Text Embeddings V2 through the Bedrock runtime API.

    Credentials come from the standard AWS chain (e.g. AWS_PROFILE). The client
    is created at the first call, with retries and backoff on throttling.
    Titan embeds one text per call: the documents are embedded by a pool of
    threads (boto3 clients are thread safe).
    """

    def __init__(
        self, *, model_id: str, region_name: str, dimensions: int, max_concurrency: int
    ):
        """Initializes the embedder, without connecting yet.

        Args:
            model_id: Bedrock model id, e.g. "amazon.titan-embed-text-v2:0".
            region_name: Bedrock region.
            dimensions: Size of the vectors (256, 512 or 1024).
            max_concurrency: Calls made at the same time while embedding documents.
        """
        self.model_id = model_id
        self.region_name = region_name
        self.dimensions = dimensions
        self.max_concurrency = max_concurrency
        self.name = f"bedrock-titan:{model_id}:{dimensions}"
        self._client: Any = None
        self._client_lock = threading.Lock()

    def embed_documents(self, documents: list[str]) -> np.ndarray:
        """Returns the vectors of the documents, in their order, with up to max_concurrency calls at a time.

        Raises:
            EmbeddingError: If a call still fails after the retries.
        """
        _LOGGER.info(
            "Embedding %s documents with %s, %s calls at a time",
            len(documents),
            self.name,
            self.max_concurrency,
        )
        with futures.ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            return np.vstack(list(pool.map(self._embed, documents)))

    def embed_query(self, query: str) -> np.ndarray:
        """Returns the vector of the query."""
        return self._embed(query)

    def _embed(self, text: str) -> np.ndarray:
        """Calls the model; the vector comes back already normalized.

        Raises:
            EmbeddingError: If the API call fails.
        """
        body = json.dumps(
            {"inputText": text, "dimensions": self.dimensions, "normalize": True}
        )
        try:
            response = self._get_client().invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
        except (
            botocore_exceptions.BotoCoreError,
            botocore_exceptions.ClientError,
        ) as e:
            raise EmbeddingError(f"Embedding with {self.model_id} failed: {e}") from e
        return np.asarray(json.loads(response["body"].read())["embedding"], dtype=float)

    def _get_client(self) -> Any:
        with self._client_lock:
            if self._client is None:
                self._client = boto3.client(
                    "bedrock-runtime",
                    region_name=self.region_name,
                    config=botocore_config.Config(
                        retries={"max_attempts": MAX_ATTEMPTS, "mode": RETRY_MODE}
                    ),
                )
            return self._client


def create_embedder(config: "EmbedderConfig") -> BedrockTitanEmbedder:
    """Creates the embedder of config/knowledge_base.json."""
    return BedrockTitanEmbedder(
        model_id=config.model_id,
        region_name=config.region_name,
        dimensions=config.dimensions,
        max_concurrency=config.max_concurrency,
    )
