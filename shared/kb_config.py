"""Configuration of the knowledge base, from config/knowledge_base.json.

The ingestion reads embedder and chunking, the agent reads the retrieval.
"""

import pathlib
from typing import Literal

import pydantic


class EmbedderConfig(pydantic.BaseModel):
    """Amazon Titan Text Embeddings model on Bedrock.

    Attributes:
        model_id: Bedrock model id.
        region_name: Bedrock region.
        dimensions: Size of the vectors.
        max_concurrency: Calls made at the same time by the ingestion; keep it within the
            account quota, the retries absorb the occasional throttling.
    """

    model_id: str = pydantic.Field(min_length=1)
    region_name: str = pydantic.Field(min_length=1)
    dimensions: Literal[256, 512, 1024]
    max_concurrency: int = pydantic.Field(ge=1, le=32)


class ChunkingConfig(pydantic.BaseModel):
    """Size of the chunks."""

    max_chars: int = pydantic.Field(gt=0)
    overlap_chars: int = pydantic.Field(ge=0)


class RetrievalConfig(pydantic.BaseModel):
    """Vector search of the search tool.

    Attributes:
        top_k: Most chunks returned by a search.
        min_score: Cosine similarity under which a chunk is not relevant; it depends
            on the embedding model, so review it when the model changes.
    """

    top_k: int = pydantic.Field(gt=0)
    min_score: float = pydantic.Field(ge=0, le=1)


class KnowledgeBaseConfig(pydantic.BaseModel):
    """Content of config/knowledge_base.json."""

    embedder: EmbedderConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig

    @classmethod
    def from_file(cls, path: pathlib.Path) -> "KnowledgeBaseConfig":
        """Loads and validates the configuration.

        Raises:
            pydantic.ValidationError: If the file does not match the schema.
        """
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
