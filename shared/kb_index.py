"""Format of the knowledge base index: chunks and vectors, plus the metadata of the model that made them.

Written by the ingestion, read by the search_documentation tool: it is the
only contract between the two. Three data-only files in the index folder:

    chunks.json        the chunks, readable
    vectors.npy        one vector per chunk, same order (loaded without pickle)
    index_meta.json    embedding model and dimensions, chunk count, build time

A new index is written in a sibling folder and swapped in only when complete:
a reader finds either the whole old index or the whole new one. Inside it the
metadata are written last, as a seal of a complete index.

The metadata make the compatibility explicit: the agent embeds the queries
with the model of its configuration, and check_compatibility stops it when
that model is not the one that built the vectors.
"""

import dataclasses
import datetime as dt
import json
import pathlib
import shutil
from typing import TYPE_CHECKING

import numpy as np
import pydantic

if TYPE_CHECKING:
    from shared.embeddings import Embedder

CHUNKS_FILE_NAME = "chunks.json"
VECTORS_FILE_NAME = "vectors.npy"
METADATA_FILE_NAME = "index_meta.json"
# Siblings of the index folder used while swapping in a new index
BUILDING_SUFFIX = ".building"
PREVIOUS_SUFFIX = ".previous"


class IndexNotFoundError(FileNotFoundError):
    """Raised when the knowledge base index has not been built yet."""


class InvalidIndexError(ValueError):
    """Raised when the index files are not consistent with each other."""


class IndexMismatchError(RuntimeError):
    """Raised when the index was built with another embedding model or size than the configured one."""


@dataclasses.dataclass(frozen=True)
class KbChunk:
    """A chunk of a knowledge base document, with the fields used to filter and cite it.

    Attributes:
        chunk_id: Deterministic id, "<doc_id>_<chunk_num>".
        doc_id: Id of the source document.
        scope_id: Scope of the document, used as search pre-filter.
        title: Title of the source document.
        section: Heading of the section the chunk comes from.
        source: Public URL of the source document.
        chunk_num: Position of the chunk in the document.
        text: Text of the chunk.
    """

    chunk_id: str
    doc_id: str
    scope_id: str
    title: str
    section: str
    source: str
    chunk_num: int
    text: str


class IndexMetadata(pydantic.BaseModel):
    """What the vectors of the index were built with.

    Attributes:
        embedding_model: Model id of the embedder.
        dimensions: Size of the vectors.
        chunks: Number of indexed chunks.
        built_at: Build time of the index.
    """

    embedding_model: str
    dimensions: int
    chunks: int
    built_at: dt.datetime


@dataclasses.dataclass
class KbIndex:
    """The chunks and their L2-normalized vectors, in the same row order."""

    chunks: list[KbChunk]
    vectors: np.ndarray

    def save(self, index_dir: pathlib.Path, *, embedder: "Embedder") -> pathlib.Path:
        """Writes the index in a sibling folder, then swaps it with the current one.

        Returns:
            The index folder.
        """
        building_dir = index_dir.with_name(index_dir.name + BUILDING_SUFFIX)
        previous_dir = index_dir.with_name(index_dir.name + PREVIOUS_SUFFIX)
        # Leftovers of an interrupted build
        for leftover_dir in (building_dir, previous_dir):
            if leftover_dir.exists():
                shutil.rmtree(leftover_dir)

        building_dir.mkdir(parents=True)
        (building_dir / CHUNKS_FILE_NAME).write_text(
            json.dumps(
                [dataclasses.asdict(chunk) for chunk in self.chunks],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        np.save(building_dir / VECTORS_FILE_NAME, self.vectors)
        metadata = IndexMetadata(
            embedding_model=embedder.model_id,
            dimensions=embedder.dimensions,
            chunks=len(self.chunks),
            built_at=dt.datetime.now(dt.UTC),
        )
        (building_dir / METADATA_FILE_NAME).write_text(
            metadata.model_dump_json(indent=2), encoding="utf-8"
        )

        # Two renames: the folder is missing only between them, for an instant
        if index_dir.exists():
            index_dir.rename(previous_dir)
        building_dir.rename(index_dir)
        if previous_dir.exists():
            shutil.rmtree(previous_dir)
        return index_dir

    @classmethod
    def load(cls, index_dir: pathlib.Path) -> "KbIndex":
        """Reads chunks and vectors written by the ingestion.

        Raises:
            IndexNotFoundError: If an index file does not exist.
            InvalidIndexError: If chunks and vectors are not one per one.
        """
        missing = [
            name
            for name in (CHUNKS_FILE_NAME, VECTORS_FILE_NAME)
            if not (index_dir / name).exists()
        ]
        if missing:
            raise IndexNotFoundError(
                f"Knowledge base index incomplete in {index_dir} ({missing}): run the ingestion"
            )
        raw_chunks = json.loads(
            (index_dir / CHUNKS_FILE_NAME).read_text(encoding="utf-8")
        )
        chunks = [KbChunk(**raw_chunk) for raw_chunk in raw_chunks]
        # Data only: a crafted file cannot run code while it is loaded
        vectors = np.load(index_dir / VECTORS_FILE_NAME, allow_pickle=False)
        if vectors.shape[0] != len(chunks):
            raise InvalidIndexError(
                f"{len(chunks)} chunks but {vectors.shape[0]} vectors in {index_dir}"
            )
        return cls(chunks=chunks, vectors=vectors)


def check_compatibility(
    index_dir: pathlib.Path, *, embedder: "Embedder"
) -> IndexMetadata:
    """Checks that the index was built with the model and size of the given embedder.

    The region is not compared: the same model gives the same vectors in every region.

    Returns:
        The metadata of the index.

    Raises:
        IndexNotFoundError: If the index or its metadata do not exist.
        IndexMismatchError: If model or dimensions differ.
    """
    path = index_dir / METADATA_FILE_NAME
    if not path.exists():
        raise IndexNotFoundError(
            f"Index metadata not found in {index_dir}: run the ingestion first"
        )
    metadata = IndexMetadata.model_validate_json(path.read_text(encoding="utf-8"))
    if (metadata.embedding_model, metadata.dimensions) != (
        embedder.model_id,
        embedder.dimensions,
    ):
        raise IndexMismatchError(
            f"The index was built with {metadata.embedding_model} ({metadata.dimensions} dimensions), "
            f"the configuration uses {embedder.model_id} ({embedder.dimensions} dimensions): "
            "run `make ingest` to rebuild it"
        )
    return metadata
