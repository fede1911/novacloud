"""Tests of the ingestion pipeline and of the retriever on a temporary knowledge base."""

import json
import pathlib

import pytest

from ingestion import pipeline
from shared import kb_index, kb_search
from tests import fakes

pytestmark = pytest.mark.integration

SYNC_DOC = "---\ntitle: Sincronizzazione\nurl: https://help.example/sync\n---\n## Cartella ferma\nRiprendi la sincronizzazione dall'icona."
BILL_DOC = "---\ntitle: Fatture\nurl: https://help.example/bill\n---\n## Scaricare\nLe fatture sono nell'area Fatturazione."


@pytest.fixture
def source_dir(tmp_path) -> pathlib.Path:
    for scope_id, name, text in [
        ("novadrive", "sync.md", SYNC_DOC),
        ("novabill", "fatture.md", BILL_DOC),
    ]:
        (tmp_path / "kb" / scope_id).mkdir(parents=True)
        (tmp_path / "kb" / scope_id / name).write_text(text, encoding="utf-8")
    (tmp_path / "kb" / "novabill" / "note.txt").write_text(
        "not markdown", encoding="utf-8"
    )
    return tmp_path / "kb"


def _run(
    source_dir: pathlib.Path, index_dir: pathlib.Path
) -> pipeline.IngestionSummary:
    return pipeline.run(
        source_dir=source_dir,
        index_dir=index_dir,
        scope_ids=["novadrive", "novabill"],
        embedder=fakes.HashingEmbedder(),
        max_chars=500,
        overlap_chars=50,
    )


class TestIngestionPipeline:
    """The pipeline indexes the Markdown documents of the known products."""

    def test_run_indexes_markdown_and_skips_other_files(self, source_dir, tmp_path):
        """Test: two Markdown documents and one text file.
        Expected: one chunk per product, the text file skipped.
        """
        # Act
        summary = _run(source_dir, tmp_path / "index")

        # Assert
        assert (summary.documents, summary.chunks_per_scope) == (
            2,
            {"novabill": 1, "novadrive": 1},
        )

    def test_document_without_frontmatter_stops_the_run(self, source_dir, tmp_path):
        """Test: a Markdown document without frontmatter.
        Expected: InvalidDocumentError naming the file.
        """
        # Arrange
        (source_dir / "novadrive" / "broken.md").write_text(
            "# Nessun frontmatter", encoding="utf-8"
        )

        # Act / Assert
        with pytest.raises(pipeline.InvalidDocumentError, match="broken.md"):
            _run(source_dir, tmp_path / "index")


class TestRetriever:
    """The retriever searches only the chunks of the requested product, over the threshold."""

    def test_search_is_filtered_by_product(self, source_dir, tmp_path):
        """Test: query on the words of the NovaBill document, searched in NovaDrive.
        Expected: no NovaBill chunk among the results.
        """
        # Arrange
        _run(source_dir, tmp_path / "index")
        kb_retriever = kb_search.Retriever(
            index_dir=tmp_path / "index",
            embedder=fakes.HashingEmbedder(),
            top_k=5,
            min_score=0.0,
        )

        # Act
        hits = kb_retriever.search("fatture area fatturazione", scope_id="novadrive")

        # Assert
        assert {hit.chunk.scope_id for hit in hits} == {"novadrive"}

    def test_chunks_under_the_threshold_are_dropped(self, source_dir, tmp_path):
        """Test: query without words in common with the documents and a positive threshold.
        Expected: no results.
        """
        # Arrange
        _run(source_dir, tmp_path / "index")
        kb_retriever = kb_search.Retriever(
            index_dir=tmp_path / "index",
            embedder=fakes.HashingEmbedder(),
            top_k=5,
            min_score=0.5,
        )

        # Act
        hits = kb_retriever.search("ricetta carbonara", scope_id="novadrive")

        # Assert
        assert hits == []


class _OtherSizeEmbedder(fakes.HashingEmbedder):
    """Same model as the index, another vector size: stands for a changed configuration."""

    dimensions = 128


class TestIndexCompatibility:
    """The metadata of the index record the model, and the check rejects another one."""

    def test_metadata_record_model_and_chunks(self, source_dir, tmp_path):
        """Test: index built with the hashing embedder.
        Expected: metadata with its model, dimensions and the number of chunks.
        """
        # Arrange
        _run(source_dir, tmp_path / "index")

        # Act
        metadata = kb_index.check_compatibility(
            tmp_path / "index", embedder=fakes.HashingEmbedder()
        )

        # Assert
        assert (metadata.embedding_model, metadata.dimensions, metadata.chunks) == (
            "hashing-test",
            64,
            2,
        )

    def test_other_dimensions_are_rejected(self, source_dir, tmp_path):
        """Test: configuration with another vector size than the index.
        Expected: IndexMismatchError asking to rebuild the index.
        """
        # Arrange
        _run(source_dir, tmp_path / "index")

        # Act / Assert
        with pytest.raises(kb_index.IndexMismatchError, match="make ingest"):
            kb_index.check_compatibility(
                tmp_path / "index", embedder=_OtherSizeEmbedder()
            )

    def test_retriever_refuses_an_incompatible_index(self, source_dir, tmp_path):
        """Test: search with an embedder that does not match the index.
        Expected: IndexMismatchError, no search on incomparable vectors.
        """
        # Arrange
        _run(source_dir, tmp_path / "index")
        kb_retriever = kb_search.Retriever(
            index_dir=tmp_path / "index",
            embedder=_OtherSizeEmbedder(),
            top_k=5,
            min_score=0.0,
        )

        # Act / Assert
        with pytest.raises(kb_index.IndexMismatchError):
            kb_retriever.search("fatture", scope_id="novabill")

    def test_index_with_missing_vectors_is_invalid(self, source_dir, tmp_path):
        """Test: chunks file with one chunk more than the vectors.
        Expected: InvalidIndexError, the index is not used.
        """
        # Arrange
        _run(source_dir, tmp_path / "index")
        chunks_path = tmp_path / "index" / kb_index.CHUNKS_FILE_NAME
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks_path.write_text(json.dumps([*chunks, chunks[0]]), encoding="utf-8")

        # Act / Assert
        with pytest.raises(kb_index.InvalidIndexError, match="3 chunks but 2 vectors"):
            kb_index.KbIndex.load(tmp_path / "index")


class TestIndexSwap:
    """A rebuild replaces the index as a whole and leaves no working folders."""

    def test_rebuild_replaces_the_index_without_leftovers(self, source_dir, tmp_path):
        """Test: second build after removing a document.
        Expected: the new index has only the remaining chunks; no .building or .previous folder.
        """
        # Arrange
        _run(source_dir, tmp_path / "index")
        (source_dir / "novabill" / "fatture.md").unlink()

        # Act
        _run(source_dir, tmp_path / "index")

        # Assert
        assert [
            chunk.scope_id for chunk in kb_index.KbIndex.load(tmp_path / "index").chunks
        ] == ["novadrive"]
        assert sorted(path.name for path in tmp_path.iterdir()) == ["index", "kb"]
