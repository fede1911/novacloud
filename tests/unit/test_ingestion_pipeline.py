"""Tests of the chunking of the ingestion pipeline."""

import pytest

from ingestion import pipeline

pytestmark = pytest.mark.unit


def _document(body: str) -> pipeline.SourceDocument:
    return pipeline.SourceDocument(
        doc_id="novaid-guide",
        scope_id="novaid",
        title="Guida",
        url="https://help.example/guide",
        body=body,
    )


def _bodies(chunks) -> list[str]:
    return [chunk.text.split("\n", 1)[1] for chunk in chunks]


class TestChunkDocument:
    """One chunk per section; long sections split at natural boundaries, with overlap."""

    def test_sections_become_chunks_with_context(self):
        """Test: document with an H1 and two H2 sections.
        Expected: one chunk per section, the H1 left out of the section, title and section as prefix.
        """
        # Arrange
        document = _document(
            "# Guida\n\n## Primo\nTesto uno.\n\n## Secondo\nTesto due."
        )

        # Act
        chunks = pipeline.chunk_document(document, max_chars=500, overlap_chars=50)

        # Assert
        assert [(chunk.chunk_id, chunk.section, chunk.text) for chunk in chunks] == [
            ("novaid-guide_0", "Primo", "Guida > Primo\nTesto uno."),
            ("novaid-guide_1", "Secondo", "Guida > Secondo\nTesto due."),
        ]

    def test_subsection_keeps_the_path_of_its_headings(self):
        """Test: H3 under an H2, then another H2.
        Expected: "H2 > H3" for the subsection, the new H2 alone after it.
        """
        # Arrange
        document = _document("## App\n### Notifiche\nTesto.\n## Offline\nAltro testo.")

        # Act
        chunks = pipeline.chunk_document(document, max_chars=500, overlap_chars=50)

        # Assert
        assert [chunk.section for chunk in chunks] == ["App > Notifiche", "Offline"]

    def test_long_section_is_split_at_paragraphs_with_sentence_overlap(self):
        """Test: two paragraphs that do not fit together, overlap smaller than a paragraph.
        Expected: one chunk per paragraph, the second starting with the last sentence of the first.
        """
        # Arrange
        document = _document(
            "## Lungo\nPrima frase. Seconda frase.\n\nTerza frase del secondo paragrafo."
        )

        # Act
        chunks = pipeline.chunk_document(document, max_chars=50, overlap_chars=16)

        # Assert
        assert _bodies(chunks) == [
            "Prima frase. Seconda frase.",
            "Seconda frase.\n\nTerza frase del secondo paragrafo.",
        ]

    def test_long_sentence_is_split_at_words(self):
        """Test: one sentence longer than the chunk size, no overlap possible.
        Expected: pieces within the size limit, no word cut.
        """
        # Arrange
        document = _document("## Lungo\nuno due tre quattro cinque sei sette otto")

        # Act
        chunks = pipeline.chunk_document(document, max_chars=15, overlap_chars=1)

        # Assert
        assert _bodies(chunks) == ["uno due tre", "quattro cinque", "sei sette otto"]

    def test_overlap_not_smaller_than_max_raises(self):
        """Test: overlap equal to the maximum size.
        Expected: ValueError.
        """
        # Act / Assert
        with pytest.raises(ValueError, match="overlap_chars"):
            pipeline.chunk_document(_document("testo"), max_chars=10, overlap_chars=10)
