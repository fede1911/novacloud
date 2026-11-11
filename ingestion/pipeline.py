"""Ingestion of the knowledge base: Markdown documents -> chunks -> Titan vectors -> index.

    data/kb/<scope_id>/<name>.md   (YAML frontmatter with title and url)
        -> load_documents  -> chunk_document -> embedder.embed_documents -> KbIndex.save

Every run rebuilds the whole index: with a few dozen documents it takes
seconds, and there is no state to keep in sync.
"""

import collections
import dataclasses
import pathlib
import re

import yaml

from shared import embeddings, kb_index, logger

_LOGGER = logger.get_logger(__name__)

DOCUMENT_SUFFIX = ".md"
FRONTMATTER_DELIMITER = "---"
REQUIRED_FRONTMATTER_FIELDS = ("title", "url")
_SECTION_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Headings that become part of the section path: the H1 repeats the document title
SECTION_PATH_LEVELS = (2, 3)
SECTION_PATH_SEPARATOR = " > "


class InvalidDocumentError(ValueError):
    """Raised when a source document has no valid frontmatter."""


@dataclasses.dataclass(frozen=True)
class SourceDocument:
    """A knowledge base document, parsed.

    Attributes:
        doc_id: Stable id, "<scope_id>-<file stem>".
        scope_id: Product of the document, from its folder.
        title: Title, from the frontmatter.
        url: Public URL, from the frontmatter, cited in the answers.
        body: Markdown body.
    """

    doc_id: str
    scope_id: str
    title: str
    url: str
    body: str


@dataclasses.dataclass(frozen=True)
class IngestionSummary:
    """Outcome of a run."""

    documents: int
    chunks_per_scope: dict[str, int]
    embedder: str
    index_dir: str


def load_documents(
    source_dir: pathlib.Path, *, scope_ids: list[str]
) -> list[SourceDocument]:
    """Reads the Markdown documents of every known product, sorted by path.

    Files outside a product folder or without the .md suffix are skipped with a warning.

    Raises:
        InvalidDocumentError: If a Markdown document has no valid frontmatter.
    """
    documents = []
    for path in sorted(source_dir.glob("*/*")):
        scope_id = path.parent.name
        if scope_id not in scope_ids or path.suffix != DOCUMENT_SUFFIX:
            _LOGGER.warning(
                "Skipped %s: not a Markdown document of a known product", path
            )
            continue
        documents.append(parse_document(path, scope_id=scope_id))
    return documents


def parse_document(path: pathlib.Path, *, scope_id: str) -> SourceDocument:
    """Parses the YAML frontmatter and the Markdown body of a document.

    Raises:
        InvalidDocumentError: If the frontmatter is missing, invalid or lacks a required field.
    """
    parts = path.read_text(encoding="utf-8").split(FRONTMATTER_DELIMITER, 2)
    if len(parts) < 3 or parts[0].strip():
        raise InvalidDocumentError(f"{path}: missing YAML frontmatter")
    try:
        frontmatter = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError as e:
        raise InvalidDocumentError(f"{path}: invalid YAML frontmatter") from e
    missing_fields = [
        field for field in REQUIRED_FRONTMATTER_FIELDS if not frontmatter.get(field)
    ]
    if missing_fields:
        raise InvalidDocumentError(
            f"{path}: missing frontmatter fields {missing_fields}"
        )
    return SourceDocument(
        doc_id=f"{scope_id}-{path.stem}",
        scope_id=scope_id,
        title=str(frontmatter["title"]),
        url=str(frontmatter["url"]),
        body=parts[2].strip(),
    )


def chunk_document(
    document: SourceDocument, *, max_chars: int, overlap_chars: int
) -> list[kb_index.KbChunk]:
    """Splits a document into chunks: one per section, more for the long ones.

    - Section: the path of its headings, e.g. "Sincronizzazione lenta" or "App > Notifiche".
    - Long sections are split at paragraph ends; paragraphs too long at sentence ends;
      sentences too long at word boundaries. A word is never cut.
    - Consecutive chunks of a section share the last paragraphs or sentences of the
      previous one, up to overlap_chars.
    - Every chunk starts with "Title > Section", so it carries its own context.

    Raises:
        ValueError: If overlap_chars is not smaller than max_chars.
    """
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")

    chunks = []
    for section, section_body in _sections(document):
        for piece in _split(
            section_body, max_chars=max_chars, overlap_chars=overlap_chars
        ):
            chunk_num = len(chunks)
            chunks.append(
                kb_index.KbChunk(
                    chunk_id=f"{document.doc_id}_{chunk_num}",
                    doc_id=document.doc_id,
                    scope_id=document.scope_id,
                    title=document.title,
                    section=section,
                    source=document.url,
                    chunk_num=chunk_num,
                    text=f"{document.title}{SECTION_PATH_SEPARATOR}{section}\n{piece}",
                )
            )
    return chunks


def run(
    *,
    source_dir: pathlib.Path,
    index_dir: pathlib.Path,
    scope_ids: list[str],
    embedder: embeddings.Embedder,
    max_chars: int,
    overlap_chars: int,
) -> IngestionSummary:
    """Rebuilds the index from the source documents.

    Args:
        source_dir: Folder with one subfolder of documents per product.
        index_dir: Folder of the index.
        scope_ids: Products of the catalog.
        embedder: Embeds the chunks; its model and dimensions go in the index metadata.
        max_chars: Maximum length of the body of a chunk.
        overlap_chars: Characters shared by consecutive windows of a long section.

    Raises:
        InvalidDocumentError: If a document is invalid.
        EmbeddingError: If the embedding API fails.
        ValueError: If there is nothing to index.
    """
    documents = load_documents(source_dir, scope_ids=scope_ids)
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(
            document, max_chars=max_chars, overlap_chars=overlap_chars
        )
    ]
    if not chunks:
        raise ValueError(f"No documents to index in {source_dir}")

    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    index_path = kb_index.KbIndex(chunks=chunks, vectors=vectors).save(
        index_dir, embedder=embedder
    )
    _LOGGER.info(
        "Index with %s chunks of %s documents written to %s",
        len(chunks),
        len(documents),
        index_path,
    )
    return IngestionSummary(
        documents=len(documents),
        chunks_per_scope=dict(collections.Counter(chunk.scope_id for chunk in chunks)),
        embedder=embedder.name,
        index_dir=str(index_path),
    )


def _sections(document: SourceDocument) -> list[tuple[str, str]]:
    """Returns (section path, body) pairs; the text under no H2/H3 belongs to the document title."""
    matches = list(_SECTION_HEADING.finditer(document.body))
    sections = [
        (
            document.title,
            document.body[: matches[0].start() if matches else len(document.body)],
        )
    ]
    path: dict[int, str] = {}
    for position, match in enumerate(matches):
        level, heading = len(match.group(1)), match.group(2)
        path = {lvl: text for lvl, text in path.items() if lvl < level} | {
            level: heading
        }
        section_path = SECTION_PATH_SEPARATOR.join(
            path[lvl] for lvl in SECTION_PATH_LEVELS if lvl in path
        )
        end = (
            matches[position + 1].start()
            if position + 1 < len(matches)
            else len(document.body)
        )
        sections.append(
            (section_path or document.title, document.body[match.end() : end])
        )
    return [(section, body.strip()) for section, body in sections if body.strip()]


def _split(text: str, *, max_chars: int, overlap_chars: int) -> list[str]:
    """Packs paragraphs, sentences or words into pieces of at most max_chars, with overlap."""
    if len(text) <= max_chars:
        return [text]

    units = _units(text, max_chars=max_chars)
    pieces: list[str] = []
    current: list[str] = []
    for unit in units:
        if current and len("\n\n".join([*current, unit])) > max_chars:
            pieces.append("\n\n".join(current))
            current = _overlap(current, overlap_chars=overlap_chars)
            # The overlap gives way when it does not leave room for the next unit
            while current and len("\n\n".join([*current, unit])) > max_chars:
                current = current[1:]
        current.append(unit)
    pieces.append("\n\n".join(current))
    return pieces


def _units(text: str, *, max_chars: int) -> list[str]:
    """Splits a text into paragraphs; paragraphs too long into sentences; sentences too long into word runs."""
    units = []
    for paragraph in (
        part.strip() for part in _PARAGRAPH_BREAK.split(text) if part.strip()
    ):
        if len(paragraph) <= max_chars:
            units.append(paragraph)
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            units.extend(
                [sentence]
                if len(sentence) <= max_chars
                else _word_runs(sentence, max_chars=max_chars)
            )
    return units


def _word_runs(text: str, *, max_chars: int) -> list[str]:
    """Splits a text at word boundaries into runs of at most max_chars."""
    runs: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            runs.append(current)
            candidate = word
        current = candidate
    if current:
        runs.append(current)
    return runs


def _overlap(units: list[str], *, overlap_chars: int) -> list[str]:
    """Returns the tail of the previous piece to repeat at the start of the next one.

    Whole units if they fit in overlap_chars; otherwise the last sentences of the last unit.
    """
    kept: list[str] = []
    for unit in reversed(units):
        if len("\n\n".join([unit, *kept])) > overlap_chars:
            break
        kept.insert(0, unit)
    if kept or not units:
        return kept

    sentences: list[str] = []
    for sentence in reversed(_SENTENCE_END.split(units[-1])):
        if len(" ".join([sentence, *sentences])) > overlap_chars:
            break
        sentences.insert(0, sentence)
    return [" ".join(sentences)] if sentences else []
