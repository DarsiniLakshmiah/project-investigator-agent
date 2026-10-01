"""Deterministic chunking of Phase 4 parsed documents (pure Python, no Spark).

Three strategies (configs/retrieval/chunking.yaml):

* ``fixed`` (baseline A): recursive character splitting of the whole document text in
  reading order, with overlap. Ignores section boundaries.
* ``structure`` (baseline B): a chunk never crosses a section; paragraphs are packed up
  to ``max_chars``; every table is its own unit, split into row groups with the table's
  header lines repeated.
* ``parent_child`` (baseline C): small ``structure``-style children are retrieved; each
  child points to a PARENT chunk (consecutive children of one section, up to
  ``parent_max_chars``) used only for context assembly.

Provenance: every chunk keeps the document, its source hash, the pages and elements
(block / table ids) it was built from, and its section. Page headers/footers marked as
furniture in Phase 4 and operations-portal template markers are excluded. Nothing is
invented: metadata the parsed document does not carry stays NULL.

Identity: ``chunk_id`` = stable hash of (strategy, strategy version, document, role,
ordinal). The same parsed document with the same configuration always yields the same
ids; a configuration change changes the strategy version and therefore the ids.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date

from worldbank_copilot.lakehouse.identity import stable_id
from worldbank_copilot.parsing.models import BlockLabel, ParsedDocument, ParsedTable, TextBlock
from worldbank_copilot.retrieval.config import ChunkingConfig, StrategyConfig

TEXT, TABLE = "TEXT", "TABLE"
RETRIEVAL, PARENT = "RETRIEVAL", "PARENT"
CHUNK_TABLE = "document_chunks"
_SKIP_LABELS = {BlockLabel.PAGE_HEADER, BlockLabel.PAGE_FOOTER}
_SENTENCE = re.compile(r"(?<=[.;:!?])\s+")
_MAX_HEADER_LINES = 3
# Machine template markers printed in World Bank operations-portal papers, e.g.
# "@#&OPS~Doctype~OPS^dynamics@restrhybridsummarychanges#doctemplate" (no content).
_TEMPLATE_MARKER = re.compile(r"^@#&OPS~")


@dataclass(frozen=True)
class Element:
    """One block or table of a parsed document, in reading order."""

    element_id: str
    kind: str  # TEXT | TABLE
    text: str
    page_number: int
    page_numbers: tuple[int, ...]
    section_id: str | None
    reading_order: int
    table: ParsedTable | None = None


@dataclass
class Chunk:
    strategy: str
    strategy_version: str
    role: str  # RETRIEVAL | PARENT
    chunk_type: str  # TEXT | TABLE
    ordinal: int
    text: str
    elements: list[Element]
    section_id: str | None
    section_title: str | None
    parent_ordinal: int | None = None
    table_id: str | None = None
    part: int | None = None  # row-group / split index within an element
    extra_pages: set[int] = field(default_factory=set)

    @property
    def page_numbers(self) -> list[int]:
        pages = {p for e in self.elements for p in e.page_numbers} | self.extra_pages
        return sorted(pages)


def document_label(doc: ParsedDocument) -> str:
    when = doc.document_date or doc.report_date
    suffix = f" ({when.isoformat()})" if isinstance(when, date) else ""
    return f"{doc.display_name()}{suffix}"


def elements(doc: ParsedDocument) -> list[Element]:
    """Non-furniture text blocks and tables in reading order."""
    out: list[Element] = []
    for block in doc.blocks:
        text = block.text_clean.strip()
        if block.is_furniture or block.label in _SKIP_LABELS or not text:
            continue
        if _TEMPLATE_MARKER.match(text):
            continue
        out.append(_block_element(block))
    for table in doc.tables:
        if table.text.strip():
            out.append(
                Element(
                    table.table_id,
                    TABLE,
                    table.text.strip(),
                    table.page_number,
                    tuple(table.page_numbers),
                    table.section_id,
                    table.reading_order,
                    table,
                )
            )
    return sorted(out, key=lambda e: (e.reading_order, e.element_id))


def _block_element(block: TextBlock) -> Element:
    return Element(
        block.block_id,
        TEXT,
        block.text_clean.strip(),
        block.page_number,
        tuple(block.page_numbers),
        block.section_id,
        block.reading_order,
    )


def _is_heading(element: Element, doc: ParsedDocument) -> bool:
    if element.kind != TEXT:
        return False
    return any(s.heading_block_id == element.element_id for s in doc.sections)


# ---------------------------------------------------------------------------
# Splitting primitives
# ---------------------------------------------------------------------------


def split_text(text: str, max_chars: int) -> list[str]:
    """Split one text into pieces <= max_chars: sentences first, then words, then chars."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    pieces: list[str] = []
    for unit in _pack(_SENTENCE.split(text), max_chars, " "):
        if len(unit) <= max_chars:
            pieces.append(unit)
            continue
        for word_unit in _pack(unit.split(), max_chars, " "):
            pieces.extend(word_unit[i : i + max_chars] for i in range(0, len(word_unit), max_chars))
    return [p for p in (p.strip() for p in pieces) if p]


def _pack(units: Iterable[str], max_chars: int, sep: str) -> list[str]:
    """Greedy packing of units into strings <= max_chars (a longer unit stays alone)."""
    out: list[str] = []
    current = ""
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        candidate = f"{current}{sep}{unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                out.append(current)
            current = unit
    if current:
        out.append(current)
    return out


def table_parts(table_text: str, max_chars: int) -> list[str]:
    """Row groups of a rendered table, each repeating the table's header lines.

    Header lines are the leading lines without digits (title / column names), at most
    three. Rows are never split across groups unless a single row exceeds the limit.
    """
    lines = [line for line in table_text.split("\n") if line.strip()]
    header: list[str] = []
    for line in lines[:_MAX_HEADER_LINES]:
        if re.search(r"\d", line):
            break
        header.append(line)
    rows = lines[len(header) :]
    head = "\n".join(header)
    if len(head) > max_chars // 2:  # an oversized header is not repeated
        head, rows = "", lines
    budget = max_chars - (len(head) + 1 if head else 0)
    if not rows:
        return split_text(head, max_chars) if head else []
    groups: list[str] = []
    for row in rows:
        for piece in split_text(row, budget) if len(row) > budget else [row]:
            groups.append(piece)
    packed = _pack(groups, budget, "\n")
    return [f"{head}\n{p}" if head else p for p in packed]


def split_with_overlap(text: str, max_chars: int, overlap: int) -> list[tuple[int, int]]:
    """Character spans (start, end) of <= max_chars, cut at whitespace, with overlap."""
    spans: list[tuple[int, int]] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            cut = text.rfind(" ", start + max_chars // 2, end)
            cut = max(cut, text.rfind("\n", start + max_chars // 2, end))
            if cut > start:
                end = cut
        spans.append((start, end))
        if end >= n:
            break
        next_start = max(end - overlap, start + 1)
        space = text.find(" ", next_start, end)
        start = space + 1 if 0 <= space < end else next_start
    return spans


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def _fixed(doc: ParsedDocument, cfg: StrategyConfig, name: str, version: str) -> list[Chunk]:
    els = [e for e in elements(doc)]
    text_parts, offsets, cursor = [], [], 0
    for e in els:
        text_parts.append(e.text)
        offsets.append((cursor, cursor + len(e.text), e))
        cursor += len(e.text) + 2
    full = "\n\n".join(text_parts)
    chunks = []
    for ordinal, (start, end) in enumerate(
        split_with_overlap(full, cfg.max_chars, cfg.overlap_chars)
    ):
        covered = [e for s, t, e in offsets if s < end and t > start]
        first = covered[0] if covered else None
        section = doc.section(first.section_id) if first else None
        kind = TABLE if covered and all(e.kind == TABLE for e in covered) else TEXT
        chunks.append(
            Chunk(
                name,
                version,
                RETRIEVAL,
                kind,
                ordinal,
                full[start:end].strip(),
                covered,
                first.section_id if first else None,
                section.title if section else None,
            )
        )
    return chunks


def _sectioned(
    doc: ParsedDocument, name: str, version: str, text_max: int, table_max: int
) -> list[Chunk]:
    """Section-bounded chunks: packed paragraphs and table row groups."""
    chunks: list[Chunk] = []
    pending: list[Element] = []

    def section_title(section_id: str | None) -> str | None:
        section = doc.section(section_id)
        return section.title if section else None

    def emit_group(group: list[Element]) -> None:
        if group:
            sid = group[0].section_id
            chunks.append(
                Chunk(
                    name,
                    version,
                    RETRIEVAL,
                    TEXT,
                    len(chunks),
                    "\n".join(e.text for e in group),
                    list(group),
                    sid,
                    section_title(sid),
                )
            )

    def flush() -> None:
        """Emit pending paragraphs in reading order, packed up to text_max."""
        group: list[Element] = []
        size = 0
        for e in pending:
            if len(e.text) > text_max:
                emit_group(group)
                group, size = [], 0
                for part_index, piece in enumerate(split_text(e.text, text_max)):
                    chunks.append(
                        Chunk(
                            name,
                            version,
                            RETRIEVAL,
                            TEXT,
                            len(chunks),
                            piece,
                            [e],
                            e.section_id,
                            section_title(e.section_id),
                            part=part_index,
                        )
                    )
                continue
            if group and size + 1 + len(e.text) > text_max:
                emit_group(group)
                group, size = [], 0
            group.append(e)
            size += len(e.text) + 1
        emit_group(group)
        pending.clear()

    current_section: object = object()
    for e in elements(doc):
        if e.section_id != current_section:
            flush()
            current_section = e.section_id
        if _is_heading(e, doc):
            continue  # the heading is carried as section_title / context header
        if e.kind == TABLE:
            flush()
            for part_index, piece in enumerate(table_parts(e.text, table_max)):
                chunks.append(
                    Chunk(
                        name,
                        version,
                        RETRIEVAL,
                        TABLE,
                        len(chunks),
                        piece,
                        [e],
                        e.section_id,
                        section_title(e.section_id),
                        table_id=e.element_id,
                        part=part_index,
                    )
                )
            continue
        pending.append(e)
    flush()
    return chunks


def _parents(children: list[Chunk], name: str, version: str, max_chars: int) -> list[Chunk]:
    """Group consecutive children of one section into PARENT chunks <= max_chars."""
    parents: list[Chunk] = []
    current: list[Chunk] = []
    size = 0

    def close() -> None:
        if current:
            parent = Chunk(
                name,
                version,
                PARENT,
                TEXT,
                len(parents),
                "\n\n".join(c.text for c in current),
                [e for c in current for e in c.elements],
                current[0].section_id,
                current[0].section_title,
            )
            if all(c.chunk_type == TABLE for c in current):
                parent.chunk_type = TABLE
            for child in current:
                child.parent_ordinal = parent.ordinal
            parents.append(parent)

    for child in children:
        new_section = current and child.section_id != current[0].section_id
        if new_section or (current and size + 2 + len(child.text) > max_chars):
            close()
            current, size = [], 0
        current.append(child)
        size += len(child.text) + 2
    close()
    return parents


def chunk_document(doc: ParsedDocument, config: ChunkingConfig, strategy: str) -> list[Chunk]:
    """All chunks (retrieval + parent) of one document for one strategy."""
    cfg = config.strategies[strategy]
    version = config.version(strategy)
    if cfg.kind == "fixed":
        chunks = _fixed(doc, cfg, strategy, version)
        parents: list[Chunk] = []
    elif cfg.kind == "structure":
        chunks = _sectioned(doc, strategy, version, cfg.max_chars, cfg.table_max_chars)
        parents = []
    else:
        chunks = _sectioned(doc, strategy, version, cfg.child_max_chars, cfg.table_max_chars)
        parents = []
    chunks = _clean(chunks, config.min_chars)
    if cfg.kind == "parent_child":
        parents = _parents(chunks, strategy, version, cfg.parent_max_chars)
    return chunks + parents


def _clean(chunks: list[Chunk], min_chars: int) -> list[Chunk]:
    """Drop fragments shorter than min_chars and exact duplicates within the document."""
    seen: set[str] = set()
    out: list[Chunk] = []
    for chunk in chunks:
        key = re.sub(r"\s+", " ", chunk.text).strip().lower()
        if len(key) < min_chars or key in seen:
            continue
        seen.add(key)
        chunk.ordinal = len(out)
        out.append(chunk)
    return out


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def chunk_id(chunk: Chunk, document_id: str) -> str:
    return stable_id(
        CHUNK_TABLE, chunk.strategy, chunk.strategy_version, document_id, chunk.role, chunk.ordinal
    )


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_rows(doc: ParsedDocument, config: ChunkingConfig, strategy: str) -> list[dict]:
    """Contract-shaped bodies for silver.document_chunks (identity added by finalize)."""
    chunks = chunk_document(doc, config, strategy)
    label = document_label(doc)
    parent_ids = {c.ordinal: chunk_id(c, doc.document_id) for c in chunks if c.role == PARENT}
    rows = []
    for chunk in chunks:
        header = " | ".join(p for p in (doc.project_id, label, chunk.section_title) if p)
        search_text = f"{header}\n{chunk.text}" if config.context_header else chunk.text
        methods = {e.kind for e in chunk.elements}
        rows.append(
            {
                "chunk_id": chunk_id(chunk, doc.document_id),
                "chunk_strategy": chunk.strategy,
                "strategy_version": chunk.strategy_version,
                "chunk_role": chunk.role,
                "chunk_type": chunk.chunk_type,
                "chunk_ordinal": chunk.ordinal,
                "part_index": chunk.part,
                "parent_chunk_id": parent_ids.get(chunk.parent_ordinal)
                if chunk.parent_ordinal is not None
                else None,
                "project_id": doc.project_id,
                "document_id": doc.document_id,
                "document_type": doc.document_type,
                "document_label": label,
                "document_date": doc.document_date,
                "isr_sequence": doc.isr_sequence,
                "report_number": doc.report_number,
                "source_file": doc.filename,
                "source_relative_path": doc.source_ref.relative_path,
                "source_hash": doc.source_hash,
                "parser_config_hash": doc.parser.config_hash,
                "page_number": chunk.page_numbers[0] if chunk.page_numbers else None,
                "page_numbers": chunk.page_numbers,
                "section_id": chunk.section_id,
                "section_title": chunk.section_title,
                "table_id": chunk.table_id,
                "element_ids": [e.element_id for e in chunk.elements],
                "extraction_method": "DOCLING_TABLE"
                if methods == {TABLE}
                else ("DOCLING_TEXT" if methods == {TEXT} else "DOCLING_TEXT_AND_TABLE"),
                "chunk_text": chunk.text,
                "search_text": search_text,
                "text_sha256": text_sha256(search_text),
                "char_count": len(chunk.text),
            }
        )
    return rows
