"""Internal parsed-document representation (parser-independent).

A ``DocumentParser`` returns ``ParsedContent`` (pages, blocks, tables, sections).
The parsing pipeline wraps it into a ``ParsedDocument`` together with inventory
identity, validated metadata and quality issues.

Provenance rule: every block and table carries ``page_number`` (1-based, as in
the PDF) and ``page_numbers`` (all pages it spans), plus its section. Nothing is
stored only as one document-wide string, so "ISR Sequence 18, page 7" can always
be produced from stored data.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BlockLabel(StrEnum):
    TITLE = "title"
    SECTION_HEADER = "section_header"
    TEXT = "text"
    LIST_ITEM = "list_item"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"
    OTHER = "other"


class BoundingBox(_Model):
    left: float
    top: float
    right: float
    bottom: float
    coord_origin: str = "BOTTOMLEFT"


class TextBlock(_Model):
    block_id: str
    page_number: int
    page_numbers: list[int]
    label: BlockLabel
    source_label: str | None = None  # parser's own label, kept for transparency
    heading_level: int | None = None
    text_raw: str
    text_clean: str
    bbox: BoundingBox | None = None
    section_id: str | None = None
    is_furniture: bool = False  # page header/footer/number: excluded from clean text
    furniture_reason: str | None = None
    reading_order: int


class ParsedTable(_Model):
    table_id: str
    page_number: int
    page_numbers: list[int]
    caption: str | None = None
    n_rows: int
    n_cols: int
    header_rows: int = 0
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)  # full grid incl. header rows
    markdown: str | None = None  # parser's raw rendering
    text: str  # readable, row-wise rendering for extraction/RAG
    bbox: BoundingBox | None = None
    section_id: str | None = None
    reading_order: int


class Section(_Model):
    section_id: str
    title: str
    level: int | None = None
    numbering: str | None = None
    heading_block_id: str | None = None  # None for the implicit front-matter section
    start_page: int
    end_page: int
    block_ids: list[str] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)


class ParsedPage(_Model):
    page_number: int
    width: float | None = None
    height: float | None = None
    block_ids: list[str] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)
    heading_block_ids: list[str] = Field(default_factory=list)
    text_raw: str
    text_clean: str
    char_count_clean: int
    picture_count: int = 0
    is_empty: bool
    # Share of the PDF text layer's word tokens on this page that also appear in the
    # parsed raw page text (None if the PDF text layer was not available).
    pdf_text_coverage: float | None = None


class ParsedContent(_Model):
    """What a ``DocumentParser`` returns (no project metadata yet)."""

    page_count: int
    pages: list[ParsedPage]
    blocks: list[TextBlock]
    tables: list[ParsedTable]
    sections: list[Section]
    parser_warnings: list[str] = Field(default_factory=list)

    def section(self, section_id: str | None) -> Section | None:
        return next((s for s in self.sections if s.section_id == section_id), None)


class ParseStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class ValidationStatus(StrEnum):
    CONFIRMED = "CONFIRMED"
    CORRECTED_FROM_DOCUMENT = "CORRECTED_FROM_DOCUMENT"
    MANIFEST_ONLY = "MANIFEST_ONLY"
    DOCUMENT_ONLY = "DOCUMENT_ONLY"
    CONFLICT = "CONFLICT"
    UNKNOWN = "UNKNOWN"


class ExtractedValue(_Model):
    """A metadata value found in document text, with where it was found."""

    value: Any
    page_number: int
    evidence: str
    method: str
    confident: bool = True


class FieldValidation(_Model):
    field: str
    manifest_value: Any = None
    document_value: Any = None
    resolved_value: Any = None
    status: ValidationStatus
    resolution_reason: str
    evidence: ExtractedValue | None = None
    alternatives: list[ExtractedValue] = Field(default_factory=list)


class DocumentSourceRef(_Model):
    project_id: str
    document_id: str
    filename: str
    relative_path: str
    source_sha256: str
    source_size_bytes: int


class ParserInfo(_Model):
    name: str
    version: str
    config: dict[str, Any]
    config_hash: str


class ParseIssue(_Model):
    code: str
    severity: str  # ERROR | WARNING | INFO
    message: str
    page_number: int | None = None


class ParsedDocument(_Model):
    schema_version: str = SCHEMA_VERSION
    document_id: str
    project_id: str
    filename: str
    parse_status: ParseStatus
    error: str | None = None
    parser: ParserInfo
    parsed_at: str
    parse_seconds: float | None = None
    source_ref: DocumentSourceRef
    source_hash: str

    # Resolved metadata (see ``metadata_validation`` for how each was decided).
    document_type: str | None = None
    title: str | None = None
    document_date: date | None = None
    document_date_basis: str | None = None
    report_date: date | None = None  # ISR header date
    archive_date: date | None = None  # ISR "Archived on" date
    report_number: str | None = None
    isr_sequence: int | None = None
    loan_number: str | None = None
    normalized_loan_number: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    metadata_validation: list[FieldValidation] = Field(default_factory=list)

    page_count: int = 0
    pages: list[ParsedPage] = Field(default_factory=list)
    blocks: list[TextBlock] = Field(default_factory=list)
    tables: list[ParsedTable] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    quality_issues: list[ParseIssue] = Field(default_factory=list)
    parser_warnings: list[str] = Field(default_factory=list)

    def block(self, block_id: str) -> TextBlock:
        return next(b for b in self.blocks if b.block_id == block_id)

    def section(self, section_id: str | None) -> Section | None:
        return next((s for s in self.sections if s.section_id == section_id), None)

    def display_name(self) -> str:
        """Short human label, e.g. 'ISR Sequence 18' or 'RESTRUCTURING_PAPER RES01944'."""
        if self.document_type == "ISR" and self.isr_sequence is not None:
            return f"ISR Sequence {self.isr_sequence}"
        label = (self.document_type or "Document").replace("_", " ").title()
        return f"{label} {self.report_number}" if self.report_number else label


class EvidenceLocation(_Model):
    """Everything needed to cite a block or table later."""

    project_id: str
    document_id: str
    filename: str
    document_type: str | None
    page_number: int
    page_numbers: list[int]
    section_title: str | None
    element_id: str
    label: str  # "Source: ISR Sequence 18, page 7"


def evidence_location(
    document: ParsedDocument, element: TextBlock | ParsedTable
) -> EvidenceLocation:
    section = document.section(element.section_id)
    element_id = element.block_id if isinstance(element, TextBlock) else element.table_id
    pages = element.page_numbers
    page_text = f"page {pages[0]}" if len(pages) == 1 else f"pages {pages[0]}-{pages[-1]}"
    return EvidenceLocation(
        project_id=document.project_id,
        document_id=document.document_id,
        filename=document.filename,
        document_type=document.document_type,
        page_number=element.page_number,
        page_numbers=pages,
        section_title=section.title if section else None,
        element_id=element_id,
        label=f"Source: {document.display_name()}, {page_text}",
    )
