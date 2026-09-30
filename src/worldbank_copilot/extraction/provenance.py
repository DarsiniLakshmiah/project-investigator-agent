"""Provenance, method and status vocabulary for document-derived facts.

Every extracted business value carries an ``EvidenceRef`` pointing at the exact
parsed element it came from (document, page, section, table/row/column or block)
plus the extraction method. Statuses are deterministic labels, not confidence
scores.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from worldbank_copilot.parsing.models import ParsedDocument

SNIPPET_MAX = 240  # short evidence snippets only; never whole passages


class ExtractionMethod(StrEnum):
    DOCLING_TABLE = "DOCLING_TABLE"
    DOCLING_TEXT = "DOCLING_TEXT"
    PDF_TEXT_FALLBACK = "PDF_TEXT_FALLBACK"
    DERIVED = "DERIVED"  # computed from other extracted values (rule documented)


class ExtractionStatus(StrEnum):
    EXACT = "EXACT"  # value as printed
    NORMALIZED = "NORMALIZED"  # printed value mapped to a controlled form
    DERIVED_FROM_EXPLICIT_SOURCE = "DERIVED_FROM_EXPLICIT_SOURCE"
    AMBIGUOUS = "AMBIGUOUS"  # source present but not deterministically interpretable
    MISSING = "MISSING"  # not present in the source
    CONFLICT = "CONFLICT"  # sources disagree; left unresolved


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str
    document_id: str
    document_type: str | None
    document_date: date | None
    filename: str
    source_hash: str
    page_number: int
    section: str | None = None
    table_id: str | None = None
    row: int | None = None
    column: int | None = None
    block_id: str | None = None
    source_text: str | None = None
    extraction_method: ExtractionMethod
    label: str  # e.g. "ISR Sequence 18, page 2"


class ExtractionIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    severity: str  # ERROR | WARNING | INFO
    message: str
    evidence: EvidenceRef | None = None
    details: dict = Field(default_factory=dict)


def snippet(text: str | None, limit: int = SNIPPET_MAX) -> str | None:
    if text is None:
        return None
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def evidence(
    doc: ParsedDocument,
    page_number: int,
    method: ExtractionMethod,
    *,
    section: str | None = None,
    table_id: str | None = None,
    row: int | None = None,
    column: int | None = None,
    block_id: str | None = None,
    text: str | None = None,
) -> EvidenceRef:
    return EvidenceRef(
        project_id=doc.project_id,
        document_id=doc.document_id,
        document_type=doc.document_type,
        document_date=doc.document_date,
        filename=doc.filename,
        source_hash=doc.source_hash,
        page_number=page_number,
        section=section,
        table_id=table_id,
        row=row,
        column=column,
        block_id=block_id,
        source_text=snippet(text),
        extraction_method=method,
        label=f"{doc.display_name()}, page {page_number}",
    )


def issue(
    code: str, severity: str, message: str, ev: EvidenceRef | None = None, **details
) -> ExtractionIssue:
    return ExtractionIssue(
        code=code, severity=severity, message=message, evidence=ev, details=details
    )
