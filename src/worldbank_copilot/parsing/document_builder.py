"""Build a ``ParsedDocument`` from parser content + inventory record, with per-document checks."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from worldbank_copilot.common.identifiers import normalize_loan_number
from worldbank_copilot.common.quality import CheckCode
from worldbank_copilot.parsing.cleaning import normalize_unicode
from worldbank_copilot.parsing.metadata import extract_metadata
from worldbank_copilot.parsing.models import (
    DocumentSourceRef,
    ParsedContent,
    ParsedDocument,
    ParseIssue,
    ParserInfo,
    ParseStatus,
    ValidationStatus,
)
from worldbank_copilot.parsing.reconcile import reconcile, resolved

LOW_TEXT_CHARS_PER_PAGE = 400
USABLE_TEXT_MIN_CHARS = 1000
KEY_TYPES = {"ISR", "APPRAISAL_DOCUMENT", "RESTRUCTURING_PAPER", "ADDITIONAL_FINANCING"}
_DROPPED = re.compile(r"(\d+) of (\d+) pdf cells .*dropped from the table")
_TOKEN = re.compile(r"[a-z0-9]{3,}")
DOC_COVERAGE_WARNING = 0.95
PAGE_COVERAGE_FLAG = 0.85


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(normalize_unicode(text).lower()))


def _content_lines(pdf_page_texts: list[str]) -> list[str]:
    """Each page's PDF text minus running header/footer lines.

    A line (digits masked) repeated on at least max(3, 50% of pages) pages is a running
    header/footer; parsers may legitimately omit repeats of it, so it is not counted as
    content.
    """

    def signature(line: str) -> str:
        return re.sub(r"\d+", "#", " ".join(line.lower().split()))

    pages = [[ln for ln in text.splitlines() if ln.strip()] for text in pdf_page_texts]
    counts: dict[str, int] = {}
    for lines in pages:
        for sig in {signature(ln) for ln in lines}:
            counts[sig] = counts.get(sig, 0) + 1
    threshold = max(3, len(pages) * 0.5)
    return ["\n".join(ln for ln in lines if counts[signature(ln)] < threshold) for lines in pages]


def apply_text_coverage(
    content: ParsedContent, pdf_page_texts: list[str] | None
) -> tuple[ParsedContent, float | None]:
    """Compare each page's PDF text layer with everything the parser kept on that page.

    Coverage = share of distinct PDF-text-layer word tokens (3+ chars, running
    headers/footers excluded) that also appear in the parsed blocks/tables on that page
    (raw text, furniture included). It measures content lost by the parser, e.g. table
    cells dropped during structure matching.
    """
    if not pdf_page_texts:
        return content, None
    pdf_page_texts = _content_lines(pdf_page_texts)
    parsed_text: dict[int, list[str]] = {}
    for element in [*content.blocks, *content.tables]:
        text = element.text_raw if hasattr(element, "text_raw") else element.text
        for page in element.page_numbers:
            parsed_text.setdefault(page, []).append(text)
    pages, found, total = [], 0, 0
    for page in content.pages:
        if page.page_number > len(pdf_page_texts):
            pages.append(page)
            continue
        pdf_tokens = _tokens(pdf_page_texts[page.page_number - 1])
        if not pdf_tokens:
            pages.append(page.model_copy(update={"pdf_text_coverage": None}))
            continue
        hit = len(pdf_tokens & _tokens("\n".join(parsed_text.get(page.page_number, []))))
        found, total = found + hit, total + len(pdf_tokens)
        pages.append(page.model_copy(update={"pdf_text_coverage": round(hit / len(pdf_tokens), 4)}))
    overall = round(found / total, 4) if total else None
    return content.model_copy(update={"pages": pages}), overall


def source_ref(record: dict[str, Any]) -> DocumentSourceRef:
    return DocumentSourceRef(
        project_id=record["project_id"],
        document_id=record["document_id"],
        filename=record["filename"],
        relative_path=record["relative_path"],
        source_sha256=record["sha256"],
        source_size_bytes=record["file_size_bytes"],
    )


def manifest_values(record: dict[str, Any]) -> dict[str, Any]:
    """Inventory values (filename rules + curated manifest) to validate against content."""
    unclassified = record.get("classification_method") == "UNCLASSIFIED"
    return {
        "project_id": record["project_id"],
        "document_type": None if unclassified else record.get("document_type"),
        "document_date": record.get("document_date"),
        "isr_sequence": record.get("isr_sequence"),
        "report_number": record.get("report_number"),
        "loan_number": record.get("raw_loan_number"),
    }


def failed_document(
    record: dict[str, Any], parser: ParserInfo, parsed_at: str, seconds: float, error: str
) -> ParsedDocument:
    return ParsedDocument(
        document_id=record["document_id"],
        project_id=record["project_id"],
        filename=record["filename"],
        parse_status=ParseStatus.FAILED,
        error=error,
        parser=parser,
        parsed_at=parsed_at,
        parse_seconds=round(seconds, 2),
        source_ref=source_ref(record),
        source_hash=record["sha256"],
        document_type=record.get("document_type"),
        isr_sequence=record.get("isr_sequence"),
        quality_issues=[ParseIssue(code=CheckCode.PARSE_FAILED, severity="ERROR", message=error)],
    )


def content_of(document: ParsedDocument) -> ParsedContent:
    return ParsedContent(
        page_count=document.page_count,
        pages=document.pages,
        blocks=document.blocks,
        tables=document.tables,
        sections=document.sections,
        parser_warnings=document.parser_warnings,
    )


def build_document(
    record: dict[str, Any],
    content: ParsedContent,
    parser: ParserInfo,
    parsed_at: str,
    seconds: float | None,
    pdf_page_count: int | None = None,
    pdf_page_texts: list[str] | None = None,
) -> ParsedDocument:
    content, coverage = apply_text_coverage(content, pdf_page_texts)
    if pdf_page_texts is not None and pdf_page_count is None:
        pdf_page_count = len(pdf_page_texts)
    evidence = extract_metadata(content)
    validations = reconcile(manifest_values(record), evidence)
    document_date = resolved(validations, "document_date")
    if isinstance(document_date, str):
        document_date = date.fromisoformat(document_date)
    date_validation = next(v for v in validations if v.field == "document_date")
    if date_validation.status is ValidationStatus.MANIFEST_ONLY:
        date_basis = record.get("date_basis")
    else:
        date_basis = evidence.document_date_basis if document_date else None
    loan = resolved(validations, "loan_number")

    document = ParsedDocument(
        document_id=record["document_id"],
        project_id=record["project_id"],
        filename=record["filename"],
        parse_status=ParseStatus.SUCCESS,
        parser=parser,
        parsed_at=parsed_at,
        parse_seconds=round(seconds, 2) if seconds is not None else None,
        source_ref=source_ref(record),
        source_hash=record["sha256"],
        document_type=resolved(validations, "document_type"),
        title=evidence.title.value if evidence.title else None,
        document_date=document_date,
        document_date_basis=date_basis,
        report_date=evidence.report_date.value if evidence.report_date else None,
        archive_date=evidence.archive_date.value if evidence.archive_date else None,
        report_number=resolved(validations, "report_number"),
        isr_sequence=resolved(validations, "isr_sequence"),
        loan_number=loan,
        normalized_loan_number=normalize_loan_number(loan) if loan else None,
        metadata={
            "extras": evidence.extras,
            "notes": evidence.notes,
            "project_ids_mentioned": evidence.project_ids_mentioned,
            "type_alternatives": [a.model_dump(mode="json") for a in evidence.type_alternatives],
            "inventory": {
                k: record.get(k)
                for k in ("classification_method", "manifest_status", "filename_rule", "date_basis")
            },
            "pdf_page_count": pdf_page_count,
            "pdf_text_coverage": coverage,
        },
        metadata_validation=validations,
        page_count=content.page_count,
        pages=content.pages,
        blocks=content.blocks,
        tables=content.tables,
        sections=content.sections,
        parser_warnings=content.parser_warnings,
    )
    return document.model_copy(update={"quality_issues": document_checks(document)})


def _issue(code: CheckCode, severity: str, message: str, page: int | None = None) -> ParseIssue:
    return ParseIssue(code=code, severity=severity, message=message, page_number=page)


def document_checks(doc: ParsedDocument) -> list[ParseIssue]:
    issues: list[ParseIssue] = []
    pdf_pages = doc.metadata.get("pdf_page_count")
    if pdf_pages is not None and pdf_pages != doc.page_count:
        issues.append(
            _issue(
                CheckCode.PDF_PAGE_COUNT_MISMATCH,
                "ERROR",
                f"PDF has {pdf_pages} pages; parser returned {doc.page_count}",
            )
        )
    if doc.page_count <= 0:
        issues.append(_issue(CheckCode.NO_USABLE_TEXT, "ERROR", "no pages parsed"))
        return issues

    empty = [p.page_number for p in doc.pages if p.is_empty]
    if empty:
        severity = "WARNING" if len(empty) / doc.page_count > 0.25 else "INFO"
        issues.append(
            _issue(
                CheckCode.EMPTY_PAGES,
                severity,
                f"{len(empty)} page(s) without text or tables: {empty}",
            )
        )
    clean_chars = sum(p.char_count_clean for p in doc.pages)
    if doc.document_type in KEY_TYPES and clean_chars < USABLE_TEXT_MIN_CHARS:
        issues.append(
            _issue(
                CheckCode.NO_USABLE_TEXT, "ERROR", f"only {clean_chars} characters of clean text"
            )
        )
    elif clean_chars / doc.page_count < LOW_TEXT_CHARS_PER_PAGE:
        issues.append(
            _issue(
                CheckCode.LOW_TEXT_EXTRACTION,
                "WARNING",
                f"{clean_chars / doc.page_count:.0f} clean characters per page",
            )
        )

    coverage = doc.metadata.get("pdf_text_coverage")
    if coverage is not None and coverage < DOC_COVERAGE_WARNING:
        low = [
            f"p{p.page_number}={p.pdf_text_coverage:.2f}"
            for p in doc.pages
            if p.pdf_text_coverage is not None and p.pdf_text_coverage < PAGE_COVERAGE_FLAG
        ]
        issues.append(_issue(CheckCode.LOW_TEXT_COVERAGE, "WARNING",
                             f"parsed text covers {coverage:.1%} of the PDF text layer's tokens; "
                             f"pages below {PAGE_COVERAGE_FLAG:.0%}: {low or 'none'}"))  # fmt: skip

    for element in [*doc.blocks, *doc.tables]:
        if not element.page_numbers or not all(
            1 <= p <= doc.page_count for p in element.page_numbers
        ):
            element_id = getattr(element, "block_id", None) or getattr(element, "table_id", None)
            issues.append(
                _issue(
                    CheckCode.PAGE_PROVENANCE_INVALID,
                    "ERROR",
                    f"{element_id} has invalid pages {element.page_numbers}",
                )
            )

    for validation in doc.metadata_validation:
        if validation.field == "project_id" and validation.resolved_value != doc.project_id:
            issues.append(
                _issue(
                    CheckCode.DOCUMENT_PROJECT_MISMATCH,
                    "ERROR",
                    f"document identifies project {validation.document_value!r} "
                    f"but is filed under {doc.project_id}",
                )
            )
        elif validation.status is ValidationStatus.CORRECTED_FROM_DOCUMENT:
            issues.append(
                _issue(
                    CheckCode.METADATA_CORRECTED_FROM_DOCUMENT,
                    "WARNING",
                    f"{validation.field}: {validation.manifest_value!r} -> "
                    f"{validation.resolved_value!r} ({validation.resolution_reason})",
                    validation.evidence.page_number if validation.evidence else None,
                )
            )
        elif validation.status is ValidationStatus.CONFLICT:
            issues.append(
                _issue(
                    CheckCode.METADATA_CONFLICT,
                    "WARNING",
                    f"{validation.field}: manifest {validation.manifest_value!r} vs "
                    f"document {validation.document_value!r}; unresolved",
                )
            )
    if doc.document_type in (None, "OTHER"):
        issues.append(
            _issue(
                CheckCode.DOCUMENT_TYPE_UNRESOLVED, "ERROR", "document type could not be resolved"
            )
        )

    if doc.document_type == "ISR":
        for name in ("isr_sequence", "archive_date", "report_number"):
            if getattr(doc, name) is None:
                issues.append(
                    _issue(CheckCode.MISSING_EXPECTED_METADATA, "ERROR", f"ISR without {name}")
                )
        if doc.report_date is None:
            issues.append(
                _issue(
                    CheckCode.MISSING_EXPECTED_METADATA,
                    "WARNING",
                    "ISR header report date not found",
                )
            )
        elif doc.archive_date and doc.report_date != doc.archive_date:
            issues.append(
                _issue(
                    CheckCode.ISR_DATES_DIFFER,
                    "INFO",
                    f"header report date {doc.report_date} vs archive date "
                    f"{doc.archive_date}; both kept",
                )
            )
    elif doc.document_date is None:
        issues.append(
            _issue(
                CheckCode.MISSING_EXPECTED_METADATA,
                "INFO",
                "no reliable document date in the document "
                + "; ".join(doc.metadata.get("notes", [])),
            )
        )

    dropped = [m for w in doc.parser_warnings if (m := _DROPPED.search(w))]
    if dropped:
        cells = sum(int(m.group(1)) for m in dropped)
        issues.append(
            _issue(
                CheckCode.TABLE_CELLS_DROPPED,
                "WARNING",
                f"parser dropped {cells} PDF cells from {len(dropped)} table(s) "
                "during table-structure matching",
            )
        )
    other = [w for w in doc.parser_warnings if not _DROPPED.search(w)]
    if other:
        issues.append(
            _issue(CheckCode.PARSER_WARNING, "INFO", f"{len(other)} parser warning(s): {other[:3]}")
        )
    return issues
