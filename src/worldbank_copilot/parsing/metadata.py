"""Deterministic document metadata extraction by document type (no LLM).

Each document type has a handler that reads the *raw* page text (all content
layers, because running headers carry ISR report dates) and returns
``DocumentEvidence``: the type detected from content plus the metadata values it
can identify, each with page number and evidence text.

Handlers identify structure and metadata only. They do not extract business
entities (issues, risks, results, restructuring reasons): those belong to later
phases. Closing dates are collected as *candidates* with provenance and never
written to Silver here.

Known traps, handled explicitly:
* restructuring papers say "APPROVED ON <date>": that is the original approval
  date, not the document date, so it is recorded separately;
* ISRs have both a header report date and an "Archived on" date, which can differ
  (e.g. Seq 8: 6/14/2019 vs 21-Feb-2019); both are kept;
* month-only cover dates ("February 2023") are not turned into a day.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime

from worldbank_copilot.common.identifiers import normalize_loan_number
from worldbank_copilot.parsing.models import BlockLabel, ExtractedValue, ParsedContent

_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_FULL_DATE = re.compile(rf"\b({_MONTHS})\s+(\d{{1,2}}),\s+(\d{{4}})\b", re.IGNORECASE)
_MONTH_ONLY = re.compile(rf"\b({_MONTHS})\s+(\d{{4}})\b", re.IGNORECASE)
_APPROVED_ON = re.compile(
    rf"APPROVED\s+ON\s+((?:{_MONTHS})\s+\d{{1,2}},\s+\d{{4}}|\d{{1,2}}-[A-Za-z]{{3}}-\d{{4}})",
    re.IGNORECASE,
)
_PROJECT_IN_PARENS = re.compile(r"\((P\d{6})\)")
_PROJECT_ANY = re.compile(r"\b(P\d{6})\b")
_REPORT_NO = re.compile(r"REPORT\s+NO\.?\s*:?\s*([A-Z]{2,}[A-Z]*\d+[A-Z0-9]*)", re.IGNORECASE)
_LOAN_NO = re.compile(r"LOAN\s+(?:NUMBER|NO\.?)\s*:?\s*(\d{4}\s*-\s*[A-Z]{2})\b", re.IGNORECASE)
_LOAN_REFS = re.compile(r"\b(IBRD-?\d{5}|\d{4}-IN)\b")
_SEQ = re.compile(r"Seq\.?\s*No\.?\s*:?\s*(\d{1,3})\b", re.IGNORECASE)
_ARCHIVED = re.compile(r"Archived\s+on\s+(\d{1,2}-[A-Za-z]{3}-\d{4})", re.IGNORECASE)
_ISR_NUMBER = re.compile(r"\b(ISR\d{4,6})\b")
_ISR_TITLE = re.compile(
    r"Implementation\s+Status\s*(?:&|and)\s*Results\s+Report\s+(.+?)\s*\(P\d{6}\)",
    re.IGNORECASE | re.DOTALL,
)
_ISR_HEADER_DATE = re.compile(
    r"(\d{1,2}/\d{1,2}/\d{4}|[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})\s+Page\s+\d+\s+of\s+\d+"
)
_SHORT_DATE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4}|[A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})\b")
# Requires "Label: date". Without the colon, two-column key/value layouts such as
# "Approval Date Current Closing Date 31-Mar-2016 30-Nov-2022" (restructuring BASIC DATA)
# would wrongly pair the approval date with the closing-date label.
_CLOSING_TEXT = re.compile(
    r"\b((?:Original|Current|Revised|Proposed)\s+Closing\s+Date)\s*:\s*"
    r"(\d{1,2}-[A-Za-z]{3}-\d{4})",
    re.IGNORECASE,
)
_DMY = re.compile(r"^\d{1,2}-[A-Za-z]{3}-\d{4}$")

# Content-based type detection, checked in priority order on page 1 (then page 2).
TYPE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ISR", re.compile(r"Implementation\s+Status\s*(?:&|and)\s*Results\s+Report", re.I)),
    ("ADDITIONAL_FINANCING", re.compile(
        r"PROJECT\s+PAPER\s+ON\s+A\s+PROPOSED\s+ADDITIONAL\s+(?:LOAN|FINANCING|CREDIT)", re.I)),
    ("RESTRUCTURING_PAPER", re.compile(r"RESTRUCTURING\s+PAPER", re.I)),
    ("APPRAISAL_DOCUMENT", re.compile(r"(?:PROJECT|PROGRAM)\s+APPRAISAL\s+DOCUMENT", re.I)),
    ("CANCELLATION", re.compile(r"(?:NOTICE|LETTER)\s+OF\s+(?:PARTIAL\s+)?CANCELLATION", re.I)),
    ("PERFORMANCE_INDICATORS", re.compile(r"Performance\s+Monitoring\s+Indicators", re.I)),
    ("LOAN_AGREEMENT", re.compile(r"LOAN\s+NUMBER\s+\d{4}\s*-\s*[A-Z]{2}", re.I)),
    ("ESSA", re.compile(r"ENVIRONMENT(?:AL)?\s+AND\s+SOCIAL\s+SYSTEMS?\s+ASSESSMENT", re.I)),
    ("FIDUCIARY_ASSESSMENT", re.compile(r"FIDUCIARY\s+SYSTEMS?\s+ASSESSMENT", re.I)),
    ("TECHNICAL_ASSESSMENT", re.compile(r"TECHNICAL\s+ASSESSMENT", re.I)),
)  # fmt: skip

_COVER_DATE_TYPES = {
    "APPRAISAL_DOCUMENT",
    "ADDITIONAL_FINANCING",
    "PERFORMANCE_INDICATORS",
    "TECHNICAL_ASSESSMENT",
    "FIDUCIARY_ASSESSMENT",
    "ESSA",
    "CANCELLATION",
}


@dataclass
class DocumentEvidence:
    document_type: ExtractedValue | None = None
    type_alternatives: list[ExtractedValue] = field(default_factory=list)
    project_id: ExtractedValue | None = None
    project_ids_mentioned: list[str] = field(default_factory=list)
    title: ExtractedValue | None = None
    document_date: ExtractedValue | None = None
    document_date_basis: str | None = None
    report_date: ExtractedValue | None = None
    archive_date: ExtractedValue | None = None
    report_number: ExtractedValue | None = None
    isr_sequence: ExtractedValue | None = None
    loan_number: ExtractedValue | None = None
    extras: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


_STAMPS = re.compile(r"(?:public disclosure (?:authorized|copy)|(?:for )?official use only)",
                     re.IGNORECASE)  # fmt: skip


def _strip_stamps(text: str) -> str:
    """Remove disclosure stamps that interleave with titles in the raw text layer."""
    return _flat(_STAMPS.sub(" ", text)).strip()


def _snippet(text: str, match: re.Match[str], width: int = 60) -> str:
    start, end = max(0, match.start() - width), min(len(text), match.end() + width)
    return text[start:end].strip()


def _page_texts(content: ParsedContent, first: int = 3) -> list[tuple[int, str]]:
    return [(p.page_number, _flat(p.text_raw)) for p in content.pages[:first]]


def _find(content: ParsedContent, pattern: re.Pattern[str], method: str, *, pages: int = 3,
          group: int = 1, convert=lambda v: v) -> ExtractedValue | None:  # fmt: skip
    for number, text in _page_texts(content, pages):
        match = pattern.search(text)
        if match:
            try:
                value = convert(match.group(group))
            except ValueError:
                continue
            return ExtractedValue(value=value, page_number=number,
                                  evidence=_snippet(text, match), method=method)  # fmt: skip
    return None


def _parse_long_date(text: str) -> date:
    return datetime.strptime(_flat(text).title(), "%B %d, %Y").date()


def _parse_any(text: str) -> date:
    text = _flat(text).strip()
    for fmt in ("%m/%d/%Y", "%b %d, %Y", "%d-%b-%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(text.title() if fmt.startswith("%B") else text, fmt).date()
        except ValueError:
            continue
    raise ValueError(text)


def detect_type(content: ParsedContent) -> tuple[ExtractedValue | None, list[ExtractedValue]]:
    for pages in (1, 2):
        hits = []
        for number, text in _page_texts(content, pages):
            for doc_type, pattern in TYPE_PATTERNS:
                match = pattern.search(text)
                if match:
                    hits.append(ExtractedValue(value=doc_type, page_number=number,
                                               evidence=_snippet(text, match, 40),
                                               method=f"type_pattern:{doc_type}"))  # fmt: skip
        if hits:
            ordered = sorted(hits, key=lambda h: [t for t, _ in TYPE_PATTERNS].index(h.value))
            return ordered[0], ordered[1:]
    return None, []


def _project_ids(content: ParsedContent, evidence: DocumentEvidence) -> None:
    mentioned: list[str] = []
    for _, text in _page_texts(content, 3):
        mentioned += _PROJECT_ANY.findall(text)
    evidence.project_ids_mentioned = list(dict.fromkeys(mentioned))
    evidence.project_id = _find(content, _PROJECT_IN_PARENS, "project_id_in_parentheses") or _find(
        content, _PROJECT_ANY, "project_id_token"
    )


def _isr(content: ParsedContent, ev: DocumentEvidence) -> None:
    ev.isr_sequence = _find(content, _SEQ, "isr_seq_no", pages=2, convert=int)
    ev.archive_date = _find(
        content,
        _ARCHIVED,
        "isr_archived_on",
        pages=2,
        convert=lambda v: datetime.strptime(v.title(), "%d-%b-%Y").date(),
    )
    ev.report_number = _find(content, _ISR_NUMBER, "isr_number", pages=2)
    ev.title = _find(content, _ISR_TITLE, "isr_title", pages=1, convert=_strip_stamps)
    ev.report_date = _find(content, _ISR_HEADER_DATE, "isr_header_date_before_page_n_of_m",
                           pages=2, convert=_parse_any)  # fmt: skip
    if ev.report_date is None:
        ev.report_date = _isr_header_date_from_furniture(content)
    ev.document_date = ev.archive_date
    ev.document_date_basis = "isr_archived_date" if ev.archive_date else None
    if ev.report_date and ev.archive_date and ev.report_date.value != ev.archive_date.value:
        ev.notes.append(
            f"ISR header date {ev.report_date.value} differs from archive date "
            f"{ev.archive_date.value}; both kept (document_date uses the archive date)."
        )


def _isr_header_date_from_furniture(content: ParsedContent) -> ExtractedValue | None:
    """Fallback: a single distinct date in page-1/2 running headers/footers."""
    found: dict[date, ExtractedValue] = {}
    for block in content.blocks:
        if block.page_number > 2:
            break
        if block.label not in (BlockLabel.PAGE_HEADER, BlockLabel.PAGE_FOOTER):
            continue
        for match in _SHORT_DATE.finditer(block.text_raw):
            try:
                value = _parse_any(match.group(1))
            except ValueError:
                continue
            found.setdefault(value, ExtractedValue(
                value=value, page_number=block.page_number, evidence=block.text_raw[:120],
                method="isr_header_date_in_running_header"))  # fmt: skip
    if len(found) == 1:
        return next(iter(found.values()))
    return None


def _cover_date(content: ParsedContent, ev: DocumentEvidence) -> None:
    number, text = _page_texts(content, 1)[0] if content.pages else (1, "")
    approved = {m.start() for m in _APPROVED_ON.finditer(text)}
    for match in _FULL_DATE.finditer(text):
        preceded_by_approved = any(abs(match.start() - a) < 20 for a in approved)
        if preceded_by_approved:
            continue
        ev.document_date = ExtractedValue(value=_parse_long_date(match.group(0)),
                                          page_number=number, evidence=_snippet(text, match),
                                          method="cover_full_date")  # fmt: skip
        ev.document_date_basis = "cover_date"
        return
    month = _MONTH_ONLY.search(text)
    if month:
        month_number = datetime.strptime(month.group(1).title(), "%B").month
        ev.extras["cover_month"] = {
            "value": f"{month.group(2)}-{month_number:02d}",
            "page_number": number,
            "evidence": _snippet(text, month),
        }
        ev.notes.append("Cover shows a month only; document_date left empty (day unknown).")


def _approved_on(content: ParsedContent, ev: DocumentEvidence) -> None:
    for number, text in _page_texts(content, 1):
        match = _APPROVED_ON.search(text)
        if match:
            ev.extras["approved_on_date"] = {
                "value": _parse_any(match.group(1)).isoformat(),
                "page_number": number,
                "evidence": _snippet(text, match),
                "meaning": "approval date of the operation being restructured (not the "
                "document date)",
            }


def _closing_date_candidates(content: ParsedContent) -> list[dict]:
    """Closing-date mentions with provenance. Candidates only; never applied to Silver."""
    candidates = []
    for page in content.pages:
        text = _flat(page.text_raw)
        for match in _CLOSING_TEXT.finditer(text):
            candidates.append({"label": _flat(match.group(1)).title(), "raw_value": match.group(2),
                               "page_number": page.page_number, "source": "text",
                               "evidence": _snippet(text, match)})  # fmt: skip
    for table in content.tables:
        for col, header in enumerate(table.headers):
            if "closing" not in header.lower():
                continue
            for row in table.rows[table.header_rows :]:
                if col < len(row) and _DMY.match(row[col] or ""):
                    candidates.append({"label": header, "raw_value": row[col],
                                       "page_number": table.page_number, "source": "table",
                                       "table_id": table.table_id,
                                       "row_label": row[0] if row else None})  # fmt: skip
    return candidates


def _loan_refs(content: ParsedContent) -> list[dict]:
    refs: dict[str, dict] = {}
    for number, text in _page_texts(content, 5):
        for raw in _LOAN_REFS.findall(text):
            key = raw.replace("-", "") if raw.startswith("IBRD") else raw
            refs.setdefault(key, {"raw": raw, "normalized": normalize_loan_number(key),
                                  "first_page": number})  # fmt: skip
    return list(refs.values())


def extract_metadata(content: ParsedContent) -> DocumentEvidence:
    ev = DocumentEvidence()
    ev.document_type, ev.type_alternatives = detect_type(content)
    _project_ids(content, ev)
    doc_type = ev.document_type.value if ev.document_type else None

    if doc_type == "ISR":
        _isr(content, ev)
    else:
        ev.report_number = _find(content, _REPORT_NO, "report_no", pages=2)
        ev.loan_number = _find(content, _LOAN_NO, "loan_number", pages=2,
                               convert=lambda v: re.sub(r"\s+", "", v).upper())  # fmt: skip
        if ev.document_type:
            pattern = dict(TYPE_PATTERNS)[doc_type]
            ev.title = _find(content, pattern, "type_title_phrase", pages=2,
                             group=0, convert=lambda v: _flat(v).upper())  # fmt: skip
        if doc_type in _COVER_DATE_TYPES:
            _cover_date(content, ev)
        if doc_type == "RESTRUCTURING_PAPER":
            _approved_on(content, ev)
            ev.notes.append("Restructuring papers carry no document date on the cover; "
                            "document_date left empty.")  # fmt: skip
        if doc_type == "LOAN_AGREEMENT":
            ev.notes.append("Loan agreement date is not reliably printed on the cover.")
    ev.extras["closing_date_candidates"] = _closing_date_candidates(content)
    ev.extras["loan_references"] = _loan_refs(content)
    return ev
