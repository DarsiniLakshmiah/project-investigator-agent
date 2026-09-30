"""Appraisal-stage risks (silver.appraisal_risks): what the source itself identified.

Framing is preserved:

* FORMAL_RISK_RATING: a row of a Systematic Operations Risk-rating Tool (SORT)
  table in a PAD / Program Appraisal Document / AF Project Paper.
* ASSESSMENT_FINDING: an explicit entry in a risk table of an appraisal-stage
  assessment (technical, fiduciary, environmental & social): a table with a
  "Risk"/"Risks" column (plus mitigation or rating columns), or a Technical
  Assessment "Risk - n / Risk Rating / Mitigation Actions" block.

Extraction is constrained to those tables/blocks. Narrative paragraphs are not
turned into risks (documented limitation). Whether a risk later materialised is
decided in Gold, not here.
"""

from __future__ import annotations

import hashlib
import re

from worldbank_copilot.common.quality import CheckCode
from worldbank_copilot.extraction.models import AppraisalRisk, ExtractedRating
from worldbank_copilot.extraction.provenance import (
    EvidenceRef,
    ExtractionIssue,
    ExtractionMethod,
    ExtractionStatus,
    evidence,
    issue,
    snippet,
)
from worldbank_copilot.extraction.ratings import RISK_RATINGS, RatingScale, normalize_rating
from worldbank_copilot.extraction.tables import clean_cell, section_title, sections_matching
from worldbank_copilot.extraction.text_source import LoggedTextSource
from worldbank_copilot.parsing.models import ParsedDocument, ParsedTable

APPRAISAL_TYPES = {
    "APPRAISAL_DOCUMENT",
    "ADDITIONAL_FINANCING",
    "TECHNICAL_ASSESSMENT",
    "FIDUCIARY_ASSESSMENT",
    "ESSA",
}
FORMAL = "FORMAL_RISK_RATING"
FINDING = "ASSESSMENT_FINDING"

_TA_RISK = re.compile(r"^Risk\s*-\s*(\d+)\s*(.*)$", re.I)
_TA_RATING = re.compile(r"^Risk Rating\s*(.*)$", re.I)
_TA_MITIGATION = re.compile(r"^Mitigation(?: Actions)?\s*(.*)$", re.I)
# Only a parenthetical that is exactly a rating, e.g. "Groundwater depletion (Moderate)".
_PAREN_RATING = re.compile(rf"\((?P<r>{'|'.join(RISK_RATINGS)})\)")
_NUMBERING = re.compile(r"^\s*(?:\d+\.?)\s*")


def _risk_id(doc: ParsedDocument, *parts: str) -> str:
    digest = hashlib.sha1("|".join([doc.document_id, *parts]).encode()).hexdigest()[:10]
    return f"{doc.project_id}-{digest}"


def _rating(raw: str | None, ref: EvidenceRef, issues: list[ExtractionIssue]) -> ExtractedRating:
    normalized, status = normalize_rating(raw, RatingScale.RISK)
    if status is ExtractionStatus.AMBIGUOUS:
        issues.append(
            issue(
                CheckCode.UNKNOWN_RATING_VALUE,
                "WARNING",
                f"unrecognised risk rating {raw!r} kept raw",
                ref,
            )
        )
    return ExtractedRating(
        raw_rating=clean_cell(raw) or None,
        normalized_rating=normalized,
        status=status,
        evidence=ref,
    )


def _category_for(doc: ParsedDocument, section: str | None) -> str:
    title = (section or "").lower()
    if doc.document_type == "TECHNICAL_ASSESSMENT" or "technical" in title:
        return "Technical"
    if doc.document_type == "FIDUCIARY_ASSESSMENT" or "fiduciary" in title:
        return "Fiduciary - Procurement" if "procurement" in title else "Fiduciary"
    if "social" in title and "environment" not in title:
        return "Social"
    if "environment" in title and "social" not in title:
        return "Environmental"
    if doc.document_type == "ESSA":
        return "Environmental and Social"
    return "Other"


def _record(
    doc, framing, category, description, rating, mitigation, ref, method, activity=None, issues=None
) -> AppraisalRisk:
    status = ExtractionStatus.EXACT
    if rating is not None and rating.status is ExtractionStatus.NORMALIZED:
        status = ExtractionStatus.NORMALIZED
    return AppraisalRisk(
        project_id=doc.project_id,
        risk_id=_risk_id(doc, framing, category or "", description or "", activity or ""),
        framing=framing,
        risk_category=category,
        risk_description=description,
        activity=activity,
        risk_rating=rating,
        mitigation_text=mitigation,
        identified_date=doc.document_date,
        source_document=doc.filename,
        source_document_type=doc.document_type,
        source_page=ref.page_number,
        source_section=ref.section,
        source_text=ref.source_text,
        extraction_method=method,
        status=status,
        source_refs=[ref],
        quality_issues=issues or [],
    )


# ---------------------------------------------------------------------------
# SORT (formal ratings)
# ---------------------------------------------------------------------------


def _is_sort_table(doc: ParsedDocument, table: ParsedTable) -> bool:
    title = (section_title(doc, table.section_id) or "").lower()
    head = " ".join(clean_cell(c) for r in table.rows[:2] for c in r).lower()
    return "risk category" in head or (
        "risk" in head
        and "rating" in head
        and table.n_cols <= 3
        and ("sort" in title or "risk-rating" in title or "risk rating tool" in title)
    )


def extract_sort(doc: ParsedDocument, issues: list[ExtractionIssue]) -> list[AppraisalRisk]:
    out: dict[str, AppraisalRisk] = {}
    active_cols = None
    for table in doc.tables:
        if _is_sort_table(doc, table):
            active_cols = table.n_cols
        elif not (
            active_cols
            and table.n_cols == active_cols
            and table.rows
            and re.match(r"^\d+\.", clean_cell(table.rows[0][0]))
        ):
            active_cols = None if table.n_cols != active_cols else active_cols
            continue
        section = section_title(doc, table.section_id)
        for index, row in enumerate(table.rows):
            cells = [clean_cell(c) for c in row]
            texts = [c for c in cells if c and re.search(r"[A-Za-z]{3}", c)]
            if not texts:
                continue
            category = _NUMBERING.sub("", texts[0]).strip()
            raw_rating = cells[-1]
            if category.lower().startswith(("risk category", "risk", "systematic")) or (
                category == raw_rating
            ):
                continue
            ref = evidence(
                doc,
                table.page_number,
                ExtractionMethod.DOCLING_TABLE,
                section=section,
                table_id=table.table_id,
                row=index,
                column=len(cells) - 1,
                text=" | ".join(cells),
            )
            rating = _rating(raw_rating, ref, issues) if raw_rating not in ("", "-") else None
            key = category.lower()
            if key in out:
                existing = out[key]
                existing.source_refs.append(ref)
                old = existing.risk_rating.normalized_rating if existing.risk_rating else None
                new = rating.normalized_rating if rating else None
                if old != new:
                    existing.status = ExtractionStatus.CONFLICT
                    existing.quality_issues.append(
                        issue(
                            CheckCode.EXTRACTION_CONFLICT,
                            "WARNING",
                            f"SORT '{category}' rated {old!r} and {new!r} in the same document",
                            ref,
                        )
                    )
                continue
            out[key] = _record(
                doc, FORMAL, category, None, rating, None, ref, ExtractionMethod.DOCLING_TABLE
            )
    return list(out.values())


# ---------------------------------------------------------------------------
# Assessment risk tables
# ---------------------------------------------------------------------------


_MAX_HEADER_CELL = 60


_RISK_LABEL = re.compile(
    r"^(?:s\.?\s?n\.?\s+|sl\.?\s*no\.?\s+)?(?:key\s+|identified\s+)?"
    r"risks?(?:\s+description)?$"
)
_RATING_LABEL = re.compile(r"^(?:\w+\s+)?(?:risk\s+)?rating$")


def _risk_columns(header: list[str]) -> dict[str, int] | None:
    """Map a risk-table header row to column roles; None if it is not one.

    Header cells must be short labels of known forms ("Risk", "Risks", "SN Risk",
    "Mitigation ...", "Risk Rating", "Impact", "Justification ..."), so a body row
    that merely mentions risk or mitigation is never taken as a header.
    """
    texts = [clean_cell(c).lower().rstrip(":") for c in header]
    if any(len(t) > _MAX_HEADER_CELL for t in texts):
        return None
    cols: dict[str, int] = {}
    for i, text in enumerate(texts):
        if _RISK_LABEL.match(text):
            cols.setdefault("risk", i)
        elif text.startswith("mitigation"):
            cols.setdefault("mitigation", i)
        elif _RATING_LABEL.match(text):
            cols.setdefault("rating", i)
        elif text.startswith("impact"):
            cols.setdefault("impact", i)
        elif text.startswith("justification"):
            cols.setdefault("justification", i)
        elif i == 0 and ("activit" in text or "investment" in text):
            cols.setdefault("activity", i)
    if "risk" in cols and ("mitigation" in cols or "rating" in cols):
        return cols
    if "impact" in cols and "rating" in cols:  # ESSA risk-analysis tables
        return cols
    return None


def _cell(cells: list[str], cols: dict[str, int], role: str) -> str | None:
    index = cols.get(role)
    if index is None or index >= len(cells):
        return None
    return cells[index] or None


def _append(record: AppraisalRisk, cells: list[str], cols: dict[str, int], ref: EvidenceRef):
    """Continuation fragment of a row split across a page break."""
    key = "risk" if "risk" in cols else "impact"
    for role, field in (
        (key, "risk_description"),
        ("mitigation", "mitigation_text"),
        ("justification", "rating_justification"),
    ):
        extra = _cell(cells, cols, role)
        if extra:
            setattr(record, field, " ".join(filter(None, [getattr(record, field), extra])))
    record.source_refs.append(ref)


def extract_risk_tables(doc: ParsedDocument, issues: list[ExtractionIssue]) -> list[AppraisalRisk]:
    """Rows of explicit risk tables (Risk + Mitigation / Risk Rating columns).

    Continuation tables (no header) are accepted only on the page immediately
    after a recognised risk table with the same column count.
    """
    out: list[AppraisalRisk] = []
    active: tuple[int, int, dict[str, int]] | None = None  # (n_cols, page, cols)
    for table in doc.tables:
        if _is_sort_table(doc, table):
            active = None
            continue
        header_index, cols = None, None
        for i, row in enumerate(table.rows[:2]):
            cols = _risk_columns(row)
            if cols:
                header_index = i
                break
        if cols is None:
            if not (
                active
                and table.n_cols == active[0]
                and table.page_number in (active[1], active[1] + 1)
            ):
                active = None
                continue
            cols, header_index = active[2], -1
        active = (table.n_cols, table.page_number, cols)
        section = section_title(doc, table.section_id)
        category = _category_for(doc, section)
        first_data_row = True
        for index, row in enumerate(table.rows[header_index + 1 :], start=header_index + 1):
            cells = [clean_cell(c) for c in row]
            distinct = {c for c in cells if c}
            if not distinct or _risk_columns(row):
                continue
            ref = evidence(
                doc,
                table.page_number,
                ExtractionMethod.DOCLING_TABLE,
                section=section,
                table_id=table.table_id,
                row=index,
                text=" | ".join(cells),
            )
            raw_rating = _cell(cells, cols, "rating")
            label_cells = {c for i, c in enumerate(cells) if c and i != cols.get("rating")}
            if len(label_cells) == 1 and len(cells) > 2:
                label = next(iter(label_cells))
                if raw_rating and "aggregate" in label.lower():
                    out.append(
                        _record(
                            doc,
                            FINDING,
                            category,
                            label,
                            _rating(raw_rating, ref, issues),
                            None,
                            ref,
                            ExtractionMethod.DOCLING_TABLE,
                        )
                    )
                continue  # results-area / group label row
            activity = _cell(cells, cols, "activity")
            description = _cell(cells, cols, "risk") or _cell(cells, cols, "impact")
            if cols.get("risk") == 0 and description:
                description = re.sub(r"^\d+\s+", "", description)  # merged 'SN Risk' cells
            previous = out[-1] if out and out[-1].source_document == doc.filename else None
            # A page-top row with no rating whose activity cell is empty or starts
            # mid-sentence continues the last row of the previous page.
            fragment = (not description) or (
                "activity" in cols
                and not raw_rating
                and (not activity or (first_data_row and activity[0].islower()))
            )
            if fragment and previous is not None and (first_data_row or not description):
                _append(previous, cells, cols, ref)
                first_data_row = False
                continue
            first_data_row = False
            if not description:
                continue
            if raw_rating is None:
                match = _PAREN_RATING.search(description)
                raw_rating = match.group("r") if match else None
            rating = _rating(raw_rating, ref, issues) if raw_rating else None
            record = _record(
                doc,
                FINDING,
                category,
                description,
                rating,
                _cell(cells, cols, "mitigation"),
                ref,
                ExtractionMethod.DOCLING_TABLE,
                activity=activity,
            )
            record.rating_justification = _cell(cells, cols, "justification")
            if description[0].islower():
                record.status = ExtractionStatus.AMBIGUOUS
                record.quality_issues.append(
                    issue(
                        CheckCode.EXTRACTION_AMBIGUOUS,
                        "INFO",
                        "risk text starts mid-sentence: the source row is split across a page "
                        "break and Docling row alignment cannot be verified; kept as printed",
                        ref,
                    )
                )
            out.append(record)
    return out


# ---------------------------------------------------------------------------
# Technical Assessment "Risk - n" blocks
# ---------------------------------------------------------------------------


def _ta_from_rows(doc, rows, page, table_id, section, method, issues, text_lines=None):
    """Parse 'Risk - n <desc> / Risk Rating <r> / Mitigation Actions <text>' sequences."""
    found: dict[int, AppraisalRisk] = {}
    current = None
    for index, (label, value) in enumerate(rows):
        risk = _TA_RISK.match(label)
        if risk:
            number = int(risk.group(1))
            description = (value or risk.group(2) or "").strip()
            ref = evidence(
                doc,
                page,
                method,
                section=section,
                table_id=table_id,
                row=index,
                text=f"{label} {value or ''}",
            )
            current = _record(doc, FINDING, "Technical", description, None, None, ref, method)
            current.risk_id = _risk_id(doc, "TA", str(number))
            found[number] = current
            continue
        if current is None:
            continue
        rating = _TA_RATING.match(label)
        if rating:
            current.risk_rating = _rating(value or rating.group(1), current.source_refs[0], issues)
            continue
        mitigation = _TA_MITIGATION.match(label)
        if mitigation:
            current.mitigation_text = snippet((value or mitigation.group(1) or "").strip(), 1200)
            continue
        if text_lines is not None and current.mitigation_text is not None:
            current.mitigation_text = snippet(f"{current.mitigation_text} {label}".strip(), 1200)
    return found


def extract_ta_blocks(
    doc: ParsedDocument, text: LoggedTextSource, issues: list[ExtractionIssue]
) -> list[AppraisalRisk]:
    if doc.document_type != "TECHNICAL_ASSESSMENT":
        return []
    found: dict[int, AppraisalRisk] = {}
    for table in doc.tables:
        rows = [(clean_cell(r[0]), clean_cell(r[1]) if len(r) > 1 else "") for r in table.rows]
        if any(_TA_RISK.match(label) for label, _ in rows):
            found.update(
                _ta_from_rows(
                    doc,
                    rows,
                    table.page_number,
                    table.table_id,
                    section_title(doc, table.section_id),
                    ExtractionMethod.DOCLING_TABLE,
                    issues,
                )
            )
    # Numbered risks that Docling left outside the table (printed as headings/text).
    outside = [s for s in sections_matching(doc, r"^Risk\s*-\s*\d+")]
    for section in outside:
        number = int(_TA_RISK.match(section.title).group(1))
        if number in found:
            continue
        lines = (
            text.page_lines(
                section.start_page,
                element=f"TA risk {number}",
                reason="risk block outside the Docling table",
            )
            or []
        )
        start = next(
            (
                i
                for i, ln in enumerate(lines)
                if _TA_RISK.match(ln) and int(_TA_RISK.match(ln).group(1)) == number
            ),
            None,
        )
        if start is None:
            issues.append(
                issue(
                    CheckCode.EXTRACTION_AMBIGUOUS,
                    "WARNING",
                    f"TA risk {number}: block not found in PDF text layer",
                )
            )
            continue
        block, rows = lines[start:], []
        for line in block:
            if rows and _TA_RISK.match(line):
                break
            if re.match(r"^\d+\.\s+[A-Z]", line) and rows:
                break  # next numbered heading
            label = line
            if m := _TA_RATING.match(line):
                rows.append(("Risk Rating", m.group(1)))
            elif m := _TA_MITIGATION.match(line):
                rows.append(("Mitigation Actions", m.group(1).replace("Actions", "").strip()))
            elif _TA_RISK.match(line):
                rows.append((line, ""))
            elif line.strip().lower() != "actions":
                rows.append((label, None))
        parsed = _ta_from_rows(
            doc,
            rows,
            section.start_page,
            None,
            section.title,
            ExtractionMethod.PDF_TEXT_FALLBACK,
            issues,
            text_lines=True,
        )
        found.update({k: v for k, v in parsed.items() if k == number})
        issues.append(
            issue(
                CheckCode.PDF_TEXT_FALLBACK_USED,
                "INFO",
                f"TA risk {number} read from the PDF text layer (outside table)",
                evidence(doc, section.start_page, ExtractionMethod.PDF_TEXT_FALLBACK),
            )
        )
    return [found[k] for k in sorted(found)]


def extract_appraisal_risks(
    doc: ParsedDocument, text: LoggedTextSource
) -> tuple[list[AppraisalRisk], list[ExtractionIssue]]:
    if doc.document_type not in APPRAISAL_TYPES:
        return [], []
    issues: list[ExtractionIssue] = []
    risks = (
        extract_sort(doc, issues)
        if doc.document_type in ("APPRAISAL_DOCUMENT", "ADDITIONAL_FINANCING")
        else []
    )
    ta = extract_ta_blocks(doc, text, issues)
    table_findings = [
        r
        for r in extract_risk_tables(doc, issues)
        if not (ta and _TA_RISK.match(r.risk_description or ""))
    ]
    return [*risks, *ta, *table_findings], issues
