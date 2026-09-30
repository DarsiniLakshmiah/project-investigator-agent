"""Formal project events (silver.project_events).

Sources are authoritative formal documents only:

* Restructuring papers: PROPOSED CHANGES flags, Loan Closing tables,
  Cancellations tables, and the stated rationale, stored verbatim (cleaned).
  No causality is inferred beyond what the paper states.
* Additional-financing project paper: AF amount (kept separate from the
  original loan), Changed/Not Changed flags and its Loan Closing table.
* ISRs: per-loan Key Dates (approval, effectiveness, closing revisions) and
  "Restructuring History" lines, which carry the formal approval dates.

Restructuring papers in this corpus are undated. Their events keep
``event_date = NULL`` and a RESTRUCTURING_DATE_UNRESOLVED issue.
``restructuring_date_candidates`` stores a deterministic *candidate*
(``candidate_event_date``, status DERIVED_FROM_EXPLICIT_SOURCE) linking to an
ISR-dated restructuring. ``event_date`` is never set from it.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from typing import NamedTuple

from worldbank_copilot.common.quality import CheckCode
from worldbank_copilot.extraction.isr import restructuring_dates
from worldbank_copilot.extraction.models import IsrSnapshot, ProjectEvent
from worldbank_copilot.extraction.provenance import (
    EvidenceRef,
    ExtractionIssue,
    ExtractionMethod,
    ExtractionStatus,
    evidence,
    issue,
)
from worldbank_copilot.extraction.tables import (
    DMY,
    clean_cell,
    parse_dmy,
    parse_number,
    section_text,
    section_title,
    sections_matching,
    statement_loan_number,
)
from worldbank_copilot.parsing.models import ParsedDocument

FORMAL_PAPER_TYPES = {"RESTRUCTURING_PAPER", "ADDITIONAL_FINANCING"}

EVENT_TYPES = (
    "APPROVAL",
    "EFFECTIVENESS",
    "RESTRUCTURING",
    "ADDITIONAL_FINANCING",
    "CANCELLATION",
    "CLOSING_DATE_CHANGE",
    "RESULTS_FRAMEWORK_CHANGE",
    "COMPONENT_CHANGE",
    "FUND_REALLOCATION",
)

# Flag label (lower-case prefix) -> event type emitted when the flag is Yes/changed.
_FLAG_EVENTS = {
    "results": "RESULTS_FRAMEWORK_CHANGE",
    "results framework": "RESULTS_FRAMEWORK_CHANGE",
    "components and cost": "COMPONENT_CHANGE",
    "components": "COMPONENT_CHANGE",
    "reallocations": "FUND_REALLOCATION",
    "reallocation between disbursement categories": "FUND_REALLOCATION",
}
_CHECK = {"✔", "✓", "x", "X", "🗸"}
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_CITED_DATE = re.compile(rf"\b(?:dated|on)\s+(?P<d>(?:{_MONTHS})\s+\d{{1,2}},\s*\d{{4}})")
_RATIONALE = r"rationale for restructuring|summary of project status and proposed changes"
_DESCRIPTION = r"description of proposed changes"
_RESTRUCTURING_LEVEL = re.compile(r"Restructuring Level (\d+)", re.I)
_APPROVAL_CELL = re.compile(rf"^Approval Date\s+({DMY})$")


def _event_id(project_id: str, *parts) -> str:
    digest = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:10]
    return f"{project_id}-EV-{digest}"


def _event(doc: ParsedDocument, event_type: str, ref: EvidenceRef, **fields) -> ProjectEvent:
    status = fields.pop("status", ExtractionStatus.EXACT)
    return ProjectEvent(
        project_id=doc.project_id,
        event_id=_event_id(
            doc.project_id,
            doc.document_id,
            event_type,
            fields.get("loan_number"),
            ref.table_id,
            ref.row,
            ref.block_id,
        ),
        event_type=event_type,
        source_document=doc.filename,
        source_page=ref.page_number,
        source_section=ref.section,
        source_text=ref.source_text,
        extraction_method=ref.extraction_method,
        status=status,
        source_refs=[ref],
        **fields,
    )


def _long_date(text: str) -> date | None:
    try:
        return datetime.strptime(" ".join(text.replace(",", ", ").split()), "%B %d, %Y").date()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Formal papers: flags
# ---------------------------------------------------------------------------


def change_flags(doc: ParsedDocument) -> tuple[dict[str, bool], list[EvidenceRef]]:
    """PROPOSED CHANGES (Yes/No) or Changed/Not Changed (tick) tables."""
    flags: dict[str, bool] = {}
    refs: list[EvidenceRef] = []
    in_flags = False
    for table in doc.tables:
        header = [clean_cell(c).lower() for c in table.rows[0]] if table.rows else []
        yes_no = "proposed changes" in header
        ticked = header[1:3] == ["changed", "not changed"]
        continuation = in_flags and all(
            clean_cell(r[i]) in ("Yes", "No", "") for r in table.rows for i in range(1, len(r), 2)
        )
        if not (yes_no or ticked or continuation):
            in_flags = False
            continue
        in_flags = yes_no or continuation
        section = section_title(doc, table.section_id)
        rows = table.rows[1:] if (yes_no or ticked) else table.rows
        for row in rows:
            cells = [clean_cell(c) for c in row]
            if ticked:
                label, changed, unchanged = (cells + ["", "", ""])[:3]
                if label and (changed in _CHECK) != (unchanged in _CHECK):
                    flags[label] = changed in _CHECK
                continue
            for i in range(0, len(cells) - 1, 2):
                label, value = cells[i], cells[i + 1]
                if label and value in ("Yes", "No"):
                    flags[label] = value == "Yes"
        refs.append(
            evidence(
                doc,
                table.page_number,
                ExtractionMethod.DOCLING_TABLE,
                section=section,
                table_id=table.table_id,
                text="; ".join(f"{k}: {'Yes' if v else 'No'}" for k, v in flags.items()),
            )
        )
    return flags, refs


def _flag(flags: dict[str, bool], *names: str) -> bool | None:
    for label, value in flags.items():
        if label.lower() in names:
            return value
    return None


# ---------------------------------------------------------------------------
# Formal papers: tables
# ---------------------------------------------------------------------------


def _header_index(header: list[str], *words: str) -> int | None:
    for i, cell in enumerate(header):
        text = clean_cell(cell).lower()
        if all(w in text for w in words):
            return i
    return None


class LoanClosingRow(NamedTuple):
    loan_number: str
    original: date | None
    current: date | None
    proposed: date | None
    ref: EvidenceRef


def loan_closing_rows(doc: ParsedDocument) -> list[LoanClosingRow]:
    """Rows of 'Loan Closing' tables (Original / Revised|Current / Proposed Closing)."""
    rows = []
    for table in doc.tables:
        header = table.rows[0] if table.rows else []
        original = _header_index(header, "original closing")
        proposed = _header_index(header, "proposed closing")
        if original is None or proposed is None:
            continue
        current = _header_index(header, "revised closing")
        if current is None:
            current = _header_index(header, "current closing")
        section = section_title(doc, table.section_id)
        for index, row in enumerate(table.rows[1:], start=1):
            cells = [clean_cell(c) for c in row]
            loan = statement_loan_number(cells[0]) if cells else None
            if not loan:
                continue

            def at(col: int | None, _cells: list[str] = cells) -> date | None:
                return parse_dmy(_cells[col]) if col is not None and col < len(_cells) else None

            ref = evidence(
                doc,
                table.page_number,
                ExtractionMethod.DOCLING_TABLE,
                section=section,
                table_id=table.table_id,
                row=index,
                text=" | ".join(cells),
            )
            rows.append(LoanClosingRow(loan, at(original), at(current), at(proposed), ref))
    return rows


def closing_changes(doc: ParsedDocument) -> list[ProjectEvent]:
    """One CLOSING_DATE_CHANGE per Loan Closing row with a proposed date."""
    return [
        _event(
            doc,
            "CLOSING_DATE_CHANGE",
            row.ref,
            loan_number=row.loan_number,
            old_closing_date=row.current or row.original,
            new_closing_date=row.proposed,
            change_description=(
                f"Original closing {row.original}; current closing "
                f"{row.current}; proposed closing {row.proposed} (as printed)"
            ),
        )
        for row in loan_closing_rows(doc)
        if row.proposed is not None
    ]


def cancellations(doc: ParsedDocument) -> list[ProjectEvent]:
    """Cancellations tables: rows with a non-zero cancellation amount."""
    events = []
    for table in doc.tables:
        header = table.rows[0] if table.rows else []
        amount_col = _header_index(header, "cancellation amount")
        if amount_col is None:
            continue
        currency_col = _header_index(header, "currency")
        date_col = _header_index(header, "value date")
        reason_col = _header_index(header, "reason")
        section = section_title(doc, table.section_id)
        for index, row in enumerate(table.rows[1:], start=1):
            cells = [clean_cell(c) for c in row]
            loan = statement_loan_number(cells[0]) if cells else None
            amount = parse_number(cells[amount_col]) if amount_col < len(cells) else None
            if not loan or not amount:
                continue
            value_date = parse_dmy(cells[date_col]) if date_col is not None else None
            ref = evidence(
                doc,
                table.page_number,
                ExtractionMethod.DOCLING_TABLE,
                section=section,
                table_id=table.table_id,
                row=index,
                column=amount_col,
                text=" | ".join(cells),
            )
            events.append(
                _event(
                    doc,
                    "CANCELLATION",
                    ref,
                    loan_number=loan,
                    event_date=value_date,
                    event_date_basis="VALUE_DATE_OF_CANCELLATION" if value_date else None,
                    cancelled_amount=amount,
                    cancelled_currency=cells[currency_col] if currency_col is not None else None,
                    reason_text=(cells[reason_col] or None) if reason_col is not None else None,
                    status=ExtractionStatus.NORMALIZED,  # wrap-injected spaces removed
                )
            )
    return events


def additional_financing(doc: ParsedDocument) -> list[ProjectEvent]:
    """AF amount from the AF paper's financing summary (IBRD/IDA row)."""
    events = []
    approval, af_project, af_type = None, None, None
    for table in doc.tables:
        text_rows = [[clean_cell(c) for c in r] for r in table.rows]
        for r_index, cells in enumerate(text_rows):
            for cell in cells:
                if match := _APPROVAL_CELL.match(cell):
                    approval = approval or (parse_dmy(match.group(1)), table, r_index)
        header = [c.lower() for c in text_rows[0]] if text_rows else []
        if "additional financing type" in header and len(text_rows) > 1:
            af_project = (
                text_rows[1][header.index("project id")] if "project id" in header else None
            )
            af_type = text_rows[1][header.index("additional financing type")]
    for table in doc.tables:
        header = [clean_cell(c).lower() for c in table.rows[0]] if table.rows else []
        col = _header_index(table.rows[0] if table.rows else [], "proposed additional financing")
        if col is None:
            continue
        for index, row in enumerate(table.rows[1:], start=1):
            cells = [clean_cell(c) for c in row]
            if not cells[0].lower().startswith("of which ibrd"):
                continue
            ref = evidence(
                doc,
                table.page_number,
                ExtractionMethod.DOCLING_TABLE,
                section=section_title(doc, table.section_id),
                table_id=table.table_id,
                row=index,
                column=col,
                text=f"{' | '.join(header)} || {' | '.join(cells)}",
            )
            event = _event(
                doc,
                "ADDITIONAL_FINANCING",
                ref,
                event_date=approval[0] if approval else None,
                event_date_basis="APPROVAL_DATE_PRINTED_IN_AF_PAPER" if approval else None,
                additional_financing_amount=parse_number(cells[col]),
                additional_financing_currency="USD_MILLIONS",
                change_description=(
                    f"AF project {af_project}; AF type: {af_type}"
                    if af_project or af_type
                    else None
                ),
            )
            if approval:
                _, table_a, row_a = approval
                event.source_refs.append(
                    evidence(
                        doc,
                        table_a.page_number,
                        ExtractionMethod.DOCLING_TABLE,
                        table_id=table_a.table_id,
                        row=row_a,
                        text=" | ".join(clean_cell(c) for c in table_a.rows[row_a]),
                    )
                )
            events.append(event)
    return events


def _section_ref(doc: ParsedDocument, pattern: str) -> tuple[str | None, EvidenceRef | None]:
    parts, ref = [], None
    for section in sections_matching(doc, pattern):
        text = section_text(doc, section)
        if not text.strip() or "...." in section.title:
            continue  # table-of-contents entries
        parts.append(text)
        ref = ref or evidence(
            doc, section.start_page, ExtractionMethod.DOCLING_TEXT, section=section.title, text=text
        )
    return ("\n".join(parts) or None), ref


def cited_dates(doc: ParsedDocument) -> list[tuple[date, str]]:
    """Explicit 'dated <Month d, yyyy>' / 'on <Month d, yyyy>' dates in the text."""
    out = []
    for block in doc.blocks:
        for match in _CITED_DATE.finditer(block.text_clean or ""):
            parsed = _long_date(match.group("d"))
            if parsed:
                out.append((parsed, block.block_id))
    return out


def extract_formal_events(doc: ParsedDocument) -> tuple[list[ProjectEvent], list[ExtractionIssue]]:
    if doc.document_type not in FORMAL_PAPER_TYPES:
        return [], []
    issues: list[ExtractionIssue] = []
    events: list[ProjectEvent] = []
    flags, flag_refs = change_flags(doc)
    closing = closing_changes(doc)
    cancelled = cancellations(doc)
    events += closing + cancelled

    if doc.document_type == "ADDITIONAL_FINANCING":
        af = additional_financing(doc)
        if not af:
            issues.append(
                issue(
                    CheckCode.EXTRACTION_AMBIGUOUS,
                    "WARNING",
                    f"{doc.filename}: AF amount table not found",
                )
            )
        events += af
        date_fields = (
            {"event_date": af[0].event_date, "event_date_basis": af[0].event_date_basis}
            if af
            else {}
        )
        for event in closing:
            event.event_date, event.event_date_basis = (
                date_fields.get("event_date"),
                date_fields.get("event_date_basis"),
            )
    else:
        reason, reason_ref = _section_ref(doc, _RATIONALE)
        description, _ = _section_ref(doc, _DESCRIPTION)
        anchor = (
            reason_ref
            or (flag_refs[0] if flag_refs else None)
            or evidence(doc, 1, ExtractionMethod.DOCLING_TEXT, text=doc.filename)
        )
        restructuring = _event(
            doc,
            "RESTRUCTURING",
            anchor,
            loan_number=None,
            change_flags=flags,
            results_framework_changed=_flag(flags, "results"),
            components_changed=_flag(flags, "components", "components and cost"),
            fund_reallocation=_flag(flags, "reallocations"),
            reason_text=reason,
            change_description=description,
        )
        restructuring.source_refs += flag_refs
        if not flags:
            restructuring.quality_issues.append(
                issue(
                    CheckCode.EXTRACTION_LIMITATION,
                    "INFO",
                    "no PROPOSED CHANGES flags table in this paper format; only explicit tables "
                    "(loan closing / cancellations) and the stated summary are extracted",
                    anchor,
                )
            )
        events.insert(0, restructuring)

    # Flag-driven change events (one per distinct event type flagged Yes/changed).
    emitted = set()
    for label, value in flags.items():
        event_type = _FLAG_EVENTS.get(label.lower())
        if not value or not event_type or event_type in emitted:
            continue
        emitted.add(event_type)
        ref = next((r for r in flag_refs if label in (r.source_text or "")), flag_refs[0])
        event = _event(
            doc,
            event_type,
            ref,
            change_flags={label: True},
            change_description=f"'{label}' marked as changed in the paper's change summary table",
        )
        events.append(event)

    # Consistency: flags vs explicit tables in the same paper.
    for label, table_events, name in (
        ("loan cancellations", cancelled, "Cancellations"),
        ("cancellations proposed", cancelled, "Cancellations"),
        ("loan closing date extension", closing, "Loan Closing"),
        ("loan closing date(s)", closing, "Loan Closing"),
    ):
        value = _flag(flags, label)
        if value is not None and value != bool(table_events):
            issues.append(
                issue(
                    CheckCode.EXTRACTION_CONFLICT,
                    "WARNING",
                    f"{doc.filename}: flag '{label}'={value} but {len(table_events)} "
                    f"{name} table event(s) extracted",
                    flag_refs[0] if flag_refs else None,
                )
            )

    af_events = [e for e in events if e.event_type == "ADDITIONAL_FINANCING"]
    for event in events:
        if event.event_date is not None:
            continue
        if af_events and af_events[0].event_date:
            # Changes proposed in the AF paper share its printed approval date.
            event.event_date = af_events[0].event_date
            event.event_date_basis = af_events[0].event_date_basis
        elif doc.document_type == "RESTRUCTURING_PAPER":
            event.event_date_basis = "UNDATED_RESTRUCTURING_PAPER"
    return events, issues


# ---------------------------------------------------------------------------
# ISR-derived events
# ---------------------------------------------------------------------------


def isr_events(
    snapshots: list[IsrSnapshot], documents: dict[str, ParsedDocument]
) -> tuple[list[ProjectEvent], list[ExtractionIssue]]:
    """APPROVAL / EFFECTIVENESS per loan, dated RESTRUCTURING, closing revisions."""
    issues: list[ExtractionIssue] = []
    events: list[ProjectEvent] = []
    ordered = sorted(snapshots, key=lambda s: s.isr_sequence or 0)
    per_loan: dict[tuple[str, str], list] = {}
    for snap in ordered:
        for kd in snap.loan_key_dates:
            for field, event_type in (
                ("approval_date", "APPROVAL"),
                ("effectiveness_date", "EFFECTIVENESS"),
            ):
                value = getattr(kd, field)
                if value:
                    per_loan.setdefault((kd.loan_number, event_type), []).append((value, kd, snap))
    for (loan, event_type), seen in sorted(per_loan.items()):
        values = {v for v, _, _ in seen}
        value, kd, snap = seen[-1]  # latest ISR reporting it
        doc = documents[snap.document_id]
        event = _event(
            doc,
            event_type,
            kd.evidence,
            loan_number=loan,
            event_date=value,
            event_date_basis=f"ISR_KEY_DATES_{event_type}",
        )
        event.source_refs.append(seen[0][1].evidence)
        if len(values) > 1:
            event.status = ExtractionStatus.CONFLICT
            event.event_date = None
            event.quality_issues.append(
                issue(
                    CheckCode.EXTRACTION_CONFLICT,
                    "WARNING",
                    f"{loan} {event_type.lower()} date differs across ISRs: "
                    f"{sorted(str(v) for v in values)}",
                    kd.evidence,
                )
            )
        events.append(event)

    # Dated restructurings (ISR 'Restructuring History').
    dated: dict[date, list] = {}
    for snap in ordered:
        for when, ref in restructuring_dates(snap):
            dated.setdefault(when, []).append((ref, snap))
    for when, seen in sorted(dated.items()):
        ref, snap = seen[-1]
        level = _RESTRUCTURING_LEVEL.search(ref.source_text or "")
        event = _event(
            documents[snap.document_id],
            "RESTRUCTURING",
            ref,
            event_date=when,
            event_date_basis="ISR_RESTRUCTURING_HISTORY_APPROVED_ON",
            change_description=f"Restructuring Level {level.group(1)}" if level else None,
        )
        event.source_refs.insert(0, seen[0][0])
        events.append(event)

    # Revised-closing changes between consecutive ISRs (first reported date).
    last: dict[str, tuple] = {}
    for snap in ordered:
        for kd in snap.loan_key_dates:
            current = kd.revised_closing_date
            if current is None:
                continue
            previous = last.get(kd.loan_number)
            if (
                previous is None
                and kd.original_closing_date
                and current != kd.original_closing_date
            ):
                old = kd.original_closing_date
            elif previous is not None and previous[0] != current:
                old = previous[0]
            else:
                last[kd.loan_number] = (current, snap)
                continue
            events.append(
                _event(
                    documents[snap.document_id],
                    "CLOSING_DATE_CHANGE",
                    kd.evidence,
                    loan_number=kd.loan_number,
                    old_closing_date=old,
                    new_closing_date=current,
                    event_date=snap.canonical_report_date,
                    event_date_basis="FIRST_REPORTED_IN_ISR (report date, not approval date)",
                    change_description=(
                        f"Revised closing first reported in ISR sequence {snap.isr_sequence}"
                    ),
                )
            )
            last[kd.loan_number] = (current, snap)
    return events, issues


def restructuring_date_candidates(
    paper_events: list[ProjectEvent],
    isr_restructurings: list[ProjectEvent],
    documents: dict[str, ParsedDocument],
) -> list[ExtractionIssue]:
    """Candidate paper -> ISR restructuring date links (reported, never applied).

    Rule: a paper cannot predate the latest 'dated/on <date>' it cites; the
    candidate is the earliest ISR-listed restructuring approval on or after that
    bound. Uniqueness is reported. The link is not asserted as fact.
    """
    issues = []
    isr_dates = sorted(e.event_date for e in isr_restructurings if e.event_date)
    for event in paper_events:
        if event.event_type != "RESTRUCTURING" or event.event_date is not None:
            continue
        doc = next(d for d in documents.values() if d.filename == event.source_document)
        cited = cited_dates(doc)
        bound = max((d for d, _ in cited), default=None)
        candidate = next((d for d in isr_dates if bound and d >= bound), None)
        details = {
            "isr_restructuring_dates": [str(d) for d in isr_dates],
            "latest_cited_date": str(bound) if bound else None,
            "candidate_date": str(candidate) if candidate else None,
            "rule": (
                "earliest ISR restructuring approval on/after the latest date cited in the paper"
            ),
        }
        issue_ = issue(
            CheckCode.RESTRUCTURING_DATE_UNRESOLVED,
            "WARNING",
            f"{event.source_document}: restructuring paper is undated; approval date "
            f"not asserted (candidate {candidate or 'none'})",
            event.source_refs[0],
            **details,
        )
        event.quality_issues.append(issue_)
        event.status = ExtractionStatus.AMBIGUOUS
        if candidate is not None:
            event.candidate_event_date = candidate
            event.candidate_date_basis = (
                "EARLIEST_ISR_RESTRUCTURING_APPROVAL_ON_OR_AFTER_LATEST_CITED_DATE"
            )
            event.candidate_date_status = ExtractionStatus.DERIVED_FROM_EXPLICIT_SOURCE
        issues.append(issue_)
    return issues


__all__ = [
    "EVENT_TYPES",
    "extract_formal_events",
    "isr_events",
    "restructuring_date_candidates",
    "change_flags",
    "closing_changes",
    "loan_closing_rows",
    "LoanClosingRow",
    "cancellations",
    "additional_financing",
    "cited_dates",
]
