"""Formal events, original-closing reconciliation, enrichment and cross-source checks."""

from datetime import date
from decimal import Decimal
from types import SimpleNamespace

from pydantic import BaseModel
from tests.support.extraction_builders import (
    RATINGS_ROWS,
    S,
    T,
    Tbl,
    isr_doc,
    make_doc,
    text_source,
)

from worldbank_copilot.common.quality import CheckCode
from worldbank_copilot.extraction.closing_dates import (
    apply_project_enrichment,
    reconcile_original_closing,
)
from worldbank_copilot.extraction.crosscheck import crosscheck_project
from worldbank_copilot.extraction.events import (
    extract_formal_events,
    isr_events,
    restructuring_date_candidates,
)
from worldbank_copilot.extraction.isr import extract_isr_snapshot
from worldbank_copilot.extraction.provenance import ExtractionMethod, ExtractionStatus

FLAGS = [
    ["Operation Information", "Proposed Changes", "Operation Information", "Proposed Changes"],
    ["Results", "Yes", "Loan Closing Date Extension", "Yes"],
    ["Development Objective", "No", "Loan Cancellations", "Yes"],
    ["Components", "No", "Reallocations", "Yes"],
]
FLAGS_CONTINUATION = [["Risks", "No", "Procurement", "No"]]
CLOSING = [
    [
        "Loan/Credit/Trust Fund",
        "Status",
        "Original Closing",
        "Revised Closing(s)",
        "Proposed Closing",
        "Proposed Deadline for Withdrawal Applications",
    ],
    ["IBRD-86010", "Closed", "30-Nov-2022", "22-Nov-2024", "", ""],
    ["IBRD-93240", "Effective", "30-Jun-2026", "30-Jun-2026", "30-Sep-2027", "30-Jan-2028"],
]
CANCEL = [
    [
        "Loan/Credit/ Trust Fund",
        "Status",
        "Currency",
        "Current Amount",
        "Cancellation Amount",
        "Value Date of Cancellation",
        "New Amount",
        "Reason",
    ],
    ["IBRD-86010- 001", "Disbursing", "USD", "100.00", "0.00", "", "100.00", ""],
    [
        "IBRD-93240- 001",
        "Disbursing",
        "JPY",
        "24,069,800,0 00.00",
        "4,011,633,25 0.00",
        "06-Aug-2024",
        "20,058,166,75 0.00",
        "SYNTHETIC REASON",
    ],
]


def _paper(tables, pages=None, **kw):
    pages = pages or [
        [
            (S, "B. Rationale for Restructuring"),
            (T, "Synthetic rationale sentence, email dated June 22, 2026."),
        ],
        [(S, "II. DESCRIPTION OF PROPOSED CHANGES"), (T, "Synthetic description.")],
        [(S, "III. PROPOSED CHANGES"), (T, "Flags.")],
        [(S, "Loan Closing"), (T, "Closing."), (S, "Cancellations"), (T, "Cancel.")],
    ]
    defaults = {"document_type": "RESTRUCTURING_PAPER", "filename": "res.pdf"}
    defaults.update(kw)
    return make_doc(pages, tables, **defaults)


def _standard_paper():
    return _paper(
        [
            Tbl(3, FLAGS, after="Flags."),
            Tbl(3, FLAGS_CONTINUATION),
            Tbl(4, CLOSING, after="Closing."),
            Tbl(4, CANCEL, after="Cancel."),
        ]
    )


def _by_type(events):
    out = {}
    for e in events:
        out.setdefault(e.event_type, []).append(e)
    return out


def test_restructuring_paper_flags_reason_and_no_causal_inference():
    events, issues = extract_formal_events(_standard_paper())
    restructuring = _by_type(events)["RESTRUCTURING"][0]
    assert restructuring.change_flags["Results"] is True
    assert restructuring.change_flags["Development Objective"] is False
    assert restructuring.change_flags["Procurement"] is False  # continuation table
    assert (
        restructuring.results_framework_changed,
        restructuring.fund_reallocation,
        restructuring.components_changed,
    ) == (True, True, False)
    assert restructuring.reason_text.startswith("Synthetic rationale sentence")
    assert restructuring.change_description == "Synthetic description."
    assert restructuring.event_date is None
    assert restructuring.event_date_basis == "UNDATED_RESTRUCTURING_PAPER"
    assert not issues  # flags agree with the explicit tables


def test_closing_date_change_from_loan_closing_table():
    events, _ = extract_formal_events(_standard_paper())
    (change,) = _by_type(events)["CLOSING_DATE_CHANGE"]
    assert change.loan_number == "IBRD93240"
    assert (change.old_closing_date, change.new_closing_date) == (
        date(2026, 6, 30),
        date(2027, 9, 30),
    )
    ref = change.source_refs[0]
    assert (ref.page_number, ref.table_id, ref.extraction_method) == (
        4,
        "t0003",
        ExtractionMethod.DOCLING_TABLE,
    )


def test_cancellation_keeps_currency_value_date_and_reason():
    events, _ = extract_formal_events(_standard_paper())
    (cancel,) = _by_type(events)["CANCELLATION"]
    assert cancel.loan_number == "IBRD93240"
    assert (cancel.cancelled_amount, cancel.cancelled_currency) == (Decimal("4011633250.00"), "JPY")
    assert (cancel.event_date, cancel.event_date_basis) == (
        date(2024, 8, 6),
        "VALUE_DATE_OF_CANCELLATION",
    )
    assert cancel.reason_text == "SYNTHETIC REASON"


def test_flag_events_and_flag_table_conflict():
    events, _ = extract_formal_events(_standard_paper())
    types = _by_type(events)
    assert "RESULTS_FRAMEWORK_CHANGE" in types and "FUND_REALLOCATION" in types
    assert "COMPONENT_CHANGE" not in types
    # Flag says cancellation but no cancellation table row -> conflict observation.
    _, issues = extract_formal_events(_paper([Tbl(3, FLAGS, after="Flags.")]))
    assert any(
        i.code == CheckCode.EXTRACTION_CONFLICT and "loan cancellations" in i.message
        for i in issues
    )


def test_additional_financing_kept_separate_with_printed_date():
    pages = [
        [(S, "BASIC INFORMATION - ADDITIONAL FINANCING"), (T, "Basic.")],
        [(S, "SUMMARY (Total Financing)"), (T, "Summary.")],
        [(S, "Summary of changes"), (T, "Changes.")],
    ]
    basic = [
        ["Project ID", "Project Name", "Additional Financing Type", "Urgent Need"],
        ["P000001", "Synthetic AF", "Cost Overrun/Financing Gap", "No"],
    ]
    approval = [["Financing instrument X", "Product line Y", "Approval Date 04-Jan-2022"]]
    summary = [
        ["", "Current Financing", "Proposed Additional Financing", "Total"],
        ["Total Project Cost", "153.00", "221.85", "374.85"],
        ["of which IBRD/IDA", "100.00", "150.00", "250.00"],
    ]
    changes = [
        ["", "Changed", "Not Changed"],
        ["Results Framework", "✔", ""],
        ["Cancellations Proposed", "", "✔"],
    ]
    doc = _paper(
        [
            Tbl(1, basic, after="Basic."),
            Tbl(1, approval),
            Tbl(2, summary, after="Summary."),
            Tbl(3, changes, after="Changes."),
        ],
        pages=pages,
        document_type="ADDITIONAL_FINANCING",
        filename="af.pdf",
    )
    events, issues = extract_formal_events(doc)
    types = _by_type(events)
    (af,) = types["ADDITIONAL_FINANCING"]
    assert (af.additional_financing_amount, af.additional_financing_currency) == (
        Decimal("150.00"),
        "USD_MILLIONS",
    )
    assert af.event_date == date(2022, 1, 4)
    assert af.event_date_basis == "APPROVAL_DATE_PRINTED_IN_AF_PAPER"
    assert "P000001" in af.change_description and len(af.source_refs) == 2
    assert "RESTRUCTURING" not in types
    assert types["RESULTS_FRAMEWORK_CHANGE"][0].event_date == date(2022, 1, 4)
    assert not issues


def _isr(sequence, key_rows, history=(), report=None):
    header = [
        "Project",
        "Loan/Credit/TF",
        "Status",
        "Approval Date",
        "Signing Date",
        "Effectiveness Date",
        "Orig. Closing Date",
        "Rev. Closing Date",
    ]
    extra = [[(S, "Restructuring History"), *[(T, h) for h in history]]] if history else []
    doc = isr_doc(
        [
            Tbl(1, RATINGS_ROWS, after="Ratings follow."),
            Tbl(4, [header, *key_rows], after="Dates."),
        ],
        extra_pages=extra,
        isr_sequence=sequence,
        document_id=f"isr-{sequence}",
        report_date=report or date(2020 + sequence, 1, 1),
    )
    return doc, extract_isr_snapshot(doc, text_source(doc))


ROW_A = [
    "P130544",
    "IBRD-86010",
    "Effective",
    "31-Mar-2016",
    "24-May-2016",
    "22-Aug-2016",
    "30-Nov-2022",
    "30-Nov-2022",
]
ROW_B = ROW_A[:7] + ["22-Nov-2024"]


def test_isr_events_approval_restructuring_and_closing_revision():
    d1, s1 = _isr(1, [ROW_A])
    d2, s2 = _isr(2, [ROW_B], history=["Restructuring Level 2 Approved on 20-May-2021"])
    events, _ = isr_events([s2, s1], {d.document_id: d for d in (d1, d2)})
    types = _by_type(events)
    assert types["APPROVAL"][0].event_date == date(2016, 3, 31)
    assert types["EFFECTIVENESS"][0].event_date == date(2016, 8, 22)
    # One line from the fixture's base history section plus one added in ISR 2.
    assert [e.event_date for e in types["RESTRUCTURING"]] == [date(2021, 5, 20), date(2024, 7, 23)]
    restructuring = types["RESTRUCTURING"][0]
    assert restructuring.event_date_basis == "ISR_RESTRUCTURING_HISTORY_APPROVED_ON"
    (change,) = types["CLOSING_DATE_CHANGE"]
    assert (change.old_closing_date, change.new_closing_date) == (
        date(2022, 11, 30),
        date(2024, 11, 22),
    )
    assert change.event_date == s2.canonical_report_date
    assert change.event_date_basis.startswith("FIRST_REPORTED_IN_ISR")


def test_conflicting_isr_key_dates_are_not_resolved():
    other = ROW_A[:3] + ["01-Apr-2016"] + ROW_A[4:]
    d1, s1 = _isr(1, [ROW_A])
    d2, s2 = _isr(2, [other])
    events, _ = isr_events([s1, s2], {d.document_id: d for d in (d1, d2)})
    approval = _by_type(events)["APPROVAL"][0]
    assert approval.status is ExtractionStatus.CONFLICT and approval.event_date is None


def test_restructuring_date_is_candidate_only():
    paper = _standard_paper()
    events, _ = extract_formal_events(paper)
    d1, s1 = _isr(
        1,
        [ROW_A],
        history=[
            "Restructuring Level 2 Approved on 10-Dec-2024",
            "Restructuring Level 2 Approved on 29-Jun-2026",
        ],
    )
    dated, _ = isr_events([s1], {d1.document_id: d1})
    issues = restructuring_date_candidates(events, dated, {paper.document_id: paper})
    restructuring = _by_type(events)["RESTRUCTURING"][0]
    assert restructuring.event_date is None and restructuring.status is ExtractionStatus.AMBIGUOUS
    (unresolved,) = issues
    assert unresolved.code == CheckCode.RESTRUCTURING_DATE_UNRESOLVED
    assert unresolved.details["latest_cited_date"] == "2026-06-22"
    assert unresolved.details["candidate_date"] == "2026-06-29"
    # The candidate is exposed as a derived value; the source event date stays unknown.
    assert restructuring.event_date is None
    assert restructuring.candidate_event_date == date(2026, 6, 29)
    assert restructuring.candidate_date_status is ExtractionStatus.DERIVED_FROM_EXPLICIT_SOURCE
    assert restructuring.candidate_date_basis.startswith("EARLIEST_ISR_RESTRUCTURING")


def _loan(**kw):
    base = {
        "project_id": "P130544",
        "raw_loan_number": "IBRD86010",
        "original_principal_usd": Decimal("100000000"),
        "cancelled_amount_usd": Decimal(0),
        "disbursed_amount_usd": Decimal("100000000"),
        "board_approval_date": date(2016, 3, 31),
        "agreement_signing_date": date(2016, 5, 24),
        "effective_date": date(2016, 8, 22),
        "closing_date": date(2024, 11, 22),
        "snapshot_date": date(2026, 8, 31),
    }
    base.update(kw)
    return SimpleNamespace(**base)


def test_original_closing_established_only_when_candidates_agree():
    d1, s1 = _isr(1, [ROW_A])
    d2, s2 = _isr(2, [ROW_B])
    paper = _standard_paper()
    records, issues = reconcile_original_closing("P130544", [s1, s2], [paper], [_loan()])
    loan = next(
        r for r in records if r.target_table == "silver_loans" and r.loan_number == "IBRD86010"
    )
    assert (loan.value, loan.status) == (
        date(2022, 11, 30),
        ExtractionStatus.DERIVED_FROM_EXPLICIT_SOURCE,
    )
    assert len(loan.evidence) >= 2  # ISR and restructuring-paper evidence
    project = next(r for r in records if r.target_table == "silver_projects")
    assert project.value == date(2022, 11, 30) and project.loan_number == "IBRD86010"
    assert CheckCode.ORIGINAL_CLOSING_DATE_ESTABLISHED in {i.code for i in issues}

    conflicting = ROW_A[:6] + ["30-Nov-2023", "30-Nov-2023"]
    d3, s3 = _isr(3, [conflicting])
    records, issues = reconcile_original_closing("P130544", [s1, s3], [], [_loan()])
    project = next(r for r in records if r.target_table == "silver_projects")
    assert project.value is None
    assert CheckCode.ORIGINAL_CLOSING_DATE_UNRESOLVED in {i.code for i in issues}


class _Project(BaseModel):
    project_id: str
    original_closing_date: date | None = None


def test_enrichment_is_explicit_and_never_mutates_silver():
    d1, s1 = _isr(1, [ROW_A])
    records, _ = reconcile_original_closing("P130544", [s1], [], [_loan()])
    silver = [_Project(project_id="P130544")]
    enriched, lineage = apply_project_enrichment(silver, records)
    assert silver[0].original_closing_date is None
    assert enriched[0].original_closing_date == date(2022, 11, 30)
    assert lineage[0]["applied"] is True and lineage[0]["evidence"]
    kept = [_Project(project_id="P130544", original_closing_date=date(2030, 1, 1))]
    enriched, lineage = apply_project_enrichment(kept, records)
    assert enriched[0].original_closing_date == date(2030, 1, 1)
    assert lineage[0]["applied"] is False


def test_cross_source_agreement_difference_and_not_comparable():
    d1, s1 = _isr(1, [ROW_B])
    events, _ = extract_formal_events(_standard_paper())
    loans = [
        _loan(),
        _loan(
            raw_loan_number="IBRD93240",
            board_approval_date=date(2021, 12, 21),
            cancelled_amount_usd=Decimal("24937499.48"),
        ),
    ]
    observations = crosscheck_project("P130544", [s1], events, loans)
    codes = [(o.code, o.details.get("field")) for o in observations]
    assert (CheckCode.CROSS_SOURCE_DIFFERENCE, "loan_numbers") in codes  # ISR lists one loan
    assert (CheckCode.CROSS_SOURCE_AGREEMENT, "closing_date") in codes
    assert (CheckCode.CROSS_SOURCE_NOT_COMPARABLE, "cancelled_amount") in codes
    loans[0].closing_date = date(2025, 1, 1)
    observations = crosscheck_project("P130544", [s1], [], loans)
    difference = next(o for o in observations if o.details.get("field") == "closing_date")
    assert difference.code == CheckCode.CROSS_SOURCE_DIFFERENCE
    assert difference.details["document_value"] == "2024-11-22"
    assert difference.details["structured_value"] == "2025-01-01"
