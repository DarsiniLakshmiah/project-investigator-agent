"""ISR snapshot extraction: metadata, ratings, date policy, narratives, loans, SORT."""

from datetime import date
from decimal import Decimal

from tests.support.extraction_builders import RATINGS_ROWS, Tbl, isr_doc, text_source

from worldbank_copilot.common.quality import CheckCode
from worldbank_copilot.extraction.isr import extract_isr_snapshot
from worldbank_copilot.extraction.provenance import ExtractionMethod, ExtractionStatus
from worldbank_copilot.extraction.ratings import RatingScale, normalize_rating
from worldbank_copilot.extraction.text_source import FallbackLog

DISB = [
    [
        "Project",
        "Loan/Credit/TF",
        "Status",
        "Currency",
        "Original",
        "Revised",
        "Cancelled",
        "Disbursed",
        "Undisbursed",
        "% Disbursed",
    ],
    [
        "P130544",
        "IBRD-86010",
        "Effective",
        "USD",
        "100.00",
        "100.00",
        "0.00",
        "54.07",
        "45.93",
        "54%",
    ],
]
KEY_DATES = [
    [
        "Project",
        "Loan/Credit/TF",
        "Status",
        "Approval Date",
        "Signing Date",
        "Effectiveness Date",
        "Orig. Closing Date",
        "Rev. Closing Date",
    ],
    [
        "P130544",
        "IBRD-86010",
        "Effective",
        "31-Mar-2016",
        "24-May-2016",
        "22-Aug-2016",
        "30-Nov-2022",
        "22-Nov-2024",
    ],
]
SORT = [
    ["Risk Category", "Rating at Approval", "Previous Rating", "Current Rating"],
    ["Political and Governance", "Moderate", "Substantial", "Substantial"],
    ["Fiduciary", "Substantial", "Substantial", "Moderate"],
]
SORT_CONTINUATION = [
    ["Stakeholders", "Moderate", "Moderate", "Moderate"],
    ["Overall", "Substantial", "Substantial", "Substantial"],
]


def _full_doc(**kw):
    return isr_doc(
        [
            Tbl(1, RATINGS_ROWS, after="Ratings follow."),
            Tbl(1, SORT),
            Tbl(2, SORT_CONTINUATION),
            Tbl(4, DISB, after="Loans."),
            Tbl(4, KEY_DATES, after="Dates."),
        ],
        **kw,
    )


def test_isr_metadata_and_source_pages():
    doc = _full_doc()
    snap = extract_isr_snapshot(doc, text_source(doc))
    assert (snap.project_id, snap.isr_sequence, snap.document_id) == ("P130544", 5, doc.document_id)
    assert snap.source_document == doc.filename
    assert 1 in snap.source_pages and 4 in snap.source_pages
    assert all(ref.source_hash == doc.source_hash for ref in snap.source_refs)


def test_ratings_from_table_keep_raw_and_previous():
    doc = _full_doc()
    snap = extract_isr_snapshot(doc, text_source(doc))
    assert snap.pdo_rating.normalized_rating == "Moderately Satisfactory"
    assert snap.previous_pdo_rating.normalized_rating == "Moderately Unsatisfactory"
    assert snap.implementation_progress_rating.raw_rating == "Satisfactory"
    assert snap.overall_risk_rating.normalized_rating == "Substantial"
    ev = snap.pdo_rating.evidence
    assert (ev.page_number, ev.table_id, ev.extraction_method) == (
        1,
        "t0001",
        ExtractionMethod.DOCLING_TABLE,
    )


def test_rating_normalization_is_controlled_vocabulary():
    assert normalize_rating("Moderate", RatingScale.RISK) == (
        "Moderate",
        ExtractionStatus.NORMALIZED,
    )
    assert normalize_rating("--", RatingScale.RISK) == (None, ExtractionStatus.MISSING)
    # Wrong scale and abbreviations are never guessed.
    assert normalize_rating("Substantial", RatingScale.PERFORMANCE)[1] is ExtractionStatus.AMBIGUOUS
    assert normalize_rating("MS", RatingScale.PERFORMANCE) == (None, ExtractionStatus.AMBIGUOUS)


def test_unknown_rating_value_stays_visible():
    rows = [r[:] for r in RATINGS_ROWS]
    rows[2][2] = "Excellent"
    doc = isr_doc([Tbl(1, rows, after="Ratings follow.")])
    snap = extract_isr_snapshot(doc, text_source(doc))
    assert snap.implementation_progress_rating is None or (
        snap.implementation_progress_rating.raw_rating == "Excellent"
    )
    codes = {i.code for i in snap.quality_issues}
    assert CheckCode.UNKNOWN_RATING_VALUE in codes


def test_ratings_text_fallback_with_glyphs_is_logged():
    doc = isr_doc([])
    lines = {
        1: [
            "Progress towards achievement of PDO Moderately Unsatisfactory "
            "Moderately Satisfactory",
            "Overall Implementation Progress (IP) Satisfactory Satisfactory",
            "Overall Risk Rating Moderate Substantial",
        ]
    }
    log = FallbackLog()
    snap = extract_isr_snapshot(doc, text_source(doc, lines, log))
    assert snap.pdo_rating.normalized_rating == "Moderately Satisfactory"
    assert snap.pdo_rating.evidence.extraction_method is ExtractionMethod.PDF_TEXT_FALLBACK
    assert snap.overall_risk_rating.normalized_rating == "Substantial"
    assert log.invocations and log.invocations[0].element == "ratings"
    assert CheckCode.PDF_TEXT_FALLBACK_USED in {i.code for i in snap.quality_issues}


def test_ratings_label_block_column_major():
    doc = isr_doc([])
    lines = {
        1: [
            "Progress towards achievement of PDO",
            "Overall Implementation Progress (IP)",
            "Overall Risk Rating",
            "Moderately Unsatisfactory",
            "Satisfactory",
            "Moderate",
            "Moderately Satisfactory",
            "Satisfactory",
            "Substantial",
        ]
    }
    snap = extract_isr_snapshot(doc, text_source(doc, lines))
    assert snap.previous_pdo_rating.normalized_rating == "Moderately Unsatisfactory"
    assert snap.pdo_rating.normalized_rating == "Moderately Satisfactory"
    assert snap.overall_risk_rating.normalized_rating == "Substantial"


def test_missing_ratings_reported_not_invented():
    doc = isr_doc([])
    snap = extract_isr_snapshot(doc, text_source(doc, {}))
    assert snap.pdo_rating is None
    assert CheckCode.RATING_NOT_FOUND in {i.code for i in snap.quality_issues}


def test_date_policy_header_first_and_material_difference():
    doc = _full_doc(report_date=date(2025, 5, 31), archive_date=date(2024, 9, 11))
    snap = extract_isr_snapshot(doc, text_source(doc))
    assert snap.canonical_report_date == date(2025, 5, 31)
    assert snap.canonical_date_basis == "header_date"
    assert (snap.header_date, snap.archive_date, snap.date_difference_days) == (
        date(2025, 5, 31),
        date(2024, 9, 11),
        262,
    )
    diff = [i for i in snap.quality_issues if i.code == CheckCode.ISR_DATE_DIFFERENCE]
    assert diff and diff[0].severity == "WARNING" and diff[0].details["difference_days"] == 262


def test_date_policy_small_difference_is_info_and_archive_fallback():
    small = extract_isr_snapshot(_full_doc(), text_source(_full_doc()))
    diff = [i for i in small.quality_issues if i.code == CheckCode.ISR_DATE_DIFFERENCE]
    assert diff[0].severity == "INFO" and diff[0].details["difference_days"] == 5
    doc = _full_doc(report_date=None, archive_date=date(2020, 1, 10))
    snap = extract_isr_snapshot(doc, text_source(doc))
    assert (snap.canonical_report_date, snap.canonical_date_basis) == (
        date(2020, 1, 10),
        "archive_date",
    )


def test_narratives_kept_verbatim_with_provenance():
    doc = _full_doc()
    snap = extract_isr_snapshot(doc, text_source(doc))
    assert snap.key_issues_text == "Issue narrative B."
    assert snap.implementation_status_text == "Status narrative A."
    section = next(n for n in snap.narrative_sections if n.title == "Key Issues")
    assert section.evidence.page_number == 2


def test_disbursements_key_dates_and_totals():
    doc = _full_doc()
    snap = extract_isr_snapshot(doc, text_source(doc))
    loan = snap.loan_disbursements[0]
    assert loan.loan_number == "IBRD86010" and loan.currency == "USD"
    assert (loan.revised_musd, loan.disbursed_musd, loan.disbursed_pct_reported) == (
        Decimal("100.00"),
        Decimal("54.07"),
        Decimal("54"),
    )
    assert snap.commitment_amount_musd == Decimal("100.00")
    kd = snap.loan_key_dates[0]
    assert (kd.original_closing_date, kd.revised_closing_date) == (
        date(2022, 11, 30),
        date(2024, 11, 22),
    )
    assert kd.evidence.page_number == 4


def test_key_dates_split_header_uses_text_fallback_with_placeholders():
    split = [
        [
            "Project",
            "Loan/Credit/TF",
            "Status",
            "Approval Date",
            "Signing Date",
            "Effectiveness Date",
            "Orig.",
            "Closing Date",
            "Rev. Closing Date",
        ],
        [
            "P130544",
            "IBRD-93240",
            "Not Effective",
            "21-Dec-2021",
            "--",
            "--",
            "",
            "30-Jun-2026",
            "30-Jun-2026",
        ],
    ]
    doc = isr_doc([Tbl(4, split, after="Dates.")])
    lines = {4: ["P130544 IBRD-93240 Not Effective 21-Dec-2021 -- -- 30-Jun-2026 30-Jun-2026"]}
    snap = extract_isr_snapshot(doc, text_source(doc, lines))
    kd = snap.loan_key_dates[0]
    assert kd.approval_date == date(2021, 12, 21) and kd.signing_date is None
    assert kd.original_closing_date == date(2026, 6, 30)
    assert kd.evidence.extraction_method is ExtractionMethod.PDF_TEXT_FALLBACK


def test_sort_table_continues_across_page_break():
    doc = _full_doc()
    snap = extract_isr_snapshot(doc, text_source(doc))
    categories = [s.risk_category for s in snap.sort_ratings]
    assert categories == ["Political and Governance", "Fiduciary", "Stakeholders", "Overall"]
    fiduciary = snap.sort_ratings[1]
    assert (
        fiduciary.rating_at_approval.normalized_rating,
        fiduciary.current_rating.normalized_rating,
    ) == ("Substantial", "Moderate")


def test_restructuring_history_requires_full_line():
    doc = _full_doc()
    snap = extract_isr_snapshot(doc, text_source(doc))
    assert [r.source_text for r in snap.restructuring_history] == [
        "Restructuring Level 2 Approved on 23-Jul-2024"
    ]
    assert snap.restructuring_history[0].page_number == 4


def test_historical_disbursed_column_kept_as_printed():
    rows = [DISB[0][:9] + ["Historical Disbursed", "% Disbursed"],
            ["P506272", "IBRD-98350", "Effective", "USD", "426.00", "386.20", "0.00", "130.91",
             "255.30", "131.96", "33.90%"]]  # fmt: skip
    doc = isr_doc([Tbl(4, rows, after="Loans.")])
    loan = extract_isr_snapshot(doc, text_source(doc)).loan_disbursements[0]
    assert (loan.disbursed_musd, loan.historical_disbursed_musd, loan.disbursed_pct_reported) == (
        Decimal("130.91"),
        Decimal("131.96"),
        Decimal("33.90"),
    )
