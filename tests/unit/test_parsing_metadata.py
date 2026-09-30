"""Document-type handlers, ISR metadata and manifest reconciliation (no Docling)."""

from datetime import date

import pytest

from support.parsing_builders import H, S, T, content_from_pages, isr_pages
from worldbank_copilot.parsing.metadata import detect_type, extract_metadata
from worldbank_copilot.parsing.models import ExtractedValue, ParsedTable, ValidationStatus
from worldbank_copilot.parsing.reconcile import reconcile, reconcile_field


def test_isr_metadata_old_format():
    ev = extract_metadata(content_from_pages(isr_pages()))
    assert ev.document_type.value == "ISR"
    assert ev.isr_sequence.value == 5 and ev.isr_sequence.page_number == 1
    assert ev.archive_date.value == date(2017, 11, 29)
    assert ev.report_date.value == date(2017, 11, 29)
    assert ev.report_number.value == "ISR30462"
    assert ev.project_id.value == "P130544"
    assert ev.document_date_basis == "isr_archived_date"
    assert ev.title.value == "IN Karnataka Urban Water Supply Modernization Project"


def test_isr_header_and_archive_dates_both_preserved():
    ev = extract_metadata(content_from_pages(isr_pages(seq=8, archived="21-Feb-2019",
                                                       header_date="6/14/2019")))  # fmt: skip
    assert ev.report_date.value == date(2019, 6, 14)
    assert ev.archive_date.value == date(2019, 2, 21)
    assert ev.document_date.value == date(2019, 2, 21)  # archive date is the document date
    assert any("differs from archive date" in n for n in ev.notes)


def test_isr_new_format_without_header_date():
    ev = extract_metadata(
        content_from_pages(
            isr_pages(seq=20, archived="11-Oct-2024", header_date=None, isr_number="ISR01465")
        )
    )
    assert ev.isr_sequence.value == 20
    assert ev.report_date is None
    assert ev.archive_date.value == date(2024, 10, 11)


def test_isr_month_name_header_date():
    ev = extract_metadata(
        content_from_pages(isr_pages(seq=22, archived="23-May-2025", header_date="May 23, 2025"))
    )
    assert ev.report_date.value == date(2025, 5, 23)


def test_restructuring_paper_does_not_use_approval_date():
    content = content_from_pages([[
        (T, "REPORT NO.: RES33572"), (T, "RESTRUCTURING PAPER ON A PROPOSED PROJECT RESTRUCTURING"),
        (T, "OF IN KARNATAKA URBAN WATER SUPPLY MODERNIZATION PROJECT"),
        (T, "APPROVED ON MARCH 31, 2016"), (T, "TO GOVERNMENT OF INDIA"),
    ]])  # fmt: skip
    ev = extract_metadata(content)
    assert ev.document_type.value == "RESTRUCTURING_PAPER"
    assert ev.report_number.value == "RES33572"
    assert ev.document_date is None
    assert ev.extras["approved_on_date"]["value"] == "2016-03-31"
    assert "not the document date" in ev.extras["approved_on_date"]["meaning"]


def test_additional_financing_beats_restructuring_and_uses_cover_date():
    content = content_from_pages([[
        (T, "Report No: PAD4503"),
        (T, "PROJECT PAPER ON A PROPOSED ADDITIONAL LOAN IN THE AMOUNT OF US$ 150 MILLION"),
        (T, "AND RESTRUCTURING PAPER"), (T, "November 22, 2021"),
    ]])  # fmt: skip
    ev = extract_metadata(content)
    assert ev.document_type.value == "ADDITIONAL_FINANCING"
    assert [a.value for a in ev.type_alternatives] == ["RESTRUCTURING_PAPER"]
    assert ev.document_date.value == date(2021, 11, 22)
    assert ev.report_number.value == "PAD4503"


def test_appraisal_uppercase_cover_date():
    ev = extract_metadata(
        content_from_pages(
            [[(T, "Report No: PADHP00139"), (T, "PROGRAM APPRAISAL DOCUMENT"), (T, "MAY 30, 2025")]]
        )
    )
    assert ev.document_type.value == "APPRAISAL_DOCUMENT"
    assert ev.document_date.value == date(2025, 5, 30)
    assert ev.report_number.value == "PADHP00139"


def test_month_only_cover_date_is_not_a_day():
    ev = extract_metadata(
        content_from_pages([[(T, "TECHNICAL ASSESSMENT"), (T, "MARCH 2025 FINAL VERSION")]])
    )
    assert ev.document_type.value == "TECHNICAL_ASSESSMENT"
    assert ev.document_date is None
    assert ev.extras["cover_month"]["value"] == "2025-03"


def test_loan_agreement_and_garbled_loan_number():
    la = extract_metadata(content_from_pages([[(T, "LOAN NUMBER 9835-IN"), (T, "Loan Agreement")]]))
    assert la.document_type.value == "LOAN_AGREEMENT" and la.loan_number.value == "9835-IN"
    pmi = extract_metadata(content_from_pages([[
        (T, "INDIA: Loan No. 860 I-IN (Karnataka Urban Water Supply Modernization Project)"),
        (T, "Performance Monitoring Indicators"), (T, "May 24, 2016")]]))  # fmt: skip
    assert pmi.document_type.value == "PERFORMANCE_INDICATORS"
    assert pmi.loan_number is None  # OCR damage is not repaired
    assert pmi.document_date.value == date(2016, 5, 24)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Environment and Social Systems Assessment (ESSA)", "ESSA"),
        ("ENVIRONMENTAL AND SOCIAL SYSTEMS ASSESSMENT", "ESSA"),
        ("Integrated Fiduciary Systems Assessment", "FIDUCIARY_ASSESSMENT"),
        ("NOTICE OF PARTIAL CANCELLATION", "CANCELLATION"),
    ],
)
def test_type_patterns(text, expected):
    detected, _ = detect_type(content_from_pages([[(T, text)]]))
    assert detected.value == expected


def test_unknown_type():
    detected, _ = detect_type(content_from_pages([[(T, "Minutes of a meeting")]]))
    assert detected is None


def test_closing_date_candidates_from_text_and_tables():
    table = ParsedTable(table_id="t0001", page_number=1, page_numbers=[1], n_rows=2, n_cols=3,
                        header_rows=1, headers=["Ln/Cr/Tf", "Original Closing", "Proposed Closing"],
                        rows=[["Ln/Cr/Tf", "Original Closing", "Proposed Closing"],
                              ["IBRD-86010", "30-Nov-2022", "22-Nov-2024"]],
                        text="", reading_order=5)  # fmt: skip
    content = content_from_pages(isr_pages(), tables=[table])
    candidates = extract_metadata(content).extras["closing_date_candidates"]
    labels = {(c["label"], c["raw_value"], c["source"]) for c in candidates}
    assert ("Original Closing Date", "30-Nov-2022", "text") in labels
    assert ("Proposed Closing", "22-Nov-2024", "table") in labels
    assert all(c["page_number"] == 1 for c in candidates)


def test_key_value_layout_does_not_misattribute_closing_date():
    # Restructuring BASIC DATA: labels first, then values. The approval date must not be
    # reported as the current closing date.
    content = content_from_pages(
        [[(T, "RESTRUCTURING PAPER"),
          (T, "Approval Date Current Closing Date 31-Mar-2016 30-Nov-2022")]]
    )  # fmt: skip
    assert extract_metadata(content).extras["closing_date_candidates"] == []


def test_header_date_fallback_requires_single_distinct_date():
    pages = [[(H, "Report 6/14/2019"), (H, "Printed 7/1/2019"),
              (T, "Implementation Status & Results Report X (P130544) Seq No: 8"),
              (T, "Archived on 21-Feb-2019 ISR35489"), (S, "Key Dates")]]  # fmt: skip
    ev = extract_metadata(content_from_pages(pages))
    assert ev.report_date is None  # two different dates: ambiguous, not guessed


# --- reconciliation ------------------------------------------------------------


def _ev(value, confident=True):
    return ExtractedValue(value=value, page_number=1, evidence="e", method="m",
                          confident=confident)  # fmt: skip


@pytest.mark.parametrize(
    ("manifest", "document", "status", "resolved"),
    [
        (None, None, ValidationStatus.UNKNOWN, None),
        (5, None, ValidationStatus.MANIFEST_ONLY, 5),
        (None, _ev(5), ValidationStatus.DOCUMENT_ONLY, 5),
        (5, _ev(5), ValidationStatus.CONFIRMED, 5),
        (5, _ev(6), ValidationStatus.CORRECTED_FROM_DOCUMENT, 6),
        (5, _ev(6, confident=False), ValidationStatus.CONFLICT, None),
    ],
)
def test_reconciliation_statuses(manifest, document, status, resolved):
    result = reconcile_field("isr_sequence", manifest, document)
    assert result.status is status and result.resolved_value == resolved
    assert result.manifest_value == manifest
    assert result.document_value == (document.value if document else None)
    assert result.resolution_reason


def test_correction_records_both_values():
    result = reconcile_field("isr_sequence", 5, _ev(6))
    assert (result.manifest_value, result.document_value, result.resolved_value) == (5, 6, 6)
    assert "superseded but recorded" in result.resolution_reason


def test_loan_numbers_compared_normalized_and_dates_as_dates():
    assert reconcile_field("loan_number", "8601-in", _ev("8601-IN")).status is (
        ValidationStatus.CONFIRMED
    )
    assert reconcile_field("document_date", "2017-11-29", _ev(date(2017, 11, 29))).status is (
        ValidationStatus.CONFIRMED
    )


def test_reconcile_all_fields():
    ev = extract_metadata(content_from_pages(isr_pages()))
    manifest = {"project_id": "P130544", "document_type": "ISR", "document_date": "2017-11-29",
                "isr_sequence": 5, "report_number": None, "loan_number": None}  # fmt: skip
    statuses = {v.field: v.status for v in reconcile(manifest, ev)}
    assert statuses == {
        "project_id": ValidationStatus.CONFIRMED,
        "document_type": ValidationStatus.CONFIRMED,
        "document_date": ValidationStatus.CONFIRMED,
        "isr_sequence": ValidationStatus.CONFIRMED,
        "report_number": ValidationStatus.DOCUMENT_ONLY,
        "loan_number": ValidationStatus.UNKNOWN,
    }
