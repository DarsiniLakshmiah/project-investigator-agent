"""Phase 6: persisted row builders (identity, provenance, precision, quality, semantics)."""

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from tests.support.extraction_builders import isr_doc

from worldbank_copilot.common.quality import CheckCode
from worldbank_copilot.extraction.models import (
    ExtractedRating,
    IsrSnapshot,
    ProjectEvent,
    ResultObservation,
)
from worldbank_copilot.extraction.provenance import (
    ExtractionMethod,
    ExtractionStatus,
    evidence,
    issue,
)
from worldbank_copilot.lakehouse.contracts import RECORD_HASH, RECORD_ID, require_valid
from worldbank_copilot.lakehouse.records import (
    LoadContext,
    QualityInput,
    bronze_dataset,
    delta_name,
    events_dataset,
    isr_dataset,
    quality_dataset,
    record_quality_inputs,
    results_dataset,
    silver_dataset,
)
from worldbank_copilot.transformations.bronze import BronzeTable, SourceMetadata
from worldbank_copilot.transformations.silver_models import SilverLoan, SourceRef


def ctx(run="run-1", when=datetime(2026, 9, 30, 12, tzinfo=UTC)):
    return LoadContext(
        "snap", run, when, "test", {"loans.csv": "a" * 64, "P130544/isr.pdf": "0" * 64}
    )


DOC = isr_doc([], isr_sequence=5, document_id="doc-5")
REF = evidence(
    DOC,
    3,
    ExtractionMethod.DOCLING_TABLE,
    section="Results",
    table_id="t0003",
    row=4,
    column=2,
    text="Indicator A | 10 | 20",
)


def _loan(**kw):
    base = dict(
        project_id="P130544",
        raw_loan_number="IBRD93240",
        normalized_loan_number="IBRD-9324-0",
        loan_base_number="9324",
        loan_suffix="0",
        lender="IBRD",
        loan_type="FSL",
        loan_status="Disbursing",
        borrower="B",
        country_code="IN",
        currency_of_commitment=None,
        original_principal_usd=Decimal("150000000"),
        cancelled_amount_usd=Decimal("24937499.48"),
        disbursed_amount_usd=Decimal("32713450.09"),
        undisbursed_amount_usd=None,
        repaid_to_ibrd_usd=None,
        due_to_ibrd_usd=None,
        exchange_adjustment_usd=None,
        borrowers_obligation_usd=None,
        loans_held_usd=None,
        principal_components_total_usd=None,
        principal_components_difference_usd=None,
        board_approval_date=date(2021, 12, 21),
        agreement_signing_date=None,
        effective_date=date(2024, 10, 28),
        closing_date=date(2027, 9, 30),
        last_disbursement_date=None,
        first_repayment_date=None,
        last_repayment_date=None,
        snapshot_date=date(2026, 8, 31),
        valuation_caveats=[],
        source_file="loans.csv",
        source_refs=[
            SourceRef(
                bronze_table="bronze_loans_raw",
                source_file="loans.csv",
                source_row=7,
                ingestion_run_id="snapshot-x",
            )
        ],
    )
    base.update(kw)
    return SilverLoan(**base)


def test_delta_names_drop_the_layer_prefix_only():
    assert delta_name("silver_loans", "silver") == "loans"
    assert delta_name("bronze_loans_raw", "bronze") == "loans_raw"
    assert delta_name("isr_snapshots", "silver") == "isr_snapshots"


def test_silver_rows_preserve_decimal_precision_dates_and_nulls():
    dataset = silver_dataset("silver_loans", [_loan()], ctx())
    require_valid(dataset.contract, dataset.rows)
    row = dataset.rows[0]
    assert row["cancelled_amount_usd"] == Decimal("24937499.48")
    assert isinstance(row["disbursed_amount_usd"], Decimal)  # never float
    assert row["board_approval_date"] == date(2021, 12, 21)
    assert row["undisbursed_amount_usd"] is None and row["agreement_signing_date"] is None
    assert row["source_record_keys"] == ["bronze_loans_raw:loans.csv::7"]
    assert '"ingestion_run_id":"snapshot-x"' in row["source_refs_json"]


def test_identity_is_stable_across_runs_and_content_changes_update_the_hash():
    first = silver_dataset("silver_loans", [_loan()], ctx()).rows[0]
    rerun = silver_dataset(
        "silver_loans", [_loan()], ctx("run-2", datetime(2027, 1, 1, tzinfo=UTC))
    ).rows[0]
    assert (first[RECORD_ID], first[RECORD_HASH]) == (rerun[RECORD_ID], rerun[RECORD_HASH])
    assert first["_load_run_id"] != rerun["_load_run_id"]
    changed = silver_dataset("silver_loans", [_loan(closing_date=date(2028, 1, 1))], ctx()).rows[0]
    assert changed[RECORD_ID] == first[RECORD_ID] and changed[RECORD_HASH] != first[RECORD_HASH]


def test_bronze_rows_are_source_aligned_with_source_hash():
    meta = SourceMetadata(
        source_name="loans", source_file="loans.csv", ingested_at="t", ingestion_run_id="r"
    )
    fields = [
        "_source_file",
        "_source_sheet",
        "_source_row",
        "_ingested_at",
        "_ingestion_run_id",
        "project_id",
        "original_principal_amount_usd",
        "_extra_fields",
    ]
    record = {
        "_source_file": "loans.csv",
        "_source_sheet": None,
        "_source_row": 7,
        "_ingested_at": "2026-09-30T00:00:00+00:00",
        "_ingestion_run_id": "snapshot-x",
        "project_id": "P130544",
        "original_principal_amount_usd": "150000000",
        "_extra_fields": {"Sold 3rd Party (US$)": "0"},
    }
    dataset = bronze_dataset(BronzeTable("bronze_loans_raw", [record], meta, fields), ctx())
    require_valid(dataset.contract, dataset.rows)
    row = dataset.rows[0]
    assert dataset.contract.name == "loans_raw"
    assert row["original_principal_amount_usd"] == "150000000"  # text exactly as read
    assert row["_source_sha256"] == "a" * 64
    # The ingestion timestamp is operational: a later load does not change the content hash.
    later = dict(record, _ingested_at="2027-01-01T00:00:00+00:00")
    again = bronze_dataset(BronzeTable("bronze_loans_raw", [later], meta, fields), ctx())
    assert again.rows[0][RECORD_HASH] == row[RECORD_HASH]
    unknown = dict(record, _source_file="other.csv")
    with pytest.raises(KeyError, match="not in the source snapshot"):
        bronze_dataset(BronzeTable("bronze_loans_raw", [unknown], meta, fields), ctx())


def _observation(**kw):
    base = dict(
        project_id="P130544",
        indicator_key="P130544-IND-1",
        indicator_name_raw="Indicator A (Number)",
        indicator_name_normalized="indicator a (number)",
        unit="Number",
        current_value="20",
        isr_sequence=5,
        observation_date=date(2020, 1, 15),
        source_document=DOC.filename,
        source_page=3,
        source_table="t0003",
        layout="WIDE",
        extraction_method=ExtractionMethod.DOCLING_TABLE,
        status=ExtractionStatus.EXACT,
        source_ref=REF,
        identity_basis="EXACT_NAME",
    )
    base.update(kw)
    return ResultObservation(**base)


def test_results_keep_queryable_provenance_columns():
    dataset = results_dataset([_observation()], ctx())
    require_valid(dataset.contract, dataset.rows)
    row = dataset.rows[0]
    assert (
        row["evidence_document_id"],
        row["evidence_page_number"],
        row["evidence_section"],
        row["evidence_table_id"],
        row["evidence_row"],
        row["evidence_column"],
    ) == ("doc-5", 3, "Results", "t0003", 4, 2)
    assert row["evidence_text"] == "Indicator A | 10 | 20"
    assert row["evidence_extraction_method"] == "DOCLING_TABLE"
    assert row["evidence_hash"] == DOC.source_hash and row["document_id"] == "doc-5"
    assert row["status"] == "EXACT" and row["quality_issue_codes"] == []


def test_undated_restructuring_keeps_null_event_date_and_derived_candidate():
    event = ProjectEvent(
        project_id="P130544",
        event_id="ev1",
        event_type="RESTRUCTURING",
        event_date=None,
        event_date_basis="UNDATED_RESTRUCTURING_PAPER",
        candidate_event_date=date(2024, 12, 10),
        candidate_date_basis="EARLIEST_ISR_RESTRUCTURING_APPROVAL_ON_OR_AFTER_LATEST_CITED_DATE",
        candidate_date_status=ExtractionStatus.DERIVED_FROM_EXPLICIT_SOURCE,
        source_document="res.pdf",
        source_page=6,
        source_section=None,
        source_text=None,
        extraction_method=ExtractionMethod.DOCLING_TEXT,
        status=ExtractionStatus.AMBIGUOUS,
        source_refs=[REF],
        quality_issues=[issue(CheckCode.RESTRUCTURING_DATE_UNRESOLVED, "WARNING", "undated", REF)],
    )
    dataset = events_dataset([event], ctx())
    require_valid(dataset.contract, dataset.rows)
    row = dataset.rows[0]
    assert row["event_date"] is None
    assert row["candidate_event_date"] == date(2024, 12, 10)
    assert row["candidate_date_status"] == "DERIVED_FROM_EXPLICIT_SOURCE"
    assert row["status"] == "AMBIGUOUS"
    assert row["quality_issue_codes"] == ["RESTRUCTURING_DATE_UNRESOLVED"]


def _snapshot(seq, header, archive):
    rating = ExtractedRating(
        raw_rating="Satisfactory",
        normalized_rating="Satisfactory",
        status=ExtractionStatus.EXACT,
        evidence=REF,
    )
    return IsrSnapshot(
        project_id="P179039",
        document_id="doc-5",
        isr_sequence=seq,
        isr_number=None,
        archive_date=archive,
        header_date=header,
        canonical_report_date=header or archive,
        canonical_date_basis="header_date" if header else "archive_date",
        date_difference_days=(header - archive).days if header and archive else None,
        pdo_rating=rating,
        source_document=DOC.filename,
        source_pages=[1],
        source_refs=[REF],
    )


def test_isr_rows_keep_both_dates_and_sequence_ordering():
    snaps = [
        _snapshot(5, date(2025, 5, 31), date(2024, 9, 11)),
        _snapshot(6, date(2025, 3, 12), date(2025, 3, 12)),
    ]
    dataset = isr_dataset(snaps, {"doc-5": DOC}, ctx())
    require_valid(dataset.contract, dataset.rows)
    by_seq = sorted(dataset.rows, key=lambda r: r["isr_sequence"])
    assert [r["isr_sequence"] for r in by_seq] == [5, 6]
    assert by_seq[0]["canonical_report_date"] > by_seq[1]["canonical_report_date"]  # kept as is
    assert (
        by_seq[0]["header_date"],
        by_seq[0]["archive_date"],
        by_seq[0]["date_difference_days"],
    ) == (date(2025, 5, 31), date(2024, 9, 11), 262)
    assert by_seq[0]["pdo_rating"] == "Satisfactory"
    assert by_seq[0]["pdo_rating_extraction_method"] == "DOCLING_TABLE"
    assert by_seq[0]["pdo_rating_page_number"] == 3


def test_quality_observations_link_records_and_deduplicate():
    obs = _observation(
        status=ExtractionStatus.AMBIGUOUS,
        quality_issues=[issue(CheckCode.EXTRACTION_AMBIGUOUS, "WARNING", "cells merged", REF)],
    )
    datasets = {"silver.project_results": results_dataset([obs], ctx())}
    linked = record_quality_inputs(datasets, {"silver.project_results": [obs]})
    report_copy = QualityInput(
        "silver_documents",
        "EXTRACTION_AMBIGUOUS",
        "WARNING",
        "P130544",
        REF.label,
        "cells merged",
        evidence=REF,
        details={},
    )
    other = QualityInput(
        "bronze",
        "SOURCE_VALUE_CONFLICT",
        "WARNING",
        "P130544",
        "loans",
        "differs",
        details={"field": "x"},
    )
    dataset = quality_dataset([*linked, report_copy, other, other], ctx())
    require_valid(dataset.contract, dataset.rows)
    assert len(dataset.rows) == 2
    linked_row = next(r for r in dataset.rows if r["related_record_id"])
    assert linked_row["related_table"] == "silver.project_results"
    assert linked_row["related_record_id"] == datasets["silver.project_results"].rows[0][RECORD_ID]
    assert (linked_row["page_number"], linked_row["table_id"]) == (3, "t0003")
    assert next(r for r in dataset.rows if r["layer"] == "bronze")["occurrence_count"] == 2
