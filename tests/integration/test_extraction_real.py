"""Phase 5 extraction over the real parsed documents (Docling is not re-run).

Run after `python scripts/parse_documents.py`, with: pytest -m integration
Skipped when parsed outputs or source files are not present.
"""

from collections import Counter
from datetime import date
from decimal import Decimal

import pytest

from worldbank_copilot.common import load_project_registry, load_settings
from worldbank_copilot.common.quality import CheckCode
from worldbank_copilot.extraction.pipeline import build_quality_report, run_extraction

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def extracted(tmp_path_factory):
    settings = load_settings("local", env={})
    if not (settings.local_output_root / "parsed").is_dir():
        pytest.skip("no parsed outputs; run scripts/parse_documents.py first")
    registry = load_project_registry(settings.config_dir)
    try:
        from worldbank_copilot.ingestion.pipeline import ingest_bronze
        from worldbank_copilot.transformations.silver import build_silver

        silver = build_silver(ingest_bronze(settings, registry), registry)
        run = run_extraction(
            settings,
            registry,
            silver.tables["silver_loans"].rows,
            silver.tables["silver_projects"].rows,
            output_root=tmp_path_factory.mktemp("silver_documents"),
        )
    except FileNotFoundError as exc:
        pytest.skip(f"source files not available: {exc}")
    return run, build_quality_report(run)


def test_all_35_isr_snapshots_with_ratings(extracted):
    run, report = extracted
    counts = Counter(s.project_id for s in run.snapshots)
    assert counts == {"P130544": 24, "P179039": 8, "P506272": 3}
    for snap in run.snapshots:
        assert snap.pdo_rating and snap.pdo_rating.normalized_rating, snap.isr_sequence
        assert snap.implementation_progress_rating.normalized_rating
        assert snap.overall_risk_rating.normalized_rating
        assert snap.canonical_report_date is not None
    assert not report.by_check(CheckCode.ISR_SNAPSHOT_MISSING)
    assert run.hashes_unchanged is True and not report.has_errors


def test_isr_date_anomaly_is_reported(extracted):
    run, report = extracted
    seq5 = next(s for s in run.snapshots if s.project_id == "P179039" and s.isr_sequence == 5)
    assert (seq5.header_date, seq5.archive_date, seq5.date_difference_days) == (
        date(2025, 5, 31),
        date(2024, 9, 11),
        262,
    )
    assert any(
        "canonical date of ISR 5" in o.message
        for o in report.by_check(CheckCode.SEQUENCE_DATE_ANOMALY)
    )


def test_formal_events_from_p130544_papers(extracted):
    run, report = extracted
    events = [e for e in run.events if e.project_id == "P130544"]
    cancellation = next(e for e in events if e.event_type == "CANCELLATION")
    assert (
        cancellation.loan_number,
        cancellation.cancelled_amount,
        cancellation.cancelled_currency,
        cancellation.event_date,
    ) == ("IBRD93240", Decimal("4011633250.00"), "JPY", date(2024, 8, 6))
    af = next(e for e in events if e.event_type == "ADDITIONAL_FINANCING")
    assert af.additional_financing_amount == Decimal("150.00")
    dated = sorted(e.event_date for e in events if e.event_type == "RESTRUCTURING" and e.event_date)
    assert dated == [date(2021, 5, 20), date(2024, 7, 23), date(2024, 12, 10), date(2026, 6, 29)]
    assert len(report.by_check(CheckCode.RESTRUCTURING_DATE_UNRESOLVED)) == 4
    assert report.by_check(CheckCode.CROSS_SOURCE_NOT_COMPARABLE)


def test_original_closing_dates_established(extracted):
    run, _ = extracted
    projects = {
        e.project_id: e.value for e in run.enrichment if e.target_table == "silver_projects"
    }
    assert projects == {
        "P130544": date(2022, 11, 30),
        "P179039": date(2028, 6, 1),
        "P506272": date(2030, 12, 31),
    }


def test_results_and_risks_have_provenance(extracted):
    run, _ = extracted
    assert {r.project_id for r in run.results} == {"P130544", "P179039", "P506272"}
    for row in run.results + run.risks:
        refs = [row.source_ref] if hasattr(row, "source_ref") else row.source_refs
        assert refs and all(ref.page_number >= 1 and ref.source_hash for ref in refs)
    frames = Counter((r.project_id, r.framing) for r in run.risks)
    assert frames[("P130544", "FORMAL_RISK_RATING")] >= 10
    assert frames[("P179039", "ASSESSMENT_FINDING")] > 0
    ta = [
        r
        for r in run.risks
        if r.project_id == "P179039" and r.source_document_type == "TECHNICAL_ASSESSMENT"
    ]
    assert len(ta) == 4 and ta[-1].extraction_method.value == "PDF_TEXT_FALLBACK"
