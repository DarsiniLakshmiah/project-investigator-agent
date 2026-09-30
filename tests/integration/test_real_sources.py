"""Checks against the real local source files. Run with: pytest -m integration

Skipped when the files are not present (e.g. on CI or a fresh clone).
"""

import hashlib
from pathlib import Path

import pytest

from worldbank_copilot.common import load_project_registry, load_settings
from worldbank_copilot.ingestion.data_quality import CheckCode, build_data_quality_report
from worldbank_copilot.ingestion.pipeline import ingest_bronze, resolve_source_files
from worldbank_copilot.ingestion.procurement import ProcurementCoverageStatus

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_run():
    settings = load_settings("local", env={})
    try:
        files = resolve_source_files(settings)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"real source files not available: {exc}")
    structured = [files.projects_workbook, files.loans_snapshot, files.procurement_contract_awards]
    before = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in structured}
    registry = load_project_registry(settings.config_dir)
    result = ingest_bronze(settings, registry)
    after = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in structured}
    return registry, result, build_data_quality_report(result, registry), before, after


def test_sources_unchanged(real_run):
    _, _, _, before, after = real_run
    assert before == after


def test_projects_and_loans(real_run):
    _, result, _, _, _ = real_run
    assert result.tables["bronze_projects_raw"].count_by_project() == {
        "P130544": 1, "P179039": 1, "P506272": 1,
    }  # fmt: skip
    loans = {
        (r["project_id"], r["raw_loan_number"], r["normalized_loan_number"])
        for r in result.tables["bronze_loans_raw"].records
    }
    assert loans == {
        ("P130544", "IBRD86010", "IBRD-8601-0"),
        ("P130544", "IBRD93240", "IBRD-9324-0"),
        ("P179039", "IBRD94960", "IBRD-9496-0"),
        ("P506272", "IBRD98350", "IBRD-9835-0"),
    }


def test_procurement_coverage(real_run):
    _, result, _, _, _ = real_run
    coverage = {c.project_id: (c.coverage_status, c.row_count) for c in result.procurement_coverage}
    assert coverage == {
        "P130544": (ProcurementCoverageStatus.RECORDS_PRESENT, 17),
        "P179039": (ProcurementCoverageStatus.NOT_COVERED_BY_THIS_DATASET, 0),
        "P506272": (ProcurementCoverageStatus.NOT_COVERED_BY_THIS_DATASET, 0),
    }


def test_isr_inventory(real_run):
    _, result, _, _, _ = real_run
    isr = result.isr_completeness
    assert {pid: (v.found_count, v.is_complete) for pid, v in isr.items()} == {
        "P130544": (24, True), "P179039": (8, True), "P506272": (3, True),
    }  # fmt: skip
    counts = result.document_inventory.table.count_by_project()
    assert counts == {"P130544": 32, "P179039": 13, "P506272": 8}


def test_data_quality(real_run):
    _, _, report, _, _ = real_run
    assert not report.has_errors
    commitment = report.by_check(CheckCode.PROJECT_COMMITMENT_SOURCE_DIFFERENCE)
    assert {o.project_id for o in commitment} == {"P130544"}
    assert not report.by_check(CheckCode.UNCLASSIFIED_DOCUMENT)
    assert len(report.by_check(CheckCode.DOCUMENT_LOAN_REFERENCE_LINKED)) == 3
