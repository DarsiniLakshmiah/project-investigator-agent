"""End-to-end Bronze ingestion over the synthetic source tree (no real data)."""

import importlib.util
import json
from pathlib import Path

import pytest

from worldbank_copilot.common import load_settings
from worldbank_copilot.common.exceptions import SourceFileError
from worldbank_copilot.ingestion.data_quality import build_data_quality_report
from worldbank_copilot.ingestion.pipeline import resolve_source_files, write_bronze
from worldbank_copilot.ingestion.procurement import ProcurementCoverageStatus
from worldbank_copilot.ingestion.validation_report import format_text_report, to_json_dict
from worldbank_copilot.transformations.bronze import LocalJsonlBronzeWriter

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_ingestion_produces_all_bronze_tables(synthetic_run):
    _, _, result = synthetic_run
    names = {t.name for t in result.all_tables()}
    assert names == {
        "bronze_projects_raw", "bronze_themes_raw", "bronze_sectors_raw",
        "bronze_geo_locations_raw", "bronze_financers_raw", "bronze_loans_raw",
        "bronze_procurement_raw", "bronze_document_inventory", "bronze_procurement_coverage",
    }  # fmt: skip
    assert result.run_id == "test-run"
    assert result.tables["bronze_loans_raw"].count_by_project() == {
        "P130544": 2, "P179039": 1, "P506272": 1,
    }  # fmt: skip
    for record in result.tables["bronze_projects_raw"].records:
        assert record["_ingested_at"] == "2026-09-26T00:00:00+00:00"
        assert record["_source_file"] == "all.xlsx"


def test_coverage_and_isr_in_result(synthetic_run):
    _, _, result = synthetic_run
    statuses = {c.project_id: c.coverage_status for c in result.procurement_coverage}
    assert statuses == {
        "P130544": ProcurementCoverageStatus.RECORDS_PRESENT,
        "P179039": ProcurementCoverageStatus.NOT_COVERED_BY_THIS_DATASET,
        "P506272": ProcurementCoverageStatus.NOT_COVERED_BY_THIS_DATASET,
    }
    isr = result.isr_completeness
    assert (isr["P130544"].found_count, isr["P130544"].sequences) == (2, [1, 2])
    assert isr["P130544"].is_complete and isr["P179039"].is_complete


def test_document_inventory_in_result(synthetic_run):
    _, _, result = synthetic_run
    docs = {r["filename"]: r for r in result.document_inventory.table.records}
    assert docs["RAD696381150.pdf"]["document_type"] == "LOAN_AGREEMENT"
    assert docs["RAD696381150.pdf"]["relative_path"] == "P130544/RAD696381150.pdf"
    dated = docs["ISR-Disclosable-P130544-04-10-2017-1491820303083.pdf"]
    assert (dated["isr_sequence"], dated["document_date"]) == (2, "2017-04-10")


def test_ambiguous_source_file_fails(synthetic_config):
    settings = load_settings(config_dir=synthetic_config, env={})
    extra = (
        settings.structured_root
        / "ibrd_statement_of_loans_and_guarantees_latest_available_snapshot_y.csv"
    )
    extra.write_text("x", encoding="utf-8")
    with pytest.raises(SourceFileError, match="2 files match"):
        resolve_source_files(settings)


def test_missing_source_file_fails(synthetic_config):
    settings = load_settings(
        config_dir=synthetic_config, env={"WBC_PROJECTS_WORKBOOK": "nope.xlsx"}
    )
    with pytest.raises(SourceFileError, match="nope.xlsx"):
        resolve_source_files(settings)


def test_source_files_are_not_modified(synthetic_config, synthetic_run):
    settings, _, _ = synthetic_run
    before = {p: p.stat().st_mtime_ns for p in Path(settings.data_root).rglob("*") if p.is_file()}
    from worldbank_copilot.ingestion.pipeline import ingest_bronze

    ingest_bronze(settings, synthetic_run[1])
    after = {p: p.stat().st_mtime_ns for p in Path(settings.data_root).rglob("*") if p.is_file()}
    assert before == after


def test_local_writer_round_trip(synthetic_run, tmp_path):
    _, _, result = synthetic_run
    written = write_bronze(result, LocalJsonlBronzeWriter(tmp_path / "bronze"))
    assert len(written) == 9
    lines = (
        (tmp_path / "bronze" / "bronze_loans_raw.jsonl").read_text(encoding="utf-8").splitlines()
    )
    records = [json.loads(line) for line in lines]
    assert {r["raw_loan_number"] for r in records if r["project_id"] == "P130544"} == {
        "IBRD86010", "IBRD93240",
    }  # fmt: skip
    meta = json.loads((tmp_path / "bronze" / "bronze_loans_raw.metadata.json").read_text("utf-8"))
    assert meta["source_name"] == "ibrd_statement_of_loans"
    assert "normalized_loan_number" in meta["fields"]


def test_text_and_json_reports(synthetic_run):
    _, registry, result = synthetic_run
    report = build_data_quality_report(result, registry)
    text = format_text_report(result, report, registry)
    assert "P130544 OK" in text
    assert "IBRD86010 -> IBRD-8601-0" in text
    assert "NOT_COVERED_BY_THIS_DATASET" in text
    assert "ISR 2/2 complete" in text
    payload = to_json_dict(result, report, registry)
    json.dumps(payload)  # fully serialisable
    assert payload["data_quality"]["has_errors"] is False


def test_validate_data_script(synthetic_config, tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location(
        "validate_data", REPO_ROOT / "scripts" / "validate_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("WBC_CONFIG_DIR", str(synthetic_config))
    monkeypatch.setenv("WBC_ENV", "local")
    out_json = tmp_path / "report.json"
    code = module.main(["--json", str(out_json), "--write-bronze"])
    assert code == 0
    assert json.loads(out_json.read_text(encoding="utf-8"))["projects"] == [
        "P130544", "P179039", "P506272",
    ]  # fmt: skip
    assert (tmp_path / "out" / "bronze" / "bronze_document_inventory.jsonl").is_file()
    assert "Procurement coverage" in capsys.readouterr().out


def test_validate_data_script_silver_layer(synthetic_config, tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location(
        "validate_data", REPO_ROOT / "scripts" / "validate_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("WBC_CONFIG_DIR", str(synthetic_config))
    monkeypatch.setenv("WBC_ENV", "local")
    out_json = tmp_path / "report.json"
    assert module.main(["--json", str(out_json), "--write-silver"]) == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(payload["silver"]["tables"]["silver_loans"]) == 4
    assert payload["silver"]["data_quality"]["has_errors"] is False
    silver_dir = tmp_path / "out" / "silver"
    assert (silver_dir / "silver_loans.jsonl").is_file()
    assert (silver_dir / "silver_loans.schema.json").is_file()
    assert (silver_dir / "silver_field_lineage.json").is_file()
    out = capsys.readouterr().out
    assert "Project financial summaries: 3" in out and "unknown (NULL)" in out

    assert module.main(["--layer", "bronze"]) == 0
    assert "SILVER" not in capsys.readouterr().out
