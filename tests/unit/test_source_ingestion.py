"""Workbook, loans and procurement ingestion over small synthetic files."""

import pytest

from support.synthetic import (
    _headers,
    default_contract_rows,
    default_loan_rows,
    default_workbook_sheets,
    loan_row,
    write_csv,
    write_workbook,
)
from worldbank_copilot.common.exceptions import SourceContractError
from worldbank_copilot.ingestion.contracts import LOANS, PROCUREMENT, PROJECTS
from worldbank_copilot.ingestion.loans import ingest_loans, loans_by_project
from worldbank_copilot.ingestion.procurement import (
    ProcurementCoverageStatus,
    assess_procurement_coverage,
    ingest_procurement,
)
from worldbank_copilot.ingestion.projects import ingest_workbook

IDS = ["P130544", "P179039", "P506272"]
KW = {"ingested_at": "2026-09-26T00:00:00+00:00", "run_id": "r"}


@pytest.fixture
def workbook(tmp_path):
    return write_workbook(tmp_path / "all.xlsx", default_workbook_sheets())


# --- workbook ----------------------------------------------------------------


def test_every_sheet_becomes_its_own_bronze_table(workbook):
    tables = ingest_workbook(workbook, IDS, source_file="all.xlsx", **KW)
    assert set(tables) == {
        "bronze_projects_raw",
        "bronze_themes_raw",
        "bronze_sectors_raw",
        "bronze_geo_locations_raw",
        "bronze_financers_raw",
    }
    assert tables["bronze_projects_raw"].count_by_project() == {
        "P130544": 1,
        "P179039": 1,
        "P506272": 1,
    }
    assert tables["bronze_themes_raw"].count_by_project() == {"P130544": 2}
    assert tables["bronze_geo_locations_raw"].count_by_project() == {"P130544": 1}


def test_projects_sheet_quirks(workbook):
    projects = ingest_workbook(workbook, IDS, source_file="all.xlsx", **KW)["bronze_projects_raw"]
    meta = projects.metadata
    assert (meta.header_row, meta.secondary_header_row) == (2, 3)
    assert meta.column_mapping["project_development_objective"] == "Project Development Objective "
    assert meta.known_gaps == list(PROJECTS.known_gaps)
    row = projects.for_project("P130544")[0]
    assert row["_source_row"] == 5  # Excel row: title, header, API header, P000001, P130544
    assert row["_source_sheet"] == "World Bank Projects"
    assert row["project_development_objective"] == "<p>Objective.</p>"  # HTML kept
    # openpyxl saves an integral float as an int, so the synthetic cell reads back as
    # 100000000 (the real export stores 100000000.0; see test_raw_text_preserves_values).
    assert row["ibrd_commitment"] == "100000000"
    assert row["board_approval_date"] == "2016-03-31T00:00:00Z"  # raw date text
    # openpyxl does not save empty-string cells, so the synthetic blank reads back as None
    # (the real export returns ''); Bronze keeps whichever the source yields.
    assert projects.for_project("P506272")[0]["current_project_cost"] is None


def test_financers_alias_and_raw_whitespace(workbook):
    financers = ingest_workbook(workbook, IDS, source_file="all.xlsx", **KW)["bronze_financers_raw"]
    assert financers.metadata.column_mapping["source_project_id"] == "Project"
    currencies = [r["currency"] for r in financers.for_project("P179039")]
    assert currencies == ["USD  "]


def test_theme_values_not_cleaned(workbook):
    themes = ingest_workbook(workbook, IDS, source_file="all.xlsx", **KW)["bronze_themes_raw"]
    assert themes.for_project("P130544")[1]["level_2"] == " Water  Institutions"


def test_missing_sheet_fails_clearly(tmp_path):
    sheets = default_workbook_sheets()
    del sheets["Sectors"]
    path = write_workbook(tmp_path / "all.xlsx", sheets)
    with pytest.raises(SourceContractError, match="Sectors"):
        ingest_workbook(path, IDS, source_file="all.xlsx", **KW)


# --- loans ----------------------------------------------------------------------


@pytest.fixture
def loans_csv(tmp_path):
    return write_csv(tmp_path / "loans.csv", _headers(LOANS), default_loan_rows())


def test_one_project_many_loans_not_aggregated(loans_csv):
    table = ingest_loans(loans_csv, IDS, source_file="loans.csv", **KW)
    assert loans_by_project(table) == {
        "P130544": ["IBRD86010", "IBRD93240"],
        "P179039": ["IBRD94960"],
        "P506272": ["IBRD98350"],
    }
    assert len(table.for_project("P130544")) == 2  # two records, no project total


def test_raw_and_normalized_loan_numbers(loans_csv):
    table = ingest_loans(loans_csv, IDS, source_file="loans.csv", **KW)
    pairs = [(r["raw_loan_number"], r["normalized_loan_number"]) for r in table.records]
    assert ("IBRD86010", "IBRD-8601-0") in pairs
    assert ("IBRD93240", "IBRD-9324-0") in pairs
    fields = table.fields
    assert fields.index("normalized_loan_number") == fields.index("raw_loan_number") + 1


def test_loan_values_and_snapshot_preserved(loans_csv):
    table = ingest_loans(loans_csv, IDS, source_file="loans.csv", **KW)
    af = next(r for r in table.records if r["raw_loan_number"] == "IBRD93240")
    assert af["cancelled_amount_usd"] == "24937499.48"
    assert af["end_of_period"] == "08/31/2026"
    assert af["currency_of_commitment"] == ""  # blank kept blank; nothing inferred


def test_unrecognised_loan_number_kept_raw(tmp_path):
    path = write_csv(
        tmp_path / "loans.csv", _headers(LOANS), [loan_row("P130544", "IBRD1234A", "1")]
    )
    record = ingest_loans(path, IDS, source_file="loans.csv", **KW).records[0]
    assert (record["raw_loan_number"], record["normalized_loan_number"]) == ("IBRD1234A", None)


def test_loans_missing_required_column(tmp_path):
    headers = [h for h in _headers(LOANS) if h != "Loan Number"]
    rows = [{k: v for k, v in r.items() if k != "Loan Number"} for r in default_loan_rows()]
    path = write_csv(tmp_path / "loans.csv", headers, rows)
    with pytest.raises(SourceContractError, match="Loan Number"):
        ingest_loans(path, IDS, source_file="loans.csv", **KW)


# --- procurement -----------------------------------------------------------------


def test_procurement_filtering(tmp_path):
    path = write_csv(tmp_path / "p.csv", _headers(PROCUREMENT), default_contract_rows())
    table = ingest_procurement(path, IDS, source_file="p.csv", **KW)
    assert table.count_by_project() == {"P130544": 3}
    assert table.records[0]["supplier_contract_amount_usd"] == "35729.020000"


def test_procurement_coverage_semantics(tmp_path, repo_config_dir):
    from worldbank_copilot.common import load_project_registry

    registry = load_project_registry(repo_config_dir)
    path = write_csv(tmp_path / "p.csv", _headers(PROCUREMENT), default_contract_rows())
    table = ingest_procurement(path, IDS, source_file="p.csv", **KW)
    coverage = {c.project_id: c for c in assess_procurement_coverage(table, registry)}

    assert coverage["P130544"].coverage_status is ProcurementCoverageStatus.RECORDS_PRESENT
    assert coverage["P130544"].row_count == 3
    for pid in ("P179039", "P506272"):
        item = coverage[pid]
        assert item.coverage_status is ProcurementCoverageStatus.NOT_COVERED_BY_THIS_DATASET
        assert item.row_count == 0 and item.covered_by_dataset is False
        assert "provides no information" in item.interpretation


def test_covered_project_with_zero_rows(tmp_path, repo_config_dir):
    from worldbank_copilot.common import load_project_registry

    registry = load_project_registry(repo_config_dir)
    path = write_csv(tmp_path / "p.csv", _headers(PROCUREMENT), default_contract_rows()[:1])
    table = ingest_procurement(path, IDS, source_file="p.csv", **KW)
    coverage = {c.project_id: c for c in assess_procurement_coverage(table, registry)}
    item = coverage["P130544"]
    assert item.coverage_status is ProcurementCoverageStatus.NO_RECORDS_IN_DATASET
    assert "not evidence that no procurement took place" in item.interpretation
