"""Data-quality checks over hand-built Bronze tables and the synthetic pipeline run."""

from decimal import Decimal, InvalidOperation

import pytest

from worldbank_copilot.common import load_project_registry
from worldbank_copilot.ingestion.contracts import LOANS, PROJECTS
from worldbank_copilot.ingestion.data_quality import (
    CheckCode,
    Severity,
    build_data_quality_report,
    check_commitment_sources,
    check_loans,
    check_project_presence,
    check_table_values,
    parse_amount,
)
from worldbank_copilot.transformations.bronze import BronzeTable, SourceMetadata


def _table(name, records, **meta):
    for i, record in enumerate(records, start=2):
        record.setdefault("_source_row", i)
    metadata = SourceMetadata(
        source_name=name, source_file="f", ingested_at="t", ingestion_run_id="r", **meta
    )
    return BronzeTable(name, records, metadata)


def _project(pid="P130544", **values):
    record = {c.field: "x" for c in PROJECTS.columns}
    record.update({"project_id": pid, "source_project_id": pid, "ibrd_commitment": "100000000.0",
                   "board_approval_date": "2016-03-31T00:00:00Z",
                   "project_closing_date": "2027-09-30", "public_disclosure_date": "2012-07-16",
                   "loan_effective_date": "2016-08-22", "last_update_date": "2024-04-04",
                   "current_project_cost": "1", "ida_commitment": "0", "grant_amount": "0",
                   "total_ibrd_ida_grant_commitment": "1"})  # fmt: skip
    record.update(values)
    return record


def _loan(pid, number, original, **values):
    record = {c.field: "" for c in LOANS.columns}
    record.update({"project_id": pid, "source_project_id": pid, "raw_loan_number": number,
                   "normalized_loan_number": None, "original_principal_amount_usd": original,
                   "end_of_period": "08/31/2026"})  # fmt: skip
    record.update(values)
    return record


@pytest.fixture
def registry(repo_config_dir):
    return load_project_registry(repo_config_dir)


def test_parse_amount():
    assert parse_amount("100000000.0") == Decimal("100000000")
    assert parse_amount("  ") is None
    with pytest.raises(InvalidOperation):
        parse_amount("12,000")


def test_p130544_style_commitment_difference(registry):
    projects = _table("bronze_projects_raw", [_project()])
    loans = _table("bronze_loans_raw", [
        _loan("P130544", "IBRD86010", "100000000"),
        _loan("P130544", "IBRD93240", "150000000"),
    ])  # fmt: skip
    observations = check_commitment_sources(projects, None, loans, registry)
    assert len(observations) == 1
    obs = observations[0]
    assert obs.check is CheckCode.PROJECT_COMMITMENT_SOURCE_DIFFERENCE
    assert obs.severity is Severity.WARNING
    assert obs.project_id == "P130544"
    assert obs.details["workbook_value"] == "100000000.0"
    assert obs.details["loan_snapshot_total"] == "250000000"
    assert [loan["raw_loan_number"] for loan in obs.details["loans"]] == ["IBRD86010", "IBRD93240"]
    assert "Neither is marked wrong" in obs.explanation
    assert "100,000,000" in obs.message and "250,000,000" in obs.message


def test_matching_commitment_produces_no_observation(registry):
    projects = _table("bronze_projects_raw", [_project("P179039", ibrd_commitment="363000000.0")])
    loans = _table("bronze_loans_raw", [_loan("P179039", "IBRD94960", "363000000")])
    assert check_commitment_sources(projects, None, loans, registry) == []


def test_financers_ibrd_amount_is_also_compared(registry):
    projects = _table("bronze_projects_raw", [_project(ibrd_commitment="250000000")])
    financers = _table("bronze_financers_raw", [
        {"project_id": "P130544", "financer_id": "IBRD", "amount_usd": "100000000"},
        {"project_id": "P130544", "financer_id": "BORR", "amount_usd": "1"},
    ])  # fmt: skip
    loans = _table("bronze_loans_raw", [
        _loan("P130544", "IBRD86010", "100000000"),
        _loan("P130544", "IBRD93240", "150000000"),
    ])  # fmt: skip
    observations = check_commitment_sources(projects, financers, loans, registry)
    assert [o.details["workbook_sheet"] for o in observations] == ["Financers"]


def test_missing_and_duplicate_projects(registry):
    projects = _table("bronze_projects_raw", [_project("P130544"), _project("P130544")])
    loans = _table("bronze_loans_raw", [_loan("P130544", "IBRD86010", "1")])
    observations = check_project_presence(
        {"bronze_projects_raw": projects, "bronze_loans_raw": loans}, registry
    )
    checks = {(o.check, o.project_id, o.severity) for o in observations}
    assert (CheckCode.DUPLICATE_PROJECT_RECORD, "P130544", Severity.ERROR) in checks
    assert (CheckCode.REQUIRED_PROJECT_MISSING, "P179039", Severity.ERROR) in checks
    assert (CheckCode.PROJECT_MISSING_FROM_SOURCE, "P506272", Severity.WARNING) in checks


def test_multiple_and_duplicate_loans():
    loans = _table("bronze_loans_raw", [
        _loan("P130544", "IBRD86010", "1", normalized_loan_number="IBRD-8601-0"),
        _loan("P130544", "IBRD86010", "1", normalized_loan_number="IBRD-8601-0"),
        _loan("P179039", "BAD", "1"),
    ])  # fmt: skip
    checks = {o.check: o for o in check_loans(loans)}
    assert checks[CheckCode.MULTIPLE_LOANS_PER_PROJECT].severity is Severity.INFO
    assert checks[CheckCode.DUPLICATE_LOAN_IDENTIFIER].severity is Severity.ERROR
    assert checks[CheckCode.UNRECOGNIZED_LOAN_NUMBER].project_id == "P179039"
    assert checks[CheckCode.LOAN_SNAPSHOT_DATE].details["end_of_period"] == "08/31/2026"


def test_inconsistent_snapshot_dates():
    loans = _table("bronze_loans_raw", [
        _loan("P130544", "IBRD86010", "1", end_of_period="08/31/2026"),
        _loan("P179039", "IBRD94960", "1", end_of_period="07/31/2026"),
    ])  # fmt: skip
    assert CheckCode.SNAPSHOT_DATE_INCONSISTENT in {o.check for o in check_loans(loans)}


def test_value_checks_on_contract_columns():
    loans = _table("bronze_loans_raw", [
        _loan("P130544", "", "abc", agreement_signing_date="2016-05-24",
              loan_status="", currency_of_commitment=""),
    ], known_gaps=["current_principal"])  # fmt: skip
    by_check = {}
    for o in check_table_values(loans, LOANS):
        by_check.setdefault(o.check, []).append(o)
    assert by_check[CheckCode.MISSING_REQUIRED_IDENTIFIER][0].details["field"] == "raw_loan_number"
    assert by_check[CheckCode.MISSING_REQUIRED_IDENTIFIER][0].severity is Severity.ERROR
    assert any(o.details["field"] == "loan_status"
               for o in by_check[CheckCode.MISSING_REQUIRED_VALUE])  # fmt: skip
    assert by_check[CheckCode.UNPARSEABLE_AMOUNT][0].details["raw_value"] == "abc"
    assert by_check[CheckCode.UNPARSEABLE_DATE][0].details["raw_value"] == "2016-05-24"
    blank_fields = {o.details["field"] for o in by_check[CheckCode.BLANK_SOURCE_COLUMN]}
    assert "currency_of_commitment" in blank_fields
    assert by_check[CheckCode.SOURCE_CONCEPT_NOT_PROVIDED][0].details == {
        "concept": "current_principal"
    }


def test_html_in_objective_is_informational():
    projects = _table("bronze_projects_raw", [_project(project_development_objective="<p>x</p>")])
    html = [o for o in check_table_values(projects, PROJECTS)
            if o.check is CheckCode.HTML_IN_TEXT_FIELD]  # fmt: skip
    assert len(html) == 1 and html[0].severity is Severity.INFO


def test_full_report_on_synthetic_run(synthetic_run):
    _, registry, result = synthetic_run
    report = build_data_quality_report(result, registry)
    assert not report.has_errors
    checks = {(o.check, o.project_id) for o in report.observations}
    assert (CheckCode.PROJECT_COMMITMENT_SOURCE_DIFFERENCE, "P130544") in checks
    assert (CheckCode.MULTIPLE_LOANS_PER_PROJECT, "P130544") in checks
    assert (CheckCode.DUPLICATE_CONTRACT_IDENTIFIER, "P130544") in checks
    assert (CheckCode.FINANCER_AMOUNT_FIELDS_DIFFER, "P130544") in checks
    assert (CheckCode.DOCUMENT_LOAN_REFERENCE_LINKED, "P130544") in checks
    assert (CheckCode.HTML_IN_TEXT_FIELD, "P130544") in checks
    # Zero procurement rows for PforR projects must not produce any warning.
    assert not [
        o
        for o in report.observations
        if o.project_id in ("P179039", "P506272") and "procurement" in o.source
    ]
    # Errors sort first, then warnings, then info.
    order = [o.severity for o in report.observations]
    assert order == sorted(order, key=["ERROR", "WARNING", "INFO"].index)


def test_isr_gap_is_reported(synthetic_config, synthetic_run):
    _, registry, result = synthetic_run
    # Remove the only P179039 ISR: expected 1, found 0.
    result.isr_completeness["P179039"] = result.isr_completeness["P179039"].model_copy(
        update={"found_count": 0, "missing": [1], "sequences": [], "is_complete": False}
    )
    report = build_data_quality_report(result, registry)
    incomplete = report.by_check(CheckCode.ISR_SEQUENCE_INCOMPLETE)
    assert [o.project_id for o in incomplete] == ["P179039"]
