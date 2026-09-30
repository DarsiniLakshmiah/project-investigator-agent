"""Silver data-quality checks."""

from decimal import Decimal

import pytest

from support.bronze_builders import award_row, loan, loans_table, procurement_table
from worldbank_copilot.common.quality import CheckCode, Severity
from worldbank_copilot.transformations.silver import (
    SilverResult,
    SilverTable,
    build_loans,
    build_procurement,
    build_silver,
)
from worldbank_copilot.transformations.silver_models import SilverProcurementCoverage
from worldbank_copilot.transformations.silver_quality import (
    build_silver_quality_report,
    check_loans,
    check_procurement,
    check_projects,
    check_provenance_and_issues,
)


@pytest.fixture
def base(synthetic_run):
    _, registry, bronze = synthetic_run
    return registry, build_silver(bronze, registry)


def _checks(observations):
    return {(o.check, o.project_id, o.severity) for o in observations}


def _replace(silver: SilverResult, name: str, rows) -> SilverResult:
    tables = dict(silver.tables)
    tables[name] = SilverTable(name, silver[name].model, rows)
    return SilverResult(silver.run_id, tables)


def test_synthetic_silver_has_no_errors(base):
    registry, silver = base
    report = build_silver_quality_report(silver, registry)
    assert report.layer == "silver" and not report.has_errors
    checks = {o.check for o in report.observations}
    assert CheckCode.PROCUREMENT_AWARD_AMOUNT_UNRESOLVED in checks
    assert CheckCode.PROJECT_METADATA_INCOMPLETE in checks


def test_exactly_three_projects_required(base):
    registry, silver = base
    broken = _replace(silver, "silver_projects", silver["silver_projects"].rows[:2])
    assert (CheckCode.SILVER_PROJECT_COUNT_MISMATCH, None, Severity.ERROR) in _checks(
        check_projects(broken, registry)
    )


def test_known_unavailable_fields_listed_separately(base):
    registry, silver = base
    obs = [o for o in check_projects(silver, registry) if o.project_id == "P130544"
           and o.check is CheckCode.PROJECT_METADATA_INCOMPLETE][0]  # fmt: skip
    assert obs.details["unavailable_fields"] == ["state", "original_closing_date"]
    assert "state" not in obs.details["null_fields"]


def test_expected_and_unexpected_loans(base):
    registry, silver = base
    loans = [x for x in silver["silver_loans"].rows if x.raw_loan_number != "IBRD93240"]
    extra = build_loans(loans_table([loan("P179039", "IBRD99990", "1")]))
    broken = _replace(silver, "silver_loans", loans + extra)
    checks = _checks(check_loans(broken, registry))
    assert (CheckCode.SILVER_EXPECTED_LOAN_MISSING, "P130544", Severity.ERROR) in checks
    assert (CheckCode.SILVER_UNEXPECTED_LOAN, "P179039", Severity.WARNING) in checks


def test_duplicate_loan_and_broken_reference(base):
    registry, silver = base
    loans = silver["silver_loans"].rows
    orphan = build_loans(loans_table([loan("P000001", "IBRD11110", "1")]))
    broken = _replace(silver, "silver_loans", loans + [loans[0]] + orphan)
    checks = {o.check for o in check_loans(broken, registry)}
    assert CheckCode.SILVER_DUPLICATE_LOAN in checks
    assert CheckCode.LOAN_PROJECT_REFERENCE_BROKEN in checks


def test_negative_amount_is_an_error(base):
    registry, silver = base
    bad = build_loans(loans_table([loan("P130544", "IBRD86010", "-5")]))
    others = [x for x in silver["silver_loans"].rows if x.raw_loan_number != "IBRD86010"]
    checks = _checks(check_loans(_replace(silver, "silver_loans", others + bad), registry))
    assert (CheckCode.NEGATIVE_FINANCIAL_VALUE, "P130544", Severity.ERROR) in checks


def test_negative_exchange_adjustment_is_allowed(base):
    registry, silver = base
    rows = build_loans(loans_table([loan("P506272", "IBRD98350", "10",
                                         exchange_adjustment_usd="-2")]))  # fmt: skip
    others = [x for x in silver["silver_loans"].rows if x.raw_loan_number != "IBRD98350"]
    checks = {o.check for o in check_loans(_replace(silver, "silver_loans", others + rows),
                                           registry)}  # fmt: skip
    assert CheckCode.NEGATIVE_FINANCIAL_VALUE not in checks
    assert CheckCode.LOAN_VALUATION_CAVEAT in checks


@pytest.mark.parametrize(
    ("exchange", "severity"), [("-2613739.59", Severity.INFO), ("0", Severity.WARNING)]
)
def test_reconciliation_difference_severity(base, exchange, severity):
    registry, silver = base
    rows = build_loans(loans_table([loan("P506272", "IBRD98350", "426", disbursed_amount_usd="100",
                                         undisbursed_amount_usd="200",
                                         exchange_adjustment_usd=exchange)]))  # fmt: skip
    others = [x for x in silver["silver_loans"].rows if x.raw_loan_number != "IBRD98350"]
    obs = [o for o in check_loans(_replace(silver, "silver_loans", others + rows), registry)
           if o.check is CheckCode.LOAN_PRINCIPAL_COMPONENTS_DIFFERENCE]  # fmt: skip
    assert len(obs) == 1 and obs[0].severity is severity
    assert obs[0].details["difference_usd"] == "126"


def test_reconciling_loan_produces_no_difference(base):
    registry, silver = base
    obs = [o for o in check_loans(silver, registry)
           if o.check is CheckCode.LOAN_PRINCIPAL_COMPONENTS_DIFFERENCE]  # fmt: skip
    assert obs == []  # synthetic loans reconcile exactly


def test_date_order(base):
    registry, silver = base
    rows = build_loans(loans_table([loan("P130544", "IBRD86010", "1",
                                         effective_date_most_recent="12/31/2025")]))  # fmt: skip
    others = [x for x in silver["silver_loans"].rows if x.raw_loan_number != "IBRD86010"]
    obs = [o for o in check_loans(_replace(silver, "silver_loans", others + rows), registry)
           if o.check is CheckCode.LOAN_DATE_ORDER_VIOLATION]  # fmt: skip
    assert obs[0].details == {"earlier_field": "effective_date", "later_field": "closing_date"}


def test_last_disbursement_after_closing_is_not_a_violation(base):
    registry, silver = base
    rows = build_loans(loans_table([loan("P130544", "IBRD86010", "1",
                                         last_disbursement_date="04/07/2025")]))  # fmt: skip
    others = [x for x in silver["silver_loans"].rows if x.raw_loan_number != "IBRD86010"]
    checks = {o.check for o in check_loans(_replace(silver, "silver_loans", others + rows),
                                           registry)}  # fmt: skip
    assert CheckCode.LOAN_DATE_ORDER_VIOLATION not in checks


def test_coverage_semantics_violation_detected(base):
    _, silver = base
    rows = [
        r.model_copy(update={"award_count": 0}) if r.project_id == "P179039" else r
        for r in silver["silver_procurement_coverage"].rows
    ]
    broken = _replace(silver, "silver_procurement_coverage", rows)
    checks = _checks(check_procurement(broken))
    assert (CheckCode.PROCUREMENT_COVERAGE_SEMANTICS_VIOLATION, "P179039", Severity.ERROR) in checks


def test_coverage_rows_are_typed_as_unknown():
    field = SilverProcurementCoverage.model_fields["award_count"]
    assert "unknown, not zero" in field.description


def test_award_grain_violation(base):
    _, silver = base
    awards = silver["silver_procurement_awards"].rows
    broken = _replace(silver, "silver_procurement_awards", awards + [awards[0]])
    assert CheckCode.PROCUREMENT_GRAIN_VIOLATION in {o.check for o in check_procurement(broken)}


def test_unresolved_amount_warning_carries_bounds(base):
    _, silver = base
    (obs,) = [o for o in check_procurement(silver)
              if o.check is CheckCode.PROCUREMENT_AWARD_AMOUNT_UNRESOLVED]  # fmt: skip
    assert obs.severity is Severity.WARNING
    assert Decimal(obs.details["lower_bound_usd"]) == Decimal("35729.02")
    assert Decimal(obs.details["upper_bound_usd"]) == Decimal("71458.04")


def test_missing_provenance_is_an_error(base):
    _, silver = base
    loans = silver["silver_loans"].rows
    stripped = [loans[0].model_copy(update={"source_refs": []}), *loans[1:]]
    checks = {o.check for o in check_provenance_and_issues(
        _replace(silver, "silver_loans", stripped))}  # fmt: skip
    assert CheckCode.MISSING_PROVENANCE in checks


def test_malformed_value_becomes_observation_with_source_ref(base):
    _, silver = base
    bad = build_loans(loans_table([loan("P130544", "IBRD86010", "1e8")]))
    obs = [o for o in check_provenance_and_issues(_replace(silver, "silver_loans", bad))
           if o.check is CheckCode.SOURCE_VALUE_NOT_NORMALIZABLE]  # fmt: skip
    assert obs[0].details["raw_value"] == "1e8"
    assert obs[0].details["source_ref"]["bronze_table"] == "bronze_loans_raw"


def test_orphan_procurement_issue_reported(base):
    _, silver = base
    _, _, orphans = build_procurement(procurement_table([award_row("P130544", "", "A", "1")]))
    result = SilverResult(silver.run_id, silver.tables, table_issues=orphans)
    checks = {o.check for o in check_provenance_and_issues(result)}
    assert CheckCode.SOURCE_VALUE_NOT_NORMALIZABLE in checks
