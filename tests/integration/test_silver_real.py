"""Silver checks against the real local source files. Run with: pytest -m integration"""

from datetime import date
from decimal import Decimal

import pytest

from worldbank_copilot.common import load_project_registry, load_settings
from worldbank_copilot.common.quality import CheckCode
from worldbank_copilot.ingestion.data_quality import build_data_quality_report
from worldbank_copilot.ingestion.pipeline import ingest_bronze, resolve_source_files
from worldbank_copilot.transformations.silver import build_silver, known_award_count
from worldbank_copilot.transformations.silver_models import AwardAmountBasis
from worldbank_copilot.transformations.silver_quality import build_silver_quality_report

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_silver():
    settings = load_settings("local", env={})
    try:
        resolve_source_files(settings)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"real source files not available: {exc}")
    registry = load_project_registry(settings.config_dir)
    bronze = ingest_bronze(settings, registry)
    silver = build_silver(bronze, registry)
    return (
        bronze,
        silver,
        build_data_quality_report(bronze, registry),
        build_silver_quality_report(silver, registry),
    )


def test_projects(real_silver):
    _, silver, _, _ = real_silver
    projects = {p.project_id: p for p in silver["silver_projects"].rows}
    assert sorted(projects) == ["P130544", "P179039", "P506272"]
    p = projects["P130544"]
    assert p.project_development_objective.startswith("The Project Development Objective")
    assert "<p>" not in p.project_development_objective
    assert (p.approval_date, p.current_closing_date) == (date(2016, 3, 31), date(2027, 9, 30))
    assert p.original_closing_date is None and p.state is None
    assert p.workbook_ibrd_commitment_usd == Decimal("100000000")
    assert projects["P506272"].current_project_cost_usd is None


def test_loans_one_to_many(real_silver):
    _, silver, _, _ = real_silver
    by_project = {}
    for loan in silver["silver_loans"].rows:
        by_project.setdefault(loan.project_id, []).append(loan.raw_loan_number)
    assert by_project == {
        "P130544": ["IBRD86010", "IBRD93240"],
        "P179039": ["IBRD94960"],
        "P506272": ["IBRD98350"],
    }


def test_financial_summaries(real_silver):
    _, silver, _, _ = real_silver
    summaries = {s.project_id: s for s in silver["silver_project_financial_summary"].rows}
    assert len(summaries) == 3
    s = summaries["P130544"]
    assert s.loan_count == 2 and s.snapshot_date == date(2026, 8, 31)
    assert s.original_principal_total_usd == Decimal("250000000")
    assert s.disbursed_total_usd == Decimal("132713450.09")
    assert s.cancelled_total_usd == Decimal("24937499.48")
    assert s.workbook_ibrd_commitment_usd == Decimal("100000000")
    assert s.commitment_sources_agree is False
    assert summaries["P506272"].loans_with_valuation_caveats == ["IBRD98350"]


def test_contract_1657297(real_silver):
    _, silver, _, _ = real_silver
    award = next(a for a in silver["silver_procurement_awards"].rows if a.contract_id == "1657297")
    assert award.amount_basis is AwardAmountBasis.MULTI_SUPPLIER_UNRESOLVED
    assert award.contract_amount_usd is None
    assert (award.amount_lower_bound_usd, award.amount_upper_bound_usd) == (
        Decimal("35729.02"),
        Decimal("71458.04"),
    )
    assert award.supplier_count == 2 and len(award.source_refs) == 2


def test_procurement_coverage(real_silver):
    _, silver, _, _ = real_silver
    coverage = {c.project_id: c for c in silver["silver_procurement_coverage"].rows}
    assert (coverage["P130544"].award_count, coverage["P130544"].supplier_relationship_count) == (
        16,
        17,
    )
    for pid in ("P179039", "P506272"):
        assert known_award_count(coverage[pid]) is None


def test_quality(real_silver):
    _, _, bronze_report, silver_report = real_silver
    assert not silver_report.has_errors
    # The Bronze commitment observation is retained, not "fixed".
    assert {o.project_id for o in bronze_report.by_check(
        CheckCode.PROJECT_COMMITMENT_SOURCE_DIFFERENCE)} == {"P130544"}  # fmt: skip
    unresolved = silver_report.by_check(CheckCode.PROCUREMENT_AWARD_AMOUNT_UNRESOLVED)
    assert [o.details["contract_id"] for o in unresolved] == ["1657297"]
    diffs = silver_report.by_check(CheckCode.LOAN_PRINCIPAL_COMPONENTS_DIFFERENCE)
    assert {o.details["loan"] for o in diffs} == {"IBRD93240", "IBRD98350"}


def test_bronze_untouched_by_silver(real_silver):
    bronze, _, _, _ = real_silver
    raw = bronze.tables["bronze_projects_raw"].for_project("P130544")[0]
    assert raw["project_development_objective"].startswith("<p>")
    assert raw["ibrd_commitment"] == "100000000.0"
