"""Silver transformations over synthetic Bronze data."""

import copy
from datetime import date
from decimal import Decimal

import pytest

from support.bronze_builders import (
    award_row,
    bronze_table,
    loan,
    loans_table,
    procurement_table,
    project_row,
    projects_table,
)
from worldbank_copilot.common import load_project_registry
from worldbank_copilot.ingestion.contracts import SECTORS, THEMES
from worldbank_copilot.ingestion.procurement import (
    ProcurementCoverage,
    ProcurementCoverageStatus,
)
from worldbank_copilot.transformations.silver import (
    build_financial_summaries,
    build_loans,
    build_procurement,
    build_procurement_coverage,
    build_projects,
    build_sectors,
    build_silver,
    build_themes,
    known_award_count,
)
from worldbank_copilot.transformations.silver_lineage import FIELD_LINEAGE, explain
from worldbank_copilot.transformations.silver_models import (
    AwardAmountBasis,
    IssueKind,
    SilverLoan,
    SilverProjectFinancialSummary,
)


@pytest.fixture
def registry(repo_config_dir):
    return load_project_registry(repo_config_dir)


@pytest.fixture
def silver(synthetic_run):
    _, registry, bronze = synthetic_run
    return build_silver(bronze, registry)


# --- end-to-end over the synthetic Bronze run -----------------------------------------


def test_bronze_is_not_modified(synthetic_run):
    _, registry, bronze = synthetic_run
    before = {name: copy.deepcopy(t.records) for name, t in bronze.tables.items()}
    build_silver(bronze, registry)
    assert {name: t.records for name, t in bronze.tables.items()} == before


def test_projects_one_row_each(silver):
    projects = {p.project_id: p for p in silver["silver_projects"].rows}
    assert sorted(projects) == ["P130544", "P179039", "P506272"]
    p = projects["P130544"]
    assert p.project_development_objective == "Objective."  # HTML stripped
    assert p.approval_date == date(2016, 3, 31)  # ISO datetime -> date
    assert p.current_closing_date == date(2027, 9, 30)
    assert p.original_closing_date is None  # never derived
    assert p.state is None  # not inferred from 'Karnataka' in the name
    assert p.workbook_ibrd_commitment_usd == Decimal("100000000")
    assert (p.country_code, p.country_code_source) == ("IN", "bronze_loans_raw")
    assert projects["P506272"].current_project_cost_usd is None  # blank -> NULL


def test_bronze_value_preserved_while_silver_cleaned(synthetic_run, silver):
    _, _, bronze = synthetic_run
    raw = bronze.tables["bronze_projects_raw"].for_project("P130544")[0]
    assert raw["project_development_objective"] == "<p>Objective.</p>"
    assert silver["silver_projects"].for_project("P130544")[0].project_development_objective == (
        "Objective."
    )


def test_one_to_many_loans_are_not_collapsed(silver):
    p130544 = silver["silver_loans"].for_project("P130544")
    assert [loan.raw_loan_number for loan in p130544] == ["IBRD86010", "IBRD93240"]
    assert {loan.normalized_loan_number for loan in p130544} == {"IBRD-8601-0", "IBRD-9324-0"}
    assert all(isinstance(loan.original_principal_usd, Decimal) for loan in p130544)
    assert len(silver["silver_loans"]) == 4


def test_financial_summary_per_project_per_snapshot(silver):
    summaries = {s.project_id: s for s in silver["silver_project_financial_summary"].rows}
    assert len(summaries) == 3
    s = summaries["P130544"]
    assert s.loan_count == 2 and s.snapshot_date == date(2026, 8, 31)
    assert s.original_principal_total_usd == Decimal("250000000")
    assert s.cancelled_total_usd == Decimal("24937499.48")
    # Source of truth: the workbook value is preserved beside the loan total, not overwritten.
    assert s.workbook_ibrd_commitment_usd == Decimal("100000000")
    assert s.loan_principal_minus_workbook_commitment_usd == Decimal("150000000")
    assert s.commitment_sources_agree is False
    assert summaries["P179039"].commitment_sources_agree is True


def test_procurement_multi_supplier_award(silver):
    awards = {a.contract_id: a for a in silver["silver_procurement_awards"].rows}
    award = awards["1657297"]
    assert award.supplier_count == 2
    assert award.amount_basis is AwardAmountBasis.MULTI_SUPPLIER_UNRESOLVED
    assert award.contract_amount_usd is None  # neither dropped nor summed
    assert award.amount_lower_bound_usd == Decimal("35729.02")
    assert award.amount_upper_bound_usd == Decimal("71458.04")
    single = awards["1610810"]
    assert single.amount_basis is AwardAmountBasis.SINGLE_SUPPLIER_ROW
    assert single.contract_amount_usd == Decimal("162413524.89")
    suppliers = [
        s for s in silver["silver_procurement_suppliers"].rows if s.contract_id == "1657297"
    ]
    assert [s.supplier_name for s in suppliers] == ["SUPPLIER ONE", "SUPPLIER TWO"]


def test_procurement_coverage_is_unknown_not_zero(silver):
    coverage = {c.project_id: c for c in silver["silver_procurement_coverage"].rows}
    assert coverage["P130544"].award_count == 2
    for pid in ("P179039", "P506272"):
        row = coverage[pid]
        assert row.coverage_status == ProcurementCoverageStatus.NOT_COVERED_BY_THIS_DATASET
        assert row.award_count is None and row.supplier_relationship_count is None
        assert known_award_count(row) is None
        assert not silver["silver_procurement_awards"].for_project(pid)


def test_every_silver_row_has_provenance(silver):
    for table in silver.tables.values():
        for row in table.rows:
            assert row.source_refs, table.name


def test_loan_provenance_points_to_bronze_row(synthetic_run, silver):
    _, _, bronze = synthetic_run
    loan_row = next(r for r in silver["silver_loans"].rows if r.raw_loan_number == "IBRD93240")
    ref = loan_row.source_refs[0]
    bronze_row = next(
        r for r in bronze.tables["bronze_loans_raw"].records if r["_source_row"] == ref.source_row
    )
    assert ref.bronze_table == "bronze_loans_raw" and ref.record_key == "IBRD93240"
    assert bronze_row["raw_loan_number"] == "IBRD93240"
    assert Decimal(bronze_row["cancelled_amount_usd"]) == loan_row.cancelled_amount_usd


def test_summary_provenance_includes_every_loan(silver):
    summary = next(
        s for s in silver["silver_project_financial_summary"].rows if s.project_id == "P130544"
    )
    keys = {r.record_key for r in summary.source_refs if r.bronze_table == "bronze_loans_raw"}
    assert keys == {"IBRD86010", "IBRD93240"}


def test_lineage_explains_financial_number():
    text = explain("silver_project_financial_summary", "disbursed_total_usd")
    assert "silver_loans.disbursed_amount_usd" in text
    assert "bronze_loans_raw.disbursed_amount_usd" in text
    assert "'Disbursed Amount (US$)'" in text


def test_lineage_covers_every_silver_field(silver):
    ignored = {"source_refs", "quality_issues", "project_id", "source_file"}
    documented_elsewhere = {
        "silver_project_themes": {"parent_theme_name", "level_1_theme", "taxonomy_label"},
        "silver_procurement_suppliers": {"contract_id"},
        "silver_procurement_coverage": {"dataset", "covered_by_dataset", "source_row_count",
                                        "interpretation"},
        "silver_project_financial_summary": {"snapshot_date"},
    }  # fmt: skip
    for name, table in silver.tables.items():
        missing = (
            set(table.model.model_fields)
            - ignored
            - set(FIELD_LINEAGE[name])
            - documented_elsewhere.get(name, set())
        )
        assert not missing, (name, missing)


# --- focused unit tests on hand-built Bronze ----------------------------------------------


def test_financial_ratio_names_and_definitions():
    loans = build_loans(
        loans_table([loan("P130544", "IBRD86010", "100", disbursed_amount_usd="40",
                          undisbursed_amount_usd="40", cancelled_amount_usd="20")])
    )  # fmt: skip
    (summary,) = build_financial_summaries(loans, [])
    assert summary.disbursement_vs_original_principal_pct == Decimal("40.00")
    assert summary.disbursement_vs_net_principal_pct == Decimal("50.00")  # 40 / (100 - 20)
    fields = SilverProjectFinancialSummary.model_fields
    assert not any("completion" in f or "progress" in f for f in fields)
    assert "not physical progress" in fields["disbursement_vs_original_principal_pct"].description


def test_summary_total_is_null_when_any_input_null():
    loans = build_loans(
        loans_table([loan("P130544", "IBRD86010", "100"),
                     loan("P130544", "IBRD93240", "", disbursed_amount_usd="1")])
    )  # fmt: skip
    (summary,) = build_financial_summaries(loans, [])
    assert summary.loan_count == 2
    assert summary.original_principal_total_usd is None  # not a partial sum
    assert summary.disbursed_total_usd == Decimal("101")
    assert summary.quality_issues[0].field == "original_principal_total_usd"


def test_summary_grain_is_per_snapshot():
    loans = build_loans(
        loans_table([loan("P130544", "IBRD86010", "1", end_of_period="07/31/2026"),
                     loan("P130544", "IBRD86010", "1", end_of_period="08/31/2026")])
    )  # fmt: skip
    summaries = build_financial_summaries(loans, [])
    assert [s.snapshot_date for s in summaries] == [date(2026, 7, 31), date(2026, 8, 31)]


def test_principal_components_difference_is_reported_not_assumed():
    (row,) = build_loans(
        loans_table([loan("P506272", "IBRD98350", "426000000", disbursed_amount_usd="131961818.05",
                          undisbursed_amount_usd="252260260.43",
                          exchange_adjustment_usd="-2613739.59")])
    )  # fmt: skip
    assert row.principal_components_total_usd == Decimal("384222078.48")
    assert row.principal_components_difference_usd == Decimal("41777921.52")
    assert row.original_principal_usd == Decimal("426000000")  # untouched
    assert len(row.valuation_caveats) == 1 and "no conversion" in row.valuation_caveats[0]
    assert row.currency_of_commitment is None  # blank stays NULL; nothing inferred


def test_malformed_values_become_null_with_issue_and_provenance():
    (row,) = build_loans(
        loans_table([loan("P130544", "IBRD86010", "12,000,000",
                          closed_date_most_recent="2024-11-22")])
    )  # fmt: skip
    assert row.original_principal_usd is None
    assert row.closing_date is None
    fields = {i.field: i for i in row.quality_issues}
    issue = fields["original_principal_amount_usd"]
    assert issue.kind is IssueKind.MALFORMED and issue.raw_value == "12,000,000"
    assert issue.source_ref.source_row == 2 and issue.source_ref.bronze_table == "bronze_loans_raw"
    assert fields["closed_date_most_recent"].raw_value == "2024-11-22"


def test_sector_aggregation(registry):
    sectors = build_sectors(bronze_table("bronze_sectors_raw", SECTORS, [
        {"major_sector": "Water", "sector": "Water Supply", "sector_percent": "44.0"},
        {"major_sector": "Water", "sector": "Public Administration - Water ",
         "sector_percent": "47.0"},
        {"major_sector": "Public Administration", "sector": "FY17 - Sub-National",
         "sector_percent": "9"},
    ]))  # fmt: skip
    assert sectors[2].taxonomy_label == "FY17"
    (project,) = build_projects(projects_table([project_row()]), registry, sectors, [], [])
    assert project.sectors == ["Public Administration - Water", "Water Supply",
                               "FY17 - Sub-National"]  # fmt: skip
    assert project.major_sectors == ["Water", "Public Administration"]
    assert project.primary_sector == "Public Administration - Water"


def test_primary_sector_null_on_tie(registry):
    sectors = build_sectors(bronze_table("bronze_sectors_raw", SECTORS, [
        {"major_sector": "A", "sector": "One", "sector_percent": "50"},
        {"major_sector": "A", "sector": "Two", "sector_percent": "50"},
    ]))  # fmt: skip
    (project,) = build_projects(projects_table([project_row()]), registry, sectors, [], [])
    assert project.primary_sector is None and project.sectors == ["One", "Two"]


def _themes(rows):
    return build_themes(bronze_table("bronze_themes_raw", THEMES, rows))


def test_theme_hierarchy_aggregation(registry):
    themes, orphans = _themes([
        {"level_1": "FY17 - Urban", "percentage_1": "100"},
        {"level_1": "> FY17 - Urban", "percentage_1": "100",
         "level_2": " Water  Institutions", "percentage_2": "80"},
        {"level_1": "> FY17 - Urban", "percentage_1": "100",
         "level_2": ">  Water Institutions", "percentage_2": "80",
         "level_3": "Utility Performance", "percentage_3": "51"},
        {"level_1": "Finance", "percentage_1": "2"},
    ])  # fmt: skip
    assert orphans == []
    paths = [(t.level, t.theme_path, t.percentage) for t in themes]
    assert paths == [
        (1, ["FY17 - Urban"], Decimal("100")),
        (2, ["FY17 - Urban", "Water Institutions"], Decimal("80")),
        (3, ["FY17 - Urban", "Water Institutions", "Utility Performance"], Decimal("51")),
        (1, ["Finance"], Decimal("2")),
    ]
    assert themes[2].parent_theme_name == "Water Institutions"
    assert all(not t.quality_issues for t in themes)
    (project,) = build_projects(projects_table([project_row()]), registry, [], themes, [])
    assert project.themes == ["FY17 - Urban", "Finance"]


def test_theme_structure_issues_are_recorded():
    themes, _ = _themes([
        {"level_1": "Urban", "percentage_1": "100", "level_2": "Water", "percentage_2": "80"},
        {"level_1": "Other", "percentage_1": "5"},
        {"level_1": "Other", "percentage_1": "6"},
    ])  # fmt: skip
    water = next(t for t in themes if t.theme_name == "Water")
    reasons = {i.reason for i in water.quality_issues}
    assert "ancestor level lacks the '>' marker" in reasons
    assert "parent theme node has no row of its own" in reasons
    other = next(t for t in themes if t.theme_name == "Other")
    assert len(other.source_refs) == 2
    assert other.quality_issues[0].kind is IssueKind.CONFLICT


def test_project_never_invents_state_or_original_closing(registry):
    (project,) = build_projects(
        projects_table([project_row(project_name="Karnataka Urban Water Project")]),
        registry, [], [], [],
    )  # fmt: skip
    assert project.state is None and project.original_closing_date is None


def test_country_code_conflict_left_null(registry):
    loans = build_loans(
        loans_table(
            [
                loan("P130544", "IBRD86010", "1", country_code="IN"),
                loan("P130544", "IBRD93240", "1", country_code="BD"),
            ]
        )
    )
    (project,) = build_projects(projects_table([project_row()]), registry, [], [], loans)
    assert project.country_code is None
    assert project.quality_issues[0].kind is IssueKind.CONFLICT


def test_award_field_conflict_left_null():
    awards, suppliers, _ = build_procurement(procurement_table([
        award_row("P130544", "1", "A", "10", contract_signing_date="06/05/2021"),
        award_row("P130544", "1", "B", "10", contract_signing_date="06/06/2021"),
    ]))  # fmt: skip
    (award,) = awards
    assert award.contract_signing_date is None
    assert award.quality_issues[0].field == "contract_signing_date"
    assert len(suppliers) == 2


def test_different_supplier_amounts_bounds():
    awards, _, _ = build_procurement(procurement_table([
        award_row("P130544", "9", "A", "1130.43"),
        award_row("P130544", "9", "B", "1097.18"),
        award_row("P130544", "9", "C", "1097.18"),
    ]))  # fmt: skip
    (award,) = awards
    assert award.amount_lower_bound_usd == Decimal("1130.43")
    assert award.amount_upper_bound_usd == Decimal("3324.79")


def test_blank_contract_identifier_is_not_grouped():
    awards, _, orphans = build_procurement(procurement_table([award_row("P130544", "", "A", "1")]))
    assert awards == [] and orphans[0].field == "wb_contract_number"


def test_coverage_semantics_rule():
    coverage = [
        ProcurementCoverage(project_id="P130544", dataset="d", covered_by_dataset=True,
                            coverage_status=ProcurementCoverageStatus.NO_RECORDS_IN_DATASET,
                            row_count=0, interpretation="i"),
        ProcurementCoverage(project_id="P179039", dataset="d", covered_by_dataset=False,
                            coverage_status=ProcurementCoverageStatus.NOT_COVERED_BY_THIS_DATASET,
                            row_count=0, interpretation="i"),
    ]  # fmt: skip
    rows = {r.project_id: r for r in build_procurement_coverage(coverage, [], [], "f", "r")}
    assert rows["P130544"].award_count == 0  # covered: zero rows in dataset
    assert known_award_count(rows["P130544"]) == 0
    assert rows["P179039"].award_count is None  # not covered: unknown
    assert known_award_count(rows["P179039"]) is None


def test_decimal_serialisation_is_string(silver):
    loan_row: SilverLoan = silver["silver_loans"].rows[0]
    dumped = loan_row.model_dump(mode="json")
    assert isinstance(dumped["original_principal_usd"], str)
    assert dumped["snapshot_date"] == "2026-08-31"
