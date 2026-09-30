"""Field-level lineage for Silver: which Bronze field(s)/source column(s) and which transform.

Together with each row's ``source_refs`` this answers "where did this number come
from?", e.g.::

    >>> explain("silver_project_financial_summary", "disbursed_total_usd")
    silver_project_financial_summary.disbursed_total_usd = sum over loans of
      silver_loans.disbursed_amount_usd = Decimal(...) of
        bronze_loans_raw.disbursed_amount_usd <- source column 'Disbursed Amount (US$)'
"""

from __future__ import annotations

from dataclasses import dataclass

from worldbank_copilot.ingestion.contracts import ALL_TABULAR_CONTRACTS


@dataclass(frozen=True)
class FieldLineage:
    inputs: tuple[str, ...]  # "table.field" (Bronze or Silver)
    transform: str


def _l(transform: str, *inputs: str) -> FieldLineage:
    return FieldLineage(tuple(inputs), transform)


_P, _LN, _PR = "bronze_projects_raw", "bronze_loans_raw", "bronze_procurement_raw"
_TEXT = "clean_text (whitespace collapsed, blank -> NULL)"
_CAT = "normalize_category (whitespace collapsed, casing kept)"
_DATE_WB = "parse_source_date [%Y-%m-%d | %Y-%m-%dT%H:%M:%SZ]"
_DATE_CSV = "parse_source_date [%m/%d/%Y]"
_MONEY = "parse_money (exact Decimal, no float)"

_PROJECT = {
    "project_id": _l("registry-canonical project ID", f"{_P}.source_project_id"),
    "project_name": _l(_TEXT, f"{_P}.project_name"),
    "country": _l(_TEXT, f"{_P}.country"),
    "country_code": _l("clean_text; used only if all of the project's loans agree",
                       f"{_LN}.country_code"),
    "country_code_source": _l("constant naming the table country_code came from"),
    "region": _l(_TEXT, f"{_P}.region"),
    "state": _l("always NULL (no structured source)"),
    "financing_instrument": _l(_CAT, f"{_P}.lending_instrument"),
    "financing_type": _l(_CAT, f"{_P}.financing_type"),
    "project_status": _l(_CAT, f"{_P}.project_status"),
    "last_stage_reached": _l(_CAT, f"{_P}.last_stage_reached_name"),
    "approval_date": _l(_DATE_WB, f"{_P}.board_approval_date"),
    "effective_date": _l(_DATE_WB, f"{_P}.loan_effective_date"),
    "current_closing_date": _l(_DATE_WB, f"{_P}.project_closing_date"),
    "original_closing_date": _l("always NULL in Phase 3 (documents, Phase 4-5)"),
    "public_disclosure_date": _l(_DATE_WB, f"{_P}.public_disclosure_date"),
    "borrower": _l(_TEXT, f"{_P}.borrower"),
    "implementing_agency": _l(_TEXT, f"{_P}.implementing_agency"),
    "primary_sector": _l("sector with strictly highest sector_percent",
                         "silver_project_sectors.sector", "silver_project_sectors.sector_percent"),
    "sectors": _l("distinct sectors ordered by percent desc, then name",
                  "silver_project_sectors.sector"),
    "major_sectors": _l("distinct major sectors in first-seen order",
                        "silver_project_sectors.major_sector"),
    "themes": _l("distinct level-1 themes in source order", "silver_project_themes.theme_name"),
    "project_development_objective": _l("strip_html then clean_text",
                                        f"{_P}.project_development_objective"),
    "workbook_ibrd_commitment_usd": _l(_MONEY, f"{_P}.ibrd_commitment"),
    "workbook_ida_commitment_usd": _l(_MONEY, f"{_P}.ida_commitment"),
    "workbook_grant_amount_usd": _l(_MONEY, f"{_P}.grant_amount"),
    "workbook_total_commitment_usd": _l(_MONEY, f"{_P}.total_ibrd_ida_grant_commitment"),
    "current_project_cost_usd": _l(_MONEY, f"{_P}.current_project_cost"),
    "environmental_assessment_category": _l(_CAT, f"{_P}.environmental_assessment_category"),
    "environmental_and_social_risk": _l(_CAT, f"{_P}.environmental_and_social_risk"),
    "source_last_updated": _l(_DATE_WB, f"{_P}.last_update_date"),
    "source_file": _l("Bronze lineage", f"{_P}._source_file"),
}  # fmt: skip

_SECTOR = {
    "major_sector": _l(_CAT, "bronze_sectors_raw.major_sector"),
    "sector": _l(_CAT, "bronze_sectors_raw.sector"),
    "sector_percent": _l("parse_decimal", "bronze_sectors_raw.sector_percent"),
    "taxonomy_label": _l("explicit 'FYnn - ' name prefix, else NULL", "bronze_sectors_raw.sector"),
}

_THEME = {
    "level": _l("deepest non-blank level of the row",
                "bronze_themes_raw.level_1", "bronze_themes_raw.level_2",
                "bronze_themes_raw.level_3"),
    "theme_name": _l("clean_text of the deepest level", "bronze_themes_raw.level_1",
                     "bronze_themes_raw.level_2", "bronze_themes_raw.level_3"),
    "theme_path": _l("ancestor names with '>' marker removed, then theme_name",
                     "bronze_themes_raw.level_1", "bronze_themes_raw.level_2",
                     "bronze_themes_raw.level_3"),
    "percentage": _l("parse_decimal of the deepest level's percentage",
                     "bronze_themes_raw.percentage_1", "bronze_themes_raw.percentage_2",
                     "bronze_themes_raw.percentage_3"),
}  # fmt: skip

_LOAN = {
    "raw_loan_number": _l("as in source", f"{_LN}.raw_loan_number"),
    "normalized_loan_number": _l("common.identifiers.normalize_loan_number",
                                 f"{_LN}.normalized_loan_number"),
    "loan_base_number": _l("parse_loan_number().base_number", f"{_LN}.raw_loan_number"),
    "loan_suffix": _l("parse_loan_number().suffix", f"{_LN}.raw_loan_number"),
    "lender": _l("parse_loan_number().lender", f"{_LN}.raw_loan_number"),
    "loan_type": _l(_CAT, f"{_LN}.loan_type"),
    "loan_status": _l(_CAT, f"{_LN}.loan_status"),
    "borrower": _l(_TEXT, f"{_LN}.borrower"),
    "country_code": _l(_TEXT, f"{_LN}.country_code"),
    "currency_of_commitment": _l(_TEXT + "; never inferred", f"{_LN}.currency_of_commitment"),
    "original_principal_usd": _l(_MONEY, f"{_LN}.original_principal_amount_usd"),
    "cancelled_amount_usd": _l(_MONEY, f"{_LN}.cancelled_amount_usd"),
    "disbursed_amount_usd": _l(_MONEY, f"{_LN}.disbursed_amount_usd"),
    "undisbursed_amount_usd": _l(_MONEY, f"{_LN}.undisbursed_amount_usd"),
    "repaid_to_ibrd_usd": _l(_MONEY, f"{_LN}.repaid_to_ibrd_usd"),
    "due_to_ibrd_usd": _l(_MONEY, f"{_LN}.due_to_ibrd_usd"),
    "exchange_adjustment_usd": _l(_MONEY, f"{_LN}.exchange_adjustment_usd"),
    "borrowers_obligation_usd": _l(_MONEY, f"{_LN}.borrowers_obligation_usd"),
    "loans_held_usd": _l(_MONEY, f"{_LN}.loans_held_usd"),
    "principal_components_total_usd": _l(
        "disbursed + undisbursed + cancelled (Decimal)", "silver_loans.disbursed_amount_usd",
        "silver_loans.undisbursed_amount_usd", "silver_loans.cancelled_amount_usd"),
    "principal_components_difference_usd": _l(
        "original_principal - principal_components_total", "silver_loans.original_principal_usd",
        "silver_loans.principal_components_total_usd"),
    "board_approval_date": _l(_DATE_CSV, f"{_LN}.board_approval_date"),
    "agreement_signing_date": _l(_DATE_CSV, f"{_LN}.agreement_signing_date"),
    "effective_date": _l(_DATE_CSV, f"{_LN}.effective_date_most_recent"),
    "closing_date": _l(_DATE_CSV, f"{_LN}.closed_date_most_recent"),
    "last_disbursement_date": _l(_DATE_CSV, f"{_LN}.last_disbursement_date"),
    "first_repayment_date": _l(_DATE_CSV, f"{_LN}.first_repayment_date"),
    "last_repayment_date": _l(_DATE_CSV, f"{_LN}.last_repayment_date"),
    "snapshot_date": _l(_DATE_CSV, f"{_LN}.end_of_period"),
    "valuation_caveats": _l("rule: note when exchange adjustment is non-zero (no conversion)",
                            f"{_LN}.exchange_adjustment_usd"),
    "source_file": _l("Bronze lineage", f"{_LN}._source_file"),
}  # fmt: skip

_SUMMARY = {
    "loan_count": _l("count of silver loans", "silver_loans.raw_loan_number"),
    "raw_loan_numbers": _l("list of silver loans", "silver_loans.raw_loan_number"),
    "snapshot_date": _l("group key", "silver_loans.snapshot_date"),
    "original_principal_total_usd": _l("Decimal sum over loans; NULL if any input NULL",
                                       "silver_loans.original_principal_usd"),
    "cancelled_total_usd": _l("Decimal sum over loans; NULL if any input NULL",
                              "silver_loans.cancelled_amount_usd"),
    "disbursed_total_usd": _l("Decimal sum over loans; NULL if any input NULL",
                              "silver_loans.disbursed_amount_usd"),
    "undisbursed_total_usd": _l("Decimal sum over loans; NULL if any input NULL",
                                "silver_loans.undisbursed_amount_usd"),
    "net_principal_after_cancellation_usd": _l(
        "original_principal_total - cancelled_total",
        "silver_project_financial_summary.original_principal_total_usd",
        "silver_project_financial_summary.cancelled_total_usd"),
    "disbursement_vs_original_principal_pct": _l(
        "disbursed_total / original_principal_total * 100, 2 dp half-even",
        "silver_project_financial_summary.disbursed_total_usd",
        "silver_project_financial_summary.original_principal_total_usd"),
    "disbursement_vs_net_principal_pct": _l(
        "disbursed_total / net_principal_after_cancellation * 100, 2 dp half-even",
        "silver_project_financial_summary.disbursed_total_usd",
        "silver_project_financial_summary.net_principal_after_cancellation_usd"),
    "workbook_ibrd_commitment_usd": _l("copied for comparison",
                                       "silver_projects.workbook_ibrd_commitment_usd"),
    "loan_principal_minus_workbook_commitment_usd": _l(
        "original_principal_total - workbook_ibrd_commitment",
        "silver_project_financial_summary.original_principal_total_usd",
        "silver_projects.workbook_ibrd_commitment_usd"),
    "commitment_sources_agree": _l("difference == 0 (NULL if either side NULL)",
                                   "silver_project_financial_summary."
                                   "loan_principal_minus_workbook_commitment_usd"),
    "loans_with_valuation_caveats": _l("loans with any valuation caveat",
                                       "silver_loans.valuation_caveats"),
}  # fmt: skip

_AWARD = {
    "contract_id": _l("normalize_identifier", f"{_PR}.wb_contract_number"),
    "contract_description": _l(_TEXT + "; must agree across the award's rows",
                               f"{_PR}.contract_description"),
    "procurement_category": _l(_CAT, f"{_PR}.procurement_category"),
    "procurement_method": _l(_CAT, f"{_PR}.procurement_method"),
    "review_type": _l(_CAT, f"{_PR}.review_type"),
    "project_global_practice": _l(_CAT, f"{_PR}.project_global_practice"),
    "borrower_contract_reference": _l(_TEXT + " (free-text reference, spaces occur)",
                                      f"{_PR}.borrower_contract_reference_number"),
    "contract_signing_date": _l(_DATE_CSV, f"{_PR}.contract_signing_date"),
    "fiscal_year": _l("parse_year", f"{_PR}.fiscal_year"),
    "supplier_count": _l("number of supplier rows for the award", f"{_PR}.supplier"),
    "supplier_names": _l(_TEXT, f"{_PR}.supplier"),
    "amount_basis": _l("SINGLE_SUPPLIER_ROW if one row, else MULTI_SUPPLIER_UNRESOLVED",
                       f"{_PR}.supplier_contract_amount_usd"),
    "contract_amount_usd": _l(_MONEY + "; single-supplier awards only",
                              f"{_PR}.supplier_contract_amount_usd"),
    "amount_lower_bound_usd": _l("max of supplier-row amounts",
                                 f"{_PR}.supplier_contract_amount_usd"),
    "amount_upper_bound_usd": _l("sum of supplier-row amounts",
                                 f"{_PR}.supplier_contract_amount_usd"),
}  # fmt: skip

_SUPPLIER = {
    "supplier_name": _l(_TEXT, f"{_PR}.supplier"),
    "supplier_id": _l("normalize_identifier", f"{_PR}.supplier_id"),
    "supplier_country": _l(_TEXT, f"{_PR}.supplier_country"),
    "supplier_country_code": _l(_TEXT, f"{_PR}.supplier_country_code"),
    "supplier_row_amount_usd": _l(_MONEY, f"{_PR}.supplier_contract_amount_usd"),
}

_COVERAGE = {
    "coverage_status": _l(
        "carried from Bronze coverage", "bronze_procurement_coverage.coverage_status"
    ),
    "award_count": _l(
        "count of silver awards; NULL when not covered", "silver_procurement_awards.contract_id"
    ),
    "supplier_relationship_count": _l(
        "count of silver supplier rows; NULL when not covered",
        "silver_procurement_suppliers.supplier_name",
    ),
}

FIELD_LINEAGE: dict[str, dict[str, FieldLineage]] = {
    "silver_projects": _PROJECT,
    "silver_project_sectors": _SECTOR,
    "silver_project_themes": _THEME,
    "silver_loans": _LOAN,
    "silver_project_financial_summary": _SUMMARY,
    "silver_procurement_awards": _AWARD,
    "silver_procurement_suppliers": _SUPPLIER,
    "silver_procurement_coverage": _COVERAGE,
}

_SOURCE_COLUMN = {
    f"{c.bronze_table}.{col.field}": col.source_names[0]
    for c in ALL_TABULAR_CONTRACTS
    for col in c.columns
}


def source_column(bronze_ref: str) -> str | None:
    """Original source header for ``bronze_table.field`` (None for derived fields)."""
    return _SOURCE_COLUMN.get(bronze_ref)


def explain(table: str, field: str, _depth: int = 0) -> str:
    """Human-readable lineage chain down to the source column(s)."""
    lineage = FIELD_LINEAGE.get(table, {}).get(field)
    pad = "  " * _depth
    if lineage is None:
        column = source_column(f"{table}.{field}")
        suffix = f" <- source column {column!r}" if column else ""
        return f"{pad}{table}.{field}{suffix}"
    lines = [f"{pad}{table}.{field} = {lineage.transform}"]
    for ref in lineage.inputs:
        ref_table, ref_field = ref.split(".", 1)
        lines.append(explain(ref_table, ref_field, _depth + 1))
    return "\n".join(lines)


def lineage_records() -> list[dict]:
    """Flat lineage table (for writing alongside Silver outputs)."""
    rows = []
    for table, fields in FIELD_LINEAGE.items():
        for field, lineage in fields.items():
            rows.append(
                {
                    "silver_table": table,
                    "silver_field": field,
                    "transform": lineage.transform,
                    "inputs": list(lineage.inputs),
                    "source_columns": [c for c in map(source_column, lineage.inputs) if c],
                }
            )
    return rows
