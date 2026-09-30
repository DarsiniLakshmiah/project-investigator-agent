"""Human-readable and JSON renderings of a Silver build and its quality report."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from worldbank_copilot.common.project_registry import ProjectRegistry
from worldbank_copilot.common.quality import DataQualityReport
from worldbank_copilot.transformations.silver import SilverResult


def _money(value: Decimal | None) -> str:
    if value is None:
        return "NULL"
    return f"{value:,.2f}"


def format_silver_report(
    silver: SilverResult, report: DataQualityReport, registry: ProjectRegistry
) -> str:
    ids = registry.project_ids
    projects = silver["silver_projects"].rows
    loans = silver["silver_loans"].rows
    lines = ["", "=" * 78, "SILVER", "=" * 78, "", f"Projects: {len(projects)}"]
    for p in projects:
        lines.append(
            f"  {p.project_id}  {p.financing_instrument} | status {p.project_status} | "
            f"approved {p.approval_date} | effective {p.effective_date} | "
            f"closing {p.current_closing_date} | original closing {p.original_closing_date}"
        )

    lines += ["", "Loans:"]
    for pid in ids:
        project_loans = [loan for loan in loans if loan.project_id == pid]
        lines.append(f"  {pid}: {len(project_loans)}")
        for loan in project_loans:
            lines.append(
                f"      {loan.raw_loan_number} ({loan.normalized_loan_number}) {loan.loan_status}: "
                f"original {_money(loan.original_principal_usd)}, disbursed "
                f"{_money(loan.disbursed_amount_usd)}, undisbursed "
                f"{_money(loan.undisbursed_amount_usd)}, cancelled "
                f"{_money(loan.cancelled_amount_usd)}, components diff "
                f"{_money(loan.principal_components_difference_usd)}"
            )

    summaries = silver["silver_project_financial_summary"].rows
    lines += ["", f"Project financial summaries: {len(summaries)}"]
    for s in summaries:
        lines.append(
            f"  {s.project_id} @ {s.snapshot_date}: loans {s.loan_count} | original "
            f"{_money(s.original_principal_total_usd)} | disbursed {_money(s.disbursed_total_usd)} "
            f"| undisbursed {_money(s.undisbursed_total_usd)} | cancelled "
            f"{_money(s.cancelled_total_usd)}"
        )
        lines.append(
            "      disbursement_vs_original_principal_pct "
            f"{s.disbursement_vs_original_principal_pct}"
            f" | disbursement_vs_net_principal_pct {s.disbursement_vs_net_principal_pct} | "
            f"workbook IBRD commitment {_money(s.workbook_ibrd_commitment_usd)} | loan principal - "
            f"workbook {_money(s.loan_principal_minus_workbook_commitment_usd)} | agree "
            f"{s.commitment_sources_agree}"
        )

    awards = silver["silver_procurement_awards"].rows
    suppliers = silver["silver_procurement_suppliers"].rows
    lines += ["", "Procurement:"]
    for row in silver["silver_procurement_coverage"].rows:
        lines.append(
            f"  {row.project_id} {row.coverage_status:<28} source rows {row.source_row_count} | "
            f"awards {row.award_count if row.award_count is not None else 'unknown (NULL)'} | "
            f"supplier relationships "
            f"{row.supplier_relationship_count if row.award_count is not None else 'unknown'}"
        )
    multi = [a for a in awards if a.supplier_count > 1]
    lines.append(
        f"  awards {len(awards)}, supplier rows {len(suppliers)}, multi-supplier awards "
        f"{len(multi)}"
    )

    counts = report.counts()
    lines += [
        "",
        f"Silver data-quality observations ({counts['ERROR']} errors, {counts['WARNING']} "
        f"warnings, {counts['INFO']} info):",
    ]
    for o in report.observations:
        lines.append(f"  [{o.severity.value:<7}] {o.check.value:<38} {o.project_id or '-':<8} "
                     f"{o.message}")  # fmt: skip
    return "\n".join(lines)


def silver_json_dict(silver: SilverResult, report: DataQualityReport) -> dict[str, Any]:
    return {
        "run_id": silver.run_id,
        "tables": {
            name: [row.model_dump(mode="json") for row in table.rows]
            for name, table in silver.tables.items()
        },
        "table_issues": [i.model_dump(mode="json") for i in silver.table_issues],
        "data_quality": {
            "counts": report.counts(),
            "has_errors": report.has_errors,
            "observations": [o.model_dump(mode="json") for o in report.observations],
        },
    }
