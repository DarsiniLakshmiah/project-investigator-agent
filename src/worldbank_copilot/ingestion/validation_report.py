"""Human-readable and JSON renderings of a Bronze ingestion run + data-quality report."""

from __future__ import annotations

from collections import Counter
from typing import Any

from worldbank_copilot.common.project_registry import ProjectRegistry
from worldbank_copilot.ingestion.data_quality import CheckCode, DataQualityReport, Severity
from worldbank_copilot.ingestion.pipeline import BronzeIngestionResult

_WORKBOOK_TABLES = (
    "bronze_projects_raw",
    "bronze_themes_raw",
    "bronze_sectors_raw",
    "bronze_geo_locations_raw",
    "bronze_financers_raw",
)


def _project_status(result: BronzeIngestionResult, report: DataQualityReport, pid: str) -> str:
    rows = result.tables["bronze_projects_raw"].for_project(pid)
    blocking = {CheckCode.REQUIRED_PROJECT_MISSING, CheckCode.DUPLICATE_PROJECT_RECORD}
    if any(o.project_id == pid and o.check in blocking for o in report.observations):
        return "MISSING" if not rows else f"DUPLICATE ({len(rows)} rows)"
    row = rows[0]
    return f"OK   (sheet row {row['_source_row']}) {row['project_name']}"


def format_text_report(
    result: BronzeIngestionResult, report: DataQualityReport, registry: ProjectRegistry
) -> str:
    ids = registry.project_ids
    files = result.source_files
    lines = [
        f"Bronze ingestion run {result.run_id}  ({result.ingested_at})",
        "",
        "Sources:",
        f"  workbook     {files.relative(files.projects_workbook)}",
        f"  loans        {files.relative(files.loans_snapshot)}",
        f"  procurement  {files.relative(files.procurement_contract_awards)}",
        f"  documents    {files.documents_root}",
        "",
        "Projects:",
    ]
    lines += [f"  {pid} {_project_status(result, report, pid)}" for pid in ids]

    lines += ["", "Workbook sheets (Bronze rows per project):"]
    header = "  " + "table".ljust(28) + "".join(pid.rjust(9) for pid in ids)
    lines.append(header)
    for name in _WORKBOOK_TABLES:
        counts = result.tables[name].count_by_project()
        lines.append("  " + name.ljust(28) + "".join(str(counts.get(p, 0)).rjust(9) for p in ids))

    loans = result.tables["bronze_loans_raw"]
    periods = sorted({r["end_of_period"] for r in loans.records})
    lines += ["", f"Loans (IBRD snapshot, End of Period {', '.join(periods) or 'n/a'}):"]
    for pid in ids:
        project_loans = loans.for_project(pid)
        lines.append(f"  {pid} {len(project_loans)}")
        for r in project_loans:
            lines.append(
                f"      {r['raw_loan_number']} -> {r['normalized_loan_number']}  "
                f"status={r['loan_status']}  original={r['original_principal_amount_usd']}  "
                f"cancelled={r['cancelled_amount_usd']}  disbursed={r['disbursed_amount_usd']}  "
                f"undisbursed={r['undisbursed_amount_usd']}  "
                f"closed={r['closed_date_most_recent']}"
            )

    lines += ["", "Procurement coverage (ipf_contract_awards_india):"]
    for item in result.procurement_coverage:
        lines.append(
            f"  {item.project_id} {item.coverage_status.value:<28} rows={item.row_count}  "
            f"covered_by_dataset={item.covered_by_dataset}"
        )

    lines += ["", "Documents:"]
    inventory = result.document_inventory.table
    for pid in ids:
        docs = inventory.for_project(pid)
        isr = result.isr_completeness[pid]
        state = "complete" if isr.is_complete else "INCOMPLETE"
        types = Counter(d["document_type"] for d in docs)
        lines.append(
            f"  {pid} files={len(docs)}  ISR {isr.found_count}/{isr.expected_count} {state}"
        )
        lines.append("      " + ", ".join(f"{t} {n}" for t, n in sorted(types.items())))

    counts = report.counts()
    lines += [
        "",
        f"Data-quality observations ({counts['ERROR']} errors, {counts['WARNING']} warnings, "
        f"{counts['INFO']} info):",
    ]
    for o in report.observations:
        pid = o.project_id or "-"
        lines.append(f"  [{o.severity.value:<7}] {o.check.value:<38} {pid:<8} {o.message}")
    return "\n".join(lines)


def to_json_dict(
    result: BronzeIngestionResult, report: DataQualityReport, registry: ProjectRegistry
) -> dict[str, Any]:
    files = result.source_files
    tables = {
        t.name: {
            "rows": len(t),
            "rows_by_project": t.count_by_project(),
            "metadata": t.metadata.model_dump(),
        }
        for t in result.all_tables()
    }
    return {
        "run_id": result.run_id,
        "ingested_at": result.ingested_at,
        "sources": {
            "projects_workbook": files.relative(files.projects_workbook),
            "loans_snapshot": files.relative(files.loans_snapshot),
            "procurement_contract_awards": files.relative(files.procurement_contract_awards),
            "documents_root": str(files.documents_root),
        },
        "projects": registry.project_ids,
        "tables": tables,
        "loans": [
            {
                k: r[k]
                for k in (
                    "project_id",
                    "raw_loan_number",
                    "normalized_loan_number",
                    "loan_status",
                    "end_of_period",
                )
            }
            for r in result.tables["bronze_loans_raw"].records
        ],  # fmt: skip
        "procurement_coverage": [c.model_dump(mode="json") for c in result.procurement_coverage],
        "isr_completeness": {
            k: v.model_dump(mode="json") for k, v in result.isr_completeness.items()
        },
        "data_quality": {
            "counts": report.counts(),
            "has_errors": report.has_errors,
            "observations": [o.model_dump(mode="json") for o in report.observations],
        },
        "severity_levels": [s.value for s in Severity],
    }
