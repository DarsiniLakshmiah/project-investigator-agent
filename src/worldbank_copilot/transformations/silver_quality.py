"""Silver data-quality checks.

These validate the Silver model itself (grain, integrity, provenance, semantics).
Source disagreements already reported on Bronze (e.g. the P130544 commitment
difference) stay in the Bronze report and are visible side by side in
``silver_project_financial_summary``; they are not re-reported here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal

from worldbank_copilot.common.project_registry import ProjectRegistry
from worldbank_copilot.common.quality import (
    CheckCode,
    DataQualityReport,
    Observation,
    Severity,
    sort_observations,
)
from worldbank_copilot.ingestion.procurement import ProcurementCoverageStatus
from worldbank_copilot.transformations.silver import SilverResult
from worldbank_copilot.transformations.silver_models import (
    AwardAmountBasis,
    IssueKind,
    SilverLoan,
    ValueIssue,
)

_NON_NEGATIVE_LOAN_FIELDS = (
    "original_principal_usd",
    "cancelled_amount_usd",
    "disbursed_amount_usd",
    "undisbursed_amount_usd",
)
# Lifecycle dates that must not decrease. Last disbursement is excluded: it can
# follow the closing date (grace period), e.g. IBRD86010.
_LOAN_DATE_ORDER = (
    "board_approval_date",
    "agreement_signing_date",
    "effective_date",
    "closing_date",
)
# Fields no structured source provides; listed separately from genuine gaps.
KNOWN_UNAVAILABLE_PROJECT_FIELDS = ("state", "original_closing_date")

ISSUE_CHECK = {
    IssueKind.MALFORMED: CheckCode.SOURCE_VALUE_NOT_NORMALIZABLE,
    IssueKind.STRUCTURE: CheckCode.SOURCE_STRUCTURE_ISSUE,
    IssueKind.CONFLICT: CheckCode.SOURCE_VALUE_CONFLICT,
}


def _obs(check, severity, source, message, **kw) -> Observation:
    return Observation(check=check, severity=severity, source=source, message=message, **kw)


def check_projects(silver: SilverResult, registry: ProjectRegistry) -> list[Observation]:
    out = []
    projects = silver["silver_projects"].rows
    ids = [p.project_id for p in projects]
    if sorted(ids) != sorted(registry.project_ids):
        out.append(_obs(CheckCode.SILVER_PROJECT_COUNT_MISMATCH, Severity.ERROR,
                        "silver_projects",
                        f"silver_projects has {len(ids)} rows {ids}; expected exactly "
                        f"{registry.project_ids}"))  # fmt: skip
    for project in projects:
        missing = [
            name
            for name, value in project.model_dump(exclude={"source_refs", "quality_issues"}).items()
            if value in (None, [])
        ]
        gaps = [m for m in missing if m not in KNOWN_UNAVAILABLE_PROJECT_FIELDS]
        unavailable = [m for m in missing if m in KNOWN_UNAVAILABLE_PROJECT_FIELDS]
        out.append(_obs(
            CheckCode.PROJECT_METADATA_INCOMPLETE, Severity.INFO, "silver_projects",
            f"{project.project_id}: NULL fields {gaps or 'none'}; not available from structured "
            f"sources: {unavailable}",
            project_id=project.project_id,
            details={"null_fields": gaps, "unavailable_fields": unavailable},
        ))  # fmt: skip
        if project.country_code_source:
            out.append(_obs(
                CheckCode.FIELD_FROM_SECONDARY_SOURCE, Severity.INFO, "silver_projects",
                f"{project.project_id}: country_code {project.country_code!r} taken from "
                f"{project.country_code_source} (the workbook has no country code column)",
                project_id=project.project_id,
                details={"field": "country_code", "source": project.country_code_source},
            ))  # fmt: skip
    return out


def check_loans(silver: SilverResult, registry: ProjectRegistry) -> list[Observation]:
    out = []
    loans: list[SilverLoan] = silver["silver_loans"].rows
    project_ids = {p.project_id for p in silver["silver_projects"].rows}

    counts = Counter(loan.raw_loan_number for loan in loans)
    for number, n in counts.items():
        if n > 1:
            out.append(
                _obs(
                    CheckCode.SILVER_DUPLICATE_LOAN,
                    Severity.ERROR,
                    "silver_loans",
                    f"loan {number} appears {n} times",
                    details={"loan": number},
                )
            )
    for loan in loans:
        if loan.project_id not in project_ids:
            out.append(_obs(CheckCode.LOAN_PROJECT_REFERENCE_BROKEN, Severity.ERROR,
                            "silver_loans",
                            f"loan {loan.raw_loan_number} references project {loan.project_id} "
                            "which is not in silver_projects",
                            project_id=loan.project_id))  # fmt: skip

    for project in registry.projects:
        if project.expected_loan_numbers is None:
            continue
        actual = {loan.raw_loan_number for loan in loans if loan.project_id == project.project_id}
        expected = set(project.expected_loan_numbers)
        for number in sorted(expected - actual):
            out.append(_obs(CheckCode.SILVER_EXPECTED_LOAN_MISSING, Severity.ERROR, "silver_loans",
                            f"{project.project_id}: expected loan {number} not found",
                            project_id=project.project_id))  # fmt: skip
        for number in sorted(actual - expected):
            out.append(_obs(CheckCode.SILVER_UNEXPECTED_LOAN, Severity.WARNING, "silver_loans",
                            f"{project.project_id}: loan {number} is not in "
                            "expected_loan_numbers (new loan or configuration out of date)",
                            project_id=project.project_id))  # fmt: skip

    for loan in loans:
        pid, number = loan.project_id, loan.raw_loan_number
        for name in _NON_NEGATIVE_LOAN_FIELDS:
            value = getattr(loan, name)
            if value is not None and value < 0:
                out.append(_obs(CheckCode.NEGATIVE_FINANCIAL_VALUE, Severity.ERROR,
                                "silver_loans", f"{number}: {name} is negative ({value})",
                                project_id=pid, details={"field": name,
                                                         "value": str(value)}))  # fmt: skip
        out.extend(_reconciliation(loan))
        out.extend(_date_order(loan))
        if loan.valuation_caveats:
            out.append(_obs(CheckCode.LOAN_VALUATION_CAVEAT, Severity.INFO, "silver_loans",
                            f"{number}: " + " ".join(loan.valuation_caveats),
                            project_id=pid, details={"loan": number}))  # fmt: skip
    return out


def _reconciliation(loan: SilverLoan) -> list[Observation]:
    difference = loan.principal_components_difference_usd
    if difference is None or difference == 0:
        return []
    exchange = loan.exchange_adjustment_usd or Decimal(0)
    fx_signal = exchange != 0
    return [_obs(
        CheckCode.LOAN_PRINCIPAL_COMPONENTS_DIFFERENCE,
        Severity.INFO if fx_signal else Severity.WARNING,
        "silver_loans",
        f"{loan.raw_loan_number}: original principal {loan.original_principal_usd} vs "
        f"disbursed + undisbursed + cancelled {loan.principal_components_total_usd} "
        f"(difference {difference})",
        project_id=loan.project_id,
        explanation=(
            "The identity original = disbursed + undisbursed + cancelled is not assumed: it "
            "does not hold for a material share of loans in the snapshot "
            "(IMPLEMENTATION_PLAN.md, Phase 3). "
            + ("This loan has a non-zero exchange adjustment, consistent with (but not proof "
               "of) US$ revaluation of a non-US$ loan." if fx_signal else
               "No exchange adjustment is reported, so the source gives no explanation.")
        ),
        details={"loan": loan.raw_loan_number, "difference_usd": str(difference),
                 "exchange_adjustment_usd": str(exchange)},
    )]  # fmt: skip


def _date_order(loan: SilverLoan) -> list[Observation]:
    out = []
    dated = [(name, getattr(loan, name)) for name in _LOAN_DATE_ORDER]
    dated = [(n, d) for n, d in dated if d is not None]
    for (a, da), (b, db) in zip(dated, dated[1:], strict=False):
        if da > db:
            out.append(_obs(CheckCode.LOAN_DATE_ORDER_VIOLATION, Severity.WARNING, "silver_loans",
                            f"{loan.raw_loan_number}: {a} {da} is after {b} {db}",
                            project_id=loan.project_id,
                            details={"earlier_field": a, "later_field": b}))  # fmt: skip
    return out


def check_financial_summary(silver: SilverResult) -> list[Observation]:
    out = []
    loans_by_project = Counter(loan.project_id for loan in silver["silver_loans"].rows)
    summaries = silver["silver_project_financial_summary"].rows
    covered = Counter(s.project_id for s in summaries)
    for pid, n in loans_by_project.items():
        loan_total = sum(s.loan_count for s in summaries if s.project_id == pid)
        if covered[pid] == 0 or loan_total != n:
            out.append(
                _obs(
                    CheckCode.FINANCIAL_SUMMARY_MISMATCH,
                    Severity.ERROR,
                    "silver_project_financial_summary",
                    f"{pid}: summaries cover {loan_total} of {n} loans",
                    project_id=pid,
                )
            )
    return out


def check_procurement(silver: SilverResult) -> list[Observation]:
    out = []
    awards = silver["silver_procurement_awards"].rows
    suppliers = silver["silver_procurement_suppliers"].rows
    keys = Counter((a.project_id, a.contract_id) for a in awards)
    for key, n in keys.items():
        if n > 1:
            out.append(_obs(CheckCode.PROCUREMENT_GRAIN_VIOLATION, Severity.ERROR,
                            "silver_procurement_awards", f"award {key} appears {n} times",
                            project_id=key[0]))  # fmt: skip
    per_award = Counter((s.project_id, s.contract_id) for s in suppliers)
    for award in awards:
        if per_award[(award.project_id, award.contract_id)] != award.supplier_count:
            out.append(_obs(CheckCode.PROCUREMENT_GRAIN_VIOLATION, Severity.ERROR,
                            "silver_procurement_suppliers",
                            f"award {award.contract_id}: supplier_count {award.supplier_count} "
                            f"but {per_award[(award.project_id, award.contract_id)]} supplier "
                            "rows", project_id=award.project_id))  # fmt: skip
        if award.amount_basis is AwardAmountBasis.MULTI_SUPPLIER_UNRESOLVED:
            out.append(_obs(
                CheckCode.PROCUREMENT_AWARD_AMOUNT_UNRESOLVED, Severity.WARNING,
                "silver_procurement_awards",
                f"award {award.contract_id}: {award.supplier_count} suppliers "
                f"({', '.join(award.supplier_names)}); award amount between "
                f"{award.amount_lower_bound_usd} and {award.amount_upper_bound_usd} US$",
                project_id=award.project_id,
                explanation=(
                    "The dataset does not state whether a supplier row's amount is the whole "
                    "award (repeated per supplier, e.g. a joint venture) or that supplier's "
                    "share. contract_amount_usd is left NULL; totals must use the bounds."
                ),
                details={"contract_id": award.contract_id,
                         "lower_bound_usd": str(award.amount_lower_bound_usd),
                         "upper_bound_usd": str(award.amount_upper_bound_usd)},
            ))  # fmt: skip

    for row in silver["silver_procurement_coverage"].rows:
        not_covered = row.coverage_status == ProcurementCoverageStatus.NOT_COVERED_BY_THIS_DATASET
        has_rows = any(a.project_id == row.project_id for a in awards)
        if not_covered and (row.award_count is not None or has_rows):
            out.append(_obs(CheckCode.PROCUREMENT_COVERAGE_SEMANTICS_VIOLATION, Severity.ERROR,
                            "silver_procurement_coverage",
                            f"{row.project_id}: NOT_COVERED_BY_THIS_DATASET must have "
                            "award_count NULL and no award rows",
                            project_id=row.project_id))  # fmt: skip
    return out


def check_provenance_and_issues(silver: SilverResult) -> list[Observation]:
    out = []
    for name, table in silver.tables.items():
        for row in table.rows:
            key = getattr(row, "raw_loan_number", None) or getattr(row, "contract_id", None)
            pid = getattr(row, "project_id", None)
            if not row.source_refs:
                out.append(
                    _obs(
                        CheckCode.MISSING_PROVENANCE,
                        Severity.ERROR,
                        name,
                        f"row without source_refs ({pid}, {key})",
                        project_id=pid,
                    )
                )
            for issue in row.quality_issues:
                out.append(_issue_observation(name, issue, pid))
    for issue in silver.table_issues:
        out.append(_issue_observation("silver", issue, issue.project_id))
    return out


def _issue_observation(table: str, issue: ValueIssue, pid: str | None) -> Observation:
    ref = issue.source_ref
    return _obs(
        ISSUE_CHECK[issue.kind],
        Severity.WARNING,
        table,
        f"{issue.field}: {issue.reason}",
        project_id=pid,
        details={
            "field": issue.field,
            "raw_value": issue.raw_value,
            "source_ref": ref.model_dump() if ref else None,
        },
    )


def build_silver_quality_report(
    silver: SilverResult,
    registry: ProjectRegistry,
    *,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> DataQualityReport:
    observations = [
        *check_projects(silver, registry),
        *check_loans(silver, registry),
        *check_financial_summary(silver),
        *check_procurement(silver),
        *check_provenance_and_issues(silver),
    ]
    return DataQualityReport(
        generated_at=now().isoformat(),
        ingestion_run_id=silver.run_id,
        observations=sort_observations(observations),
        layer="silver",
    )
