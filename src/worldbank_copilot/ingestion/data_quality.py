"""Structured data-quality observations over a Bronze ingestion run.

Observations describe what the sources contain; they never repair or choose
between sources. Where sources disagree (e.g. workbook commitment vs loan
snapshot), both values are reported and neither is marked wrong, following the
source-of-truth policy in IMPLEMENTATION_PLAN.md §1a.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

from worldbank_copilot.common.dates import DateParseError, parse_date
from worldbank_copilot.common.identifiers import LoanLinkStatus, link_loan_reference
from worldbank_copilot.common.project_registry import ProjectRegistry
from worldbank_copilot.common.quality import (  # re-exported for existing callers
    CheckCode,
    DataQualityReport,
    Observation,
    Severity,
    sort_observations,
)
from worldbank_copilot.ingestion.contracts import (
    ALL_TABULAR_CONTRACTS,
    FieldKind,
    SourceContract,
)
from worldbank_copilot.ingestion.documents import ClassificationMethod, ManifestStatus
from worldbank_copilot.ingestion.loans import loans_by_project
from worldbank_copilot.transformations.bronze import BronzeTable

if TYPE_CHECKING:
    from worldbank_copilot.ingestion.pipeline import BronzeIngestionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONTRACTS_BY_TABLE = {c.bronze_table: c for c in ALL_TABULAR_CONTRACTS}
_HTML_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9]*(\s[^<>]*)?/?>")


def parse_amount(raw: str | None) -> Decimal | None:
    """Parse a source amount; ``None`` if blank. Raises ``InvalidOperation`` if malformed."""
    if raw is None or not raw.strip():
        return None
    value = Decimal(raw.strip())
    if not value.is_finite():
        raise InvalidOperation(raw)
    return value


def _fmt(value: Decimal) -> str:
    """Thousands-separated display without a spurious trailing '.0'."""
    return f"{value:,f}" if value != value.to_integral_value() else f"{int(value):,}"


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _obs(check: CheckCode, severity: Severity, source: str, message: str, **kw) -> Observation:
    return Observation(check=check, severity=severity, source=source, message=message, **kw)


# ---------------------------------------------------------------------------
# Checks over tabular Bronze tables (contract-driven)
# ---------------------------------------------------------------------------


def check_table_values(table: BronzeTable, contract: SourceContract) -> list[Observation]:
    """Required identifiers/values, date and amount parsing, markup, blank columns."""
    out: list[Observation] = []
    name = table.name
    for spec in contract.columns:
        for record in table.records:
            value = record.get(spec.field)
            pid = record["project_id"]
            where = {"field": spec.field, "source_row": record["_source_row"], "raw_value": value}
            if spec.required and _blank(value):
                if spec.kind is FieldKind.IDENTIFIER:
                    out.append(_obs(CheckCode.MISSING_REQUIRED_IDENTIFIER, Severity.ERROR, name,
                                    f"Required identifier {spec.field!r} is blank",
                                    project_id=pid, details=where))  # fmt: skip
                else:
                    out.append(_obs(CheckCode.MISSING_REQUIRED_VALUE, Severity.WARNING, name,
                                    f"Required field {spec.field!r} is blank",
                                    project_id=pid, details=where))  # fmt: skip
                continue
            if spec.kind is FieldKind.DATE:
                try:
                    parse_date(value, spec.date_formats)
                except DateParseError:
                    out.append(_obs(CheckCode.UNPARSEABLE_DATE, Severity.WARNING, name,
                                    f"{spec.field} value {value!r} matches none of the declared "
                                    f"formats {list(spec.date_formats)}",
                                    project_id=pid, details=where))  # fmt: skip
            elif spec.kind is FieldKind.AMOUNT:
                try:
                    parse_amount(value)
                except InvalidOperation:
                    out.append(_obs(CheckCode.UNPARSEABLE_AMOUNT, Severity.WARNING, name,
                                    f"{spec.field} value {value!r} is not a number",
                                    project_id=pid, details=where))  # fmt: skip

    for field_name in contract.rich_text_fields:
        for record in table.records:
            value = record.get(field_name)
            if isinstance(value, str) and _HTML_TAG.search(value):
                out.append(_obs(
                    CheckCode.HTML_IN_TEXT_FIELD, Severity.INFO, name,
                    f"{field_name} contains HTML markup; preserved as-is in Bronze",
                    project_id=record["project_id"],
                    explanation="Markup removal belongs to Silver normalisation.",
                    details={"field": field_name, "source_row": record["_source_row"]},
                ))  # fmt: skip

    if table.records:
        for spec in contract.columns:
            if spec.field in table.metadata.missing_optional_columns:
                continue
            if all(_blank(r.get(spec.field)) for r in table.records):
                out.append(_obs(
                    CheckCode.BLANK_SOURCE_COLUMN, Severity.INFO, name,
                    f"{contract.name}: column {spec.field!r} is blank for every ingested row "
                    f"({len(table.records)} rows)",
                    explanation="The source provides no value; nothing is inferred in its place.",
                    details={"field": spec.field, "rows": len(table.records)},
                ))  # fmt: skip

    for gap in table.metadata.known_gaps:
        out.append(_obs(
            CheckCode.SOURCE_CONCEPT_NOT_PROVIDED, Severity.INFO, name,
            f"{contract.name} has no {gap!r} field",
            explanation="Documented contract gap; must come from another source (e.g. project "
                        "documents) or remain unknown.",
            details={"concept": gap},
        ))  # fmt: skip

    if table.metadata.rows_without_project_id:
        out.append(_obs(
            CheckCode.SOURCE_ROWS_WITHOUT_PROJECT_ID, Severity.INFO, name,
            f"{table.metadata.rows_without_project_id} source rows have no project ID "
            "and were not ingested",
            details={"rows": table.metadata.rows_without_project_id},
        ))  # fmt: skip
    return out


def check_project_presence(
    tables: dict[str, BronzeTable], registry: ProjectRegistry
) -> list[Observation]:
    out: list[Observation] = []
    expectations = {
        "bronze_projects_raw": Severity.ERROR,
        "bronze_loans_raw": Severity.WARNING,
        "bronze_themes_raw": Severity.INFO,
        "bronze_sectors_raw": Severity.INFO,
        "bronze_financers_raw": Severity.INFO,
        "bronze_geo_locations_raw": Severity.INFO,
    }
    for table_name, severity in expectations.items():
        table = tables.get(table_name)
        if table is None:
            continue
        counts = table.count_by_project()
        for pid in registry.project_ids:
            n = counts.get(pid, 0)
            if n == 0 and table_name == "bronze_projects_raw":
                out.append(
                    _obs(
                        CheckCode.REQUIRED_PROJECT_MISSING,
                        Severity.ERROR,
                        table_name,
                        f"{pid} not found in the Projects sheet",
                        project_id=pid,
                    )
                )
            elif n == 0:
                out.append(_obs(CheckCode.PROJECT_MISSING_FROM_SOURCE, severity, table_name,
                                f"{pid} has no rows in {table.metadata.source_name}",
                                project_id=pid))  # fmt: skip
            elif n > 1 and table_name == "bronze_projects_raw":
                rows = [r["_source_row"] for r in table.for_project(pid)]
                out.append(_obs(CheckCode.DUPLICATE_PROJECT_RECORD, Severity.ERROR, table_name,
                                f"{pid} appears {n} times in the Projects sheet",
                                project_id=pid, details={"source_rows": rows}))  # fmt: skip
    return out


def check_registry_metadata(projects: BronzeTable, registry: ProjectRegistry) -> list[Observation]:
    out: list[Observation] = []
    for project in registry.projects:
        rows = projects.for_project(project.project_id)
        if len(rows) != 1:
            continue  # reported by check_project_presence
        row = rows[0]
        for config_field, source_field in (
            ("name", "project_name"),
            ("instrument", "lending_instrument"),
        ):
            configured = getattr(project, config_field)
            source_value = row.get(source_field)
            if (source_value or "").strip() != configured:
                out.append(_obs(
                    CheckCode.REGISTRY_METADATA_DIFFERENCE, Severity.WARNING, projects.name,
                    f"configs/projects.yaml {config_field} differs from the workbook",
                    project_id=project.project_id,
                    details={"registry_value": configured, "workbook_value": source_value},
                ))  # fmt: skip
    return out


def check_loans(loans: BronzeTable) -> list[Observation]:
    out: list[Observation] = []
    by_project = loans_by_project(loans)
    for pid, numbers in by_project.items():
        if len(numbers) > 1:
            out.append(_obs(
                CheckCode.MULTIPLE_LOANS_PER_PROJECT, Severity.INFO, loans.name,
                f"{pid} has {len(numbers)} loans: {', '.join(numbers)}",
                project_id=pid,
                explanation="Project-to-loan is one-to-many; loans are kept as separate records.",
                details={"raw_loan_numbers": numbers},
            ))  # fmt: skip
    counts = Counter(
        r["raw_loan_number"] for r in loans.records if not _blank(r["raw_loan_number"])
    )
    for number, n in counts.items():
        if n > 1:
            out.append(_obs(CheckCode.DUPLICATE_LOAN_IDENTIFIER, Severity.ERROR, loans.name,
                            f"Loan number {number} appears {n} times",
                            details={"raw_loan_number": number, "count": n}))  # fmt: skip
    for record in loans.records:
        if not _blank(record["raw_loan_number"]) and record.get("normalized_loan_number") is None:
            out.append(_obs(CheckCode.UNRECOGNIZED_LOAN_NUMBER, Severity.WARNING, loans.name,
                            f"Loan number {record['raw_loan_number']!r} has an unrecognised "
                            "notation; normalized_loan_number left empty",
                            project_id=record["project_id"],
                            details={"raw_loan_number": record["raw_loan_number"]}))  # fmt: skip
    periods = sorted({r["end_of_period"] for r in loans.records if not _blank(r["end_of_period"])})
    if len(periods) == 1:
        out.append(_obs(CheckCode.LOAN_SNAPSHOT_DATE, Severity.INFO, loans.name,
                        f"Loan snapshot End of Period is {periods[0]}",
                        explanation="End of Period is the snapshot date; the date in the file "
                                    "name is the download date.",
                        details={"end_of_period": periods[0],
                                 "source_file": loans.metadata.source_file}))  # fmt: skip
    elif len(periods) > 1:
        out.append(_obs(CheckCode.SNAPSHOT_DATE_INCONSISTENT, Severity.WARNING, loans.name,
                        f"Ingested loans carry {len(periods)} different End of Period values",
                        details={"end_of_period_values": periods}))  # fmt: skip
    return out


def check_procurement(procurement: BronzeTable, coverage: Iterable[Any]) -> list[Observation]:
    out: list[Observation] = []
    by_contract: dict[tuple[str, str], list[int]] = defaultdict(list)
    for record in procurement.records:
        number = record["wb_contract_number"]
        if not _blank(number):
            by_contract[(record["project_id"], number)].append(record["_source_row"])
    for (pid, number), rows in by_contract.items():
        if len(rows) > 1:
            out.append(_obs(
                CheckCode.DUPLICATE_CONTRACT_IDENTIFIER, Severity.WARNING, procurement.name,
                f"WB contract number {number} appears on {len(rows)} rows",
                project_id=pid,
                explanation="May be legitimate (e.g. several suppliers on one contract, each row "
                            "repeating the contract amount). Rows are preserved; Silver must "
                            "reconcile them before summing amounts to avoid double counting.",
                details={"wb_contract_number": number, "source_rows": rows},
            ))  # fmt: skip
    for item in coverage:
        if not item.covered_by_dataset and item.row_count > 0:
            out.append(_obs(
                CheckCode.PROCUREMENT_RECORDS_FOR_UNCOVERED_PROJECT, Severity.WARNING,
                procurement.name,
                f"{item.project_id} is configured as outside dataset coverage but has "
                f"{item.row_count} rows",
                project_id=item.project_id,
                explanation="Review procurement_coverage in configs/projects.yaml.",
            ))  # fmt: skip
    return out


def check_commitment_sources(
    projects: BronzeTable,
    financers: BronzeTable | None,
    loans: BronzeTable,
    registry: ProjectRegistry,
) -> list[Observation]:
    """Compare workbook IBRD commitment with the sum of loan original principals."""
    out: list[Observation] = []
    explanation = (
        "The source values represent different snapshots and/or coverage (e.g. the workbook "
        "may not reflect additional-financing loans). Neither is marked wrong; the "
        "source-of-truth policy (loan snapshot for loan-level finance, workbook for project "
        "metadata) is applied downstream."
    )
    for pid in registry.project_ids:
        project_loans = loans.for_project(pid)
        if not project_loans:
            continue
        try:
            principals = [parse_amount(r["original_principal_amount_usd"]) for r in project_loans]
        except InvalidOperation:
            continue  # reported by the amount check
        if any(p is None for p in principals):
            continue
        loan_total = sum(principals, Decimal(0))
        loan_detail = [
            {"raw_loan_number": r["raw_loan_number"],
             "original_principal_amount_usd": r["original_principal_amount_usd"]}
            for r in project_loans
        ]  # fmt: skip

        candidates: list[tuple[str, str, dict]] = []
        workbook_rows = projects.for_project(pid)
        if len(workbook_rows) == 1:
            candidates.append(("World Bank Projects", "IBRD Commitment", workbook_rows[0]))
        if financers is not None:
            for row in financers.for_project(pid):
                if (row.get("financer_id") or "").strip() == "IBRD":
                    candidates.append(("Financers", "Amount (USD) [Financer ID = IBRD]", row))

        for sheet, label, row in candidates:
            raw = row["ibrd_commitment"] if sheet == "World Bank Projects" else row["amount_usd"]
            try:
                workbook_value = parse_amount(raw)
            except InvalidOperation:
                continue
            if workbook_value is None or workbook_value == loan_total:
                continue
            out.append(_obs(
                CheckCode.PROJECT_COMMITMENT_SOURCE_DIFFERENCE, Severity.WARNING,
                f"{projects.name} vs {loans.name}",
                f"{pid}: workbook {sheet} {label} = {_fmt(workbook_value)} but IBRD loan "
                f"snapshot original principal total = {_fmt(loan_total)} across "
                f"{len(project_loans)} loan(s)",
                project_id=pid,
                explanation=explanation,
                details={
                    "workbook_sheet": sheet,
                    "workbook_field": label,
                    "workbook_value": raw,
                    "workbook_source_row": row["_source_row"],
                    "loan_snapshot_total": str(loan_total),
                    "loan_snapshot_field": "Original Principal Amount (US$)",
                    "loans": loan_detail,
                    "difference": str(loan_total - workbook_value),
                },
            ))  # fmt: skip
    return out


def check_financer_amounts(financers: BronzeTable) -> list[Observation]:
    out: list[Observation] = []
    for row in financers.records:
        try:
            current, usd = parse_amount(row["current_amount"]), parse_amount(row["amount_usd"])
        except InvalidOperation:
            continue
        if current is None or usd is None or current == usd:
            continue
        out.append(_obs(
            CheckCode.FINANCER_AMOUNT_FIELDS_DIFFER, Severity.WARNING, financers.name,
            f"{row['project_id']} financer {row['financer_id']!r}: Current Amount "
            f"{_fmt(current)} differs from Amount (USD) {_fmt(usd)} "
            f"(Currency {row['currency']!r})",
            project_id=row["project_id"],
            explanation="Both values are preserved; their relationship is not documented in the "
                        "source and is not resolved in Bronze.",
            details={"financer_id": row["financer_id"], "current_amount": row["current_amount"],
                     "amount_usd": row["amount_usd"], "currency": row["currency"],
                     "source_row": row["_source_row"]},
        ))  # fmt: skip
    return out


def check_blank_project_fields(projects: BronzeTable) -> list[Observation]:
    out = []
    for row in projects.records:
        blanks = [
            spec.field
            for spec in _CONTRACTS_BY_TABLE[projects.name].columns
            if not spec.required and _blank(row.get(spec.field))
        ]
        if blanks:
            out.append(_obs(CheckCode.BLANK_PROJECT_FIELDS, Severity.INFO, projects.name,
                            f"{row['project_id']}: optional fields blank in the workbook: "
                            f"{', '.join(blanks)}",
                            project_id=row["project_id"], details={"fields": blanks}))  # fmt: skip
    return out


# ---------------------------------------------------------------------------
# Document checks
# ---------------------------------------------------------------------------


def check_documents(result: BronzeIngestionResult) -> list[Observation]:
    out: list[Observation] = []
    inventory = result.document_inventory
    table = inventory.table
    name = table.name
    for pid in inventory.missing_project_dirs:
        out.append(
            _obs(
                CheckCode.DOCUMENT_DIRECTORY_MISSING,
                Severity.ERROR,
                name,
                f"Document directory for {pid} not found",
                project_id=pid,
            )
        )
    for missing in inventory.missing_manifest_files:
        out.append(_obs(CheckCode.MANIFEST_FILE_MISSING, Severity.WARNING, name,
                        f"Manifest lists {missing['filename']} but the file is not present",
                        project_id=missing["project_id"], details=missing))  # fmt: skip

    for r in table.records:
        pid, fname = r["project_id"], r["filename"]
        ref = {"filename": fname, "relative_path": r["relative_path"]}
        if r["manifest_status"] == ManifestStatus.SIZE_MISMATCH.value:
            out.append(_obs(CheckCode.MANIFEST_FILE_CHANGED, Severity.WARNING, name,
                            f"{fname}: size differs from the manifest; manifest labels ignored",
                            project_id=pid, details=ref))  # fmt: skip
        if r["classification_method"] == ClassificationMethod.UNCLASSIFIED.value:
            out.append(_obs(CheckCode.UNCLASSIFIED_DOCUMENT, Severity.WARNING, name,
                            f"{fname}: no filename rule or manifest entry; typed OTHER",
                            project_id=pid, details=ref))  # fmt: skip
        if r["classification_conflicts"]:
            conflict_details = {**ref, "conflicts": r["classification_conflicts"]}
            out.append(_obs(CheckCode.CLASSIFICATION_CONFLICT, Severity.WARNING, name,
                            f"{fname}: filename and manifest disagree; conflicting values left "
                            "empty",
                            project_id=pid, details=conflict_details))  # fmt: skip
        others = [p for p in r["filename_project_ids"] if p != pid]
        if others:
            out.append(_obs(CheckCode.FILENAME_PROJECT_MISMATCH, Severity.WARNING, name,
                            f"{fname} is in {pid}'s folder but its name mentions {others}",
                            project_id=pid, details=ref))  # fmt: skip

    by_hash: dict[str, list[str]] = defaultdict(list)
    for r in table.records:
        by_hash[r["sha256"]].append(r["relative_path"])
    for digest, paths in by_hash.items():
        if len(paths) > 1:
            out.append(_obs(CheckCode.DUPLICATE_DOCUMENT_CONTENT, Severity.WARNING, name,
                            f"{len(paths)} files have identical content",
                            details={"sha256": digest, "relative_paths": paths}))  # fmt: skip

    for pid, isr in result.isr_completeness.items():
        if (
            isr.missing
            or isr.isr_without_sequence
            or (isr.expected_count is not None and isr.found_count != isr.expected_count)
        ):
            out.append(_obs(
                CheckCode.ISR_SEQUENCE_INCOMPLETE, Severity.WARNING, name,
                f"{pid}: ISR sequences found {isr.found_count}/{isr.expected_count}; "
                f"missing {isr.missing}; {isr.isr_without_sequence} ISR(s) without a sequence",
                project_id=pid, details=isr.model_dump(),
            ))  # fmt: skip
        if isr.duplicates:
            out.append(_obs(CheckCode.ISR_SEQUENCE_DUPLICATE, Severity.WARNING, name,
                            f"{pid}: duplicate ISR sequence numbers {isr.duplicates}",
                            project_id=pid, details={"duplicates": isr.duplicates}))  # fmt: skip
    return out


def check_document_loan_references(inventory: BronzeTable, loans: BronzeTable) -> list[Observation]:
    out: list[Observation] = []
    project_loans = loans_by_project(loans)
    for r in inventory.records:
        if not r["raw_loan_number"]:
            continue
        link = link_loan_reference(r["raw_loan_number"], project_loans.get(r["project_id"], []))
        details = {"filename": r["filename"], "reference_raw": link.reference_raw,
                   "matched_raw_loan_number": link.matched_raw, "status": link.status.value,
                   "candidates": list(link.candidates)}  # fmt: skip
        if link.status is LoanLinkStatus.LINKED:
            out.append(_obs(CheckCode.DOCUMENT_LOAN_REFERENCE_LINKED, Severity.INFO,
                            f"{inventory.name} vs {loans.name}",
                            f"{r['filename']}: loan reference {link.reference_raw} links to "
                            f"{link.matched_raw}",
                            project_id=r["project_id"], details=details))  # fmt: skip
        else:
            out.append(_obs(CheckCode.DOCUMENT_LOAN_REFERENCE_UNLINKED, Severity.WARNING,
                            f"{inventory.name} vs {loans.name}",
                            f"{r['filename']}: loan reference {link.reference_raw} not linked "
                            f"({link.status.value})",
                            project_id=r["project_id"], details=details))  # fmt: skip
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_data_quality_report(
    result: BronzeIngestionResult,
    registry: ProjectRegistry,
    *,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> DataQualityReport:
    tables = result.tables
    observations: list[Observation] = []
    observations += check_project_presence(tables, registry)
    projects = tables["bronze_projects_raw"]
    loans = tables["bronze_loans_raw"]
    financers = tables.get("bronze_financers_raw")
    observations += check_registry_metadata(projects, registry)
    observations += check_blank_project_fields(projects)
    for table_name, contract in _CONTRACTS_BY_TABLE.items():
        if table_name in tables:
            observations += check_table_values(tables[table_name], contract)
    observations += check_loans(loans)
    observations += check_procurement(tables["bronze_procurement_raw"], result.procurement_coverage)
    observations += check_commitment_sources(projects, financers, loans, registry)
    if financers is not None:
        observations += check_financer_amounts(financers)
    observations += check_documents(result)
    observations += check_document_loan_references(result.document_inventory.table, loans)

    return DataQualityReport(
        generated_at=now().isoformat(),
        ingestion_run_id=result.run_id,
        observations=sort_observations(observations),
        layer="bronze",
    )
