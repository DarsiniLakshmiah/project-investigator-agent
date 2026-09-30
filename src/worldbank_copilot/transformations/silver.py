"""Silver structured transformations: BRONZE (what the source said) -> SILVER (what it means).

Bronze records are read, never modified. Every Silver row keeps ``source_refs``
back to its Bronze records (see ``silver_models`` for the provenance design and
``silver_lineage`` for field-level lineage).

Tables built here:

* ``silver_projects``: one row per project (workbook metadata).
* ``silver_project_sectors`` / ``silver_project_themes``: sector rows and the
  theme hierarchy, kept as their own tables so no distinct value is lost.
* ``silver_loans``: one row per loan (never collapsed to project level).
* ``silver_project_financial_summary``: one row per project per snapshot date,
  derived from ``silver_loans`` only.
* ``silver_procurement_awards`` / ``silver_procurement_suppliers``: one row per
  contract award / per contract-supplier relationship.
* ``silver_procurement_coverage``: dataset coverage per project.
"""

from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

from worldbank_copilot.common.identifiers import parse_loan_number
from worldbank_copilot.common.project_registry import ProjectRegistry
from worldbank_copilot.ingestion.contracts import LOANS, PROCUREMENT, PROJECTS, SourceContract
from worldbank_copilot.ingestion.procurement import ProcurementCoverageStatus
from worldbank_copilot.transformations.bronze import BronzeRecord, BronzeTable
from worldbank_copilot.transformations.normalize import (
    NormalizationError,
    clean_text,
    normalize_category,
    normalize_identifier,
    parse_decimal,
    parse_money,
    parse_source_date,
    parse_year,
    percentage,
    strip_html,
    taxonomy_label,
)
from worldbank_copilot.transformations.silver_lineage import lineage_records
from worldbank_copilot.transformations.silver_models import (
    AwardAmountBasis,
    IssueKind,
    SilverLoan,
    SilverProcurementAward,
    SilverProcurementCoverage,
    SilverProcurementSupplier,
    SilverProject,
    SilverProjectFinancialSummary,
    SilverProjectSector,
    SilverProjectTheme,
    SilverRow,
    SourceRef,
    ValueIssue,
)

SILVER_TABLES = (
    "silver_projects",
    "silver_project_sectors",
    "silver_project_themes",
    "silver_loans",
    "silver_project_financial_summary",
    "silver_procurement_awards",
    "silver_procurement_suppliers",
    "silver_procurement_coverage",
)


# ---------------------------------------------------------------------------
# Containers and writers
# ---------------------------------------------------------------------------


@dataclass
class SilverTable:
    name: str
    model: type[SilverRow]
    rows: list[Any]

    def __len__(self) -> int:
        return len(self.rows)

    def for_project(self, project_id: str) -> list[Any]:
        return [r for r in self.rows if getattr(r, "project_id", None) == project_id]


@dataclass
class SilverResult:
    run_id: str
    tables: dict[str, SilverTable]
    table_issues: list[ValueIssue] = field(default_factory=list)

    def __getitem__(self, name: str) -> SilverTable:
        return self.tables[name]


class SilverWriter(Protocol):
    def write(self, table: SilverTable) -> str: ...


class LocalJsonlSilverWriter:
    """Writes ``<table>.jsonl`` (Decimal as string, dates ISO) and ``<table>.schema.json``."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def write(self, table: SilverTable) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{table.name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for row in table.rows:
                fh.write(row.model_dump_json())
                fh.write("\n")
        schema_path = self.root / f"{table.name}.schema.json"
        schema_path.write_text(
            json.dumps(table.model.model_json_schema(), indent=2), encoding="utf-8"
        )
        return str(path)

    def write_lineage(self) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "silver_field_lineage.json"
        path.write_text(json.dumps(lineage_records(), indent=2), encoding="utf-8")
        return str(path)


def write_silver(result: SilverResult, writer: SilverWriter) -> dict[str, str]:
    return {name: writer.write(table) for name, table in result.tables.items()}


# ---------------------------------------------------------------------------
# Provenance + normalisation helpers
# ---------------------------------------------------------------------------


def source_ref(record: BronzeRecord, bronze_table: str, record_key: str | None = None) -> SourceRef:
    return SourceRef(
        bronze_table=bronze_table,
        source_file=record["_source_file"],
        source_sheet=record.get("_source_sheet"),
        source_row=record.get("_source_row"),
        ingestion_run_id=record["_ingestion_run_id"],
        record_key=record_key,
    )


class RowNormalizer:
    """Applies normalisers to one Bronze record, collecting failures as issues."""

    def __init__(self, record: BronzeRecord, ref: SourceRef):
        self.record = record
        self.ref = ref
        self.issues: list[ValueIssue] = []

    def __call__(self, field_name: str, fn: Callable[..., Any], *args: Any) -> Any:
        raw = self.record.get(field_name)
        try:
            return fn(raw, *args)
        except NormalizationError as exc:
            self.issues.append(
                ValueIssue(
                    field=field_name,
                    kind=IssueKind.MALFORMED,
                    raw_value=raw,
                    reason=str(exc),
                    source_ref=self.ref,
                    project_id=self.record.get("project_id"),
                )
            )
            return None

    def date(self, field_name: str, contract: SourceContract) -> Any:
        spec = next(c for c in contract.columns if c.field == field_name)
        return self(field_name, parse_source_date, spec.date_formats)


def _sum(values: Iterable[Decimal | None]) -> Decimal | None:
    values = list(values)
    if not values or any(v is None for v in values):
        return None
    return sum(values, Decimal(0))


def _sub(a: Decimal | None, b: Decimal | None) -> Decimal | None:
    return None if a is None or b is None else a - b


# ---------------------------------------------------------------------------
# Sectors and themes
# ---------------------------------------------------------------------------


def build_sectors(table: BronzeTable) -> list[SilverProjectSector]:
    rows = []
    for record in table.records:
        ref = source_ref(record, table.name)
        norm = RowNormalizer(record, ref)
        sector = norm("sector", normalize_category)
        rows.append(
            SilverProjectSector(
                project_id=record["project_id"],
                major_sector=norm("major_sector", normalize_category),
                sector=sector,
                sector_percent=norm("sector_percent", parse_decimal),
                taxonomy_label=taxonomy_label(sector),
                source_refs=[ref],
                quality_issues=norm.issues,
            )
        )
    return rows


def _structure_issue(field_name: str, raw: str | None, reason: str, ref: SourceRef, pid: str):
    return ValueIssue(
        field=field_name,
        kind=IssueKind.STRUCTURE,
        raw_value=raw,
        reason=reason,
        source_ref=ref,
        project_id=pid,
    )


def build_themes(table: BronzeTable) -> tuple[list[SilverProjectTheme], list[ValueIssue]]:
    """Theme rows -> distinct hierarchy nodes.

    Each source row describes one node: its deepest non-blank level. Ancestor
    levels repeat the parent name with a leading ``>`` marker, which is removed.
    """
    nodes: OrderedDict[tuple[str, tuple[str, ...]], SilverProjectTheme] = OrderedDict()
    orphan_issues: list[ValueIssue] = []
    for record in table.records:
        pid = record["project_id"]
        ref = source_ref(record, table.name)
        norm = RowNormalizer(record, ref)
        names = [clean_text(record.get(f"level_{i}")) for i in (1, 2, 3)]
        depth = max((i for i, n in enumerate(names, 1) if n), default=0)
        if depth == 0:
            orphan_issues.append(_structure_issue("level_1", None, "theme row has no name", ref,
                                                  pid))  # fmt: skip
            continue
        issues: list[ValueIssue] = []
        path: list[str] = []
        for level in range(1, depth + 1):
            name = names[level - 1]
            field_name = f"level_{level}"
            if name is None:
                issues.append(_structure_issue(field_name, None, "gap in theme hierarchy", ref,
                                               pid))  # fmt: skip
                break
            marked = name.startswith(">")
            if level < depth and not marked:
                issues.append(
                    _structure_issue(
                        field_name, name, "ancestor level lacks the '>' marker", ref, pid
                    )
                )
            if level == depth and marked:
                issues.append(
                    _structure_issue(
                        field_name, name, "deepest level carries a '>' marker", ref, pid
                    )
                )
            path.append(clean_text(name[1:]) if marked else name)
        if len(path) != depth:
            orphan_issues.extend(issues)
            continue
        pct = norm(f"percentage_{depth}", parse_decimal)
        issues.extend(norm.issues)
        key = (pid, tuple(path))
        existing = nodes.get(key)
        if existing is not None:
            existing.source_refs.append(ref)
            existing.quality_issues.extend(issues)
            if pct != existing.percentage:
                existing.quality_issues.append(ValueIssue(
                    field="percentage", kind=IssueKind.CONFLICT, raw_value=str(pct),
                    reason=f"duplicate theme node with percentage {pct} vs "
                           f"{existing.percentage}",
                    source_ref=ref, project_id=pid))  # fmt: skip
            continue
        nodes[key] = SilverProjectTheme(
            project_id=pid,
            level=depth,
            theme_name=path[-1],
            parent_theme_name=path[-2] if depth > 1 else None,
            level_1_theme=path[0],
            theme_path=path,
            percentage=pct,
            taxonomy_label=taxonomy_label(path[-1]),
            source_refs=[ref],
            quality_issues=issues,
        )
    for (pid, path), node in nodes.items():
        if len(path) > 1 and (pid, path[:-1]) not in nodes:
            node.quality_issues.append(_structure_issue(
                "theme_path", " > ".join(path), "parent theme node has no row of its own",
                node.source_refs[0], pid))  # fmt: skip
    return list(nodes.values()), orphan_issues


# ---------------------------------------------------------------------------
# Loans and financial summary
# ---------------------------------------------------------------------------


def build_loans(table: BronzeTable) -> list[SilverLoan]:
    rows = []
    for record in table.records:
        raw_number = record["raw_loan_number"]
        ref = source_ref(record, table.name, record_key=raw_number)
        norm = RowNormalizer(record, ref)
        parsed = parse_loan_number(raw_number)
        money = {
            name: norm(name, parse_money)
            for name in (
                "original_principal_amount_usd", "cancelled_amount_usd", "disbursed_amount_usd",
                "undisbursed_amount_usd", "repaid_to_ibrd_usd", "due_to_ibrd_usd",
                "exchange_adjustment_usd", "borrowers_obligation_usd", "loans_held_usd",
            )
        }  # fmt: skip
        original = money["original_principal_amount_usd"]
        components = _sum(
            [money["disbursed_amount_usd"], money["undisbursed_amount_usd"],
             money["cancelled_amount_usd"]]
        )  # fmt: skip
        difference = _sub(original, components)
        exchange = money["exchange_adjustment_usd"]
        # Valuation caveats come only from what the source signals (a non-zero exchange
        # adjustment). The principal-components difference has its own field.
        caveats = []
        if exchange is not None and exchange != 0:
            caveats.append(
                f"Exchange Adjustment (US$) is {exchange}: US$ values are affected by "
                "exchange-rate revaluation; the source does not state the currency, and "
                "no conversion is applied."
            )
        rows.append(
            SilverLoan(
                project_id=record["project_id"],
                raw_loan_number=raw_number,
                normalized_loan_number=record.get("normalized_loan_number"),
                loan_base_number=parsed.base_number if parsed else None,
                loan_suffix=parsed.suffix if parsed else None,
                lender=parsed.lender if parsed else None,
                loan_type=norm("loan_type", normalize_category),
                loan_status=norm("loan_status", normalize_category),
                borrower=norm("borrower", clean_text),
                country_code=norm("country_code", clean_text),
                currency_of_commitment=norm("currency_of_commitment", clean_text),
                original_principal_usd=original,
                cancelled_amount_usd=money["cancelled_amount_usd"],
                disbursed_amount_usd=money["disbursed_amount_usd"],
                undisbursed_amount_usd=money["undisbursed_amount_usd"],
                repaid_to_ibrd_usd=money["repaid_to_ibrd_usd"],
                due_to_ibrd_usd=money["due_to_ibrd_usd"],
                exchange_adjustment_usd=exchange,
                borrowers_obligation_usd=money["borrowers_obligation_usd"],
                loans_held_usd=money["loans_held_usd"],
                principal_components_total_usd=components,
                principal_components_difference_usd=difference,
                board_approval_date=norm.date("board_approval_date", LOANS),
                agreement_signing_date=norm.date("agreement_signing_date", LOANS),
                effective_date=norm.date("effective_date_most_recent", LOANS),
                closing_date=norm.date("closed_date_most_recent", LOANS),
                last_disbursement_date=norm.date("last_disbursement_date", LOANS),
                first_repayment_date=norm.date("first_repayment_date", LOANS),
                last_repayment_date=norm.date("last_repayment_date", LOANS),
                snapshot_date=norm.date("end_of_period", LOANS),
                valuation_caveats=caveats,
                source_file=record["_source_file"],
                source_refs=[ref],
                quality_issues=norm.issues,
            )
        )
    return rows


def build_financial_summaries(
    loans: list[SilverLoan], projects: list[SilverProject]
) -> list[SilverProjectFinancialSummary]:
    workbook = {p.project_id: p for p in projects}
    groups: OrderedDict[tuple, list[SilverLoan]] = OrderedDict()
    for loan in loans:
        groups.setdefault((loan.project_id, loan.snapshot_date), []).append(loan)

    rows = []
    for (pid, snapshot), group in groups.items():
        issues = []
        totals = {}
        for name, attr in (
            ("original_principal_total_usd", "original_principal_usd"),
            ("cancelled_total_usd", "cancelled_amount_usd"),
            ("disbursed_total_usd", "disbursed_amount_usd"),
            ("undisbursed_total_usd", "undisbursed_amount_usd"),
        ):
            values = [getattr(loan, attr) for loan in group]
            totals[name] = _sum(values)
            if totals[name] is None:
                reason = f"total not computed: {attr} is NULL for at least one loan"
                issues.append(ValueIssue(field=name, kind=IssueKind.STRUCTURE, project_id=pid,
                                         reason=reason))  # fmt: skip
        net = _sub(totals["original_principal_total_usd"], totals["cancelled_total_usd"])
        project = workbook.get(pid)
        commitment = project.workbook_ibrd_commitment_usd if project else None
        difference = _sub(totals["original_principal_total_usd"], commitment)
        refs = [ref for loan in group for ref in loan.source_refs]
        if project:
            refs += project.source_refs[:1]
        rows.append(
            SilverProjectFinancialSummary(
                project_id=pid,
                snapshot_date=snapshot,
                loan_count=len(group),
                raw_loan_numbers=[loan.raw_loan_number for loan in group],
                **totals,
                net_principal_after_cancellation_usd=net,
                disbursement_vs_original_principal_pct=percentage(
                    totals["disbursed_total_usd"], totals["original_principal_total_usd"]
                ),
                disbursement_vs_net_principal_pct=percentage(totals["disbursed_total_usd"], net),
                workbook_ibrd_commitment_usd=commitment,
                loan_principal_minus_workbook_commitment_usd=difference,
                commitment_sources_agree=None if difference is None else difference == 0,
                loans_with_valuation_caveats=[
                    loan.raw_loan_number for loan in group if loan.valuation_caveats
                ],
                source_refs=refs,
                quality_issues=issues,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


def _primary_sector(sectors: list[SilverProjectSector]) -> str | None:
    ranked = [s for s in sectors if s.sector_percent is not None and s.sector]
    if not ranked:
        return None
    top = max(s.sector_percent for s in ranked)
    leaders = {s.sector for s in ranked if s.sector_percent == top}
    return leaders.pop() if len(leaders) == 1 else None


def _distinct(values: Iterable[str | None]) -> list[str]:
    return list(OrderedDict.fromkeys(v for v in values if v))


def build_projects(
    table: BronzeTable,
    registry: ProjectRegistry,
    sectors: list[SilverProjectSector],
    themes: list[SilverProjectTheme],
    loans: list[SilverLoan],
) -> list[SilverProject]:
    rows = []
    for record in table.records:
        pid = record["project_id"]
        if pid not in registry:
            continue
        ref = source_ref(record, table.name, record_key=pid)
        norm = RowNormalizer(record, ref)
        refs = [ref]

        project_loans = [loan for loan in loans if loan.project_id == pid]
        codes = {loan.country_code for loan in project_loans if loan.country_code}
        country_code = country_code_source = None
        if len(codes) == 1:
            country_code, country_code_source = codes.pop(), "bronze_loans_raw"
            refs += [r for loan in project_loans for r in loan.source_refs]
        elif len(codes) > 1:
            norm.issues.append(ValueIssue(
                field="country_code", kind=IssueKind.CONFLICT, raw_value=", ".join(sorted(codes)),
                reason="loans disagree on the borrower country code; left NULL",
                source_ref=ref, project_id=pid))  # fmt: skip

        project_sectors = sorted(
            (s for s in sectors if s.project_id == pid),
            key=lambda s: (-(s.sector_percent or Decimal(0)), s.sector or ""),
        )

        def date_field(name: str, norm: RowNormalizer = norm) -> Any:
            return norm.date(name, PROJECTS)

        rows.append(
            SilverProject(
                project_id=pid,
                project_name=norm("project_name", clean_text),
                country=norm("country", clean_text),
                country_code=country_code,
                country_code_source=country_code_source,
                region=norm("region", clean_text),
                financing_instrument=norm("lending_instrument", normalize_category),
                financing_type=norm("financing_type", normalize_category),
                project_status=norm("project_status", normalize_category),
                last_stage_reached=norm("last_stage_reached_name", normalize_category),
                approval_date=date_field("board_approval_date"),
                effective_date=date_field("loan_effective_date"),
                current_closing_date=date_field("project_closing_date"),
                public_disclosure_date=date_field("public_disclosure_date"),
                borrower=norm("borrower", clean_text),
                implementing_agency=norm("implementing_agency", clean_text),
                primary_sector=_primary_sector(project_sectors),
                sectors=_distinct(s.sector for s in project_sectors),
                major_sectors=_distinct(s.major_sector for s in sectors if s.project_id == pid),
                themes=_distinct(t.theme_name for t in themes
                                 if t.project_id == pid and t.level == 1),  # fmt: skip
                project_development_objective=norm("project_development_objective", strip_html),
                workbook_ibrd_commitment_usd=norm("ibrd_commitment", parse_money),
                workbook_ida_commitment_usd=norm("ida_commitment", parse_money),
                workbook_grant_amount_usd=norm("grant_amount", parse_money),
                workbook_total_commitment_usd=norm("total_ibrd_ida_grant_commitment", parse_money),
                current_project_cost_usd=norm("current_project_cost", parse_money),
                environmental_assessment_category=norm("environmental_assessment_category",
                                                       normalize_category),
                environmental_and_social_risk=norm("environmental_and_social_risk",
                                                   normalize_category),
                source_last_updated=date_field("last_update_date"),
                source_file=record["_source_file"],
                source_refs=refs,
                quality_issues=norm.issues,
            )
        )  # fmt: skip
    return rows


# ---------------------------------------------------------------------------
# Procurement
# ---------------------------------------------------------------------------

_AWARD_FIELDS = {
    "contract_description": ("contract_description", clean_text),
    "procurement_category": ("procurement_category", normalize_category),
    "procurement_method": ("procurement_method", normalize_category),
    "review_type": ("review_type", normalize_category),
    "project_global_practice": ("project_global_practice", normalize_category),
    # Free-text reference (e.g. 'IN KUIDFC104978' next to 'IN-KUIDFC-97508-CW-RFB'),
    # so whitespace is cleaned rather than rejected as a malformed identifier.
    "borrower_contract_reference": ("borrower_contract_reference_number", clean_text),
    "fiscal_year": ("fiscal_year", parse_year),
}


_AWARD_ISSUE_FIELDS = {bronze for bronze, _ in _AWARD_FIELDS.values()} | {
    "wb_contract_number",
    "contract_signing_date",
}


def build_procurement(
    table: BronzeTable,
) -> tuple[list[SilverProcurementAward], list[SilverProcurementSupplier], list[ValueIssue]]:
    """Split contract-award rows into awards (grain: project + contract) and suppliers."""
    groups: OrderedDict[tuple[str, str], list[tuple[BronzeRecord, RowNormalizer]]] = OrderedDict()
    orphan_issues: list[ValueIssue] = []
    for record in table.records:
        norm = RowNormalizer(record, source_ref(record, table.name))
        contract_id = norm("wb_contract_number", normalize_identifier)
        if contract_id is None:
            orphan_issues.extend(norm.issues or [ValueIssue(
                field="wb_contract_number", kind=IssueKind.MALFORMED, raw_value=None,
                reason="blank contract identifier; row not assigned to an award",
                source_ref=norm.ref, project_id=record["project_id"])])  # fmt: skip
            continue
        norm.ref = source_ref(record, table.name, record_key=contract_id)
        groups.setdefault((record["project_id"], contract_id), []).append((record, norm))

    awards, suppliers = [], []
    for (pid, contract_id), members in groups.items():
        issues: list[ValueIssue] = []
        award_values: dict[str, Any] = {}
        for target, (bronze_field, fn) in _AWARD_FIELDS.items():
            values = [norm(bronze_field, fn) for _, norm in members]
            distinct = set(values)
            if len(distinct) == 1:
                award_values[target] = values[0]
            else:
                award_values[target] = None
                issues.append(ValueIssue(
                    field=target, kind=IssueKind.CONFLICT, project_id=pid,
                    raw_value=" | ".join(str(v) for v in values),
                    reason="supplier rows of one award disagree; left NULL",
                    source_ref=members[0][1].ref))  # fmt: skip
        dates = [norm.date("contract_signing_date", PROCUREMENT) for _, norm in members]
        if len(set(dates)) == 1:
            award_values["contract_signing_date"] = dates[0]
        else:
            award_values["contract_signing_date"] = None
            issues.append(ValueIssue(field="contract_signing_date", kind=IssueKind.CONFLICT,
                                     project_id=pid, reason="supplier rows disagree; left NULL",
                                     raw_value=" | ".join(map(str, dates)),
                                     source_ref=members[0][1].ref))  # fmt: skip

        amounts = []
        for _record, norm in members:
            award_issue_count = len(norm.issues)
            amount = norm("supplier_contract_amount_usd", parse_money)
            amounts.append(amount)
            suppliers.append(
                SilverProcurementSupplier(
                    project_id=pid,
                    contract_id=contract_id,
                    supplier_name=norm("supplier", clean_text),
                    supplier_id=norm("supplier_id", normalize_identifier),
                    supplier_country=norm("supplier_country", clean_text),
                    supplier_country_code=norm("supplier_country_code", clean_text),
                    supplier_row_amount_usd=amount,
                    source_refs=[norm.ref],
                    quality_issues=norm.issues[award_issue_count:],
                )
            )
        known = [a for a in amounts if a is not None]
        complete = len(known) == len(amounts)
        single = len(members) == 1
        for _, norm in members:  # award-field issues; supplier issues live on supplier rows
            issues.extend(
                i for i in norm.issues if i not in issues and i.field in _AWARD_ISSUE_FIELDS
            )
        awards.append(
            SilverProcurementAward(
                project_id=pid,
                contract_id=contract_id,
                **award_values,
                supplier_count=len(members),
                supplier_names=[clean_text(r.get("supplier")) or "" for r, _ in members],
                amount_basis=(
                    AwardAmountBasis.SINGLE_SUPPLIER_ROW
                    if single
                    else AwardAmountBasis.MULTI_SUPPLIER_UNRESOLVED
                ),
                contract_amount_usd=amounts[0] if single else None,
                amount_lower_bound_usd=max(known) if complete and known else None,
                amount_upper_bound_usd=sum(known, Decimal(0)) if complete and known else None,
                source_file=members[0][0]["_source_file"],
                source_refs=[norm.ref for _, norm in members],
                quality_issues=issues,
            )
        )
    return awards, suppliers, orphan_issues


def build_procurement_coverage(
    coverage: Iterable[Any],
    awards: list[SilverProcurementAward],
    suppliers: list[SilverProcurementSupplier],
    procurement_source_file: str,
    run_id: str,
) -> list[SilverProcurementCoverage]:
    award_counts: dict[str, int] = defaultdict(int)
    supplier_counts: dict[str, int] = defaultdict(int)
    for award in awards:
        award_counts[award.project_id] += 1
    for supplier in suppliers:
        supplier_counts[supplier.project_id] += 1
    rows = []
    for item in coverage:
        not_covered = item.coverage_status == ProcurementCoverageStatus.NOT_COVERED_BY_THIS_DATASET
        rows.append(
            SilverProcurementCoverage(
                project_id=item.project_id,
                dataset=item.dataset,
                covered_by_dataset=item.covered_by_dataset,
                coverage_status=str(item.coverage_status),
                source_row_count=item.row_count,
                award_count=None if not_covered else award_counts[item.project_id],
                supplier_relationship_count=(
                    None if not_covered else supplier_counts[item.project_id]
                ),
                interpretation=item.interpretation,
                source_refs=[
                    SourceRef(
                        bronze_table="bronze_procurement_coverage",
                        source_file=procurement_source_file,
                        ingestion_run_id=run_id,
                        record_key=item.project_id,
                    )
                ],
            )
        )
    return rows


def known_award_count(coverage: SilverProcurementCoverage) -> int | None:
    """Award count usable downstream; ``None`` means unknown, never zero activity."""
    if coverage.coverage_status == ProcurementCoverageStatus.NOT_COVERED_BY_THIS_DATASET:
        return None
    return coverage.award_count


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_silver(bronze: Any, registry: ProjectRegistry) -> SilverResult:
    """Build every Silver table from a ``BronzeIngestionResult`` (read-only on Bronze)."""
    tables = bronze.tables
    sectors = build_sectors(tables["bronze_sectors_raw"])
    themes, theme_issues = build_themes(tables["bronze_themes_raw"])
    loans = build_loans(tables["bronze_loans_raw"])
    projects = build_projects(tables["bronze_projects_raw"], registry, sectors, themes, loans)
    summaries = build_financial_summaries(loans, projects)
    procurement = tables["bronze_procurement_raw"]
    awards, suppliers, procurement_issues = build_procurement(procurement)
    coverage = build_procurement_coverage(
        bronze.procurement_coverage, awards, suppliers, procurement.metadata.source_file,
        bronze.run_id,
    )  # fmt: skip

    def table(name: str, model: type[SilverRow], rows: list) -> tuple[str, SilverTable]:
        return name, SilverTable(name, model, rows)

    return SilverResult(
        run_id=bronze.run_id,
        tables=dict(
            [
                table("silver_projects", SilverProject, projects),
                table("silver_project_sectors", SilverProjectSector, sectors),
                table("silver_project_themes", SilverProjectTheme, themes),
                table("silver_loans", SilverLoan, loans),
                table("silver_project_financial_summary", SilverProjectFinancialSummary,
                      summaries),
                table("silver_procurement_awards", SilverProcurementAward, awards),
                table("silver_procurement_suppliers", SilverProcurementSupplier, suppliers),
                table("silver_procurement_coverage", SilverProcurementCoverage, coverage),
            ]
        ),
        table_issues=[*theme_issues, *procurement_issues],
    )  # fmt: skip
