"""T4 get_financial_status: loans, commitments, disbursement and cancellations (Phase 9).

Two different sources are returned side by side and NEVER reconciled:

* loan statement snapshot (FACT, US$, snapshot date): gold.project_360 summary and
  silver.loans per loan;
* figures printed in an ISR (DOCUMENTED_FINDING, US$ millions, ISR date):
  silver.isr_loan_disbursements, only when ``as_of_isr`` is requested;
* optionally the documented additional-financing / cancellation events
  (silver.project_events, original currency; no conversion).

Loan numbers are normalised (``IBRD-93240`` == ``IBRD93240``). A loan of another approved
project is refused (SCOPE_REFUSED); an unknown loan is NOT_FOUND, with this project's
loans listed as candidates. A requested ISR without printed loan lines is NOT_FOUND.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict

from worldbank_copilot.tools.base import DataIntegrityError, ToolArgs, ToolContext, ToolSpec
from worldbank_copilot.tools.models import (
    ArgumentCandidate,
    Derivation,
    Fact,
    MechanicalCode,
    MechanicalFinding,
    ProvenanceClass,
    SourceRef,
    ToolOutcome,
    ToolStatus,
    derive,
    fact,
)
from worldbank_copilot.tools.project import instrument_caveats
from worldbank_copilot.tools.reader import Filter, ReadRequest

SUMMARY = (
    "original_principal_usd",
    "cancelled_usd",
    "net_principal_usd",
    "disbursed_usd",
    "undisbursed_usd",
    "financial_snapshot_date",
    "loan_count",
)
LOAN_COLUMNS = (
    "record_id",
    "raw_loan_number",
    "loan_status",
    "currency_of_commitment",
    "original_principal_usd",
    "cancelled_amount_usd",
    "disbursed_amount_usd",
    "undisbursed_amount_usd",
    "exchange_adjustment_usd",
    "board_approval_date",
    "agreement_signing_date",
    "effective_date",
    "closing_date",
    "last_disbursement_date",
    "snapshot_date",
    "valuation_caveats",
)
ISR_COLUMNS = (
    "record_id",
    "isr_sequence",
    "canonical_report_date",
    "loan_number",
    "loan_status",
    "currency",
    "original_musd",
    "revised_musd",
    "cancelled_musd",
    "disbursed_musd",
    "undisbursed_musd",
    "disbursed_pct_reported",
    "status",
    "evidence_document_id",
    "evidence_page_number",
    "evidence_section",
    "evidence_table_id",
    "evidence_extraction_method",
)
EVENT_COLUMNS = (
    "record_id",
    "event_id",
    "event_type",
    "event_date",
    "event_date_basis",
    "candidate_event_date",
    "candidate_date_status",
    "loan_number",
    "additional_financing_amount",
    "additional_financing_currency",
    "cancelled_amount",
    "cancelled_currency",
    "source_document",
    "source_page",
    "source_section",
    "extraction_method",
    "status",
)
SOURCES_NOTICE = (
    "Loan-statement figures (FACT, US$, statement snapshot date) and ISR-printed figures "
    "(DOCUMENTED_FINDING, US$ millions, ISR date) are different sources and dates; they "
    "are reported side by side and are not reconciled."
)


def normalize_loan(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


class FinanceArgs(ToolArgs):
    loan_number: str | None = None
    as_of_isr: int | Literal["latest"] | None = None
    include_events: bool = False


class LoanRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loan_number: str
    loan_status: str | None
    currency_of_commitment: str | None
    original_principal_usd: Decimal | None
    cancelled_amount_usd: Decimal | None
    disbursed_amount_usd: Decimal | None
    undisbursed_amount_usd: Decimal | None
    exchange_adjustment_usd: Decimal | None
    board_approval_date: date | None
    agreement_signing_date: date | None
    effective_date: date | None
    closing_date: date | None
    last_disbursement_date: date | None
    snapshot_date: date | None
    valuation_caveats: list[str]
    provenance_class: ProvenanceClass = ProvenanceClass.FACT
    source: SourceRef


class IsrLoanLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    isr_sequence: int
    report_date: date | None
    loan_number: str
    loan_status: str | None
    currency: str | None
    original_musd: Decimal | None
    revised_musd: Decimal | None
    cancelled_musd: Decimal | None
    disbursed_musd: Decimal | None
    undisbursed_musd: Decimal | None
    disbursed_pct_reported: Decimal | None
    unit: str = "US$ millions (as printed in the ISR)"
    provenance_class: ProvenanceClass = ProvenanceClass.DOCUMENTED_FINDING
    source: SourceRef


class FinancialEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    event_type: str
    event_date: date | None
    event_date_basis: str | None
    candidate_event_date: date | None
    candidate_date_status: str | None
    loan_number: str | None
    additional_financing_amount: Decimal | None
    additional_financing_currency: str | None
    cancelled_amount: Decimal | None
    cancelled_currency: str | None
    provenance_class: ProvenanceClass = ProvenanceClass.DOCUMENTED_FINDING
    source: SourceRef


class FinancialStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    summary: list[Fact]
    loans: list[LoanRecord]
    isr_reported: list[IsrLoanLine]
    events: list[FinancialEvent]


def _owner(ctx: ToolContext, loan: str) -> str | None:
    for project in ctx.registry.projects:
        if loan in {normalize_loan(n) for n in project.expected_loan_numbers or ()}:
            return project.project_id
    return None


def _summary(ctx: ToolContext, pid: str) -> list[Fact]:
    rows = ctx.read(
        ReadRequest(
            "gold.project_360",
            pid,
            (*SUMMARY, "disbursement_pct_of_net_principal", "record_id", "source_record_ids"),
        )
    )
    if len(rows) != 1:
        raise DataIntegrityError(f"gold.project_360 holds {len(rows)} rows for {pid}")
    row = rows[0]
    ref = SourceRef(
        table="gold.project_360",
        record_id=row["record_id"],
        supporting_record_ids=tuple(row["source_record_ids"] or ()),
    )
    facts = {}
    for name in SUMMARY:
        unit = "USD" if name.endswith("_usd") else None
        item = fact(name, row[name], ProvenanceClass.FACT, ref, unit=unit)
        if name.endswith("_usd") and item.value is not None:
            item = item.model_copy(
                update={
                    "derivation": Derivation(operation="sum_over_loans", inputs=("silver.loans",))
                }
            )
        facts[name] = item
    facts["disbursement_pct_of_net_principal"] = derive(
        "disbursement_pct_of_net_principal",
        "percentage",
        [facts["disbursed_usd"], facts["net_principal_usd"]],
        value=row["disbursement_pct_of_net_principal"],
        unit="percent",
        source=ref,
    )
    return list(facts.values())


def run(ctx: ToolContext, args: FinanceArgs) -> ToolOutcome:
    pid = args.project_id
    echo = args.model_dump(mode="json", exclude={"project_id"})
    wanted = normalize_loan(args.loan_number) if args.loan_number else None
    if wanted:
        owner = _owner(ctx, wanted)
        if owner is not None and owner != pid:
            detail = f"loan {args.loan_number} belongs to another project; scope is {pid}"
            return ToolOutcome(
                status=ToolStatus.SCOPE_REFUSED,
                filters=echo,
                mechanical=[
                    MechanicalFinding(code=MechanicalCode.PROJECT_OUT_OF_SCOPE, detail=detail)
                ],
            )
    loans = ctx.read(
        ReadRequest(
            "silver.loans",
            pid,
            LOAN_COLUMNS,
            order_by=(("raw_loan_number", "asc"), ("record_id", "asc")),
        )
    )
    if wanted:
        matching = [r for r in loans if normalize_loan(r["raw_loan_number"]) == wanted]
        if not matching:
            return ToolOutcome(
                status=ToolStatus.NOT_FOUND,
                filters=echo,
                argument_candidates=[
                    ArgumentCandidate(value=r["raw_loan_number"], label=r["raw_loan_number"])
                    for r in loans
                ],
                mechanical=[
                    MechanicalFinding(
                        code=MechanicalCode.LOAN_NOT_FOUND,
                        detail=f"loan {args.loan_number} is not among this project's loans",
                    )
                ],
            )
        loans = matching
    loan_records = [
        LoanRecord(
            loan_number=r["raw_loan_number"],
            **{
                c: r[c]
                for c in LOAN_COLUMNS
                if c not in ("record_id", "raw_loan_number", "valuation_caveats")
            },
            valuation_caveats=list(r["valuation_caveats"] or []),
            source=SourceRef(table="silver.loans", record_id=r["record_id"]),
        )
        for r in loans
    ]
    isr_lines: list[IsrLoanLine] = []
    if args.as_of_isr is not None:
        filters = [Filter("loan_number", "eq", wanted)] if wanted else []
        rows = ctx.read(
            ReadRequest(
                "silver.isr_loan_disbursements",
                pid,
                ISR_COLUMNS,
                tuple(filters),
                (("isr_sequence", "asc"), ("loan_number", "asc"), ("record_id", "asc")),
            )
        )
        sequences = sorted({r["isr_sequence"] for r in rows if r["isr_sequence"] is not None})
        target = sequences[-1] if (args.as_of_isr == "latest" and sequences) else args.as_of_isr
        selected = [r for r in rows if r["isr_sequence"] == target]
        if not selected:
            available = f"{sequences[0]}-{sequences[-1]}" if sequences else "none"
            return ToolOutcome(
                status=ToolStatus.NOT_FOUND,
                filters=echo,
                mechanical=[
                    MechanicalFinding(
                        code=MechanicalCode.ISR_NOT_FOUND,
                        detail=f"no ISR-printed loan lines for ISR {args.as_of_isr}; "
                        f"ISRs with loan lines: {available}",
                    )
                ],
            )
        isr_lines = [
            IsrLoanLine(
                isr_sequence=r["isr_sequence"],
                report_date=r["canonical_report_date"],
                loan_number=r["loan_number"],
                loan_status=r["loan_status"],
                currency=r["currency"],
                original_musd=r["original_musd"],
                revised_musd=r["revised_musd"],
                cancelled_musd=r["cancelled_musd"],
                disbursed_musd=r["disbursed_musd"],
                undisbursed_musd=r["undisbursed_musd"],
                disbursed_pct_reported=r["disbursed_pct_reported"],
                source=SourceRef(
                    table="silver.isr_loan_disbursements",
                    record_id=r["record_id"],
                    document_id=r["evidence_document_id"],
                    page_number=r["evidence_page_number"],
                    section=r["evidence_section"],
                    table_id=r["evidence_table_id"],
                    extraction_method=r["evidence_extraction_method"],
                    extraction_status=r["status"],
                ),
            )
            for r in selected
        ]
    events: list[FinancialEvent] = []
    if args.include_events:
        rows = ctx.read(
            ReadRequest(
                "silver.project_events",
                pid,
                EVENT_COLUMNS,
                (Filter("event_type", "in", ["ADDITIONAL_FINANCING", "CANCELLATION"]),),
                (("event_date", "asc"), ("event_id", "asc")),
            )
        )
        if wanted:
            rows = [
                r
                for r in rows
                if r["loan_number"] is None or normalize_loan(r["loan_number"]) == wanted
            ]
        events = [
            FinancialEvent(
                **{
                    c: r[c]
                    for c in EVENT_COLUMNS
                    if c
                    not in (
                        "record_id",
                        "source_document",
                        "source_page",
                        "source_section",
                        "extraction_method",
                        "status",
                    )
                },
                source=SourceRef(
                    table="silver.project_events",
                    record_id=r["record_id"],
                    document_id=r["source_document"],
                    page_number=r["source_page"],
                    section=r["source_section"],
                    extraction_method=r["extraction_method"],
                    extraction_status=r["status"],
                ),
            )
            for r in rows
        ]
    caveats = [
        f"loan {r.loan_number}: {', '.join(r.valuation_caveats)}"
        for r in loan_records
        if r.valuation_caveats
    ]
    caveats += instrument_caveats(ctx, pid)
    status = FinancialStatus(
        project_id=pid,
        summary=_summary(ctx, pid),
        loans=loan_records,
        isr_reported=isr_lines,
        events=events,
    )
    return ToolOutcome(
        status=ToolStatus.OK,
        items=[status],
        filters=echo,
        caveats=caveats,
        notices=[SOURCES_NOTICE] if isr_lines else [],
    )


SPEC = ToolSpec(
    name="get_financial_status",
    version="1",
    description="Loan-statement financing summary and per-loan figures; optionally the "
    "figures printed in an ISR and documented additional-financing / cancellation events.",
    args_model=FinanceArgs,
    item_model=FinancialStatus,
    tables=(
        "gold.project_360",
        "silver.loans",
        "silver.isr_loan_disbursements",
        "silver.project_events",
    ),
    run=run,
)
