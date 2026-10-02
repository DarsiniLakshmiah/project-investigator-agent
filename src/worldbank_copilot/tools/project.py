"""T1 get_project_overview: where the project stands now (Phase 9).

Source: gold.project_360 (one current-state row) + silver.projects (identity text).
Every field is a ``Fact`` with its provenance class, taken from the Gold contract's
documentation (FACT = loan statement / workbook; DOCUMENTED_FINDING = documents;
SYSTEM_DERIVED_SIGNAL = attention-rule outputs). Derived values follow Decision 4.
Exactly one Gold row is expected for an approved project; anything else is a
DATA_INTEGRITY_ERROR (never an empty answer).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from worldbank_copilot.tools.base import DataIntegrityError, ToolArgs, ToolContext, ToolSpec
from worldbank_copilot.tools.models import (
    Derivation,
    Fact,
    ProvenanceClass,
    SourceRef,
    ToolOutcome,
    ToolStatus,
    derive,
    fact,
)
from worldbank_copilot.tools.reader import ReadRequest

F, D, S = (
    ProvenanceClass.FACT,
    ProvenanceClass.DOCUMENTED_FINDING,
    ProvenanceClass.SYSTEM_DERIVED_SIGNAL,
)

# gold.project_360 column -> (group, class, unit). Derived columns are listed separately.
DIRECT: dict[str, tuple[str, ProvenanceClass, str | None]] = {
    "project_name": ("identity", F, None),
    "instrument": ("identity", F, None),
    "project_status": ("identity", F, None),
    "approval_date": ("dates", F, None),
    "effectiveness_date": ("dates", F, None),
    "original_closing_date": ("dates", D, None),
    "current_closing_date": ("dates", F, None),
    "original_principal_usd": ("financing", F, "USD"),
    "cancelled_usd": ("financing", F, "USD"),
    "net_principal_usd": ("financing", F, "USD"),
    "disbursed_usd": ("financing", F, "USD"),
    "undisbursed_usd": ("financing", F, "USD"),
    "financial_snapshot_date": ("financing", F, None),
    "loan_count": ("financing", F, "loans"),
    "loans_with_valuation_caveats": ("financing", F, None),
    "latest_isr_sequence": ("ratings", D, None),
    "latest_isr_date": ("ratings", D, None),
    "latest_do_rating": ("ratings", D, None),
    "latest_ip_rating": ("ratings", D, None),
    "latest_overall_risk_rating": ("ratings", D, None),
    "previous_do_rating": ("ratings", D, None),
    "previous_ip_rating": ("ratings", D, None),
    "latest_results_reporting_date": ("activity", D, None),
}
# Counts over documented records (DOCUMENTED_FINDING) or over signals (SIGNAL).
COUNTS: dict[str, tuple[str, ProvenanceClass, str]] = {
    "number_of_isrs": ("activity", D, "silver.isr_snapshots"),
    "number_of_project_events": ("activity", D, "silver.project_events"),
    "number_of_restructurings": ("activity", D, "ISR restructuring history (dated)"),
    "number_of_restructuring_papers": ("activity", D, "restructuring papers (undated)"),
    "formal_risk_count": ("activity", D, "gold.risk_register FORMAL_RISK_RATING"),
    "assessment_finding_count": ("activity", D, "gold.risk_register ASSESSMENT_FINDING"),
    "number_of_result_indicators": ("activity", D, "gold.result_progress"),
    "current_attention_signal_count": ("signals", S, "gold.attention_signals CURRENT"),
    "current_watch_signal_count": ("signals", S, "gold.attention_signals CURRENT WATCH"),
    "current_high_signal_count": ("signals", S, "gold.attention_signals CURRENT HIGH"),
}
DERIVED = (
    "days_extended",
    "disbursement_pct_of_net_principal",
    "do_rating_change",
    "ip_rating_change",
)
METADATA = ("original_closing_date_status", "latest_isr_date_basis")
SUMMED = {"original_principal_usd", "cancelled_usd", "disbursed_usd", "undisbursed_usd"}
SILVER_IDENTITY = ("borrower", "implementing_agency", "project_development_objective")


class OverviewArgs(ToolArgs):
    pass


class ProjectOverview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    facts: list[Fact]
    groups: dict[str, list[str]]
    original_closing_date_status: str | None
    latest_isr_date_basis: str | None


def _single(rows: list[dict[str, Any]], table: str, project_id: str) -> dict[str, Any]:
    if len(rows) != 1:
        raise DataIntegrityError(
            f"{table} holds {len(rows)} rows for approved project {project_id} (expected 1)"
        )
    return rows[0]


def _rating_change(
    name: str, latest: Fact, previous: Fact, stored: str | None, ref: SourceRef
) -> Fact:
    comparable = stored if stored in ("UPGRADE", "DOWNGRADE", "UNCHANGED") else None
    return derive(
        name,
        "rank_comparison",
        [latest, previous],
        value=comparable,
        source=ref,
        note="ordinal ranks (configs/intelligence/rating_scales.yaml), previous ISR by sequence",
        unknown_reason=f"ratings not comparable ({stored})",
    )


def run(ctx: ToolContext, args: OverviewArgs) -> ToolOutcome:
    pid = args.project_id
    gold_cols = (*DIRECT, *COUNTS, *DERIVED, *METADATA, "record_id", "source_record_ids")
    row = _single(
        ctx.read(ReadRequest("gold.project_360", pid, tuple(gold_cols))), "gold.project_360", pid
    )
    silver = _single(
        ctx.read(ReadRequest("silver.projects", pid, ("record_id", *SILVER_IDENTITY))),
        "silver.projects",
        pid,
    )
    ref = SourceRef(
        table="gold.project_360",
        record_id=row["record_id"],
        supporting_record_ids=tuple(row["source_record_ids"] or ()),
    )
    groups: dict[str, list[str]] = {}
    facts: dict[str, Fact] = {}

    def add(group: str, item: Fact) -> None:
        facts[item.name] = item
        groups.setdefault(group, []).append(item.name)

    for name, (group, cls, unit) in DIRECT.items():
        item = fact(name, row[name], cls, ref, unit=unit)
        if name in SUMMED and item.value is not None:
            item = item.model_copy(
                update={
                    "derivation": Derivation(operation="sum_over_loans", inputs=("silver.loans",))
                }
            )
        add(group, item)
    silver_ref = SourceRef(table="silver.projects", record_id=silver["record_id"])
    for name in SILVER_IDENTITY:
        add("identity", fact(name, silver[name], F, silver_ref))
    for name, (group, cls, basis) in COUNTS.items():
        add(
            group,
            Fact(
                name=name,
                value=row[name],
                provenance_class=cls,
                source=ref,
                derivation=Derivation(operation="count", inputs=(basis,)),
            )
            if row[name] is not None
            else fact(name, None, cls, ref),
        )
    add(
        "dates",
        derive(
            "days_extended",
            "difference_in_days",
            [facts["current_closing_date"], facts["original_closing_date"]],
            value=row["days_extended"],
            unit="days",
            source=ref,
        ),
    )
    add(
        "financing",
        derive(
            "disbursement_pct_of_net_principal",
            "percentage",
            [facts["disbursed_usd"], facts["net_principal_usd"]],
            value=row["disbursement_pct_of_net_principal"],
            unit="percent",
            source=ref,
        ),
    )
    add(
        "ratings",
        _rating_change(
            "do_rating_change",
            facts["latest_do_rating"],
            facts["previous_do_rating"],
            row["do_rating_change"],
            ref,
        ),
    )
    add(
        "ratings",
        _rating_change(
            "ip_rating_change",
            facts["latest_ip_rating"],
            facts["previous_ip_rating"],
            row["ip_rating_change"],
            ref,
        ),
    )
    caveats = []
    if row["loans_with_valuation_caveats"]:
        caveats.append(
            "loans with valuation caveats (US$ values move with exchange rates): "
            + ", ".join(row["loans_with_valuation_caveats"])
        )
    caveats.extend(instrument_caveats(ctx, pid))
    overview = ProjectOverview(
        project_id=pid,
        facts=list(facts.values()),
        groups=groups,
        original_closing_date_status=row["original_closing_date_status"],
        latest_isr_date_basis=row["latest_isr_date_basis"],
    )
    return ToolOutcome(status=ToolStatus.OK, items=[overview], caveats=caveats)


def instrument_caveats(ctx: ToolContext, project_id: str) -> list[str]:
    """Attention rules that do not apply to this project's instrument (never 'no issue')."""
    instrument = ctx.registry.get(project_id).instrument
    out = []
    for rule in ctx.rules.enabled.values():
        if rule.applies_to_instruments and instrument not in rule.applies_to_instruments:
            out.append(
                f"rule {rule.rule_id} applies to {', '.join(rule.applies_to_instruments)} only; "
                f"it is not evaluated for this {instrument} operation"
            )
    return out


SPEC = ToolSpec(
    name="get_project_overview",
    version="1",
    description="Current state of one project: identity, dates, financing snapshot, latest "
    "ratings, activity counts and current attention-signal counts.",
    args_model=OverviewArgs,
    item_model=ProjectOverview,
    tables=("gold.project_360", "silver.projects"),
    run=run,
)
