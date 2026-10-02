"""T5 get_results_progress: results-framework indicators (Phase 9).

Source: gold.result_progress (DOCUMENTED_FINDING observations with safe calculations).

Indicator resolution is deterministic and never guesses:

1. exact ``canonical_indicator_id``;
2. exact normalised indicator name;
3. every significant query word contained in the indicator name.

More than one series -> AMBIGUOUS_ARGUMENT with the candidates (series in pending
alias pairs are never merged); none -> NOT_FOUND.

Temporal scope: ``isr_sequence`` (an ISR absent for the indicator -> NOT_FOUND),
``"latest"`` / default (latest observation per indicator by ISR sequence) or
``history`` (every observation). Progress, gap and change are derived values
(Decision 4); when Gold's ``calculation_status`` is not OK they are UNKNOWN with that
status as the reason (e.g. DLI_LAYOUT_NOT_EVALUATED, QUALITATIVE_UNIT).
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from worldbank_copilot.tools.base import GOLD_PROVENANCE, ToolArgs, ToolContext, ToolSpec, gold_ref
from worldbank_copilot.tools.models import (
    ArgumentCandidate,
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
from worldbank_copilot.tools.reader import Filter, ReadRequest

COLUMNS = (
    "canonical_indicator_id",
    "indicator_name",
    "indicator_type",
    "unit",
    "layout",
    "identity_review_status",
    "isr_sequence",
    "reporting_date",
    "baseline_value",
    "current_value",
    "target_value",
    "baseline_number",
    "current_number",
    "target_number",
    "target_date",
    "previous_isr_sequence",
    "previous_number",
    "absolute_change",
    "progress_percentage",
    "target_gap",
    "trend_direction",
    "target_status",
    "calculation_status",
    "target_changed",
    "extraction_status",
    "table_id",
    *GOLD_PROVENANCE,
)
STOPWORDS = frozenset("the of a an and in for to by with on at from as is are what how".split())


def normalize_name(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


class ResultsArgs(ToolArgs):
    indicator: str | None = Field(default=None, min_length=2, max_length=300)
    indicator_type: Literal["PDO", "INTERMEDIATE", "DLI"] | None = None
    isr_sequence: int | Literal["latest"] | None = None
    history: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> ResultsArgs:
        if self.history and self.isr_sequence is not None:
            raise ValueError("history returns every ISR; do not combine it with isr_sequence")
        if isinstance(self.isr_sequence, int) and self.isr_sequence < 1:
            raise ValueError("isr_sequence must be >= 1")
        return self


class ResultObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    indicator_id: str
    indicator_name: str
    indicator_type: str | None
    unit: str | None
    layout: str | None
    identity_review_status: str
    isr_sequence: int | None
    reporting_date: date | None
    baseline_value: str | None  # as printed
    current_value: str | None
    target_value: str | None
    baseline_number: Decimal | None
    current_number: Decimal | None
    target_number: Decimal | None
    target_date: str | None
    previous_isr_sequence: int | None
    progress_percentage: Fact
    target_gap: Fact
    change_from_previous: Fact
    trend_direction: str
    target_status: str
    calculation_status: str
    target_changed: bool
    extraction_status: str | None
    provenance_class: ProvenanceClass
    source: SourceRef


def resolve_indicator(query: str, series: dict[str, dict[str, Any]]) -> tuple[list[str], str]:
    """Matching canonical ids and the rule that matched (deterministic)."""
    q = query.strip()
    exact = [i for i in series if i.lower() == q.lower()]
    if exact:
        return exact, "canonical_id"
    norm = normalize_name(q)
    by_name = sorted(i for i, s in series.items() if normalize_name(s["indicator_name"]) == norm)
    if by_name:
        return by_name, "exact_name"
    words = set(norm.split()) - STOPWORDS
    if not words:
        return [], "no_significant_words"
    contained = sorted(
        i for i, s in series.items() if words <= set(normalize_name(s["indicator_name"]).split())
    )
    return contained, "all_words_contained"


def _derived(row: dict[str, Any], ref: SourceRef) -> tuple[Fact, Fact, Fact]:
    D = ProvenanceClass.DOCUMENTED_FINDING
    status = row["calculation_status"]

    def number(name: str, column: str) -> Fact:
        return fact(name, row[column], D, ref, unit=row["unit"])

    base, cur, tgt = (
        number("baseline", "baseline_number"),
        number("current", "current_number"),
        number("target", "target_number"),
    )
    prev = number("previous", "previous_number")
    if status != "OK":
        reason = f"not evaluable: {status}"
        progress = Fact(
            name="progress_percentage",
            provenance_class=ProvenanceClass.UNKNOWN,
            source=ref,
            unknown_reason=reason,
        )
    else:
        progress = derive(
            "progress_percentage",
            "(current-baseline)/(target-baseline)*100",
            [base, cur, tgt],
            value=row["progress_percentage"],
            unit="percent",
            source=ref,
        )
    gap = derive(
        "target_gap",
        "target-current",
        [tgt, cur],
        value=row["target_gap"],
        unit=row["unit"],
        source=ref,
    )
    change = derive(
        "change_from_previous",
        "current-previous (same indicator, previous ISR)",
        [cur, prev],
        value=row["absolute_change"],
        unit=row["unit"],
        source=ref,
    )
    return progress, gap, change


def _latest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda r: (
            r["isr_sequence"] is not None,
            r["isr_sequence"] or 0,
            r["reporting_date"] or date.min,
            r["record_id"],
        ),
    )


def run(ctx: ToolContext, args: ResultsArgs) -> ToolOutcome:
    pid = args.project_id
    echo = args.model_dump(mode="json", exclude={"project_id"})
    filters = (Filter("indicator_type", "eq", args.indicator_type),) if args.indicator_type else ()
    rows = ctx.read(
        ReadRequest(
            "gold.result_progress",
            pid,
            COLUMNS,
            filters,
            (
                ("indicator_type", "asc"),
                ("indicator_name", "asc"),
                ("canonical_indicator_id", "asc"),
                ("isr_sequence", "asc"),
                ("record_id", "asc"),
            ),
        )
    )
    if not rows:
        return ToolOutcome(
            status=ToolStatus.EMPTY,
            filters=echo,
            mechanical=[
                MechanicalFinding(
                    code=MechanicalCode.NO_RECORDS, detail="no indicator observations"
                )
            ],
        )
    by_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_series[row["canonical_indicator_id"]].append(row)
    series = {i: obs[-1] for i, obs in by_series.items()}
    notices = []
    if args.indicator:
        ids, rule = resolve_indicator(args.indicator, series)
        if not ids:
            return ToolOutcome(
                status=ToolStatus.NOT_FOUND,
                filters=echo,
                mechanical=[
                    MechanicalFinding(
                        code=MechanicalCode.INDICATOR_NOT_FOUND,
                        detail=f"no indicator matches {args.indicator!r} ({rule})",
                    )
                ],
            )
        if len(ids) > 1:
            pending = any(
                series[i]["identity_review_status"] == "PENDING_ALIAS_REVIEW" for i in ids
            )
            return ToolOutcome(
                status=ToolStatus.AMBIGUOUS_ARGUMENT,
                filters=echo,
                argument_candidates=[
                    ArgumentCandidate(
                        value=i,
                        label=series[i]["indicator_name"],
                        note=f"{series[i]['indicator_type']}; "
                        f"{series[i]['identity_review_status']}",
                    )
                    for i in ids
                ],
                mechanical=[
                    MechanicalFinding(
                        code=MechanicalCode.INDICATOR_AMBIGUOUS,
                        detail=f"{len(ids)} indicator series match ({rule})"
                        + (
                            "; some are in pending alias pairs and are never merged"
                            if pending
                            else ""
                        ),
                    )
                ],
            )
        by_series = {ids[0]: by_series[ids[0]]}
        notices.append(f"indicator resolved by {rule}: {ids[0]}")
    selected: list[dict[str, Any]] = []
    for obs in by_series.values():
        if args.history:
            selected.extend(obs)
        elif isinstance(args.isr_sequence, int):
            selected.extend(r for r in obs if r["isr_sequence"] == args.isr_sequence)
        else:
            selected.append(_latest(obs))
    if not selected:
        return ToolOutcome(
            status=ToolStatus.NOT_FOUND,
            filters=echo,
            mechanical=[
                MechanicalFinding(
                    code=MechanicalCode.ISR_NOT_FOUND,
                    detail=f"no observation in ISR {args.isr_sequence} "
                    "for the selected indicator(s)",
                )
            ],
        )
    items = []
    for row in selected:
        ref = gold_ref("gold.result_progress", row)
        progress, gap, change = _derived(row, ref)
        items.append(
            ResultObservation(
                indicator_id=row["canonical_indicator_id"],
                indicator_name=row["indicator_name"],
                indicator_type=row["indicator_type"],
                unit=row["unit"],
                layout=row["layout"],
                identity_review_status=row["identity_review_status"],
                isr_sequence=row["isr_sequence"],
                reporting_date=row["reporting_date"],
                baseline_value=row["baseline_value"],
                current_value=row["current_value"],
                target_value=row["target_value"],
                baseline_number=row["baseline_number"],
                current_number=row["current_number"],
                target_number=row["target_number"],
                target_date=row["target_date"],
                previous_isr_sequence=row["previous_isr_sequence"],
                progress_percentage=progress,
                target_gap=gap,
                change_from_previous=change,
                trend_direction=row["trend_direction"],
                target_status=row["target_status"],
                calculation_status=row["calculation_status"],
                target_changed=bool(row["target_changed"]),
                extraction_status=row["extraction_status"],
                provenance_class=ProvenanceClass(row["provenance_class"]),
                source=ref,
            )
        )
    not_ok: dict[str, int] = defaultdict(int)
    for row in selected:
        if row["calculation_status"] != "OK":
            not_ok[row["calculation_status"]] += 1
    mechanical = [
        MechanicalFinding(
            code=MechanicalCode.NOT_EVALUABLE,
            detail=f"{n} observation(s) not evaluable: {status}",
            field="progress_percentage",
        )
        for status, n in sorted(not_ok.items())
    ]
    return ToolOutcome(
        status=ToolStatus.OK, items=items, filters=echo, mechanical=mechanical, notices=notices
    )


SPEC = ToolSpec(
    name="get_results_progress",
    version="1",
    description="Results-framework indicators: printed baseline/current/target values, safe "
    "progress calculations, trend and target status, latest or by ISR or full history.",
    args_model=ResultsArgs,
    item_model=ResultObservation,
    tables=("gold.result_progress",),
    run=run,
)
