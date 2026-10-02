"""T7 get_attention_signals: what deserves attention (Phase 9).

Source: gold.attention_signals. Every signal is a SYSTEM_DERIVED_SIGNAL: the output of a
documented, versioned, deterministic rule (configs/intelligence/attention_rules.yaml)
over sourced records. It is an observable condition worth reviewing - not a
prediction, a probability or an assessment of project success or failure. The fixed
notice below travels with every result.

EMPTY means only that no configured rule fired for the filters; rules that do not
apply to the project's instrument and deferred rule candidates are listed, so an empty
result is never read as "no problems".

Ordering: severity (HIGH, WATCH, INFO), category, observed date (latest first), id.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from worldbank_copilot.intelligence.rules import CATEGORIES, SEVERITIES
from worldbank_copilot.tools.base import (
    GOLD_PROVENANCE,
    DataIntegrityError,
    ToolArgs,
    ToolContext,
    ToolSpec,
    gold_ref,
    row_class,
)
from worldbank_copilot.tools.models import (
    MechanicalCode,
    MechanicalFinding,
    ProvenanceClass,
    SourceRef,
    ToolOutcome,
    ToolStatus,
)
from worldbank_copilot.tools.project import instrument_caveats
from worldbank_copilot.tools.reader import Filter, ReadRequest

SIGNAL_NOTICE = (
    "SYSTEM_DERIVED_SIGNAL: each item is the output of a documented deterministic rule over "
    "sourced records - an observable condition worth reviewing, not a prediction or an "
    "assessment of project success or failure."
)
EMPTY_NOTICE = (
    "No configured rule fired for these filters. This is not an assessment that the project "
    "has no implementation issues."
)
COLUMNS = (
    "signal_id",
    "rule_id",
    "rule_version",
    "signal_category",
    "signal_type",
    "subject",
    "signal_title",
    "signal_description",
    "severity",
    "signal_status",
    "observed_date",
    "effective_sequence",
    "current_value",
    "comparison_value",
    "threshold",
    "value_unit",
    "rule_description",
    "supporting_record_ids",
    "evidence_status",
    "caveats",
    *GOLD_PROVENANCE,
)
_SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITIES)}  # INFO < WATCH < HIGH


class SignalArgs(ToolArgs):
    status: Literal["CURRENT", "HISTORICAL", "ALL"] = "CURRENT"
    min_severity: Literal["INFO", "WATCH", "HIGH"] = "INFO"
    categories: tuple[str, ...] | None = Field(default=None, min_length=1)

    @field_validator("categories")
    @classmethod
    def _known(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        unknown = sorted(set(value or ()) - set(CATEGORIES))
        if unknown:
            raise ValueError(f"unknown categories {unknown}; allowed {CATEGORIES}")
        return value


class AttentionSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal_id: str
    rule_id: str
    rule_version: str
    signal_category: str
    signal_type: str
    subject: str
    signal_title: str
    signal_description: str
    severity: str
    signal_status: str
    observed_date: date | None
    effective_sequence: int | None
    current_value: str | None
    comparison_value: str | None
    threshold: str | None
    value_unit: str | None
    rule_description: str
    supporting_record_ids: list[str]
    evidence_status: str | None
    caveats: list[str]
    provenance_class: ProvenanceClass
    source: SourceRef


def run(ctx: ToolContext, args: SignalArgs) -> ToolOutcome:
    pid = args.project_id
    echo = args.model_dump(mode="json", exclude={"project_id"})
    allowed = [s for s in SEVERITIES if _SEVERITY_RANK[s] >= _SEVERITY_RANK[args.min_severity]]
    filters = [Filter("severity", "in", allowed)]
    if args.status != "ALL":
        filters.append(Filter("signal_status", "eq", args.status))
    if args.categories:
        filters.append(Filter("signal_category", "in", list(args.categories)))
    rows = ctx.read(ReadRequest("gold.attention_signals", pid, COLUMNS, tuple(filters)))
    context = instrument_caveats(ctx, pid) + [
        f"deferred rule candidate (not evaluated): {d.candidate}" for d in ctx.rules.deferred
    ]
    if not rows:
        return ToolOutcome(
            status=ToolStatus.EMPTY,
            filters=echo,
            caveats=context,
            notices=[SIGNAL_NOTICE, EMPTY_NOTICE],
            mechanical=[
                MechanicalFinding(code=MechanicalCode.NO_RECORDS, detail="no signal matched")
            ],
        )
    wrong = [r["signal_id"] for r in rows if r["provenance_class"] != "SYSTEM_DERIVED_SIGNAL"]
    if wrong:
        raise DataIntegrityError(f"attention signals with a non-signal provenance class: {wrong}")
    rows.sort(
        key=lambda r: (
            -_SEVERITY_RANK[r["severity"]],
            r["signal_category"],
            r["observed_date"] is None,
            -(r["observed_date"].toordinal() if r["observed_date"] else 0),
            r["signal_id"],
        )
    )
    items = [
        AttentionSignal(
            **{
                c: r[c]
                for c in COLUMNS
                if c not in GOLD_PROVENANCE and c not in ("supporting_record_ids", "caveats")
            },
            supporting_record_ids=list(r["supporting_record_ids"] or []),
            caveats=list(r["caveats"] or []),
            provenance_class=row_class(r),
            source=gold_ref("gold.attention_signals", r),
        )
        for r in rows
    ]
    return ToolOutcome(
        status=ToolStatus.OK, items=items, filters=echo, caveats=context, notices=[SIGNAL_NOTICE]
    )


SPEC = ToolSpec(
    name="get_attention_signals",
    version="1",
    description="Deterministic, evidence-backed attention signals (SYSTEM_DERIVED_SIGNAL); "
    "observable conditions, not predictions.",
    args_model=SignalArgs,
    item_model=AttentionSignal,
    tables=("gold.attention_signals",),
    run=run,
)
