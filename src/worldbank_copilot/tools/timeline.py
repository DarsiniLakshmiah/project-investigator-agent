"""T2 get_project_timeline: what happened and when (Phase 9).

Source: gold.project_timeline, in its canonical order (``event_sequence``).

* event type / ISR-sequence filters are pushed down to the read;
* date filters apply to the SOURCE-STATED ``event_date`` only. Undated events are
  excluded under a date filter and counted in a caveat. Derived candidate dates are used
  only when ``include_candidate_dates`` is set, and are labelled as candidates;
* ``date_sequence_anomaly`` (an ISR dated later than its successor, kept as printed) is
  surfaced as a caveat, never silently re-dated.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from worldbank_copilot.extraction.events import EVENT_TYPES
from worldbank_copilot.tools.base import (
    GOLD_PROVENANCE,
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
from worldbank_copilot.tools.reader import Filter, ReadRequest

TIMELINE_EVENT_TYPES = tuple(
    sorted({*EVENT_TYPES, "ORIGINAL_CLOSING_DATE", "ISR_REPORT", "CURRENT_CLOSING_DATE"})
)
COLUMNS = (
    "timeline_event_id",
    "event_type",
    "event_subtype",
    "event_title",
    "event_description",
    "event_date",
    "event_date_status",
    "event_date_basis",
    "candidate_event_date",
    "candidate_date_basis",
    "candidate_date_status",
    "ordering_date",
    "ordering_basis",
    "event_sequence",
    "isr_sequence",
    "loan_number",
    "date_sequence_anomaly",
    "extraction_status",
    *GOLD_PROVENANCE,
)


class TimelineArgs(ToolArgs):
    event_types: tuple[str, ...] | None = None
    date_from: date | None = None
    date_to: date | None = None
    isr_sequence_from: int | None = Field(default=None, ge=1)
    isr_sequence_to: int | None = Field(default=None, ge=1)
    include_candidate_dates: bool = False
    limit: int = Field(default=200, ge=1, le=500)

    @model_validator(mode="after")
    def _consistent(self) -> TimelineArgs:
        unknown = sorted(set(self.event_types or ()) - set(TIMELINE_EVENT_TYPES))
        if unknown:
            raise ValueError(f"unknown event types {unknown}; allowed {TIMELINE_EVENT_TYPES}")
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from is after date_to")
        if (
            self.isr_sequence_from
            and self.isr_sequence_to
            and self.isr_sequence_from > self.isr_sequence_to
        ):
            raise ValueError("isr_sequence_from is after isr_sequence_to")
        return self


class TimelineEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_sequence: int
    event_type: str
    event_subtype: str | None
    event_title: str
    event_description: str | None
    event_date: date | None  # only a source-stated date
    event_date_status: str
    event_date_basis: str | None
    candidate_event_date: date | None  # derived candidate, never an official date
    candidate_date_basis: str | None
    candidate_date_status: str | None
    date_used_for_filter: str  # EVENT_DATE | CANDIDATE_DATE | NONE
    isr_sequence: int | None
    loan_number: str | None
    date_sequence_anomaly: bool
    extraction_status: str | None
    provenance_class: ProvenanceClass
    source: SourceRef


def _in_range(when: date | None, args: TimelineArgs) -> bool:
    if when is None:
        return False
    if args.date_from and when < args.date_from:
        return False
    return not (args.date_to and when > args.date_to)


def run(ctx: ToolContext, args: TimelineArgs) -> ToolOutcome:
    filters = []
    if args.event_types:
        filters.append(Filter("event_type", "in", list(args.event_types)))
    if args.isr_sequence_from:
        filters.append(Filter("isr_sequence", "ge", args.isr_sequence_from))
    if args.isr_sequence_to:
        filters.append(Filter("isr_sequence", "le", args.isr_sequence_to))
    rows = ctx.read(
        ReadRequest(
            "gold.project_timeline",
            args.project_id,
            COLUMNS,
            tuple(filters),
            (("event_sequence", "asc"), ("timeline_event_id", "asc")),
        )
    )
    dated = args.date_from is not None or args.date_to is not None
    kept: list[tuple[dict[str, Any], str]] = []
    undated = candidates_used = 0
    for row in rows:
        if not dated:
            kept.append((row, "NONE"))
        elif _in_range(row["event_date"], args):
            kept.append((row, "EVENT_DATE"))
        elif (
            row["event_date"] is None
            and args.include_candidate_dates
            and _in_range(row["candidate_event_date"], args)
        ):
            kept.append((row, "CANDIDATE_DATE"))
            candidates_used += 1
        elif row["event_date"] is None:
            undated += 1
    echo = args.model_dump(mode="json", exclude={"project_id"})
    caveats, mechanical = [], []
    if undated:
        caveats.append(
            f"{undated} event(s) without a source-stated date were excluded by the date filter"
        )
    if candidates_used:
        caveats.append(
            f"{candidates_used} event(s) matched only on a DERIVED candidate date (not official)"
        )
    if not kept:
        return ToolOutcome(
            status=ToolStatus.EMPTY,
            filters=echo,
            caveats=caveats,
            mechanical=[
                MechanicalFinding(code=MechanicalCode.NO_RECORDS, detail="no event matched")
            ],
        )
    if len(kept) > args.limit:
        mechanical.append(
            MechanicalFinding(
                code=MechanicalCode.RESULT_TRUNCATED,
                detail=f"{len(kept)} events matched; the first {args.limit} are returned",
            )
        )
        kept = kept[: args.limit]
    anomalies = [r["isr_sequence"] for r, _ in kept if r["date_sequence_anomaly"]]
    if anomalies:
        caveats.append(
            f"ISR(s) {anomalies} are dated later than the next ISR (kept as printed); "
            "the timeline is ordered by ISR sequence"
        )
    items = [
        TimelineEvent(
            event_sequence=r["event_sequence"],
            event_type=r["event_type"],
            event_subtype=r["event_subtype"],
            event_title=r["event_title"],
            event_description=r["event_description"],
            event_date=r["event_date"],
            event_date_status=r["event_date_status"],
            event_date_basis=r["event_date_basis"],
            candidate_event_date=r["candidate_event_date"],
            candidate_date_basis=r["candidate_date_basis"],
            candidate_date_status=r["candidate_date_status"],
            date_used_for_filter=used,
            isr_sequence=r["isr_sequence"],
            loan_number=r["loan_number"],
            date_sequence_anomaly=bool(r["date_sequence_anomaly"]),
            extraction_status=r["extraction_status"],
            provenance_class=row_class(r),
            source=gold_ref("gold.project_timeline", r),
        )
        for r, used in kept
    ]
    return ToolOutcome(
        status=ToolStatus.OK, items=items, filters=echo, caveats=caveats, mechanical=mechanical
    )


SPEC = ToolSpec(
    name="get_project_timeline",
    version="1",
    description="Chronological project history (milestones, ISRs, restructurings, additional "
    "financing, cancellations, closing-date changes) with source-stated dates.",
    args_model=TimelineArgs,
    item_model=TimelineEvent,
    tables=("gold.project_timeline",),
    run=run,
)
