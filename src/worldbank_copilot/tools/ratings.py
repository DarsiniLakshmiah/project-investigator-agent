"""T3 get_rating_history: ISR ratings over time (Phase 9).

Source: silver.isr_snapshots (one row per ISR; Gold holds only the latest ratings).

* PDO ("PDO"), implementation progress ("IP") and overall risk ("OVERALL_RISK") ratings,
  each a DOCUMENTED_FINDING with its page, table and extraction method;
* ordered by ISR SEQUENCE, never by date (P179039 ISR 5 is dated out of order);
* the change from the preceding ISR (by sequence, whatever the requested range) is a
  rank comparison on configs/intelligence/rating_scales.yaml: it stays a
  DOCUMENTED_FINDING with derivation metadata, and is UNKNOWN when either rating is
  missing or off-scale. Performance: UPGRADE/DOWNGRADE; risk: INCREASED/DECREASED.
* SORT category ratings are not returned here (current SORT ratings: get_risk_register).
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from worldbank_copilot.intelligence.rules import RatingScales
from worldbank_copilot.tools.base import DataIntegrityError, ToolArgs, ToolContext, ToolSpec
from worldbank_copilot.tools.models import (
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
from worldbank_copilot.tools.reader import ReadRequest

RatingType = Literal["PDO", "IP", "OVERALL_RISK"]
COLUMN = {
    "PDO": "pdo_rating",
    "IP": "implementation_progress_rating",
    "OVERALL_RISK": "overall_risk_rating",
}
SUFFIXES = ("", "_raw", "_status", "_page_number", "_table_id", "_extraction_method")


class RatingHistoryArgs(ToolArgs):
    rating_types: tuple[RatingType, ...] = ("PDO", "IP", "OVERALL_RISK")
    isr_sequence_from: int | None = Field(default=None, ge=1)
    isr_sequence_to: int | None = Field(default=None, ge=1)
    latest_n: int | None = Field(default=None, ge=1, le=50)

    @model_validator(mode="after")
    def _consistent(self) -> RatingHistoryArgs:
        if not self.rating_types:
            raise ValueError("at least one rating type is required")
        ranged = self.isr_sequence_from is not None or self.isr_sequence_to is not None
        if ranged and self.latest_n is not None:
            raise ValueError("use either an ISR range or latest_n, not both")
        if (
            self.isr_sequence_from
            and self.isr_sequence_to
            and self.isr_sequence_from > self.isr_sequence_to
        ):
            raise ValueError("isr_sequence_from is after isr_sequence_to")
        return self


class RatingObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rating_type: RatingType
    rating: Fact  # normalized label (DOCUMENTED_FINDING) or UNKNOWN
    raw_rating: str | None
    change_from_previous: Fact | None  # None for the first ISR of the project
    compared_with_isr: int | None


class IsrRatings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    isr_sequence: int
    report_date: date | None
    report_date_basis: str | None
    document_id: str
    ratings: list[RatingObservation]
    provenance_class: ProvenanceClass = ProvenanceClass.DOCUMENTED_FINDING


def _scale(scales: RatingScales, rating_type: str) -> dict[str, int]:
    return scales.risk if rating_type == "OVERALL_RISK" else scales.performance


def _change(rating_type: str, scales: RatingScales):
    scale = _scale(scales, rating_type)
    up, down = (
        ("INCREASED", "DECREASED") if rating_type == "OVERALL_RISK" else ("UPGRADE", "DOWNGRADE")
    )

    def compute(current: str, previous: str) -> str | None:
        if current not in scale or previous not in scale:
            return None
        a, b = scale[current], scale[previous]
        return "UNCHANGED" if a == b else up if a > b else down

    return compute


def _rating_fact(row: dict[str, Any], rating_type: str) -> Fact:
    col = COLUMN[rating_type]
    ref = SourceRef(
        table="silver.isr_snapshots",
        record_id=row["record_id"],
        document_id=row["document_id"],
        page_number=row[f"{col}_page_number"],
        table_id=row[f"{col}_table_id"],
        extraction_method=row[f"{col}_extraction_method"],
        extraction_status=row[f"{col}_status"],
    )
    status = row[f"{col}_status"]
    return fact(
        f"{rating_type}_rating_isr_{row['isr_sequence']}",
        row[col],
        ProvenanceClass.DOCUMENTED_FINDING,
        ref,
        unknown_reason=f"rating not extracted (status {status})",
    )


def run(ctx: ToolContext, args: RatingHistoryArgs) -> ToolOutcome:
    columns = [
        "record_id",
        "document_id",
        "isr_sequence",
        "canonical_report_date",
        "canonical_date_basis",
    ]
    for rating_type in args.rating_types:
        columns += [COLUMN[rating_type] + s for s in SUFFIXES]
    rows = ctx.read(
        ReadRequest(
            "silver.isr_snapshots",
            args.project_id,
            tuple(columns),
            order_by=(("isr_sequence", "asc"), ("record_id", "asc")),
        )
    )
    echo = args.model_dump(mode="json", exclude={"project_id"})
    caveats, mechanical = [], []
    unsequenced = [r for r in rows if r["isr_sequence"] is None]
    rows = [r for r in rows if r["isr_sequence"] is not None]
    if unsequenced:
        caveats.append(f"{len(unsequenced)} ISR row(s) without a sequence number were skipped")
    if not rows:
        return ToolOutcome(
            status=ToolStatus.EMPTY,
            filters=echo,
            caveats=caveats,
            mechanical=[MechanicalFinding(code=MechanicalCode.NO_RECORDS, detail="no ISR ratings")],
        )
    sequences = [r["isr_sequence"] for r in rows]
    if len(set(sequences)) != len(sequences):
        raise DataIntegrityError(f"duplicate ISR sequences for {args.project_id}: {sequences}")
    previous_of = {r["isr_sequence"]: rows[i - 1] if i else None for i, r in enumerate(rows)}
    if args.latest_n:
        selected = rows[-args.latest_n :]
    else:
        lo, hi = args.isr_sequence_from or 1, args.isr_sequence_to or rows[-1]["isr_sequence"]
        selected = [r for r in rows if lo <= r["isr_sequence"] <= hi]
    if not selected:
        single = (
            args.isr_sequence_from is not None and args.isr_sequence_from == args.isr_sequence_to
        )
        available = f"available ISR sequences {rows[0]['isr_sequence']}-{rows[-1]['isr_sequence']}"
        return ToolOutcome(
            status=ToolStatus.NOT_FOUND if single else ToolStatus.EMPTY,
            filters=echo,
            mechanical=[
                MechanicalFinding(
                    code=MechanicalCode.ISR_NOT_FOUND if single else MechanicalCode.NO_RECORDS,
                    detail=f"no ISR in the requested range; {available}",
                )
            ],
        )
    items = []
    for row in selected:
        prev = previous_of[row["isr_sequence"]]
        observations = []
        for rating_type in args.rating_types:
            current = _rating_fact(row, rating_type)
            if current.provenance_class == ProvenanceClass.UNKNOWN:
                mechanical.append(
                    MechanicalFinding(
                        code=MechanicalCode.REQUIRED_FIELD_NULL,
                        detail=f"ISR {row['isr_sequence']} {rating_type}: {current.unknown_reason}",
                        field=COLUMN[rating_type],
                    )
                )
            change = None
            if prev is not None:
                change = derive(
                    f"{rating_type}_change_isr_{row['isr_sequence']}",
                    "rank_comparison",
                    [current, _rating_fact(prev, rating_type)],
                    compute=_change(rating_type, ctx.scales),
                    note="ordinal ranks only (rating_scales.yaml); preceding ISR by sequence",
                    unknown_reason="rating not on the configured ordinal scale",
                )
            observations.append(
                RatingObservation(
                    rating_type=rating_type,
                    rating=current,
                    raw_rating=row[f"{COLUMN[rating_type]}_raw"],
                    change_from_previous=change,
                    compared_with_isr=prev["isr_sequence"] if prev else None,
                )
            )
        items.append(
            IsrRatings(
                isr_sequence=row["isr_sequence"],
                report_date=row["canonical_report_date"],
                report_date_basis=row["canonical_date_basis"],
                document_id=row["document_id"],
                ratings=observations,
            )
        )
    gaps = [
        r["isr_sequence"]
        for r in selected
        if previous_of[r["isr_sequence"]] is not None
        and previous_of[r["isr_sequence"]]["isr_sequence"] != r["isr_sequence"] - 1
    ]
    if gaps:
        caveats.append(f"ISR(s) {gaps} are compared with a non-consecutive preceding ISR")
    return ToolOutcome(
        status=ToolStatus.OK, items=items, filters=echo, caveats=caveats, mechanical=mechanical
    )


SPEC = ToolSpec(
    name="get_rating_history",
    version="1",
    description="PDO, implementation-progress and overall-risk ratings by ISR sequence, with "
    "the change from the preceding ISR.",
    args_model=RatingHistoryArgs,
    item_model=IsrRatings,
    tables=("silver.isr_snapshots",),
    run=run,
)
