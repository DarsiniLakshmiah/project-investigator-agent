"""T6 get_risk_register: documented risks and assessment findings (Phase 9).

Source: gold.risk_register (DOCUMENTED_FINDING). The framing of each record is preserved:
FORMAL_RISK_RATING (a rated risk), ASSESSMENT_FINDING (a finding of an assessment) and
ISR_SORT_RATING (SORT ratings of the latest ISR). ``resolution_status`` is always
NOT_STATED: documents do not state resolution and it is never inferred.

Ordering: record type, then rating rank (highest first, unrated last), observed date,
id. ``category`` matches case-insensitively; an unknown category returns EMPTY with the
project's categories as candidates.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from worldbank_copilot.intelligence.contracts import RISK_RECORD_TYPES
from worldbank_copilot.tools.base import (
    GOLD_PROVENANCE,
    ToolArgs,
    ToolContext,
    ToolSpec,
    gold_ref,
    row_class,
)
from worldbank_copilot.tools.models import (
    ArgumentCandidate,
    MechanicalCode,
    MechanicalFinding,
    ProvenanceClass,
    SourceRef,
    ToolOutcome,
    ToolStatus,
)
from worldbank_copilot.tools.reader import Filter, ReadRequest

COLUMNS = (
    "risk_or_finding_id",
    "record_type",
    "category",
    "title",
    "description",
    "activity",
    "rating_raw",
    "rating",
    "rating_rank",
    "rating_at_approval",
    "previous_rating",
    "observed_date",
    "isr_sequence",
    "source_document_type",
    "mitigation_text",
    "resolution_status",
    "extraction_status",
    *GOLD_PROVENANCE,
)
RESOLUTION_NOTICE = (
    "resolution_status is NOT_STATED: the documents do not state whether a risk or finding "
    "was resolved, and the system does not infer it."
)
_TYPE_ORDER = {t: i for i, t in enumerate(RISK_RECORD_TYPES)}


class RiskArgs(ToolArgs):
    record_type: Literal["FORMAL_RISK_RATING", "ASSESSMENT_FINDING", "ISR_SORT_RATING"] | None = (
        None
    )
    category: str | None = Field(default=None, min_length=1, max_length=200)
    source_document_type: str | None = Field(default=None, pattern=r"^[A-Z_]+$")


class RiskRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_or_finding_id: str
    record_type: str
    category: str | None
    title: str
    description: str | None
    activity: str | None
    rating_raw: str | None
    rating: str | None
    rating_rank: int | None
    rating_at_approval: str | None
    previous_rating: str | None
    observed_date: date | None
    isr_sequence: int | None
    source_document_type: str | None
    mitigation_text: str | None
    resolution_status: str
    extraction_status: str | None
    provenance_class: ProvenanceClass
    source: SourceRef


def run(ctx: ToolContext, args: RiskArgs) -> ToolOutcome:
    pid = args.project_id
    echo = args.model_dump(mode="json", exclude={"project_id"})
    filters = []
    if args.record_type:
        filters.append(Filter("record_type", "eq", args.record_type))
    if args.source_document_type:
        filters.append(Filter("source_document_type", "eq", args.source_document_type))
    rows = ctx.read(ReadRequest("gold.risk_register", pid, COLUMNS, tuple(filters)))
    categories = sorted({r["category"] for r in rows if r["category"]})
    if args.category:
        rows = [r for r in rows if (r["category"] or "").lower() == args.category.strip().lower()]
    if not rows:
        return ToolOutcome(
            status=ToolStatus.EMPTY,
            filters=echo,
            argument_candidates=[ArgumentCandidate(value=c, label=c) for c in categories]
            if args.category
            else [],
            mechanical=[
                MechanicalFinding(code=MechanicalCode.NO_RECORDS, detail="no risk record matched")
            ],
            notices=[RESOLUTION_NOTICE],
        )
    rows.sort(
        key=lambda r: (
            _TYPE_ORDER.get(r["record_type"], 99),
            r["rating_rank"] is None,
            -(r["rating_rank"] or 0),
            r["observed_date"] is None,
            r["observed_date"] or date.min,
            r["risk_or_finding_id"],
            r["record_id"],
        )
    )
    items = [
        RiskRecord(
            **{c: r[c] for c in COLUMNS if c not in GOLD_PROVENANCE},
            provenance_class=row_class(r),
            source=gold_ref("gold.risk_register", r),
        )
        for r in rows
    ]
    return ToolOutcome(status=ToolStatus.OK, items=items, filters=echo, notices=[RESOLUTION_NOTICE])


SPEC = ToolSpec(
    name="get_risk_register",
    version="1",
    description="Documented risk ratings and assessment findings (appraisal/assessment "
    "documents) and the latest ISR's SORT ratings, framing preserved.",
    args_model=RiskArgs,
    item_model=RiskRecord,
    tables=("gold.risk_register",),
    run=run,
)
