"""Routing evaluation dataset (Phase 9C): schema, loader, baseline measurement, report.

``evaluation/routing_cases.yaml`` holds PROPOSED HUMAN labels (``label_status: DRAFT``)
- the semantic ground truth a reviewer signs off. The current deterministic router's
output is a separate concept: it is measured by ``run_baseline`` and stored in a
generated file (``evaluation/routing_baseline_9b.yaml``), never written into the human
labels. Disagreement is expected (e.g. human DOCUMENT vs router
SEMANTIC_CLASSIFICATION_REQUIRED) and is reported, not "fixed" by tuning.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from worldbank_copilot.routing.models import (
    AccessContext,
    Intent,
    ProjectStatus,
    Route,
    RouteResult,
    TemporalKind,
)

HUMAN_ROUTES = ("STRUCTURED", "DOCUMENT", "INVESTIGATION", "CLARIFY", "REFUSE")
REVIEW_FLAGS = (
    "DUAL_ROUTE",
    "WHY_BOUNDARY",
    "RATINGS_APPRAISAL_RISKS",
    "MULTI_SUBJECT",
    "AMBIGUOUS_INFORMATION_NEED",
    "LABEL_REVIEW",
    "PROGRAMME_SIZE_SEMANTICS",
    "TIME_AMBIGUITY",
    "P179039_LIMITATION",
    "NO_ANSWER_IN_CORPUS",
    "FALSE_PREMISE",
    "SECURITY",
    "TOOL_LIMITATION",
    "IMPLICIT_FOREIGN_REFERENCE",
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExpectedTool(_Model):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)  # key arguments only (subset match)


class ExpectedTemporal(_Model):
    kind: TemporalKind
    explicit: bool
    isr_sequences: tuple[int, ...] = ()
    date_from: date | None = None
    date_to: date | None = None
    # Relative scopes resolve against a governed, reproducible as-of date (never wall clock).
    as_of: date | None = None
    as_of_basis: str | None = None

    @model_validator(mode="after")
    def _relative_contract(self) -> ExpectedTemporal:
        if self.kind == TemporalKind.RELATIVE and not (
            self.as_of and self.as_of_basis and self.date_from and self.date_to
        ):
            raise ValueError("a RELATIVE scope needs as_of, as_of_basis and the resolved interval")
        return self


class Expected(_Model):
    project_status: ProjectStatus | None  # None only for invalid input (stopped earlier)
    project_id: str | None = None
    project_basis: str | None = None
    intents: tuple[Intent, ...] = ()
    temporal: ExpectedTemporal | None = None
    route: Literal["STRUCTURED", "DOCUMENT", "INVESTIGATION", "CLARIFY", "REFUSE"]
    reason_code: str
    tools: tuple[ExpectedTool, ...] = ()
    outcome_note: str | None = None


class Alternative(_Model):
    route: Literal["STRUCTURED", "DOCUMENT", "INVESTIGATION", "CLARIFY", "REFUSE"]
    intents: tuple[Intent, ...] = ()
    rationale: str


class RoutingCase(_Model):
    case_id: str
    family: str  # split group: near-duplicates / same underlying evidence stay together
    split: Literal["dev", "test"]
    source: Literal["PHASE8", "NEW"]
    phase8_ref: str | None = None
    question: str
    active_project_id: str | None
    authorized_projects: tuple[str, ...] | Literal["ALL"] = "ALL"
    expected: Expected
    label_rationale: str
    review_note: str | None = None
    review_flags: tuple[str, ...] = ()
    alternatives: tuple[Alternative, ...] = ()
    decision_information: str | None = None  # what information would change the label
    human_decision: str | None = None  # decision recorded by the human reviewer (still DRAFT)
    label_status: Literal["DRAFT", "REVIEWED"] = "DRAFT"

    @field_validator("review_flags")
    @classmethod
    def _flags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = set(value) - set(REVIEW_FLAGS)
        if unknown:
            raise ValueError(f"unknown review flags {sorted(unknown)}")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> RoutingCase:
        if (self.source == "PHASE8") != (self.phase8_ref is not None):
            raise ValueError(f"{self.case_id}: phase8_ref iff source PHASE8")
        e = self.expected
        if e.route in ("STRUCTURED", "DOCUMENT", "INVESTIGATION"):
            if e.project_status != ProjectStatus.RESOLVED or not e.intents or not e.tools:
                raise ValueError(f"{self.case_id}: executing routes need project, intent, tools")
        elif e.tools:
            raise ValueError(f"{self.case_id}: CLARIFY/REFUSE expect no tools")
        if self.alternatives and not self.decision_information:
            raise ValueError(f"{self.case_id}: alternatives need decision_information")
        return self

    def access(self, all_projects: tuple[str, ...]) -> AccessContext:
        authorized = all_projects if self.authorized_projects == "ALL" else self.authorized_projects
        return AccessContext(
            user_ref="eval",
            authorized_projects=authorized,
            active_project_id=self.active_project_id,
        )


class RoutingDataset(_Model):
    version: int
    review_status: Literal["DRAFT", "REVIEWED"] = "DRAFT"
    reviewed_on: date | None = None
    split_status: Literal["PROPOSED", "FROZEN"] = "PROPOSED"
    semantic_rules: tuple[str, ...] = ()  # human routing rules the labels follow
    cases: tuple[RoutingCase, ...]

    @model_validator(mode="after")
    def _unique(self) -> RoutingDataset:
        ids = [c.case_id for c in self.cases]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"duplicate case ids {dup}")
        splits = defaultdict(set)
        for c in self.cases:
            splits[c.family].add(c.split)
        crossing = sorted(f for f, s in splits.items() if len(s) > 1)
        if crossing:
            raise ValueError(f"families split across dev/test: {crossing}")
        statuses = {c.label_status for c in self.cases}
        if statuses != {self.review_status}:
            raise ValueError(
                f"dataset review_status {self.review_status} vs case labels {statuses}"
            )
        if self.review_status == "REVIEWED" and not (self.reviewed_on and self.semantic_rules):
            raise ValueError("a REVIEWED dataset records reviewed_on and its semantic rules")
        return self


def load_dataset(path: Path) -> RoutingDataset:
    return RoutingDataset.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def similarity(a: str, b: str) -> float:
    x, y = tokens(a), tokens(b)
    return len(x & y) / len(x | y) if x | y else 1.0


# -- baseline (current 9B deterministic router) --------------------------------------------


class BaselineResult(_Model):
    case_id: str
    route: str
    reason_code: str
    project_status: str | None
    project_id: str | None
    project_basis: str | None
    intent: str | None
    temporal_kind: str | None
    temporal_explicit: bool | None
    tools: tuple[str, ...]  # executed, or planned for INVESTIGATION
    tool_args: tuple[dict[str, Any], ...]
    executed_tools: tuple[str, ...]
    anchor_reads: int
    retrieval_executed: bool
    executed_anything: bool


def baseline_of(case: RoutingCase, result: RouteResult) -> BaselineResult:
    u = result.understanding
    if result.investigation_plan is not None:
        calls = [(c.tool, c.arguments) for c in result.investigation_plan.structured_calls]
        calls.append(("search_project_documents", {}))
    elif u.requirements is not None and result.decision.route in (Route.STRUCTURED, Route.DOCUMENT):
        calls = [(c.tool, c.arguments) for c in u.requirements.structured]
        if u.requirements.document is not None:
            calls.append(("search_project_documents", {"query": u.requirements.document.query}))
    else:
        calls = []
    return BaselineResult(
        case_id=case.case_id,
        route=result.decision.route.value,
        reason_code=result.decision.reason_code,
        project_status=u.project.status.value if u.project else None,
        project_id=u.project.project_id if u.project else None,
        project_basis=u.project.basis if u.project else None,
        intent=u.intent.intent.value if u.intent and u.intent.intent else None,
        temporal_kind=u.temporal.kind.value if u.temporal else None,
        temporal_explicit=u.temporal.explicit if u.temporal else None,
        tools=tuple(t for t, _ in calls),
        tool_args=tuple(_jsonable(a) for _, a in calls),
        executed_tools=tuple(result.executed_tools),
        anchor_reads=len(result.anchor_results),
        retrieval_executed=result.retrieval_executed,
        executed_anything=bool(
            result.executed_tools or result.anchor_results or result.retrieval_executed
        ),
    )


def _jsonable(args: dict[str, Any]) -> dict[str, Any]:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in args.items()}


def run_baseline(
    service: Any, dataset: RoutingDataset, all_projects: tuple[str, ...]
) -> list[BaselineResult]:
    return [
        baseline_of(
            case, service.handle(case.question, case.access(all_projects), request_id=case.case_id)
        )
        for case in dataset.cases
    ]


# -- comparison --------------------------------------------------------------------------------


def _args_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    for key, value in expected.items():
        got = actual.get(key)
        if isinstance(value, list):
            if sorted(map(str, value)) != sorted(map(str, got or [])):
                return False
        elif str(got) != str(value):
            return False
    return True


def compare(case: RoutingCase, base: BaselineResult) -> dict[str, Any]:
    e = case.expected
    route_match = base.route == e.route
    project_match = e.project_status is None or (
        base.project_status == e.project_status.value and base.project_id == e.project_id
    )
    temporal_match = None
    if e.temporal is not None and base.temporal_kind is not None:
        temporal_match = base.temporal_kind == e.temporal.kind.value and (
            base.temporal_explicit == e.temporal.explicit
        )
    tools_match = None
    if e.tools and base.route == e.route:
        # expected tools are a MUST-INCLUDE set with key arguments; extra planned calls
        # (e.g. an investigation plan's attention signals) are not a disagreement.
        tools_match = all(
            any(
                t == x.tool and _args_match(x.args, a)
                for t, a in zip(base.tools, base.tool_args, strict=True)
            )
            for x in e.tools
        )
    if base.route == "SEMANTIC_CLASSIFICATION_REQUIRED":
        category = f"SEMANTIC_NEEDED -> human {e.route}"
    elif not project_match:
        category = "PROJECT_RESOLUTION_DIFFERS"
    elif not route_match:
        category = f"ROUTE_DIFFERS: 9B {base.route} vs human {e.route}"
    elif base.reason_code != e.reason_code:
        category = "REASON_DIFFERS"
    elif tools_match is False:
        category = "TOOLS_OR_ARGUMENTS_DIFFER"
    elif temporal_match is False:
        category = "TEMPORAL_DIFFERS"
    else:
        category = "AGREES"
    return {
        "route_match": route_match,
        "reason_match": base.reason_code == e.reason_code,
        "project_match": project_match,
        "temporal_match": temporal_match,
        "tools_match": tools_match,
        "category": category,
    }


def metrics(dataset: RoutingDataset, baseline: list[BaselineResult]) -> dict[str, Any]:
    by_id = {b.case_id: b for b in baseline}
    rows = [(c, by_id[c.case_id], compare(c, by_id[c.case_id])) for c in dataset.cases]
    n = len(rows)
    resolved = [r for r in rows if r[1].route != "SEMANTIC_CLASSIFICATION_REQUIRED"]
    temporal = [r for r in rows if r[2]["temporal_match"] is not None]
    confusion: Counter = Counter((r[0].expected.route, r[1].route) for r in resolved)
    refusal_cases = [
        r
        for r in rows
        if r[0].expected.route in ("REFUSE", "CLARIFY")
        and r[0].expected.reason_code
        in (
            "CROSS_PROJECT",
            "MULTI_PROJECT_NOT_SUPPORTED",
            "UNSUPPORTED_PROJECT",
            "NOT_AUTHORIZED",
            "PROJECT_REQUIRED",
            "INVALID_REQUEST",
            "AMBIGUOUS_PROJECT_REFERENCE",
        )
    ]
    return {
        "cases": n,
        "deterministic_coverage": round(len(resolved) / n, 3),
        "exact_route_agreement": round(sum(r[2]["route_match"] for r in rows) / n, 3),
        "exact_route_and_reason_agreement": round(
            sum(r[2]["route_match"] and r[2]["reason_match"] for r in rows) / n, 3
        ),
        "agreement_on_resolved": round(
            sum(r[2]["route_match"] for r in resolved) / len(resolved), 3
        )
        if resolved
        else None,
        "semantic_required_rate": round(1 - len(resolved) / n, 3),
        "clarify_rate": round(sum(r[1].route == "CLARIFY" for r in rows) / n, 3),
        "refuse_rate": round(sum(r[1].route == "REFUSE" for r in rows) / n, 3),
        "project_resolution_accuracy": round(sum(r[2]["project_match"] for r in rows) / n, 3),
        "temporal_exact_match": round(
            sum(r[2]["temporal_match"] for r in temporal) / len(temporal), 3
        )
        if temporal
        else None,
        "temporal_compared": len(temporal),
        "route_confusion_on_resolved": {
            f"{h} -> {b}": k for (h, b), k in sorted(confusion.items())
        },
        "isolation_cases": len(refusal_cases),
        "isolation_zero_execution": sum(not r[1].executed_anything for r in refusal_cases),
        "categories": dict(sorted(Counter(r[2]["category"] for r in rows).items())),
    }
