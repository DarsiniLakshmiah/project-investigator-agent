"""Golden application cases: schema, loading, and governed golden-fact resolution.

A case states expected BEHAVIOUR (statuses, refusal, isolation, temporal anchoring) and,
only where a defensible answer exists, golden facts. A golden fact is never a typed value:
it is a governed query (tool, arguments, field path, filters) resolved against the real
Silver/Gold data at evaluation time, keeping the table/record identity of every value.
Document relevance comes from the Phase 8 retrieval ground truth, referenced by question
id (``relevant_evidence``), never copied.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CASES_FILE = "evaluation/app_eval_cases.jsonl"
CATEGORIES = (
    "attention",
    "ratings",
    "restructuring",
    "temporal",
    "financial",
    "results",
    "documents",
    "cross_project",
    "invalid",
    "prediction",
)
CASES_PER_CATEGORY = 5
Status = Literal[
    "ANSWER", "EVIDENCE_ONLY", "INSUFFICIENT_EVIDENCE", "CLARIFY", "REFUSE", "FAIL_CLOSED"
]


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FactSpec(_Spec):
    """A governed query whose values are the golden facts (resolved on Databricks)."""

    fact_id: str
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)  # project_id is added by code
    path: str  # dotted field path; lists are traversed
    where: dict[str, Any] = Field(default_factory=dict)  # equality; "2021*" = prefix
    kind: Literal["date", "text", "number", "id"]
    match: Literal["all", "any"] = "all"


class AnchorExpectation(_Spec):
    resolutions: list[str]  # acceptable activity.anchor_resolution values
    relations: list[Literal["BEFORE", "AFTER", "COMPARE"]] = Field(default_factory=list)
    event_date: FactSpec | None = None  # governed anchor date used to check temporal citations


class Expected(_Spec):
    allowed_statuses: list[Status] = Field(min_length=1)  # PASS
    acceptable_statuses: list[Status] = Field(default_factory=list)  # safe but weaker: PARTIAL
    acceptable_routes: list[str] = Field(default_factory=list)  # only where meaningful
    required_behavior: list[str] = Field(default_factory=list)  # human-review guidance
    required_facts: list[FactSpec] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)  # regexes on published text
    relevant_evidence: list[str] = Field(default_factory=list)  # Phase 8 question ids
    required_source_types: list[Literal["DOCUMENT", "STRUCTURED"]] = Field(default_factory=list)
    required_provenance: list[str] = Field(default_factory=list)
    useful_tools: list[str] = Field(default_factory=list)  # diagnostic: any one is capable
    foreign_projects: list[str] = Field(default_factory=list)  # must never appear
    citation_required: bool = True
    abstention_acceptable: bool = False
    clarification_expected: bool = False
    prediction_must_be_refused: bool = False
    project_isolation_required: Literal[True] = True
    unauthorized_project: bool = False
    anchor: AnchorExpectation | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Expected:
        statuses = {*self.allowed_statuses, *self.acceptable_statuses}
        if self.abstention_acceptable != ("INSUFFICIENT_EVIDENCE" in statuses):
            raise ValueError("abstention_acceptable must match the INSUFFICIENT_EVIDENCE status")
        if self.prediction_must_be_refused and self.allowed_statuses != ["REFUSE"]:
            raise ValueError("a prediction case must require REFUSE")
        if self.clarification_expected and "CLARIFY" not in self.allowed_statuses:
            raise ValueError("a clarification case must allow CLARIFY")
        return self


class Case(_Spec):
    case_id: str = Field(pattern=r"^APP_\d{3}$")
    category: Literal[CATEGORIES]
    project_id: str = Field(pattern=r"^P\d{6}$")
    question: str = Field(min_length=3)
    expected: Expected

    @property
    def golden_specs(self) -> list[FactSpec]:
        anchor = self.expected.anchor
        return [
            *self.expected.required_facts,
            *([anchor.event_date] if anchor and anchor.event_date else []),
        ]


def load_cases(path: Path, *, project_ids=None, retrieval_questions=None) -> list[Case]:
    """Load and validate the golden set (counts, ids, projects, Phase 8 references)."""
    cases = [
        Case.model_validate_json(line)
        for line in Path(path).read_text("utf-8").splitlines()
        if line.strip()
    ]
    problems = validate_cases(
        cases, project_ids=project_ids, retrieval_questions=retrieval_questions
    )
    if problems:
        raise ValueError("invalid golden cases: " + "; ".join(problems))
    return cases


def validate_cases(cases, *, project_ids=None, retrieval_questions=None) -> list[str]:
    problems = []
    if len(cases) != len(CATEGORIES) * CASES_PER_CATEGORY:
        problems.append(
            f"expected {len(CATEGORIES) * CASES_PER_CATEGORY} cases, found {len(cases)}"
        )
    counts = Counter(c.category for c in cases)
    problems += [f"{k}: {counts[k]} cases" for k in CATEGORIES if counts[k] != CASES_PER_CATEGORY]
    ids = Counter(c.case_id for c in cases)
    problems += [f"duplicate {i}" for i, n in ids.items() if n > 1]
    for c in cases:
        if project_ids is not None and c.project_id not in project_ids:
            if not (c.expected.unauthorized_project and c.expected.allowed_statuses == ["REFUSE"]):
                problems.append(f"{c.case_id}: unknown project {c.project_id}")
        if c.project_id in c.expected.foreign_projects:
            problems.append(f"{c.case_id}: own project listed as foreign")
        if retrieval_questions is not None:
            for qid in c.expected.relevant_evidence:
                q = retrieval_questions.get(qid)
                if q is None:
                    problems.append(f"{c.case_id}: unknown Phase 8 question {qid}")
                elif q.project_id != c.project_id:
                    problems.append(f"{c.case_id}: {qid} belongs to {q.project_id}")
    return problems


# -- golden-fact resolution (pure; the tool call is injected) -------------------------------
def select(item: Any, path: str) -> list[Any]:
    """Values at a dotted path; lists are traversed; missing/None values are dropped."""
    values = [item]
    for key in path.split("."):
        nxt = []
        for value in values:
            for v in value if isinstance(value, list) else [value]:
                if isinstance(v, dict) and v.get(key) is not None:
                    found = v[key]
                    nxt.extend(found if isinstance(found, list) else [found])
        values = nxt
    return [v for v in values if v is not None]


def _matches_where(item: dict, where: dict) -> bool:
    for key, wanted in where.items():
        values = [str(v) for v in select(item, key)]
        if isinstance(wanted, str) and wanted.endswith("*"):
            if not any(v.startswith(wanted[:-1]) for v in values):
                return False
        elif str(wanted) not in values:
            return False
    return True


def resolve_fact(spec: FactSpec, items: list[dict]) -> dict:
    """Golden values for one spec, with the record identity of each value."""
    values, records = [], []
    for item in items:
        if not _matches_where(item, spec.where):
            continue
        for value in select(item, spec.path):
            if str(value) not in values:
                values.append(str(value))
                source = item.get("source") or {}
                records.append({"table": source.get("table"), "record_id": source.get("record_id")})
    return {
        **spec.model_dump(),
        "values": values,
        "records": records,
        "status": "RESOLVED" if values else "UNRESOLVED",
    }


def resolve_golden(
    cases: list[Case], run_tool: Callable[[str, dict, str], tuple[str, list[dict]]]
) -> dict[str, dict[str, dict]]:
    """``{case_id: {fact_id: resolved}}``; ``run_tool`` is the governed, project-scoped read."""
    golden: dict[str, dict[str, dict]] = {}
    for case in cases:
        for spec in case.golden_specs:
            try:
                status, items = run_tool(
                    spec.tool, {**spec.arguments, "project_id": case.project_id}, case.project_id
                )
                resolved = (
                    resolve_fact(spec, items)
                    if status in ("OK", "EMPTY")
                    else {
                        **spec.model_dump(),
                        "values": [],
                        "records": [],
                        "status": f"TOOL_{status}",
                    }
                )
            except Exception as exc:  # recorded, never guessed
                resolved = {
                    **spec.model_dump(),
                    "values": [],
                    "records": [],
                    "status": f"ERROR:{type(exc).__name__}",
                }
            golden.setdefault(case.case_id, {})[spec.fact_id] = resolved
    return golden


def dump_cases(cases: list[Case]) -> str:
    return "".join(json.dumps(c.model_dump(mode="json"), ensure_ascii=False) + "\n" for c in cases)
