"""Persistable 10B contracts; no evidence, synthesis or critic execution.

Immutable: source, decision, policy, requirement records and trace identity.
Append-only: requirements and budget reservation/consumption history.
Monotonic: ledger elapsed time and future retry count.
Terminal-write-once: final response is reserved for a future finalizer; in 10B it
must be None. Future containers must remain empty. Assessments remain PENDING.
All identity properties derive from the one detached source, never competing copies.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from worldbank_copilot.investigation.policy import BudgetLedger, Contract, InvestigationPolicy
from worldbank_copilot.routing.execution import ExecutionDecision, ExecutionMode, execution_decision
from worldbank_copilot.routing.models import (
    ExecutionOutcome,
    RetrievalSpec,
    RouteResult,
    TemporalScope,
    ToolCallSpec,
)
from worldbank_copilot.tools.models import ToolResult
from worldbank_copilot.tools.timeline import TimelineEvent

BASELINE_ID = "phase8_quality_baseline@1"


def historical_source(value: RouteResult | dict) -> RouteResult:
    """Revalidate a detached snapshot, preserving the permitted anchor item type.

    RouteResult's generic ToolResult[BaseModel] cannot roundtrip polymorphic items
    by itself. Anchor reads have one authorized concrete schema in this boundary.
    Answer results are never admitted and do not become investigation state.
    """
    raw = value.model_dump(mode="json") if isinstance(value, RouteResult) else dict(value)
    raw["anchor_results"] = [
        ToolResult[TimelineEvent].model_validate(item) for item in raw.get("anchor_results", ())
    ]
    return RouteResult.model_validate(raw)


def fingerprint(kind: str, value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return kind + "_" + hashlib.sha256(payload.encode()).hexdigest()


def operation_ids(project_id: str, call: ToolCallSpec | None, document: RetrievalSpec | None):
    ids = []
    if call is not None:
        from worldbank_copilot.tools.registry import TOOL_SPECS

        specs = {s.name: s for s in TOOL_SPECS}
        args = specs[call.tool].args_model.model_validate(call.arguments).model_dump(mode="json")
        for key in ("event_types", "rating_types", "categories"):
            if args.get(key) is not None:
                args[key] = sorted(set(args[key]))
        ids.append(fingerprint("op", {"tool": call.tool, "arguments": args}))
    if document is not None:
        ids.append(
            fingerprint(
                "op",
                {
                    "project_id": project_id,
                    "profile": BASELINE_ID,
                    "query": " ".join(document.query.split()),
                },
            )
        )
    return tuple(ids)


class PurposeCode(StrEnum):
    ESTABLISH = "ESTABLISH"
    EXPLAIN = "EXPLAIN"
    RESOLVE_CONFLICT = "RESOLVE_CONFLICT"


class EvidenceRequirement(Contract):
    requirement_id: str
    objective: str = Field(min_length=1, max_length=300)
    project_id: str
    temporal_scope: TemporalScope
    structured_call: ToolCallSpec | None = None
    document_retrieval: RetrievalSpec | None = None
    required: bool = True
    parent_requirement_id: str | None = None
    purpose_code: PurposeCode
    rationale: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def coherent(self):
        if self.structured_call is None and self.document_retrieval is None:
            raise ValueError("requirement needs at least one evidence operation")
        if (
            self.structured_call is not None
            and self.structured_call.arguments.get("project_id") != self.project_id
        ):
            raise ValueError("structured project mismatch")
        if (
            self.document_retrieval is not None
            and not 1 <= len(self.document_retrieval.query) <= 1000
        ):
            raise ValueError("query length out of bounds")
        if self.requirement_id != self.expected_id():
            raise ValueError("requirement fingerprint mismatch")
        return self

    def expected_id(self):
        return requirement_id(
            self.project_id,
            self.temporal_scope,
            self.structured_call,
            self.document_retrieval,
            self.purpose_code,
            self.parent_requirement_id,
        )

    @property
    def operation_ids(self):
        return operation_ids(self.project_id, self.structured_call, self.document_retrieval)


def requirement_id(project_id, temporal_scope, call, document, purpose, parent):
    return fingerprint(
        "req",
        {
            "project_id": project_id,
            "temporal": temporal_scope.model_dump(mode="json"),
            "operations": sorted(operation_ids(project_id, call, document)),
            "purpose": purpose,
            "parent": parent,
        },
    )


class AssessmentStatus(StrEnum):
    PENDING = "PENDING"
    SATISFIED = "SATISFIED"
    PARTIAL = "PARTIAL"
    UNSATISFIED = "UNSATISFIED"
    CONFLICTING = "CONFLICTING"
    ERROR = "ERROR"


class RequirementAssessment(Contract):
    requirement_id: str
    status: AssessmentStatus = AssessmentStatus.PENDING
    supporting_evidence_ids: tuple[str, ...] = Field(default=(), max_length=100)
    unsupported_aspects: tuple[str, ...] = Field(default=(), max_length=20)
    reason_codes: tuple[str, ...] = Field(default=(), max_length=20)
    assessment_version: Literal["assessment_contract@1"] = "assessment_contract@1"

    @model_validator(mode="after")
    def bounded_text(self):
        if any(
            len(s) > 240
            for s in (*self.unsupported_aspects, *self.reason_codes, *self.supporting_evidence_ids)
        ):
            raise ValueError("assessment text exceeds bounds")
        if self.status == AssessmentStatus.PENDING and (
            self.supporting_evidence_ids or self.unsupported_aspects or self.reason_codes
        ):
            raise ValueError("pending assessment cannot imply an assessment was performed")
        return self


class InvestigationStatus(StrEnum):
    ADMITTED = "ADMITTED"
    PLANNED = "PLANNED"
    CLARIFY = "CLARIFY"
    FAIL = "FAIL"


class InvestigationState(Contract):
    source: RouteResult
    execution: ExecutionDecision
    policy: InvestigationPolicy
    requirements: tuple[EvidenceRequirement, ...] = ()
    assessments: tuple[RequirementAssessment, ...] = ()
    operation_results: tuple[()] = ()
    evidence_references: tuple[()] = ()
    snapshot_manifest: tuple[()] = ()
    claim_versions: tuple[()] = ()
    critic_reviews: tuple[()] = ()
    repair_history: tuple[()] = ()
    retry_count: Literal[0] = 0
    budget: BudgetLedger
    status: InvestigationStatus = InvestigationStatus.ADMITTED
    failure_reasons: tuple[str, ...] = Field(default=(), max_length=10)
    final_response: Literal[None] = None
    trace_reference: str = Field(min_length=1, max_length=100)

    @field_validator("source", mode="before")
    @classmethod
    def snapshot(cls, value):
        return historical_source(value)

    @model_validator(mode="after")
    def coherent(self):
        if (
            self.execution.mode != ExecutionMode.PLAN_INVESTIGATION
            or execution_decision(self.source) != self.execution
            or self.source.outcome != ExecutionOutcome.PLANNED_NOT_EXECUTED
            or self.source.investigation_plan is None
            or self.source.investigation_plan.executed
        ):
            raise ValueError("state needs an unexecuted authoritative investigation plan")
        if self.budget.policy != self.policy:
            raise ValueError("budget policy mismatch")
        ids, ops = [], []
        for r in self.requirements:
            if r.project_id != self.project_id:
                raise ValueError("requirement project mismatch")
            if r.parent_requirement_id is not None and r.parent_requirement_id not in ids:
                raise ValueError("parent must already exist")
            ids.append(r.requirement_id)
            ops.extend(r.operation_ids)
        if len(ids) != len(set(ids)) or len(ops) != len(set(ops)):
            raise ValueError("duplicate requirements or operations")
        if (
            len(ids) > self.policy.max_initial_requirements
            or len(ops) > self.policy.max_initial_operations
            or len(ops) > self.policy.max_total_operations
            or sum(r.document_retrieval is not None for r in self.requirements)
            > self.policy.max_retrieval_operations
        ):
            raise ValueError("planning policy bounds exceeded")
        if tuple(a.requirement_id for a in self.assessments) != tuple(ids):
            raise ValueError("assessment identity mismatch")
        if any(a.status != AssessmentStatus.PENDING for a in self.assessments):
            raise ValueError("10B does not assess evidence")
        return self

    @property
    def request_id(self):
        return self.source.request_id

    @property
    def project_id(self):
        return self.source.investigation_plan.project_id

    @property
    def question(self):
        return self.source.investigation_plan.question

    @property
    def intent(self):
        return self.source.understanding.intent

    @property
    def temporal_scope(self):
        return self.source.investigation_plan.temporal

    @property
    def access(self):
        return self.source.access

    @property
    def source_plan(self):
        return self.source.investigation_plan

    @property
    def retrieval_profile_id(self):
        return self.source_plan.retrieval_profile

    def transition(self, *, append=(), budget=None, status=None, failure_reasons=None):
        if self.status in (InvestigationStatus.CLARIFY, InvestigationStatus.FAIL):
            raise ValueError("terminal state cannot transition")
        next_budget = budget if budget is not None else self.budget
        if not next_budget.extends(self.budget):
            raise ValueError("ledger cannot decrease or rewrite history")
        requirements = (*self.requirements, *append)
        return InvestigationState(
            source=self.source,
            execution=self.execution,
            policy=self.policy,
            requirements=requirements,
            assessments=(
                *self.assessments,
                *(RequirementAssessment(requirement_id=r.requirement_id) for r in append),
            ),
            budget=next_budget,
            status=status or self.status,
            failure_reasons=self.failure_reasons if failure_reasons is None else failure_reasons,
            trace_reference=self.trace_reference,
        )
