"""Phase 10C evidence records, separate from immutable 10B planning history.

Retained Phase 9 envelopes keep concrete item schemas and all original fields.
Mechanical assessments never assert semantic sufficiency. No persisted clients,
credentials, authorization structures or reasoning transcripts are introduced.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, JsonValue, SerializeAsAny, field_validator, model_validator

from worldbank_copilot.investigation.models import (
    InvestigationState,
    RequirementAssessment,
    fingerprint,
)
from worldbank_copilot.investigation.policy import BudgetLedger, Contract
from worldbank_copilot.retrieval.contract import RetrievalResult
from worldbank_copilot.retrieval.models import Citation
from worldbank_copilot.routing.models import TemporalScope
from worldbank_copilot.tools.models import ProvenanceClass, SourceRef, ToolResult
from worldbank_copilot.tools.registry import TOOL_SPECS

SPECS = {s.name: s for s in TOOL_SPECS if s.tables}


class EvidenceLimits(Contract):
    """Configurable infrastructure bounds, not measured optimization results."""

    max_references: int = Field(default=10000, ge=1, le=100000)
    max_operation_bytes: int = Field(default=2000000, ge=1, le=16000000)
    max_package_bytes: int = Field(default=8000000, ge=1, le=64000000)
    projection_bytes_per_policy_token: int = Field(default=4, ge=1, le=16)


class OperationType(StrEnum):
    STRUCTURED = "STRUCTURED"
    DOCUMENT = "DOCUMENT"


class OperationStatus(StrEnum):
    OK = "OK"
    NO_EVIDENCE = "NO_EVIDENCE"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    SCOPE_REFUSED = "SCOPE_REFUSED"
    TOOL_ERROR = "TOOL_ERROR"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    INTEGRITY_ERROR = "INTEGRITY_ERROR"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    TEMPORAL_NOT_ENFORCEABLE = "TEMPORAL_NOT_ENFORCEABLE"


class MechanicalAssessment(RequirementAssessment):
    basis: Literal["MECHANICAL_EXECUTION"] = "MECHANICAL_EXECUTION"
    semantic_sufficiency: Literal["NOT_ASSESSED"] = "NOT_ASSESSED"


class EvidenceReference(Contract):
    evidence_id: str
    project_id: str
    source_type: OperationType
    identity: dict[str, JsonValue]
    provenance: ProvenanceClass
    payload: dict[str, JsonValue]
    citation: Citation | None = None
    source: SourceRef | None = None
    citation_support: Literal["SOURCE_REFERENCE", "COMPLETE", "LIMITED"]
    trust: Literal["UNTRUSTED_EVIDENCE"] = "UNTRUSTED_EVIDENCE"

    @model_validator(mode="after")
    def valid(self):
        if self.provenance == ProvenanceClass.AI_INTERPRETATION:
            raise ValueError("AI interpretations cannot enter deterministic evidence")
        if self.identity.get("project_id") != self.project_id:
            raise ValueError("reference project mismatch")
        if self.evidence_id != fingerprint("ev", self.identity):
            raise ValueError("reference identity mismatch")
        if self.citation is not None and self.citation.project_id != self.project_id:
            raise ValueError("citation project mismatch")
        return self


class EvidenceLink(Contract):
    evidence_id: str
    local_id: str | None = None


class OperationRecord(Contract):
    operation_id: str
    requirement_ids: tuple[str, ...]
    project_id: str
    operation_type: OperationType
    tool_or_profile: str
    status: OperationStatus
    reason_code: str
    attempted: bool
    reservation_id: str | None = None
    reservation_slot: int | None = None
    structured_result: SerializeAsAny[ToolResult] | None = None
    retrieval_result: RetrievalResult | None = None
    evidence_links: tuple[EvidenceLink, ...] = ()
    error_class: str | None = None
    physical_attempts: Literal[None] = None  # not exposed by the frozen interfaces

    @field_validator("structured_result", mode="before")
    @classmethod
    def typed_envelope(cls, value):
        if value is None:
            return None
        raw = value.model_dump(mode="python") if isinstance(value, ToolResult) else dict(value)
        spec = SPECS[raw["tool"]]
        return ToolResult[spec.item_model].model_validate(raw)

    @field_validator("retrieval_result", mode="before")
    @classmethod
    def retrieval_envelope(cls, value):
        return (
            None
            if value is None
            else RetrievalResult.model_validate(
                value.model_dump(mode="python") if isinstance(value, RetrievalResult) else value
            )
        )

    @model_validator(mode="after")
    def consistent(self):
        if self.structured_result is not None:
            r = self.structured_result
            if r.project_id != self.project_id or r.tool != self.tool_or_profile:
                raise ValueError("operation envelope mismatch")
        if self.retrieval_result is not None:
            r = self.retrieval_result
            if r.project_id != self.project_id or r.profile != self.tool_or_profile:
                raise ValueError("retrieval envelope mismatch")
        if self.attempted and self.reservation_id is None:
            raise ValueError("attempt needs a prior reservation")
        return self


class SnapshotManifest(Contract):
    table_versions: dict[str, int | None]
    retrieval_profile: str | None
    corpus_identity: str | None
    chunk_strategy_identity: str | None = None
    index_identity: str | None
    index_version: Literal[None] = None  # Phase 9 does not expose an atomic index revision
    globally_atomic: Literal[False] = False
    limitation: str = (
        "Per-table snapshots do not prove atomic consistency across Delta tables and AI Search."
    )


class MechanicalConflict(Contract):
    comparison_key: dict[str, JsonValue]
    evidence_ids: tuple[str, ...]
    reason_code: Literal["SAME_RECORD_FIELD_SNAPSHOT_DIFFERENT_VALUES"] = (
        "SAME_RECORD_FIELD_SNAPSHOT_DIFFERENT_VALUES"
    )


class RequirementSummary(Contract):
    requirement_id: str
    objective: str
    required: bool
    operation_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    assessment: MechanicalAssessment


class EvidencePackage(Contract):
    request_id: str
    project_id: str
    question: str
    temporal_scope: TemporalScope
    requirement_summaries: tuple[RequirementSummary, ...]
    operations: tuple[OperationRecord, ...]
    evidence_index: tuple[EvidenceReference, ...]
    conflicts: tuple[MechanicalConflict, ...]
    missing_requirements: tuple[str, ...]
    failed_requirements: tuple[str, ...]
    snapshots: SnapshotManifest
    warnings: tuple[str, ...]
    package_fingerprint: str
    semantic_sufficiency: Literal["NOT_ASSESSED"] = "NOT_ASSESSED"

    @model_validator(mode="after")
    def isolated(self):
        ids = [e.evidence_id for e in self.evidence_index]
        operation_ids = [o.operation_id for o in self.operations]
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("duplicate operation identity")
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate reference identity")
        if any(e.project_id != self.project_id for e in self.evidence_index):
            raise ValueError("package reference project mismatch")
        if any(o.project_id != self.project_id for o in self.operations):
            raise ValueError("package operation project mismatch")
        if any(link.evidence_id not in ids for o in self.operations for link in o.evidence_links):
            raise ValueError("dangling evidence reference")
        return self


class TraceKind(StrEnum):
    STARTED = "EVIDENCE_OPERATION_STARTED"
    COMPLETED = "EVIDENCE_OPERATION_COMPLETED"
    FAILED = "EVIDENCE_OPERATION_FAILED"
    ASSEMBLED = "EVIDENCE_ASSEMBLED"


class TraceEvent(Contract):
    event: TraceKind
    request_id: str
    project_id: str
    requirement_ids: tuple[str, ...] = ()
    operation_id: str | None = None
    operation_type: OperationType | None = None
    tool_or_profile: str | None = None
    status: OperationStatus | None = None
    latency_ms: float | None = None
    evidence_ids: tuple[str, ...] = ()
    table_versions: dict[str, int | None] = Field(default_factory=dict)
    index_identity: str | None = None
    error_class: str | None = None
    reserved_operations: int
    reserved_retrievals: int


class EvidenceExecutionReport(Contract):
    investigation: InvestigationState  # immutable planning history and extended ledger
    budget: BudgetLedger
    package: EvidencePackage
    events: tuple[TraceEvent, ...]
    terminal_failure: OperationStatus | None = None

    @model_validator(mode="after")
    def bounded_history(self):
        if not self.budget.extends(self.investigation.budget):
            raise ValueError("execution ledger cannot rewrite planning history")
        if (
            self.package.project_id != self.investigation.project_id
            or self.package.request_id != self.investigation.request_id
        ):
            raise ValueError("report identity mismatch")
        return self
