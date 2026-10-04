"""Typed result of ``Copilot.investigate``: everything the App and traces display."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Result(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResultStatus(StrEnum):
    ANSWER = "ANSWER"  # validated, cited claims were published
    EVIDENCE_ONLY = "EVIDENCE_ONLY"  # deterministic route: governed records, no model call
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"  # abstention: nothing publishable
    CLARIFY = "CLARIFY"
    REFUSE = "REFUSE"
    FAIL_CLOSED = "FAIL_CLOSED"


class CriticStatus(StrEnum):
    NOT_REQUIRED = "NOT_REQUIRED"  # no synthesis, or no claims to review
    DISABLED = "DISABLED"  # by configuration; claims are NOT critic-validated
    REVIEWED = "REVIEWED"  # claim-level verdicts applied (see Claim.support, removed counts)
    FAILED = "FAILED"  # critic call/output failed; fail closed


class Citation(_Result):
    evidence_id: str
    source_identity: str
    document_id: str | None = None
    document_label: str | None = None
    pages: tuple[int, ...] = ()
    table: str | None = None
    record_id: str | None = None


class Claim(_Result):
    claim_id: str
    text: str
    claim_type: str
    provenance: str
    evidence_ids: tuple[str, ...]
    citations: tuple[Citation, ...]
    support: str = "NOT_ASSESSED"  # SUPPORTED | PARTIALLY_SUPPORTED | NOT_ASSESSED
    qualifier: str | None = None  # Critic's note on what a partial claim does not support


class EvidenceItem(_Result):
    """One governed record or retrieved passage; untrusted evidence, never instructions."""

    evidence_id: str | None = None  # present for EvidencePackage references
    tool: str | None = None  # present for deterministic tool results
    provenance: tuple[str, ...]
    source_type: str | None = None
    citation: dict[str, Any] | None = None
    source: dict[str, Any] | None = None
    payload: dict[str, Any]


class AttentionSignal(_Result):
    signal_id: str
    title: str
    description: str
    category: str
    severity: str
    status: str
    observed_date: str | None = None
    current_value: str | None = None
    comparison_value: str | None = None
    provenance: str
    caveats: tuple[str, ...] = ()


class ModelCall(_Result):
    role: str
    endpoint: str | None = None
    model_identity: str | None = None
    latency_ms: float = Field(ge=0)
    input_tokens: int | None = None
    output_tokens: int | None = None
    outcome: str  # COMPLETED or the authoritative Failure category
    diagnostic_reason: str | None = None  # bounded code explaining a failure; never model text


class Validation(_Result):
    disposition: str | None = None  # finalizer disposition, when synthesized
    mechanical_validity: str | None = None
    failures: tuple[str, ...] = ()  # response-level integrity failures (fail closed)
    semantic_support: str = "NOT_ASSESSED"
    critic_status: CriticStatus = CriticStatus.NOT_REQUIRED
    claims_removed: dict[str, int] = Field(default_factory=dict)  # by reason code


class InvestigationActivity(_Result):
    """Bounded Investigator activity (counts and fixed codes only)."""

    decision_rounds: int = 0
    tool_calls: int = 0
    rejected_actions: int = 0
    evidence_retrieved: int = 0
    evidence_shown: int = 0
    temporal_anchor: str | None = None  # BEFORE | AFTER | COMPARE when applied
    evidence_filtered_by_date: int = 0


class InvestigationResult(_Result):
    request_id: str
    query: str
    project_id: str
    route: str | None = None
    reason_code: str | None = None
    intent: str | None = None
    temporal_scope: dict[str, Any] | None = None
    status: ResultStatus
    message: str
    attention_signals: tuple[AttentionSignal, ...] = ()
    claims: tuple[Claim, ...] = ()
    evidence: tuple[EvidenceItem, ...] = ()
    limitations: tuple[str, ...] = ()
    validation: Validation = Validation()
    model_calls: tuple[ModelCall, ...] = ()
    objective: str | None = None  # the Investigator's restatement (model text)
    activity: InvestigationActivity | None = None
    model_capability_note: str | None = None  # set whenever a model call was made
    latency_ms: float = Field(default=0, ge=0)
    stage_latency_ms: dict[str, float] = Field(default_factory=dict)
    trace_id: str | None = None
