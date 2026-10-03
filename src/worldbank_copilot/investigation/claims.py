"""Phase 10D bounded model-node contracts; no tools, agents or repair loop."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Protocol

from pydantic import Field

from worldbank_copilot.investigation.policy import BudgetLedger, Contract
from worldbank_copilot.routing.models import TemporalScope
from worldbank_copilot.tools.models import ProvenanceClass

PROMPT_VERSION = "bounded_synthesis_critic@1"
SCHEMA_VERSION = "candidate_claims@1"


class Failure(StrEnum):
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    MODEL_OUTPUT_INVALID = "MODEL_OUTPUT_INVALID"
    SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"
    EVIDENCE_REFERENCE_INVALID = "EVIDENCE_REFERENCE_INVALID"
    PROJECT_ISOLATION_VIOLATION = "PROJECT_ISOLATION_VIOLATION"
    PROVENANCE_VIOLATION = "PROVENANCE_VIOLATION"
    CITATION_VALIDATION_FAILED = "CITATION_VALIDATION_FAILED"
    TEMPORAL_SCOPE_VIOLATION = "TEMPORAL_SCOPE_VIOLATION"
    REQUIREMENT_REFERENCE_INVALID = "REQUIREMENT_REFERENCE_INVALID"
    CONTEXT_BUDGET_EXCEEDED = "CONTEXT_BUDGET_EXCEEDED"
    OUTPUT_BUDGET_EXCEEDED = "OUTPUT_BUDGET_EXCEEDED"
    CRITIC_REJECTED = "CRITIC_REJECTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    INTERNAL_VALIDATION_ERROR = "INTERNAL_VALIDATION_ERROR"
    OVERCLAIMED = "OVERCLAIMED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class NodeError(ValueError):
    def __init__(self, category: Failure):
        super().__init__(category.value)
        self.category = category


class ClaimType(StrEnum):
    ASSERTION = "ASSERTION"
    INTERPRETATION = "INTERPRETATION"
    UNCERTAINTY = "UNCERTAINTY"


class CitationReference(Contract):
    evidence_id: str = Field(min_length=1, max_length=100)
    source_identity: str = Field(min_length=1, max_length=100)


class CandidateClaim(Contract):
    claim_id: str = Field(pattern=r"^C[0-9]{1,3}$")
    claim_text: str = Field(min_length=1, max_length=1000)
    claim_type: ClaimType
    provenance_label: ProvenanceClass
    evidence_ids: tuple[str, ...] = Field(max_length=100)
    requirement_ids: tuple[str, ...] = Field(min_length=1, max_length=100)
    citations: tuple[CitationReference, ...] = Field(max_length=100)
    project_id: str = Field(pattern=r"^P[0-9]{6}$")
    temporal_scope: TemporalScope
    status: Literal["CANDIDATE"] = "CANDIDATE"


class SynthesisOutput(Contract):
    schema_version: Literal["candidate_claims@1"] = SCHEMA_VERSION
    candidate_claims: tuple[CandidateClaim, ...] = Field(max_length=100)
    insufficient_evidence: bool
    limitations: tuple[str, ...] = Field(max_length=20)
    # No free-text draft can bypass per-claim validation/publication.
    summary_claim_ids: tuple[str, ...] = Field(max_length=100)


class CriticCode(StrEnum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    CITATION_INVALID = "CITATION_INVALID"
    PROVENANCE_INVALID = "PROVENANCE_INVALID"
    PROJECT_SCOPE_VIOLATION = "PROJECT_SCOPE_VIOLATION"
    TEMPORAL_SCOPE_VIOLATION = "TEMPORAL_SCOPE_VIOLATION"
    OVERCLAIMED = "OVERCLAIMED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class CriticFinding(Contract):
    claim_id: str = Field(pattern=r"^C[0-9]{1,3}$")
    code: CriticCode
    evidence_ids: tuple[str, ...] = Field(max_length=100)
    concise_rationale: str = Field(min_length=1, max_length=240)


class CriticOutput(Contract):
    findings: tuple[CriticFinding, ...] = Field(max_length=100)


class Disposition(StrEnum):
    PUBLISH = "PUBLISH"
    PUBLISH_WITH_LIMITATIONS = "PUBLISH_WITH_LIMITATIONS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    REJECT_UNSUPPORTED = "REJECT_UNSUPPORTED"
    FAIL_CLOSED = "FAIL_CLOSED"


class ModelReply(Contract):
    text: str = Field(max_length=100000)
    model_identity: str = Field(min_length=1, max_length=200)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class ModelRequest(Contract):
    role: Literal["SYNTHESIZER", "CRITIC"]
    system: str
    context_json: str
    output_schema: dict
    max_output_tokens: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)
    temperature: Literal[0] = 0


class ModelAdapter(Protocol):
    def invoke(self, request: ModelRequest) -> ModelReply:
        """One bounded attempt; receives no tool/client/credential capability."""
        ...


class ModelEvent(Contract):
    request_id: str
    project_id: str
    plan_id: str
    package_fingerprint: str
    prompt_version: str = PROMPT_VERSION
    schema_version: str = SCHEMA_VERSION
    model_role: str
    model_identity: str | None = None
    input_evidence_ids: tuple[str, ...] = ()
    candidate_claim_ids: tuple[str, ...] = ()
    critic_codes: tuple[CriticCode, ...] = ()
    latency_ms: float = Field(ge=0)
    input_tokens: int | None = None
    output_tokens: int | None = None
    error_category: Failure | None = None


class FinalResponse(Contract):
    project_id: str
    package_fingerprint: str
    disposition: Disposition
    published_claims: tuple[CandidateClaim, ...] = ()
    limitations: tuple[str, ...] = ()
    failures: tuple[Failure, ...] = ()
    mechanical_validity: Literal["VALID", "INVALID"]
    semantic_support: Literal["MODEL_ASSESSED", "NOT_ASSESSED"]
    cross_source_atomicity: Literal["NOT_ESTABLISHED"] = "NOT_ESTABLISHED"


class SynthesisReport(Contract):
    final: FinalResponse
    budget: BudgetLedger
    events: tuple[ModelEvent, ...]
    synthesis: SynthesisOutput | None = None
    critic: CriticOutput | None = None
