"""Typed query-understanding state and routing decision (Phase 9B).

Every stage writes an explicit, serialisable record, so a routing decision can be
explained from harness state alone (no transcript, no model output). Nothing here holds
credentials: ``AccessContext`` carries only an opaque user reference and project scope.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from worldbank_copilot.common.project_registry import validate_project_id_format
from worldbank_copilot.tools.models import MechanicalFinding, ToolResult


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AccessContext(_Model):
    """Who asks and which projects they may see. No tokens or secrets, by construction."""

    user_ref: str | None = None  # opaque identifier (never a credential)
    authorized_projects: tuple[str, ...]
    active_project_id: str | None = None

    @field_validator("authorized_projects")
    @classmethod
    def _ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(validate_project_id_format(v) for v in value)

    @field_validator("active_project_id")
    @classmethod
    def _active(cls, value: str | None) -> str | None:
        return None if value is None or not value.strip() else value.strip().upper()


# -- 1 input ------------------------------------------------------------------------------


class InputStatus(StrEnum):
    VALID = "VALID"
    EMPTY = "EMPTY"
    TOO_LONG = "TOO_LONG"
    NO_TEXT = "NO_TEXT"


class InputValidation(_Model):
    status: InputStatus
    normalized: str
    length: int
    flags: tuple[str, ...] = ()  # e.g. INJECTION_PATTERN (recorded, never routed on)


# -- 2 project / entity resolution ----------------------------------------------------------


class MentionKind(StrEnum):
    PROJECT_ID = "PROJECT_ID"
    DOCUMENT_ID = "DOCUMENT_ID"
    REPORT_NUMBER = "REPORT_NUMBER"
    LOAN_NUMBER = "LOAN_NUMBER"
    ALIAS = "ALIAS"
    RELATIVE_REFERENCE = "RELATIVE_REFERENCE"  # "the other program": no project by itself


class Mention(_Model):
    kind: MentionKind
    span: tuple[int, int]
    text: str
    normalized: str
    project_id: str | None  # owning project; None = unknown entity
    basis: str  # registry / manifest / alias config / document-id prefix


class ProjectStatus(StrEnum):
    RESOLVED = "RESOLVED"
    MISSING = "MISSING"
    CONFLICT = "CONFLICT"
    UNSUPPORTED = "UNSUPPORTED"
    MULTI = "MULTI"
    NOT_AUTHORIZED = "NOT_AUTHORIZED"
    AMBIGUOUS_REFERENCE = "AMBIGUOUS_REFERENCE"  # unresolved relative project reference


class ProjectResolution(_Model):
    status: ProjectStatus
    project_id: str | None
    basis: str | None  # ACTIVE_SCOPE | PROJECT_ID | DOCUMENT_ID | REPORT_NUMBER | ...
    active_project_id: str | None
    mentions: tuple[Mention, ...]
    mentioned_projects: tuple[str, ...]
    detail: str


# -- 3 temporal ---------------------------------------------------------------------------


class TemporalKind(StrEnum):
    NONE = "NONE"
    LATEST = "LATEST"
    ISR_SEQUENCE = "ISR_SEQUENCE"
    ISR_RANGE = "ISR_RANGE"
    APPRAISAL = "APPRAISAL"
    YEAR = "YEAR"
    DATE = "DATE"
    DATE_RANGE = "DATE_RANGE"
    HISTORY = "HISTORY"
    EVENT_ANCHORED = "EVENT_ANCHORED"
    RELATIVE = "RELATIVE"


class TemporalStatus(StrEnum):
    RESOLVED = "RESOLVED"
    DEFAULTED = "DEFAULTED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNRESOLVED = "UNRESOLVED"  # ambiguous anchor, relative period, conflict, not found


class TemporalExpression(_Model):
    kind: TemporalKind
    span: tuple[int, int]
    text: str
    isr_sequences: tuple[int, ...] = ()
    date_from: date | None = None
    date_to: date | None = None
    anchor_event_type: str | None = None
    anchor_qualifier: str | None = None
    anchor_direction: Literal["since", "after", "before", "until"] | None = None


class AnchorCandidate(_Model):
    timeline_event_id: str
    event_type: str
    event_date: date
    title: str


class TemporalScope(_Model):
    kind: TemporalKind
    status: TemporalStatus
    explicit: bool
    defaulted: bool
    expressions: tuple[TemporalExpression, ...] = ()
    isr_sequences: tuple[int, ...] = ()
    date_from: date | None = None
    date_to: date | None = None
    reference: str = "governed latest data snapshot (never the wall clock)"
    anchor_candidates: tuple[AnchorCandidate, ...] = ()
    detail: str | None = None


# -- 4 intent -----------------------------------------------------------------------------


class Intent(StrEnum):
    PROJECT_OVERVIEW = "PROJECT_OVERVIEW"
    CURRENT_RATINGS = "CURRENT_RATINGS"
    RATING_HISTORY = "RATING_HISTORY"
    TIMELINE_EVENTS = "TIMELINE_EVENTS"
    FINANCIAL_STATUS = "FINANCIAL_STATUS"
    RESULTS_PROGRESS = "RESULTS_PROGRESS"
    RISKS = "RISKS"
    ATTENTION = "ATTENTION"
    DOCUMENT_CONTENT = "DOCUMENT_CONTENT"
    EXPLANATION = "EXPLANATION"
    CHANGE_INVESTIGATION = "CHANGE_INVESTIGATION"
    OUT_OF_DOMAIN = "OUT_OF_DOMAIN"
    PREDICTION_REQUEST = "PREDICTION_REQUEST"
    WRITE_REQUEST = "WRITE_REQUEST"


REFUSAL_INTENTS = frozenset({Intent.OUT_OF_DOMAIN, Intent.PREDICTION_REQUEST, Intent.WRITE_REQUEST})


class IntentStatus(StrEnum):
    RESOLVED = "RESOLVED"
    SEMANTIC_CLASSIFICATION_REQUIRED = "SEMANTIC_CLASSIFICATION_REQUIRED"


class RuleHit(_Model):
    rule_id: str
    group: str
    precedence: int
    span: tuple[int, int]
    text: str
    pattern_index: int
    intent: str | None = None
    subject: str | None = None
    rating_type: str | None = None
    document_type: str | None = None
    investigation: str | None = None


class IntentDecision(_Model):
    status: IntentStatus
    intent: Intent | None
    method: Literal["RULE", "NONE", "SEMANTIC"]
    decided_by: tuple[str, ...]  # rule ids (or the semantic classifier) that decided
    reason: str
    hits: tuple[RuleHit, ...]
    suppressed_hits: tuple[RuleHit, ...] = ()  # overlapped by a longer match
    subjects: tuple[str, ...] = ()
    rating_types: tuple[str, ...] = ()
    document_types: tuple[str, ...] = ()
    explanatory: bool = False
    document_scoped: bool = False
    investigation: str | None = None
    rules_version: int


class SemanticNeighbour(_Model):
    example_id: str  # a DEVELOPMENT case id (never a test case)
    intent: str
    similarity: float


class SemanticDecision(_Model):
    """Bounded output of the semantic fallback (Phase 9D). Never a project, tool or answer."""

    classifier: str
    version: str
    abstain: bool
    intent: Intent | None = None
    route: Literal["STRUCTURED", "DOCUMENT", "INVESTIGATION"] | None = None
    confidence: float = 0.0
    reason: str
    neighbours: tuple[SemanticNeighbour, ...] = ()
    latency_ms: float = 0.0


# -- 5 requirements -----------------------------------------------------------------------


class ToolCallSpec(_Model):
    tool: str
    arguments: dict[str, Any]
    purpose: str


class RetrievalSpec(_Model):
    query: str
    document_type_hints: tuple[str, ...] = ()  # recorded; not applied (Phase 8 decision)
    purpose: str


class InformationRequirements(_Model):
    route: Literal["STRUCTURED", "DOCUMENT", "INVESTIGATION"]
    structured: tuple[ToolCallSpec, ...] = ()
    document: RetrievalSpec | None = None
    temporal_supported: bool = True
    notes: tuple[str, ...] = ()


# -- 6 routing ----------------------------------------------------------------------------


class Route(StrEnum):
    STRUCTURED = "STRUCTURED"
    DOCUMENT = "DOCUMENT"
    INVESTIGATION = "INVESTIGATION"
    CLARIFY = "CLARIFY"
    REFUSE = "REFUSE"
    SEMANTIC_CLASSIFICATION_REQUIRED = "SEMANTIC_CLASSIFICATION_REQUIRED"


class RoutingDecision(_Model):
    route: Route
    reason_code: str
    detail: str
    decided_at_stage: str
    alternatives_rejected: tuple[str, ...] = ()
    clarification_options: tuple[str, ...] = ()
    suggestion: str | None = None


# -- 8 investigation plan --------------------------------------------------------------------


class PlannedToolCall(_Model):
    tool: str
    arguments: dict[str, Any]
    purpose: str
    validated: bool
    validation_error: str | None = None


class InvestigationPlan(_Model):
    project_id: str
    question: str
    investigation_kind: str  # explicit investigation wording or the structured subject(s)
    triggered_by: tuple[str, ...]
    temporal: TemporalScope
    structured_calls: tuple[PlannedToolCall, ...]
    document_retrievals: tuple[RetrievalSpec, ...]
    executed: Literal[False] = False
    notes: tuple[str, ...] = (
        "Plan only: no tool or retrieval call was executed. Execution, evidence "
        "assessment and synthesis are Phase 10.",
    )


# -- result -------------------------------------------------------------------------------


class ExecutionOutcome(StrEnum):
    EXECUTED = "EXECUTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"  # mechanical only
    DATA_INTEGRITY_ERROR = "DATA_INTEGRITY_ERROR"
    ERROR = "ERROR"
    PLANNED_NOT_EXECUTED = "PLANNED_NOT_EXECUTED"
    NOT_EXECUTED = "NOT_EXECUTED"  # CLARIFY / REFUSE / semantic classification required


class StageTiming(_Model):
    stage: str
    ms: float


class QueryUnderstanding(_Model):
    input: InputValidation
    project: ProjectResolution | None = None
    temporal: TemporalScope | None = None
    intent: IntentDecision | None = None
    requirements: InformationRequirements | None = None


class RouteResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    question: str
    access: AccessContext
    understanding: QueryUnderstanding
    decision: RoutingDecision
    outcome: ExecutionOutcome
    tool_results: list[ToolResult] = Field(default_factory=list)  # answer data calls
    executed_tools: list[str] = Field(default_factory=list)
    # reads made only to resolve an event anchor in time (after the scope check)
    anchor_results: list[ToolResult] = Field(default_factory=list)
    retrieval_executed: bool = False
    investigation_plan: InvestigationPlan | None = None
    mechanical: list[MechanicalFinding] = Field(default_factory=list)
    semantic_sufficiency: Literal["NOT_ASSESSED"] = "NOT_ASSESSED"
    timings: list[StageTiming] = Field(default_factory=list)
    versions: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
    semantic: SemanticDecision | None = None  # only when the fallback was invoked
