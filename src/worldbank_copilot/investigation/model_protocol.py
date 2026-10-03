"""Proposal-only Investigator interface. Production model choice is unresolved.

All free text is untrusted data, including objective/rationale/query strings. It
is never interpreted as control instructions, SQL, code or graph transitions.
The protocol has no executor/client/credential argument or execution capability.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Protocol

from pydantic import Field, model_validator

from worldbank_copilot.investigation.models import PurposeCode
from worldbank_copilot.investigation.policy import Contract
from worldbank_copilot.routing.models import RetrievalSpec, TemporalScope
from worldbank_copilot.tools.registry import TOOL_SPECS


class StructuredTool(StrEnum):
    OVERVIEW = "get_project_overview"
    TIMELINE = "get_project_timeline"
    RATINGS = "get_rating_history"
    FINANCE = "get_financial_status"
    RESULTS = "get_results_progress"
    RISKS = "get_risk_register"
    SIGNALS = "get_attention_signals"


class ProposedCall(Contract):
    tool: StructuredTool
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def approved_keys(self):
        spec = next(s for s in TOOL_SPECS if s.name == self.tool)
        allowed = set(spec.args_model.model_fields) - {"project_id"}
        if set(self.arguments) - allowed:
            raise ValueError("unapproved or harness-owned tool arguments")
        return self


class ProposedRequirement(Contract):
    objective: str = Field(min_length=1, max_length=300)
    temporal_scope: TemporalScope | None = None
    structured_call: ProposedCall | None = None
    document_retrieval: RetrievalSpec | None = None
    required: bool = True
    parent_requirement_id: str | None = Field(default=None, max_length=100)
    purpose_code: PurposeCode = PurposeCode.ESTABLISH
    rationale: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def bounded(self):
        if self.structured_call is None and self.document_retrieval is None:
            raise ValueError("requirement needs an operation")
        if self.document_retrieval is not None:
            d = self.document_retrieval
            if not 1 <= len(d.query.strip()) <= 1000 or not 1 <= len(d.purpose) <= 240:
                raise ValueError("unbounded retrieval text")
            if len(d.document_type_hints) > 10 or any(len(s) > 100 for s in d.document_type_hints):
                raise ValueError("unbounded retrieval hints")
        return self


class PlanningProposal(Contract):
    requirements: tuple[ProposedRequirement, ...] = Field(default=(), max_length=100)
    clarification: str | None = Field(default=None, min_length=1, max_length=300)
    decision_summary: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def shape(self):
        if bool(self.requirements) == bool(self.clarification):
            raise ValueError("provide requirements or clarification, exclusively")
        return self


class UntrustedText(Contract):
    text: str = Field(min_length=1, max_length=1000)
    trust: Literal["UNTRUSTED_DATA"] = "UNTRUSTED_DATA"


class RefinementReason(StrEnum):
    MISSING_OPERATION = "MISSING_OPERATION"
    DECOMPOSITION = "DECOMPOSITION"
    TEMPORAL_COMPARISON = "TEMPORAL_COMPARISON"


class RefinementNeed(Contract):
    """Explicit trusted harness decision, never inferred to increase model usage."""

    reason: RefinementReason
    objective: str = Field(min_length=1, max_length=240)


class PlanningContext(Contract):
    question: UntrustedText
    seed_objectives: tuple[UntrustedText, ...] = Field(max_length=100)
    requirement_ids: tuple[str, ...] = Field(max_length=100)
    temporal_scope: TemporalScope
    need: RefinementNeed
    max_additional_requirements: int = Field(ge=0)
    max_additional_operations: int = Field(ge=0)
    max_additional_retrievals: int = Field(ge=0)
    token_bound: int = Field(ge=1)


class InvestigatorModel(Protocol):
    def propose(self, context: PlanningContext) -> PlanningProposal:
        """Return bounded proposed data; do not execute evidence operations."""
        ...
