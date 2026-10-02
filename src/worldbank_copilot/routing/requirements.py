"""Stage 5: information requirements - intent + time + entities -> tool calls (Phase 9B).

The intent -> route / tools / default time / supported time mapping is declarative
(configs/routing/requirements.yaml). This module only turns it into typed tool
arguments; every argument set is validated against the tool's own Pydantic model
(dry run, no read). A temporal scope a tool cannot honour exactly is reported
(``temporal_supported=False``) and becomes CLARIFY - it is never silently ignored.
"""

from __future__ import annotations

import re
from typing import Any

from worldbank_copilot.routing.config import RequirementsConfig
from worldbank_copilot.routing.models import (
    InformationRequirements,
    Intent,
    IntentDecision,
    MentionKind,
    PlannedToolCall,
    ProjectResolution,
    RetrievalSpec,
    TemporalKind,
    TemporalScope,
    ToolCallSpec,
)
from worldbank_copilot.routing.temporal import with_default
from worldbank_copilot.tools.registry import TOOL_SPECS

SPECS = {s.name: s for s in TOOL_SPECS}
EVENT_WORDS = {
    r"\brestructur\w*": "RESTRUCTURING",
    r"\badditional financing\b": "ADDITIONAL_FINANCING",
    r"\bcancel\w*": "CANCELLATION",
    r"\bclosing[- ]date\b|\bextend\w*|\bextension\b": "CLOSING_DATE_CHANGE",
    r"\bapprov\w*": "APPROVAL",
    r"\beffective\w*": "EFFECTIVENESS",
}


def effective_kind(scope: TemporalScope) -> TemporalKind:
    """A resolved event anchor is a date range for tools."""
    return TemporalKind.DATE_RANGE if scope.kind == TemporalKind.EVENT_ANCHORED else scope.kind


def validate_call(tool: str, arguments: dict[str, Any]) -> str | None:
    """Dry-run argument validation against the tool's own model (no read)."""
    try:
        SPECS[tool].args_model.model_validate(arguments)
    except Exception as exc:  # returned as data, never swallowed
        return f"{type(exc).__name__}: {exc}"
    return None


def scoped_temporal(
    intent: Intent, scope: TemporalScope, config: RequirementsConfig
) -> TemporalScope:
    return with_default(scope, config.intents[intent].default_time)


def needs_explicit_time(
    decision: IntentDecision, scope: TemporalScope, config: RequirementsConfig
) -> bool:
    if decision.intent != Intent.CURRENT_RATINGS or scope.explicit:
        return False
    required = {r.get("rating_type") for r in config.explicit_time_required}
    return bool(required & set(decision.rating_types))


def _sequences_calls(base: dict[str, Any], scope: TemporalScope, key: str) -> list[dict[str, Any]]:
    return [{**base, key: s} for s in scope.isr_sequences] or [base]


def build_requirements(
    project_id: str,
    question: str,
    decision: IntentDecision,
    scope: TemporalScope,
    resolution: ProjectResolution,
    config: RequirementsConfig,
) -> InformationRequirements:
    intent = decision.intent
    assert intent is not None
    req = config.intents[intent]
    kind = effective_kind(scope)
    supported = kind in req.supported_time
    lowered = question.lower()
    base = {"project_id": project_id}
    calls: list[dict[str, Any]] = []
    notes: list[str] = []
    tool = req.tools[0]
    if req.route == "DOCUMENT":
        return InformationRequirements(
            route="DOCUMENT",
            document=RetrievalSpec(
                query=question,
                document_type_hints=decision.document_types,
                purpose=f"{intent.value}: documentary evidence",
            ),
            temporal_supported=supported,
            notes=("document-type hints are recorded, not applied as filters (Phase 8 decision)",)
            if decision.document_types
            else (),
        )
    if req.route == "INVESTIGATION":
        return InformationRequirements(route="INVESTIGATION", temporal_supported=supported)
    if intent in (Intent.PROJECT_OVERVIEW, Intent.CURRENT_RATINGS):
        calls = [base]
    elif intent == Intent.RATING_HISTORY:
        args = {
            **base,
            "rating_types": list(decision.rating_types or ("PDO", "IP", "OVERALL_RISK")),
        }
        if kind == TemporalKind.ISR_SEQUENCE:
            calls = [
                {**args, "isr_sequence_from": s, "isr_sequence_to": s} for s in scope.isr_sequences
            ]
        elif kind == TemporalKind.ISR_RANGE:
            calls = [
                {
                    **args,
                    "isr_sequence_from": scope.isr_sequences[0],
                    "isr_sequence_to": scope.isr_sequences[-1],
                }
            ]
        else:
            calls = [args]
    elif intent == Intent.TIMELINE_EVENTS:
        args = dict(base)
        events = sorted({v for p, v in EVENT_WORDS.items() if re.search(p, lowered)})
        if events:
            args["event_types"] = events
        if kind in (TemporalKind.ISR_SEQUENCE, TemporalKind.ISR_RANGE):
            args["isr_sequence_from"], args["isr_sequence_to"] = (
                scope.isr_sequences[0],
                scope.isr_sequences[-1],
            )
        if kind in (TemporalKind.YEAR, TemporalKind.DATE, TemporalKind.DATE_RANGE):
            if scope.date_from:
                args["date_from"] = scope.date_from.isoformat()
            if scope.date_to:
                args["date_to"] = scope.date_to.isoformat()
        calls = [args]
    elif intent == Intent.FINANCIAL_STATUS:
        args = dict(base)
        loans = [m.normalized for m in resolution.mentions if m.kind == MentionKind.LOAN_NUMBER]
        if loans:
            if len(set(loans)) > 1:
                notes.append("several loans mentioned; one call per loan")
            args["loan_number"] = loans[0]
        if re.search(r"\badditional financing\b|\bcancel\w*", lowered):
            args["include_events"] = True
        if kind == TemporalKind.ISR_SEQUENCE:
            calls = _sequences_calls(args, scope, "as_of_isr")
        elif kind == TemporalKind.LATEST and re.search(r"\bisr\b", lowered):
            calls = [{**args, "as_of_isr": "latest"}]
        else:
            calls = [args]
        if len(set(loans)) > 1:
            calls = [{**c, "loan_number": loan} for c in calls for loan in sorted(set(loans))]
    elif intent == Intent.RESULTS_PROGRESS:
        args = dict(base)
        if re.search(r"\bpdo indicators?\b", lowered):
            args["indicator_type"] = "PDO"
        elif re.search(r"\bintermediate\b", lowered):
            args["indicator_type"] = "INTERMEDIATE"
        if kind == TemporalKind.ISR_SEQUENCE:
            calls = _sequences_calls(args, scope, "isr_sequence")
        elif kind == TemporalKind.HISTORY:
            calls = [{**args, "history": True}]
        else:
            calls = [{**args, "isr_sequence": "latest"}]
        notes.append(
            "indicator selection is not resolved deterministically in 9B; all "
            "indicators matching the type/time filters are returned"
        )
    elif intent == Intent.RISKS:
        args = dict(base)
        if kind == TemporalKind.APPRAISAL:
            args["record_type"] = (
                "ASSESSMENT_FINDING"
                if re.search(r"\bfindings?\b", lowered)
                else "FORMAL_RISK_RATING"
            )
        elif kind == TemporalKind.LATEST:
            args["record_type"] = "ISR_SORT_RATING"
        calls = [args]
    elif intent == Intent.ATTENTION:
        calls = [{**base, "status": "ALL" if kind == TemporalKind.HISTORY else "CURRENT"}]
    specs = []
    for arguments in calls:
        error = validate_call(tool, arguments)
        if error:
            raise ValueError(f"requirement builder produced invalid {tool} arguments: {error}")
        specs.append(ToolCallSpec(tool=tool, arguments=arguments, purpose=intent.value))
    return InformationRequirements(
        route="STRUCTURED",
        structured=tuple(specs),
        temporal_supported=supported,
        notes=tuple(notes),
    )


def build_investigation_calls(
    project_id: str, decision: IntentDecision, config: RequirementsConfig
) -> list[PlannedToolCall]:
    kinds = (
        [decision.investigation]
        if decision.investigation in config.investigation_plans
        else ((decision.investigation or "").split("+"))
    )
    planned: list[PlannedToolCall] = []
    seen: set[str] = set()
    for kind in kinds:
        for call in config.investigation_plans.get(kind, ()):
            arguments = {"project_id": project_id, **call.args}
            key = f"{call.tool}:{sorted(arguments.items())}"
            if key in seen:
                continue
            seen.add(key)
            error = validate_call(call.tool, arguments)
            planned.append(
                PlannedToolCall(
                    tool=call.tool,
                    arguments=arguments,
                    purpose=f"establish {kind}",
                    validated=error is None,
                    validation_error=error,
                )
            )
    return planned
