"""Stage 6: deterministic route selection (Phase 9B).

Priority order (first matching row wins):

  1  input empty / too long / no text           -> REFUSE   INVALID_REQUEST
  2  project MISSING                             -> CLARIFY  PROJECT_REQUIRED
  2b relative project reference not resolvable   -> CLARIFY  AMBIGUOUS_PROJECT_REFERENCE
  3  project UNSUPPORTED (id, loan, report, scope)-> REFUSE   UNSUPPORTED_PROJECT
  4  project CONFLICT (foreign id/name/loan/doc) -> REFUSE   CROSS_PROJECT
  5  project MULTI                               -> REFUSE   MULTI_PROJECT_NOT_SUPPORTED
  6  project NOT_AUTHORIZED                      -> REFUSE   NOT_AUTHORIZED
  7  intent OUT_OF_DOMAIN / WRITE_REQUEST        -> REFUSE   OUT_OF_DOMAIN / READ_ONLY
  8  intent PREDICTION_REQUEST                   -> REFUSE   PREDICTION_NOT_SUPPORTED
  9  intent not resolved by rules                -> SEMANTIC_CLASSIFICATION_REQUIRED
 10  time UNRESOLVED / required but absent / not supported by the tools -> CLARIFY
 11  requirements route                          -> STRUCTURED | DOCUMENT | INVESTIGATION

Rows 1-6 are decided before any tool or retrieval call can run.
"""

from __future__ import annotations

from worldbank_copilot.routing.models import (
    InputStatus,
    Intent,
    IntentStatus,
    ProjectStatus,
    QueryUnderstanding,
    Route,
    RoutingDecision,
    TemporalStatus,
)

PROJECT_REFUSALS = {
    ProjectStatus.UNSUPPORTED: "UNSUPPORTED_PROJECT",
    ProjectStatus.CONFLICT: "CROSS_PROJECT",
    ProjectStatus.MULTI: "MULTI_PROJECT_NOT_SUPPORTED",
    ProjectStatus.NOT_AUTHORIZED: "NOT_AUTHORIZED",
}
INTENT_REFUSALS = {
    Intent.OUT_OF_DOMAIN: ("OUT_OF_DOMAIN", None),
    Intent.WRITE_REQUEST: ("READ_ONLY", None),
    Intent.PREDICTION_REQUEST: (
        "PREDICTION_NOT_SUPPORTED",
        "The copilot does not predict success or failure. It can show observable, "
        "rule-based attention signals (get_attention_signals).",
    ),
}


def decide_scope(u: QueryUnderstanding) -> RoutingDecision | None:
    """Rows 1-6: input and project scope. None = proceed."""
    if u.input.status != InputStatus.VALID:
        return RoutingDecision(
            route=Route.REFUSE,
            reason_code="INVALID_REQUEST",
            detail=f"input {u.input.status.value}",
            decided_at_stage="input",
        )
    p = u.project
    if p is None:  # input stage only
        return None
    if p.status == ProjectStatus.MISSING:
        return RoutingDecision(
            route=Route.CLARIFY,
            reason_code="PROJECT_REQUIRED",
            detail=p.detail,
            decided_at_stage="project",
        )
    if p.status == ProjectStatus.AMBIGUOUS_REFERENCE:
        return RoutingDecision(
            route=Route.CLARIFY,
            reason_code="AMBIGUOUS_PROJECT_REFERENCE",
            detail=p.detail,
            decided_at_stage="project",
        )
    if p.status in PROJECT_REFUSALS:
        return RoutingDecision(
            route=Route.REFUSE,
            reason_code=PROJECT_REFUSALS[p.status],
            detail=p.detail,
            decided_at_stage="project",
        )
    return None


def decide_intent(u: QueryUnderstanding) -> RoutingDecision | None:
    """Rows 7-9. None = proceed."""
    i = u.intent
    assert i is not None
    if i.intent in INTENT_REFUSALS:
        code, suggestion = INTENT_REFUSALS[i.intent]
        return RoutingDecision(
            route=Route.REFUSE,
            reason_code=code,
            detail=i.reason,
            decided_at_stage="intent",
            suggestion=suggestion,
        )
    if i.status == IntentStatus.SEMANTIC_CLASSIFICATION_REQUIRED:
        return RoutingDecision(
            route=Route.SEMANTIC_CLASSIFICATION_REQUIRED,
            reason_code="INTENT_NOT_RESOLVED_BY_RULES",
            detail=i.reason,
            decided_at_stage="intent",
            alternatives_rejected=("forced deterministic default intent",),
        )
    return None


def decide_route(u: QueryUnderstanding, *, explicit_time_missing: bool) -> RoutingDecision:
    """Rows 10-11 (after intent, temporal and requirements)."""
    t, r, i = u.temporal, u.requirements, u.intent
    assert t is not None and i is not None
    if t.status == TemporalStatus.UNRESOLVED:
        options = tuple(f"{c.title} ({c.event_date.isoformat()})" for c in t.anchor_candidates)
        return RoutingDecision(
            route=Route.CLARIFY,
            reason_code="AMBIGUOUS_TIME",
            detail=t.detail or "time not resolved",
            decided_at_stage="temporal",
            clarification_options=options,
        )
    if explicit_time_missing:
        return RoutingDecision(
            route=Route.CLARIFY,
            reason_code="TIME_REQUIRED",
            detail="an overall risk rating needs a time: latest ISR or appraisal",
            decided_at_stage="temporal",
            clarification_options=("latest ISR", "at appraisal"),
        )
    assert r is not None
    if not r.temporal_supported:
        return RoutingDecision(
            route=Route.CLARIFY,
            reason_code="TEMPORAL_NOT_SUPPORTED",
            detail=f"{t.kind.value} cannot be applied exactly by the tools for {i.intent.value}",
            decided_at_stage="requirements",
        )
    route = Route(r.route)
    rejected = {
        Route.STRUCTURED: ("DOCUMENT", "INVESTIGATION"),
        Route.DOCUMENT: ("STRUCTURED", "INVESTIGATION"),
        Route.INVESTIGATION: ("STRUCTURED", "DOCUMENT"),
    }[route]
    return RoutingDecision(
        route=route,
        reason_code=f"INTENT_{i.intent.value}",
        detail=i.reason,
        decided_at_stage="requirements",
        alternatives_rejected=rejected,
    )
