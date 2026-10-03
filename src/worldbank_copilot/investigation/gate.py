"""Server-side admission of trusted Phase 9 results; never an HTTP/model input gate.

AdmissionContext must be supplied by trusted application wiring, not a proposal.
Request/auth binding and deterministic replay detect inconsistent historical state;
this is not cryptographic authentication of a caller-controlled RouteResult.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum
from pathlib import Path

from pydantic import ValidationError

from worldbank_copilot.investigation.models import (
    BASELINE_ID,
    InvestigationState,
    historical_source,
)
from worldbank_copilot.investigation.policy import (
    BudgetLedger,
    Contract,
    InvestigationPolicy,
    Reservation,
)
from worldbank_copilot.retrieval.config import load_retrieval_settings
from worldbank_copilot.retrieval.contract import load_quality_baseline
from worldbank_copilot.routing.config import RoutingConfig
from worldbank_copilot.routing.entities import EntityIndex, resolve_project, validate_input
from worldbank_copilot.routing.execution import (
    ExecutionContractError,
    ExecutionDecision,
    ExecutionMode,
    execution_decision,
)
from worldbank_copilot.routing.intents import IntentEngine
from worldbank_copilot.routing.models import (
    PROVENANCE_REQUIREMENTS,
    AccessContext,
    AnchorCandidate,
    ExecutionOutcome,
    InputStatus,
    ProjectStatus,
    RetrievalSpec,
    RouteResult,
    TemporalKind,
    TemporalStatus,
)
from worldbank_copilot.routing.requirements import (
    build_investigation_calls,
    build_requirements,
    needs_explicit_time,
    scoped_temporal,
    validate_call,
)
from worldbank_copilot.routing.router import decide_route
from worldbank_copilot.routing.service import ORDINALS
from worldbank_copilot.routing.temporal import parse_temporal
from worldbank_copilot.tools.models import ToolStatus
from worldbank_copilot.tools.timeline import TimelineEvent


class AdmissionOutcome(StrEnum):
    ADMITTED = "ADMITTED"
    DISABLED = "DISABLED"
    CLARIFY = "CLARIFY"
    REFUSE = "REFUSE"
    FAIL = "FAIL"
    NOT_INVESTIGATION = "NOT_INVESTIGATION"


class AdmissionReason(StrEnum):
    VALID_PLAN = "VALID_PLAN"
    FEATURE_DISABLED = "FEATURE_DISABLED"
    PHASE9_CLARIFY = "PHASE9_CLARIFY"
    PHASE9_REFUSE = "PHASE9_REFUSE"
    OTHER_EXECUTION_MODE = "OTHER_EXECUTION_MODE"
    INVALID_SOURCE = "INVALID_SOURCE"
    REQUEST_MISMATCH = "REQUEST_MISMATCH"
    AUTHORIZATION_MISMATCH = "AUTHORIZATION_MISMATCH"
    PLAN_MISMATCH = "PLAN_MISMATCH"
    TEMPORAL_MISMATCH = "TEMPORAL_MISMATCH"
    UNSUPPORTED_KIND = "UNSUPPORTED_KIND"
    INVALID_CALL = "INVALID_CALL"
    PREVIOUS_EXECUTION = "PREVIOUS_EXECUTION"
    PROFILE_MISMATCH = "PROFILE_MISMATCH"
    POLICY_BOUNDS = "POLICY_BOUNDS"


class AdmissionResult(Contract):
    outcome: AdmissionOutcome
    reason: AdmissionReason
    state: InvestigationState | None = None


@dataclass(frozen=True)
class AdmissionContext:
    request_id: str
    question: str
    access: AccessContext
    config: RoutingConfig
    entities: EntityIndex
    config_dir: Path
    policy: InvestigationPolicy
    feature_enabled: bool = False


def _anchor_matches(source, parsed):
    if len(source.anchor_results) != 1:
        return False
    result = source.anchor_results[0]
    scope = source.understanding.temporal
    expr = next(e for e in parsed.expressions if e.kind == TemporalKind.EVENT_ANCHORED)
    if (
        result.tool != "get_project_timeline"
        or result.project_id != source.investigation_plan.project_id
        or result.request_id != source.request_id
        or result.status != ToolStatus.OK
        or result.filters.get("event_types") != [expr.anchor_event_type]
    ):
        return False
    try:
        events = [TimelineEvent.model_validate(e) for e in result.items]
    except ValidationError:
        return False
    if any(e.event_type != expr.anchor_event_type for e in events):
        return False
    events = [e for e in events if e.event_date is not None]
    qualifier = expr.anchor_qualifier
    if qualifier and qualifier.isdigit():
        events = [e for e in events if e.event_date.year == int(qualifier)]
    elif qualifier in ORDINALS:
        index = ORDINALS[qualifier]
        events = events[index : index + 1]
    elif qualifier in ("last", "latest", "most recent"):
        events = events[-1:]
    if len(events) != 1:
        return False
    event = events[0]
    candidate = AnchorCandidate(
        timeline_event_id=event.source.record_id or str(event.event_sequence),
        event_type=event.event_type,
        event_date=event.event_date,
        title=event.event_title,
    )
    lo = hi = None
    if expr.anchor_direction == "since":
        lo = event.event_date
    elif expr.anchor_direction == "after":
        lo = event.event_date + timedelta(days=1)
    elif expr.anchor_direction == "before":
        hi = event.event_date - timedelta(days=1)
    else:
        hi = event.event_date
    expected = parsed.model_copy(
        update={
            "status": TemporalStatus.RESOLVED,
            "date_from": lo,
            "date_to": hi,
            "anchor_candidates": (candidate,),
            "detail": (
                f"anchored on {event.event_title} ({event.event_date.isoformat()}, source-stated)"
            ),
        }
    )
    return scope == expected


def admit(
    result: RouteResult, context: AdmissionContext, decision: ExecutionDecision | None = None
) -> AdmissionResult:
    def reject(reason):
        return AdmissionResult(outcome=AdmissionOutcome.FAIL, reason=reason)

    try:
        # Revalidate nested models, preserving the concrete recorded anchor schema.
        source = historical_source(result)
        policy = InvestigationPolicy.model_validate(context.policy)
        execution = execution_decision(source)
    except (ValidationError, ExecutionContractError, ValueError, TypeError, KeyError):
        return reject(AdmissionReason.INVALID_SOURCE)
    if source.versions != context.config.versions:
        return reject(AdmissionReason.INVALID_SOURCE)
    if source.request_id != context.request_id or source.question != context.question:
        return reject(AdmissionReason.REQUEST_MISMATCH)
    if source.access != context.access:
        return reject(AdmissionReason.AUTHORIZATION_MISMATCH)
    if decision is not None and decision != execution:
        return reject(AdmissionReason.INVALID_SOURCE)
    if execution.mode != ExecutionMode.PLAN_INVESTIGATION:
        outcome, reason = {
            ExecutionMode.CLARIFY: (AdmissionOutcome.CLARIFY, AdmissionReason.PHASE9_CLARIFY),
            ExecutionMode.REFUSE: (AdmissionOutcome.REFUSE, AdmissionReason.PHASE9_REFUSE),
        }.get(
            execution.mode,
            (AdmissionOutcome.NOT_INVESTIGATION, AdmissionReason.OTHER_EXECUTION_MODE),
        )
        return AdmissionResult(outcome=outcome, reason=reason)
    if (
        source.outcome != ExecutionOutcome.PLANNED_NOT_EXECUTED
        or source.executed_tools
        or source.tool_results
        or source.retrieval_executed
        or source.mechanical
        or source.error
    ):
        return reject(AdmissionReason.PREVIOUS_EXECUTION)
    u, plan = source.understanding, source.investigation_plan
    normalized = validate_input(context.question, context.config)
    if (
        normalized.status != InputStatus.VALID
        or normalized != u.input
        or plan.question != normalized.normalized
    ):
        return reject(AdmissionReason.REQUEST_MISMATCH)
    project = resolve_project(normalized.normalized, context.access, context.entities)
    if (
        project != u.project
        or project.status != ProjectStatus.RESOLVED
        or project.project_id not in context.access.authorized_projects
        or plan.project_id != project.project_id
    ):
        return reject(AdmissionReason.AUTHORIZATION_MISMATCH)
    parsed = parse_temporal(normalized.normalized)
    intent = IntentEngine(context.config.rules).decide(normalized.normalized, parsed)
    if intent != u.intent:
        return reject(AdmissionReason.PLAN_MISMATCH)
    kind = plan.investigation_kind
    supported = context.config.requirements.investigation_plans
    if not kind or (kind not in supported and not all(k in supported for k in kind.split("+"))):
        return reject(AdmissionReason.UNSUPPORTED_KIND)
    if kind != intent.investigation or plan.triggered_by != intent.decided_by:
        return reject(AdmissionReason.PLAN_MISMATCH)
    if plan.temporal != u.temporal or u.temporal.status == TemporalStatus.UNRESOLVED:
        return reject(AdmissionReason.TEMPORAL_MISMATCH)
    if parsed.kind == TemporalKind.EVENT_ANCHORED:
        # Revalidate the recorded source-dated anchor without repeating the read.
        if not _anchor_matches(source, parsed):
            return reject(AdmissionReason.TEMPORAL_MISMATCH)
    elif (
        source.anchor_results
        or scoped_temporal(intent.intent, parsed, context.config.requirements) != u.temporal
    ):
        return reject(AdmissionReason.TEMPORAL_MISMATCH)
    expected_requirements = build_requirements(
        project.project_id,
        normalized.normalized,
        intent,
        u.temporal,
        project,
        context.config.requirements,
    )
    expected_understanding = u.model_copy(update={"requirements": expected_requirements})
    if (
        expected_requirements != u.requirements
        or decide_route(
            expected_understanding,
            explicit_time_missing=needs_explicit_time(
                intent, u.temporal, context.config.requirements
            ),
        )
        != source.decision
    ):
        return reject(AdmissionReason.PLAN_MISMATCH)
    expected_calls = tuple(
        build_investigation_calls(project.project_id, intent, context.config.requirements)
    )
    if any(
        validate_call(c.tool, c.arguments) is not None or not c.validated or c.validation_error
        for c in plan.structured_calls
    ):
        return reject(AdmissionReason.INVALID_CALL)
    expected_document = RetrievalSpec(
        query=normalized.normalized,
        document_type_hints=intent.document_types,
        purpose="documentary evidence explaining the established change",
    )
    if (
        plan.structured_calls != expected_calls
        or plan.document_retrievals != (expected_document,)
        or plan.provenance_requirements != PROVENANCE_REQUIREMENTS
    ):
        return reject(AdmissionReason.PLAN_MISMATCH)
    if plan.document_retrievals:
        if plan.retrieval_profile != BASELINE_ID:
            return reject(AdmissionReason.PROFILE_MISMATCH)
        try:
            profile = load_quality_baseline(
                context.config_dir, load_retrieval_settings(context.config_dir)
            )
            if profile.profile_id != BASELINE_ID:
                return reject(AdmissionReason.PROFILE_MISMATCH)
        except Exception:
            return reject(AdmissionReason.PROFILE_MISMATCH)
    operations = len(plan.structured_calls) + len(plan.document_retrievals)
    if (
        operations > policy.max_initial_requirements
        or operations > policy.max_initial_operations
        or operations > policy.max_total_operations
        or len(plan.document_retrievals) > policy.max_retrieval_operations
    ):
        return reject(AdmissionReason.POLICY_BOUNDS)
    try:
        budget = BudgetLedger(policy=policy)
        if source.anchor_results:
            budget = budget.reserve(
                Reservation(
                    reservation_id="phase9_anchor_reads", operations=len(source.anchor_results)
                )
            )
        if not context.feature_enabled:
            return AdmissionResult(
                outcome=AdmissionOutcome.DISABLED, reason=AdmissionReason.FEATURE_DISABLED
            )
        state = InvestigationState(
            source=source,
            execution=execution,
            policy=policy,
            budget=budget,
            trace_reference=source.request_id,
        )
    except ValueError:
        return reject(AdmissionReason.POLICY_BOUNDS)
    return AdmissionResult(
        outcome=AdmissionOutcome.ADMITTED, reason=AdmissionReason.VALID_PLAN, state=state
    )
