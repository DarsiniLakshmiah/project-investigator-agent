"""Deterministic template planning and opt-in, proposal-only model refinement.

Template planning is the default. Only a trusted, explicit RefinementNeed can
consult a future model adapter. No heuristic decides that an agent is needed.
Broad HISTORY/NONE scopes allow contextual snapshots, without claiming that they
establish history or semantic sufficiency. Restrictive scopes require real tool
arguments/query filters; recorded retrieval hints are never treated as filters.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from pydantic import ValidationError

from worldbank_copilot.investigation.gate import AdmissionContext, AdmissionOutcome, admit
from worldbank_copilot.investigation.model_protocol import (
    InvestigatorModel,
    PlanningContext,
    PlanningProposal,
    ProposedCall,
    ProposedRequirement,
    RefinementNeed,
    UntrustedText,
)
from worldbank_copilot.investigation.models import (
    BASELINE_ID,
    EvidenceRequirement,
    InvestigationState,
    InvestigationStatus,
    PurposeCode,
    fingerprint,
    requirement_id,
)
from worldbank_copilot.investigation.policy import AttemptStatus, Consumption, Contract, Reservation
from worldbank_copilot.retrieval.config import load_retrieval_settings
from worldbank_copilot.retrieval.query import process_query
from worldbank_copilot.routing.entities import resolve_project
from worldbank_copilot.routing.models import (
    AccessContext,
    ProjectStatus,
    TemporalKind,
    TemporalStatus,
    ToolCallSpec,
)
from worldbank_copilot.routing.requirements import effective_kind
from worldbank_copilot.tools.registry import TOOL_SPECS

SPECS = {s.name: s for s in TOOL_SPECS}


class PlanningOutcome(StrEnum):
    PLANNED = "PLANNED"
    CLARIFY = "CLARIFY"
    FAIL = "FAIL"


class PlanningReason(StrEnum):
    TEMPLATE = "TEMPLATE"
    VALID_PROPOSAL = "VALID_PROPOSAL"
    MODEL_CLARIFICATION = "MODEL_CLARIFICATION"
    INVALID_PROPOSAL = "INVALID_PROPOSAL"
    SCOPE_MISMATCH = "SCOPE_MISMATCH"
    TEMPORAL_NOT_ENFORCEABLE = "TEMPORAL_NOT_ENFORCEABLE"
    POLICY_BOUNDS = "POLICY_BOUNDS"
    MODEL_ERROR = "MODEL_ERROR"
    CONTEXT_BOUNDS = "CONTEXT_BOUNDS"
    INVALID_STATE = "INVALID_STATE"


class PlanningResult(Contract):
    outcome: PlanningOutcome
    reason: PlanningReason
    state: InvestigationState
    clarification: str | None = None


class PlanningValidationError(ValueError):
    def __init__(self, reason: PlanningReason):
        super().__init__(reason.value)
        self.reason = reason


def _fail(state, reason, *, clarification=None):
    outcome = (
        PlanningOutcome.CLARIFY
        if reason in (PlanningReason.TEMPORAL_NOT_ENFORCEABLE, PlanningReason.MODEL_CLARIFICATION)
        else PlanningOutcome.FAIL
    )
    status = (
        InvestigationStatus.CLARIFY
        if outcome == PlanningOutcome.CLARIFY
        else InvestigationStatus.FAIL
    )
    failed = state.transition(status=status, failure_reasons=(reason.value,))
    return PlanningResult(outcome=outcome, reason=reason, state=failed, clarification=clarification)


def _scope_text(text, state, context):
    access = AccessContext(
        user_ref=state.access.user_ref,
        authorized_projects=state.access.authorized_projects,
        active_project_id=state.project_id,
    )
    resolution = resolve_project(text, access, context.entities)
    if resolution.status != ProjectStatus.RESOLVED or resolution.project_id != state.project_id:
        raise PlanningValidationError(PlanningReason.SCOPE_MISMATCH)


def _temporal(proposed, state, call, document, context):
    source = state.temporal_scope
    scope = proposed.temporal_scope or source
    if scope.status == TemporalStatus.UNRESOLVED:
        raise PlanningValidationError(PlanningReason.TEMPORAL_NOT_ENFORCEABLE)
    source_kind, kind = effective_kind(source), effective_kind(scope)
    broad = (TemporalKind.HISTORY, TemporalKind.NONE)
    if source_kind not in broad and scope != source:
        raise PlanningValidationError(PlanningReason.TEMPORAL_NOT_ENFORCEABLE)
    if source_kind in broad and kind not in (*broad, TemporalKind.LATEST, TemporalKind.APPRAISAL):
        # Arbitrary model dates/ISR sequences need future trusted anchor authorization.
        raise PlanningValidationError(PlanningReason.TEMPORAL_NOT_ENFORCEABLE)
    if source_kind in broad and scope != source:
        # Permit only a plain, bounded snapshot scope within the existing project.
        # Model-supplied dates/expressions/anchors must not masquerade as metadata.
        if (
            scope.date_from
            or scope.date_to
            or scope.isr_sequences
            or scope.expressions
            or scope.anchor_candidates
            or scope.explicit
            or scope.defaulted
            or scope.reference != source.reference
            or scope.detail is not None
            or scope.status != TemporalStatus.RESOLVED
        ):
            raise PlanningValidationError(PlanningReason.TEMPORAL_NOT_ENFORCEABLE)
    if scope.date_from and scope.date_to and scope.date_from > scope.date_to:
        raise PlanningValidationError(PlanningReason.TEMPORAL_NOT_ENFORCEABLE)
    if call is not None:
        args = SPECS[call.tool].args_model.model_validate(call.arguments).model_dump(mode="json")
        date_args = (args.get("date_from"), args.get("date_to"))
        seq_args = (args.get("isr_sequence_from"), args.get("isr_sequence_to"))
        single_seq = args.get("as_of_isr", args.get("isr_sequence"))
        if kind in broad:
            if (
                any(date_args)
                or any(seq_args)
                or isinstance(single_seq, int)
                or args.get("latest_n")
            ):
                raise PlanningValidationError(PlanningReason.TEMPORAL_NOT_ENFORCEABLE)
        elif kind in (TemporalKind.DATE, TemporalKind.YEAR, TemporalKind.DATE_RANGE):
            expected = (
                scope.date_from.isoformat() if scope.date_from else None,
                scope.date_to.isoformat() if scope.date_to else None,
            )
            if (
                call.tool != "get_project_timeline"
                or not any(expected)
                or date_args != expected
                or any(seq_args)
            ):
                raise PlanningValidationError(PlanningReason.TEMPORAL_NOT_ENFORCEABLE)
        elif kind in (TemporalKind.ISR_SEQUENCE, TemporalKind.ISR_RANGE):
            seqs = scope.isr_sequences
            if not seqs or any(n < 1 for n in seqs) or tuple(sorted(set(seqs))) != seqs:
                raise PlanningValidationError(PlanningReason.TEMPORAL_NOT_ENFORCEABLE)
            if call.tool in ("get_project_timeline", "get_rating_history"):
                valid = seq_args == (seqs[0], seqs[-1]) and not any(date_args)
                if kind == TemporalKind.ISR_SEQUENCE and len(seqs) > 1:
                    valid = False  # one range does not enforce a disjoint selection
            elif call.tool == "get_results_progress":
                valid = len(seqs) == 1 and single_seq == seqs[0]
            else:
                valid = False
            if not valid:
                raise PlanningValidationError(PlanningReason.TEMPORAL_NOT_ENFORCEABLE)
        elif kind == TemporalKind.LATEST:
            valid = (
                call.tool == "get_project_overview"
                or call.tool == "get_attention_signals"
                and args.get("status") == "CURRENT"
                or call.tool == "get_financial_status"
                and single_seq in (None, "latest")
                and not args.get("include_events")
                or call.tool == "get_results_progress"
                and single_seq == "latest"
                and not args.get("history")
                or call.tool == "get_rating_history"
                and args.get("latest_n") == 1
                and not any(seq_args)
            )
            if not valid or any(date_args) or any(seq_args):
                raise PlanningValidationError(PlanningReason.TEMPORAL_NOT_ENFORCEABLE)
        elif kind == TemporalKind.APPRAISAL:
            # Record type alone does not guarantee appraisal-only temporal coverage.
            raise PlanningValidationError(PlanningReason.TEMPORAL_NOT_ENFORCEABLE)
        else:
            raise PlanningValidationError(PlanningReason.TEMPORAL_NOT_ENFORCEABLE)
        loan = args.get("loan_number")
        if loan:
            _scope_text(loan, state, context)
    if document is not None:
        if state.retrieval_profile_id != BASELINE_ID:
            raise PlanningValidationError(PlanningReason.SCOPE_MISMATCH)
        _scope_text(document.query, state, context)
        query_config = load_retrieval_settings(context.config_dir).query
        query = process_query(document.query, state.project_id, query_config)
        if kind in broad:
            if query.isr_sequences or query.latest_isr:
                raise PlanningValidationError(PlanningReason.TEMPORAL_NOT_ENFORCEABLE)
        elif kind in (TemporalKind.ISR_SEQUENCE, TemporalKind.ISR_RANGE):
            expected = scope.isr_sequences
            if kind == TemporalKind.ISR_RANGE and expected:
                expected = tuple(range(expected[0], expected[-1] + 1))
            if not expected or query.isr_sequences != expected or query.latest_isr:
                raise PlanningValidationError(PlanningReason.TEMPORAL_NOT_ENFORCEABLE)
        elif kind == TemporalKind.LATEST:
            # Only actual latest-ISR query filtering exists, not generic latest/date filtering.
            if not query.latest_isr or query.isr_sequences:
                raise PlanningValidationError(PlanningReason.TEMPORAL_NOT_ENFORCEABLE)
        else:
            raise PlanningValidationError(PlanningReason.TEMPORAL_NOT_ENFORCEABLE)
    return scope


def validate_proposal(
    proposal: PlanningProposal, state: InvestigationState, context: AdmissionContext
):
    proposal = PlanningProposal.model_validate(proposal)
    requirements = []
    for proposed in proposal.requirements:
        call = None
        if proposed.structured_call is not None:
            p = proposed.structured_call
            args = SPECS[p.tool].args_model.model_validate(
                {**p.arguments, "project_id": state.project_id}
            )
            call = ToolCallSpec(
                tool=p.tool.value,
                arguments=args.model_dump(mode="json"),
                purpose=proposed.objective,
            )
        document = proposed.document_retrieval
        scope = _temporal(proposed, state, call, document, context)
        parent = proposed.parent_requirement_id
        if parent is not None and parent not in {r.requirement_id for r in state.requirements}:
            raise PlanningValidationError(PlanningReason.INVALID_PROPOSAL)
        req_id = requirement_id(
            state.project_id, scope, call, document, proposed.purpose_code, parent
        )
        requirements.append(
            EvidenceRequirement(
                requirement_id=req_id,
                objective=proposed.objective,
                project_id=state.project_id,
                temporal_scope=scope,
                structured_call=call,
                document_retrieval=document,
                required=proposed.required,
                parent_requirement_id=parent,
                purpose_code=proposed.purpose_code,
                rationale=proposed.rationale,
            )
        )
    # State's canonical invariant owns duplicate/count checks.
    state.transition(append=tuple(requirements))
    return tuple(requirements)


def _valid_context(state, context):
    admission = admit(state.source, context, state.execution)
    return (
        admission.outcome == AdmissionOutcome.ADMITTED
        and state.policy == context.policy
        and state.request_id == context.request_id
    )


def template_plan(state: InvestigationState, context: AdmissionContext) -> PlanningResult:
    if (
        state.status != InvestigationStatus.ADMITTED
        or state.requirements
        or not _valid_context(state, context)
    ):
        return PlanningResult(
            outcome=PlanningOutcome.FAIL, reason=PlanningReason.INVALID_STATE, state=state
        )
    proposed = []
    for c in state.source_plan.structured_calls:
        proposed.append(
            ProposedRequirement(
                objective=c.purpose,
                rationale="Phase 9 template; contextual operations do not assess sufficiency.",
                structured_call=ProposedCall(
                    tool=c.tool,
                    arguments={k: v for k, v in c.arguments.items() if k != "project_id"},
                ),
            )
        )
    for d in state.source_plan.document_retrievals:
        proposed.append(
            ProposedRequirement(
                objective=d.purpose,
                rationale="Phase 9 template; date/type hints are recorded, not applied as filters.",
                document_retrieval=d,
                purpose_code=PurposeCode.EXPLAIN,
            )
        )
    try:
        proposal = PlanningProposal(
            requirements=tuple(proposed),
            decision_summary="Use Phase 9 evidence operations without model planning.",
        )
        requirements = validate_proposal(proposal, state, context)
        # Reserving does not mean any evidence was executed or assessed.
        operations = sum(len(r.operation_ids) for r in requirements)
        budget = state.budget.reserve(
            Reservation(
                reservation_id="initial_plan",
                operations=operations,
                initial_operations=operations,
                retrieval_operations=sum(r.document_retrieval is not None for r in requirements),
            )
        )
        planned = state.transition(
            append=requirements, budget=budget, status=InvestigationStatus.PLANNED
        )
    except PlanningValidationError as exc:
        return _fail(
            state,
            exc.reason,
            clarification="These evidence operations cannot enforce the temporal restriction.",
        )
    except (ValueError, KeyError, TypeError):
        return _fail(state, PlanningReason.INVALID_PROPOSAL)
    return PlanningResult(
        outcome=PlanningOutcome.PLANNED, reason=PlanningReason.TEMPLATE, state=planned
    )


def plan(
    state: InvestigationState,
    context: AdmissionContext,
    *,
    need: RefinementNeed | None = None,
    model: InvestigatorModel | None = None,
    count_context_tokens: Callable[[PlanningContext], int] | None = None,
) -> PlanningResult:
    """Token measurement is supplied by trusted adapter wiring, not the proposal.

    Future live adapters must enforce output limits/pricing before invocation. 10B
    supplies no such adapter. Fake adapters exercise the accounting contract only.
    """
    if state.status not in (
        InvestigationStatus.ADMITTED,
        InvestigationStatus.PLANNED,
    ) or not _valid_context(state, context):
        return PlanningResult(
            outcome=PlanningOutcome.FAIL, reason=PlanningReason.INVALID_STATE, state=state
        )
    result = (
        template_plan(state, context)
        if state.status == InvestigationStatus.ADMITTED
        else PlanningResult(
            outcome=PlanningOutcome.PLANNED, reason=PlanningReason.TEMPLATE, state=state
        )
    )
    if result.outcome != PlanningOutcome.PLANNED or need is None:
        return result
    state = result.state
    if state.status != InvestigationStatus.PLANNED or model is None or count_context_tokens is None:
        return _fail(state, PlanningReason.INVALID_STATE)
    policy = state.policy
    current_ops = sum(len(r.operation_ids) for r in state.requirements)
    ctx = PlanningContext(
        question=UntrustedText(text=state.question),
        seed_objectives=tuple(UntrustedText(text=r.objective) for r in state.requirements),
        requirement_ids=tuple(r.requirement_id for r in state.requirements),
        temporal_scope=state.temporal_scope,
        need=need,
        max_additional_requirements=policy.max_initial_requirements - len(state.requirements),
        max_additional_operations=min(policy.max_initial_operations, policy.max_total_operations)
        - current_ops,
        max_additional_retrievals=policy.max_retrieval_operations
        - sum(r.document_retrieval is not None for r in state.requirements),
        token_bound=policy.max_investigator_context_tokens,
    )
    try:
        # The byte cap bounds context independently of model-specific token measurement.
        if len(ctx.model_dump_json().encode()) > policy.max_investigator_context_tokens * 4:
            return _fail(state, PlanningReason.CONTEXT_BOUNDS)
        tokens = count_context_tokens(ctx)
        if type(tokens) is not int or not 0 < tokens <= policy.max_investigator_context_tokens:
            return _fail(state, PlanningReason.CONTEXT_BOUNDS)
        reservation_id = fingerprint(
            "planning",
            {"calls": len(state.budget.reservations), "need": need.model_dump(mode="json")},
        )
        budget = state.budget.reserve(
            Reservation(
                reservation_id=reservation_id,
                model_calls=1,
                tokens=tokens + policy.max_planning_output_tokens,
            )
        )
        state = state.transition(budget=budget)
    except (ValueError, TypeError):
        return _fail(state, PlanningReason.POLICY_BOUNDS)
    try:
        proposal = PlanningProposal.model_validate(model.propose(ctx))
        if proposal.clarification:
            budget = state.budget.consume(
                Consumption(reservation_id=reservation_id, status=AttemptStatus.SUCCEEDED)
            )
            return _fail(
                state.transition(budget=budget),
                PlanningReason.MODEL_CLARIFICATION,
                clarification=proposal.clarification,
            )
        requirements = validate_proposal(proposal, state, context)
        operations = sum(len(r.operation_ids) for r in requirements)
        budget = state.budget.consume(
            Consumption(reservation_id=reservation_id, status=AttemptStatus.SUCCEEDED)
        )
        budget = budget.reserve(
            Reservation(
                reservation_id=fingerprint(
                    "proposal", {"ids": [r.requirement_id for r in requirements]}
                ),
                operations=operations,
                initial_operations=operations,
                retrieval_operations=sum(r.document_retrieval is not None for r in requirements),
            )
        )
        planned = state.transition(append=requirements, budget=budget)
        return PlanningResult(
            outcome=PlanningOutcome.PLANNED, reason=PlanningReason.VALID_PROPOSAL, state=planned
        )
    except Exception as exc:
        # Failed/malformed proposals still consume the reserved model attempt. No raw
        # exception/model text is persisted as a rationale or operational instruction.
        budget = state.budget.consume(
            Consumption(reservation_id=reservation_id, status=AttemptStatus.FAILED)
        )
        state = state.transition(budget=budget)
        reason = (
            exc.reason
            if isinstance(exc, PlanningValidationError)
            else PlanningReason.INVALID_PROPOSAL
            if isinstance(exc, (ValidationError, ValueError, KeyError, TypeError))
            else PlanningReason.MODEL_ERROR
        )
        return _fail(state, reason)
