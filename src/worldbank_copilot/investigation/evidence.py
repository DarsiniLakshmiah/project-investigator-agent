"""Execute only admitted evidence operations through frozen Phase 9 interfaces.

An executor instance owns one request-scoped context per investigation and caches
completed batches for local replay protection. This is not distributed exactly-once
execution. No planner, model, SQL, reader bypass or adaptive retrieval exists here.
Planning reservations are reused; failures consume their allowances without refunds.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from decimal import Decimal

from pydantic import BaseModel, ValidationError

from worldbank_copilot.investigation.assembly import (
    AssemblyIntegrityError,
    assemble,
    merge_references,
    references,
)
from worldbank_copilot.investigation.evidence_models import (
    SPECS,
    EvidenceExecutionReport,
    EvidenceLimits,
    OperationRecord,
    OperationStatus,
    OperationType,
    TraceEvent,
    TraceKind,
)
from worldbank_copilot.investigation.gate import AdmissionContext
from worldbank_copilot.investigation.models import (
    BASELINE_ID,
    EvidenceRequirement,
    InvestigationState,
    InvestigationStatus,
    fingerprint,
    operation_ids,
)
from worldbank_copilot.investigation.planning import (
    PlanningValidationError,
    _temporal,
    _valid_context,
)
from worldbank_copilot.investigation.policy import AttemptStatus, Consumption, Reservation
from worldbank_copilot.retrieval.config import load_retrieval_settings
from worldbank_copilot.retrieval.contract import (
    DocumentRetrieval,
    RetrievalRequest,
    RetrievalResult,
    RetrievalStatus,
    load_quality_baseline,
)
from worldbank_copilot.retrieval.models import ScopeViolation
from worldbank_copilot.routing.models import TemporalKind, ToolCallSpec
from worldbank_copilot.routing.requirements import effective_kind
from worldbank_copilot.tools.base import ToolContext
from worldbank_copilot.tools.executor import ToolExecutor
from worldbank_copilot.tools.models import ProvenanceClass, ToolResult, ToolStatus
from worldbank_copilot.tools.reader import TABLES


class EvidenceExecutionError(AssemblyIntegrityError):
    def __init__(self, status: OperationStatus, reason_code: str, *, budget=None, events=()):
        super().__init__(reason_code)
        self.status, self.reason_code = status, reason_code
        self.budget, self.events = budget, events


def _owned(value, project_id):
    """Check owner metadata recursively; free-text passages remain inert data."""
    if isinstance(value, BaseModel):
        return _owned(value.model_dump(mode="python"), project_id)
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "project_id" and child is not None and child != project_id:
                raise EvidenceExecutionError(
                    OperationStatus.SCOPE_REFUSED, "DOWNSTREAM_PROJECT_MISMATCH"
                )
            if key == "document_id" and isinstance(child, str):
                owner = re.match(r"^(P[0-9]{6})", child, re.IGNORECASE)
                if owner and owner.group(1).upper() != project_id:
                    raise EvidenceExecutionError(
                        OperationStatus.SCOPE_REFUSED, "DOWNSTREAM_SOURCE_MISMATCH"
                    )
            if key == "provenance_class" and child == ProvenanceClass.AI_INTERPRETATION:
                raise EvidenceExecutionError(
                    OperationStatus.INTEGRITY_ERROR, "PROVENANCE_VIOLATION"
                )
            _owned(child, project_id)
        if value.get("name") == "project_id" and value.get("value") != project_id:
            raise EvidenceExecutionError(
                OperationStatus.SCOPE_REFUSED, "DOWNSTREAM_PROJECT_MISMATCH"
            )
    elif isinstance(value, (tuple, list)):
        for child in value:
            _owned(child, project_id)


def _structured_status(status):
    return {
        ToolStatus.OK: OperationStatus.OK,
        ToolStatus.EMPTY: OperationStatus.NO_EVIDENCE,
        ToolStatus.NOT_FOUND: OperationStatus.NO_EVIDENCE,
        ToolStatus.NOT_COVERED: OperationStatus.NO_EVIDENCE,
        ToolStatus.INSUFFICIENT_EVIDENCE: OperationStatus.NO_EVIDENCE,
        ToolStatus.AMBIGUOUS_ARGUMENT: OperationStatus.CLARIFICATION_REQUIRED,
        ToolStatus.INVALID_ARGUMENT: OperationStatus.CLARIFICATION_REQUIRED,
        ToolStatus.SCOPE_REFUSED: OperationStatus.SCOPE_REFUSED,
        ToolStatus.DATA_INTEGRITY_ERROR: OperationStatus.INTEGRITY_ERROR,
    }.get(status, OperationStatus.TOOL_ERROR)


class EvidenceExecutor:
    def __init__(
        self,
        admission: AdmissionContext,
        executor: ToolExecutor,
        retrieval: DocumentRetrieval | None,
        context_factory: Callable[[str], ToolContext],
        *,
        offline: bool = False,
        limits: EvidenceLimits | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not isinstance(executor, ToolExecutor) or (
            retrieval is not None and not isinstance(retrieval, DocumentRetrieval)
        ):
            raise TypeError("use approved Phase 9 execution interfaces")
        self.admission, self.executor, self.retrieval = admission, executor, retrieval
        self.context_factory, self.clock = context_factory, clock
        self.limits = limits or EvidenceLimits()
        self.offline = offline  # trusted platform wiring, never a requirement/model field
        self._reports = {}
        self._started = set()  # failed/unfinished batches cannot silently be retried
        if not offline:
            admission.policy.require_live_cost_configuration()

    def _validate(self, state, requirement, ctx):
        if (
            state.project_id not in self.admission.access.authorized_projects
            or state.access != self.admission.access
            or ctx.request_id != state.request_id
            or state.project_id not in ctx.registry
            or requirement.project_id != state.project_id
        ):
            raise EvidenceExecutionError(
                OperationStatus.SCOPE_REFUSED, "AUTHORIZATION_OR_SCOPE_MISMATCH"
            )
        req = EvidenceRequirement.model_validate(requirement)
        if req.requirement_id != req.expected_id():
            raise EvidenceExecutionError(
                OperationStatus.INTEGRITY_ERROR, "REQUIREMENT_FINGERPRINT_MISMATCH"
            )
        if req.structured_call is not None:
            call = req.structured_call
            if call.tool not in SPECS or self.executor.specs.get(call.tool) != SPECS[call.tool]:
                raise EvidenceExecutionError(OperationStatus.INTEGRITY_ERROR, "UNAPPROVED_TOOL")
            SPECS[call.tool].args_model.model_validate(call.arguments)
        _temporal(req, state, req.structured_call, req.document_retrieval, self.admission)
        if req.document_retrieval is not None:
            if self.retrieval is None:
                raise EvidenceExecutionError(
                    OperationStatus.RETRIEVAL_ERROR, "RETRIEVAL_NOT_CONFIGURED"
                )
            baseline = load_quality_baseline(
                self.admission.config_dir, load_retrieval_settings(self.admission.config_dir)
            )
            if (
                self.retrieval.profile != baseline
                or self.retrieval.profile.profile_id != BASELINE_ID
            ):
                raise EvidenceExecutionError(OperationStatus.INTEGRITY_ERROR, "PROFILE_MISMATCH")
        return req

    def execute(self, state: InvestigationState, requirements=None) -> EvidenceExecutionReport:
        if not isinstance(state, InvestigationState):
            raise EvidenceExecutionError(OperationStatus.INTEGRITY_ERROR, "INVALID_STATE")
        state = InvestigationState.model_validate(state)
        if state.status != InvestigationStatus.PLANNED or not _valid_context(state, self.admission):
            raise EvidenceExecutionError(
                OperationStatus.SCOPE_REFUSED, "ADMISSION_CONTEXT_MISMATCH"
            )
        selected = state.requirements if requirements is None else tuple(requirements)
        owned = {r.requirement_id: r for r in state.requirements}
        if not selected or any(
            r.requirement_id not in owned or r != owned[r.requirement_id] for r in selected
        ):
            raise EvidenceExecutionError(
                OperationStatus.INTEGRITY_ERROR, "UNAUTHORIZED_REQUIREMENT"
            )
        signature = fingerprint(
            "batch",
            {
                "state": state.model_dump(mode="json"),
                "selected": sorted({r.requirement_id for r in selected}),
            },
        )
        key = state.request_id
        if key in self._reports:
            previous_signature, report = self._reports[key]
            if signature != previous_signature:
                raise EvidenceExecutionError(
                    OperationStatus.INTEGRITY_ERROR, "REQUEST_REPLAY_PLAN_CHANGED"
                )
            return report.model_copy()
        if key in self._started:
            raise EvidenceExecutionError(OperationStatus.INTEGRITY_ERROR, "BATCH_ALREADY_ATTEMPTED")
        self._started.add(key)
        ctx = self.context_factory(state.request_id)
        budget = state.budget
        started = self.clock()
        remaining = state.policy.overall_deadline_seconds - float(budget.elapsed_seconds)
        ctx.deadline = (
            min(ctx.deadline, started + remaining)
            if ctx.deadline is not None
            else started + remaining
        )
        operations = {}
        for req in selected:
            ids = iter(req.operation_ids)
            for kind, operation in (
                (OperationType.STRUCTURED, req.structured_call),
                (OperationType.DOCUMENT, req.document_retrieval),
            ):
                if operation is None:
                    continue
                op_id = next(ids)
                if op_id not in operations:
                    operations[op_id] = [req, kind, operation, set()]
                operations[op_id][3].add(req.requirement_id)
        records, refs, events = [], (), []
        terminal = None
        credits = {}
        used, succeeded = {}, {}
        consumed = {c.reservation_id for c in budget.consumptions}
        for r in budget.reservations:
            if r.reservation_id not in consumed and (
                r.reservation_id == "initial_plan" or r.reservation_id.startswith("proposal_")
            ):
                credits[r.reservation_id] = r
                used[r.reservation_id] = {"operations": 0, "retrieval": 0}
        known_snapshots = {}
        seed_ids = set()
        for seed in state.source_plan.structured_calls:
            seed_ids.update(
                operation_ids(
                    state.project_id,
                    ToolCallSpec(tool=seed.tool, arguments=seed.arguments, purpose=seed.purpose),
                    None,
                )
            )
        for seed in state.source_plan.document_retrievals:
            seed_ids.update(operation_ids(state.project_id, None, seed))

        def event(kind, record=None, latency=None):
            return TraceEvent(
                event=kind,
                request_id=state.request_id,
                project_id=state.project_id,
                requirement_ids=() if record is None else record.requirement_ids,
                operation_id=None if record is None else record.operation_id,
                operation_type=None if record is None else record.operation_type,
                tool_or_profile=None if record is None else record.tool_or_profile,
                status=None if record is None else record.status,
                latency_ms=latency,
                evidence_ids=()
                if record is None
                else tuple(link.evidence_id for link in record.evidence_links),
                table_versions=dict(known_snapshots),
                index_identity=None
                if self.retrieval is None
                else self.retrieval.profile.vector_search_index,
                error_class=None if record is None else record.error_class,
                reserved_operations=sum(r.operations for r in budget.reservations),
                reserved_retrievals=sum(r.retrieval_operations for r in budget.reservations),
            )

        for op_id, (req, kind, operation, owners) in operations.items():
            details = dict(
                operation_id=op_id,
                requirement_ids=tuple(sorted(owners)),
                project_id=state.project_id,
                operation_type=kind,
                tool_or_profile=operation.tool if kind == OperationType.STRUCTURED else BASELINE_ID,
            )
            attempted, reservation_id, slot, before = False, None, None, self.clock()
            try:
                elapsed = budget.elapsed_seconds + Decimal(str(max(0, before - started)))
                if elapsed >= state.policy.overall_deadline_seconds:
                    raise EvidenceExecutionError(
                        OperationStatus.BUDGET_EXHAUSTED, "DEADLINE_EXCEEDED"
                    )
                budget = budget.advance_elapsed(elapsed)
                started = before
                self._validate(state, req, ctx)
                if op_id not in req.operation_ids:
                    raise EvidenceExecutionError(
                        OperationStatus.INTEGRITY_ERROR, "OPERATION_FINGERPRINT_MISMATCH"
                    )
                retrieval_count = int(kind == OperationType.DOCUMENT)
                for rid, credit in credits.items():
                    if (
                        rid == "initial_plan"
                        and op_id not in seed_ids
                        or rid.startswith("proposal_")
                        and op_id in seed_ids
                    ):
                        continue
                    usage = used[rid]
                    # Reserve separate structured/retrieval capacity within the batch allowance.
                    if (
                        usage["operations"] < credit.operations
                        and usage["retrieval"] + retrieval_count <= credit.retrieval_operations
                        and (
                            retrieval_count
                            or usage["operations"] - usage["retrieval"]
                            < credit.operations - credit.retrieval_operations
                        )
                    ):
                        reservation_id = rid
                        slot = usage["operations"]
                        usage["operations"] += 1
                        usage["retrieval"] += retrieval_count
                        break
                if reservation_id is None:
                    reservation_id = op_id
                    try:
                        budget = budget.reserve(
                            Reservation(
                                reservation_id=reservation_id,
                                operations=1,
                                initial_operations=1,
                                retrieval_operations=retrieval_count,
                            )
                        )
                    except ValueError as exc:
                        reservation_id = None
                        raise EvidenceExecutionError(
                            OperationStatus.BUDGET_EXHAUSTED, "OPERATION_BUDGET_EXHAUSTED"
                        ) from exc
                    credits[reservation_id] = budget.reservations[-1]
                    used[reservation_id] = {"operations": 1, "retrieval": retrieval_count}
                    slot = 0
                attempted = True
                pending = OperationRecord(
                    **details,
                    status=OperationStatus.OK,
                    reason_code="ATTEMPT_RESERVED",
                    attempted=True,
                    reservation_id=reservation_id,
                    reservation_slot=slot,
                )
                events.append(event(TraceKind.STARTED, pending))
                if kind == OperationType.STRUCTURED:
                    raw = self.executor.run(
                        operation.tool,
                        operation.arguments,
                        ctx,
                        scope_project_id=state.project_id,
                        authorized_projects=self.admission.access.authorized_projects,
                    )
                    _owned(raw, state.project_id)
                    # Revalidate facts and provenance with the registry item schema.
                    raw_data = raw.model_dump(mode="python")
                    result = ToolResult[SPECS[operation.tool].item_model].model_validate(raw_data)
                    if (
                        result.request_id != state.request_id
                        or result.tool_version != SPECS[operation.tool].version
                    ):
                        raise EvidenceExecutionError(
                            OperationStatus.INTEGRITY_ERROR, "TOOL_ENVELOPE_MISMATCH"
                        )
                    status = _structured_status(result.status)
                    record = OperationRecord(
                        **details,
                        status=status,
                        reason_code=result.status.value,
                        attempted=True,
                        reservation_id=reservation_id,
                        reservation_slot=slot,
                        structured_result=result,
                    )
                else:
                    request = RetrievalRequest.from_spec(
                        state.project_id,
                        operation,
                        intent=state.intent.intent,
                        temporal_scope=req.temporal_scope,
                    )
                    raw = self.retrieval.retrieve(
                        request, ctx, authorized_projects=self.admission.access.authorized_projects
                    )
                    _owned(raw, state.project_id)
                    result = RetrievalResult.model_validate(raw.model_dump(mode="python"))
                    if (
                        result.profile != BASELINE_ID
                        or result.top_k != self.retrieval.profile.final_k
                        or result.query_used != request.query
                        or result.recorded_hints.temporal_scope != request.temporal_scope
                        or result.recorded_hints.document_type_hints != request.document_type_hints
                    ):
                        raise EvidenceExecutionError(
                            OperationStatus.INTEGRITY_ERROR, "RETRIEVAL_ENVELOPE_MISMATCH"
                        )
                    if result.status == RetrievalStatus.OK:
                        kind_scope = effective_kind(req.temporal_scope)
                        applied = result.filters_applied or {}
                        if applied.get("project_id") != state.project_id:
                            raise EvidenceExecutionError(
                                OperationStatus.SCOPE_REFUSED, "RETRIEVAL_FILTER_SCOPE_MISMATCH"
                            )
                        sequences = tuple(applied.get("isr_sequences") or ())
                        if kind_scope == TemporalKind.LATEST and not sequences:
                            raise EvidenceExecutionError(
                                OperationStatus.INTEGRITY_ERROR, "TEMPORAL_FILTER_NOT_APPLIED"
                            )
                        if kind_scope in (TemporalKind.ISR_SEQUENCE, TemporalKind.ISR_RANGE):
                            expected = req.temporal_scope.isr_sequences
                            if kind_scope == TemporalKind.ISR_RANGE and expected:
                                expected = tuple(range(expected[0], expected[-1] + 1))
                            if sequences != expected:
                                raise EvidenceExecutionError(
                                    OperationStatus.INTEGRITY_ERROR, "TEMPORAL_FILTER_NOT_APPLIED"
                                )
                    status = OperationStatus(result.status.value)
                    record = OperationRecord(
                        **details,
                        status=status,
                        reason_code=result.status.value,
                        attempted=True,
                        reservation_id=reservation_id,
                        reservation_slot=slot,
                        retrieval_result=result,
                    )
                if len(record.model_dump_json().encode()) > self.limits.max_operation_bytes:
                    raise AssemblyIntegrityError("OPERATION_BOUND_EXCEEDED")
                incoming, links = references(record)
                proposed_refs = merge_references(refs, incoming, self.limits.max_references)
                new_snapshots = dict(known_snapshots)
                if record.structured_result is not None:
                    for table, version in record.structured_result.data_snapshot.items():
                        if table not in TABLES or (version is not None and version < 0):
                            raise AssemblyIntegrityError("INVALID_SNAPSHOT_IDENTITY")
                        if table in new_snapshots and new_snapshots[table] != version:
                            raise AssemblyIntegrityError("SNAPSHOT_DRIFT")
                        new_snapshots[table] = version
                record = OperationRecord(
                    **{**record.model_dump(mode="python"), "evidence_links": links}
                )
                # Commit retained output only after the whole component passes integrity checks.
                refs, known_snapshots = proposed_refs, new_snapshots
                succeeded.setdefault(reservation_id, []).append(
                    status in (OperationStatus.OK, OperationStatus.NO_EVIDENCE)
                )
            except Exception as exc:
                if isinstance(exc, EvidenceExecutionError):
                    status, reason = exc.status, exc.reason_code
                elif isinstance(exc, PlanningValidationError):
                    status, reason = OperationStatus.TEMPORAL_NOT_ENFORCEABLE, exc.reason.value
                elif isinstance(exc, AssemblyIntegrityError):
                    status, reason = OperationStatus.INTEGRITY_ERROR, str(exc)
                elif isinstance(exc, ScopeViolation):
                    status, reason = OperationStatus.SCOPE_REFUSED, "SCOPE_VIOLATION"
                elif isinstance(exc, ValidationError):
                    status, reason = OperationStatus.INTEGRITY_ERROR, "CONTRACT_VALIDATION_FAILED"
                else:
                    status = (
                        OperationStatus.RETRIEVAL_ERROR
                        if kind == OperationType.DOCUMENT
                        else OperationStatus.TOOL_ERROR
                    )
                    reason = "DEPENDENCY_EXCEPTION"
                record = OperationRecord(
                    **details,
                    status=status,
                    reason_code=reason,
                    attempted=attempted,
                    reservation_id=reservation_id,
                    reservation_slot=slot,
                    error_class=type(exc).__name__,
                )
                if reservation_id is not None:
                    succeeded.setdefault(reservation_id, []).append(False)
            records.append(record)
            latency = max(0, (self.clock() - before) * 1000)
            events.append(
                event(
                    TraceKind.COMPLETED
                    if record.status in (OperationStatus.OK, OperationStatus.NO_EVIDENCE)
                    else TraceKind.FAILED,
                    record,
                    latency,
                )
            )
            if record.status in (
                OperationStatus.SCOPE_REFUSED,
                OperationStatus.INTEGRITY_ERROR,
                OperationStatus.TEMPORAL_NOT_ENFORCEABLE,
                OperationStatus.BUDGET_EXHAUSTED,
            ):
                terminal = record.status
                break
        for rid, credit in credits.items():
            attempts = succeeded.get(rid, ())
            if attempts:
                complete = len(attempts) == credit.operations and all(attempts)
                budget = budget.consume(
                    Consumption(
                        reservation_id=rid,
                        status=AttemptStatus.SUCCEEDED if complete else AttemptStatus.FAILED,
                    )
                )
        actual_elapsed = budget.elapsed_seconds + Decimal(str(max(0, self.clock() - started)))
        if actual_elapsed >= state.policy.overall_deadline_seconds:
            terminal = terminal or OperationStatus.BUDGET_EXHAUSTED
        budget = budget.advance_elapsed(
            min(actual_elapsed, Decimal(state.policy.overall_deadline_seconds))
        )
        profile = None if self.retrieval is None else self.retrieval.profile
        try:
            package = assemble(state, tuple(records), refs, profile, self.limits)
        except AssemblyIntegrityError as exc:
            # A bounded assembly failure must not lose the charged attempts.
            raise EvidenceExecutionError(
                OperationStatus.INTEGRITY_ERROR, str(exc), budget=budget, events=tuple(events)
            ) from exc
        events.append(event(TraceKind.ASSEMBLED))
        report = EvidenceExecutionReport(
            investigation=state,
            budget=budget,
            package=package,
            events=tuple(events),
            terminal_failure=terminal,
        )
        self._reports[key] = signature, report
        return report.model_copy()
