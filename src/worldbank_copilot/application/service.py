"""One request entry point, composing the frozen router/evidence/model stages."""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from worldbank_copilot.application.guardrails import ScopedTools
from worldbank_copilot.application.observability import Recorder, TraceRecord
from worldbank_copilot.application.projection import bounded_report
from worldbank_copilot.investigation.bounded import investigate
from worldbank_copilot.investigation.claims import FinalResponse
from worldbank_copilot.investigation.evidence import EvidenceExecutor, _owned
from worldbank_copilot.investigation.gate import AdmissionContext, AdmissionOutcome, admit
from worldbank_copilot.investigation.planning import PlanningOutcome, template_plan
from worldbank_copilot.investigation.policy import Contract, InvestigationPolicy, Reservation
from worldbank_copilot.investigation.synthesis import run_nodes
from worldbank_copilot.routing.models import AccessContext, ExecutionOutcome, Route


class AnswerRequest(Contract):
    question: str = Field(min_length=1, max_length=1000)
    project_id: str = Field(pattern=r"^P[0-9]{6}$")
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex, pattern=r"^[a-f0-9]{32}$")

    @field_validator("question")
    @classmethod
    def bounded_input(cls, value):
        if re.search(
            r"(?i)(Bearer\s+\S+|dapi[0-9a-f]{20,}|(?:api_key|password|access_token)\s*[:=])", value
        ):
            raise ValueError("credential-shaped input is not accepted")
        return value


class Answer(Contract):
    request_id: str
    project_id: str
    route: str
    status: Literal["ANSWER", "EVIDENCE_ONLY", "UNKNOWN", "CLARIFY", "REFUSE", "FAIL"]
    message: str
    final: FinalResponse | None = None
    evidence: tuple[dict, ...] = ()
    trace: TraceRecord | None = None


@dataclass
class Application:
    router: object
    documents: object
    config_dir: Path
    allowed_projects: tuple[str, ...]
    policy: InvestigationPolicy
    synthesizer: object | None = None
    critic: object | None = None
    investigator: object | None = None
    models_enabled: bool = False
    mlflow_enabled: bool = False
    offline: bool = False
    # User-supplied bound in configured pricing units, not invented billed cost.
    worst_case_model_cost: Decimal | None = None

    def answer_question(self, request: AnswerRequest, *, mode="C") -> Answer:
        request = AnswerRequest.model_validate(request)
        if mode not in ("A", "B", "C"):
            raise ValueError("unknown evaluation configuration")
        rid = uuid.uuid4().hex
        if request.project_id not in self.allowed_projects:
            recorder = Recorder(self.mlflow_enabled)
            started = time.monotonic()
            with recorder.span("answer_question") as span:
                trace = TraceRecord(
                    request_id=rid,
                    session_id=request.session_id,
                    project_id=request.project_id,
                    route="REFUSE",
                    latency_ms=max(0, time.monotonic() - started) * 1000,
                    model_calls=0,
                    retrieval_calls=0,
                    investigator_used=False,
                    investigator_actions=0,
                    critic_used=False,
                    claim_count=0,
                    outcome="REFUSE",
                    mlflow_trace_id=recorder.trace_id,
                )
                recorder.publish(span, trace)
            return Answer(
                request_id=rid,
                project_id=request.project_id,
                route="REFUSE",
                status="REFUSE",
                message="Unsupported or unauthorized project.",
                trace=trace,
            )
        recorder = Recorder(self.mlflow_enabled)
        started = time.monotonic()
        events, evidence = [], ()
        used, actions, retrievals = False, 0, 0
        required_total, required_supported = 0, 0
        route = "REFUSE"
        result = None
        with recorder.span("answer_question") as root_span:
            try:
                # The selected project is the whole request authorization scope.
                access = AccessContext(
                    authorized_projects=(request.project_id,), active_project_id=request.project_id
                )

                def scoped_context(request_id):
                    ctx = self.router.context_factory(request_id)
                    deadline = started + self.policy.overall_deadline_seconds
                    ctx.deadline = (
                        min(ctx.deadline, deadline) if ctx.deadline is not None else deadline
                    )
                    return ctx

                tools = ScopedTools(self.router.executor)
                router = replace(self.router, context_factory=scoped_context, executor=tools)
                with recorder.span("routing"):
                    routed = router.handle(request.question, access, request_id=rid)
                route = routed.decision.route.value
                retrievals = tools.document_calls
                if routed.understanding.project and routed.understanding.project.project_id not in (
                    None,
                    request.project_id,
                ):
                    raise ValueError("router scope mismatch")
                if route in (Route.CLARIFY, Route.REFUSE, Route.SEMANTIC_CLASSIFICATION_REQUIRED):
                    status = "REFUSE" if route == Route.REFUSE else "CLARIFY"
                    result = Answer(
                        request_id=rid,
                        project_id=request.project_id,
                        route=route,
                        status=status,
                        message=routed.decision.detail,
                    )
                elif route != Route.INVESTIGATION:
                    # Direct source display is deterministic; it makes no entailment claim.
                    payload = tuple(
                        item.model_dump(mode="json")
                        for tool in routed.tool_results
                        for item in tool.items
                    )
                    _owned(payload, request.project_id)
                    evidence = payload
                    result = Answer(
                        request_id=rid,
                        project_id=request.project_id,
                        route=route,
                        status="CLARIFY"
                        if any(t.status.value == "AMBIGUOUS_ARGUMENT" for t in routed.tool_results)
                        else (
                            "EVIDENCE_ONLY"
                            if routed.outcome == ExecutionOutcome.EXECUTED
                            else "UNKNOWN"
                        ),
                        message="Source evidence; semantic support is NOT_ASSESSED."
                        if payload
                        else "UNKNOWN or scope clarification required: no qualifying evidence.",
                        evidence=payload,
                    )
                else:
                    admission = AdmissionContext(
                        rid,
                        request.question,
                        access,
                        self.router.config,
                        self.router.index,
                        self.config_dir,
                        self.policy,
                        feature_enabled=True,
                    )
                    admitted = admit(routed, admission)
                    if admitted.outcome != AdmissionOutcome.ADMITTED:
                        raise ValueError("admission failed")
                    planned = template_plan(admitted.state, admission)
                    if planned.outcome != PlanningOutcome.PLANNED:
                        result = Answer(
                            request_id=rid,
                            project_id=request.project_id,
                            route=route,
                            status="CLARIFY",
                            message="Evidence operations cannot enforce the requested scope.",
                        )
                    else:

                        def executor():
                            return EvidenceExecutor(
                                admission,
                                self.router.executor,
                                self.documents,
                                router.context_factory,
                                offline=self.offline,
                            )

                        with recorder.span("evidence"):
                            report = executor().execute(planned.state)
                        retrievals += sum(
                            o.operation_type == "DOCUMENT" and o.attempted
                            for o in report.package.operations
                        )
                        roots = {
                            r.requirement_id
                            for r in report.investigation.requirements
                            if r.required and r.parent_requirement_id is None
                        }
                        required_total = len(roots)
                        required_supported = sum(
                            s.requirement_id in roots and bool(s.evidence_ids)
                            for s in report.package.requirement_summaries
                        )
                        if report.terminal_failure is not None:
                            raise ValueError("initial evidence execution failed")
                        if mode != "A" and self.models_enabled:
                            if self.synthesizer is None or self.critic is None:
                                raise ValueError("model configuration incomplete")
                            if not self.offline:
                                self.policy.require_live_cost_configuration()
                                if self.worst_case_model_cost is None:
                                    raise ValueError("explicit model cost upper bound required")
                                # Charge all possible model allowance before the first call.
                                # One conservative accounting token makes this a valid reservation.
                                ledger = report.budget.reserve(
                                    Reservation(
                                        reservation_id="model_cost_allowance",
                                        tokens=1,
                                        cost=self.worst_case_model_cost
                                        * self.policy.max_model_calls,
                                    )
                                )
                                report = type(report)(
                                    **{**report.model_dump(mode="python"), "budget": ledger}
                                )
                            if mode == "C" and self.investigator is not None:
                                with recorder.span("investigator"):
                                    extra = investigate(
                                        report,
                                        admission,
                                        self.investigator,
                                        executor,
                                        offline=self.offline,
                                    )
                                report, used, actions = extra.report, extra.used, extra.action_count
                                if used:
                                    events.append(
                                        {
                                            "role": "INVESTIGATOR",
                                            "model": extra.model_identity,
                                            "latency_ms": extra.latency_ms,
                                            "input_tokens": extra.input_tokens,
                                            "output_tokens": extra.output_tokens,
                                            "outcome": extra.outcome,
                                        }
                                    )
                                retrievals += extra.retrieval_calls
                                if extra.outcome not in ("NOT_NEEDED", "STOP", "EVIDENCE"):
                                    raise ValueError("investigator stopped safely")
                            roots = {
                                r.requirement_id
                                for r in report.investigation.requirements
                                if r.required and r.parent_requirement_id is None
                            }
                            required_total = len(roots)
                            required_supported = sum(
                                s.requirement_id in roots and bool(s.evidence_ids)
                                for s in report.package.requirement_summaries
                            )
                            ledger = report.budget.advance_elapsed(
                                max(
                                    report.budget.elapsed_seconds,
                                    Decimal(str(time.monotonic() - started)),
                                )
                            )
                            report = type(report)(
                                **{**report.model_dump(mode="python"), "budget": ledger}
                            )
                            report = bounded_report(report)
                            with recorder.span("synthesis_validation_critic_finalization"):
                                synthesis = run_nodes(
                                    report,
                                    _ObservedAdapter(self.synthesizer, recorder),
                                    _ObservedAdapter(self.critic, recorder),
                                    offline=self.offline,
                                )
                            events.extend(
                                {
                                    "role": e.model_role,
                                    "model": e.model_identity,
                                    "latency_ms": e.latency_ms,
                                    "input_tokens": e.input_tokens,
                                    "output_tokens": e.output_tokens,
                                    "critic_codes": [c.value for c in e.critic_codes],
                                    "outcome": e.error_category.value
                                    if e.error_category
                                    else "COMPLETED",
                                }
                                for e in synthesis.events
                            )
                            final = synthesis.final
                            evidence = tuple(
                                e.model_dump(mode="json") for e in report.package.evidence_index
                            )
                            status = "ANSWER" if final.published_claims else "UNKNOWN"
                            if final.disposition == "FAIL_CLOSED":
                                status = "FAIL"
                            result = Answer(
                                request_id=rid,
                                project_id=request.project_id,
                                route=route,
                                status=status,
                                message="Validated claims below."
                                if final.published_claims
                                else "UNKNOWN: no publishable claims.",
                                final=final,
                                evidence=evidence,
                            )
                        else:
                            evidence = tuple(
                                e.model_dump(mode="json") for e in report.package.evidence_index
                            )
                            result = Answer(
                                request_id=rid,
                                project_id=request.project_id,
                                route=route,
                                status="EVIDENCE_ONLY" if evidence else "UNKNOWN",
                                evidence=evidence,
                                message="Deterministic evidence baseline; model stages disabled.",
                            )
            except Exception as exc:
                # Public failure contains a category only; no arbitrary exception/SDK text.
                result = Answer(
                    request_id=rid,
                    project_id=request.project_id,
                    route=route,
                    status="FAIL",
                    message="Request failed safely: " + type(exc).__name__,
                )
            trace = TraceRecord(
                request_id=rid,
                session_id=request.session_id,
                project_id=request.project_id,
                route=route,
                latency_ms=max(0, time.monotonic() - started) * 1000,
                model_calls=len(events),
                retrieval_calls=retrievals,
                investigator_used=used,
                investigator_actions=actions,
                critic_used=any(e["role"] == "CRITIC" for e in events),
                claim_count=len(result.final.published_claims) if result.final else 0,
                evidence_ids=tuple(e["evidence_id"] for e in evidence if "evidence_id" in e),
                outcome=result.status,
                stages_ms=recorder.stages,
                model_metrics=tuple(events),
                required_total=required_total,
                required_with_evidence=required_supported,
                mlflow_trace_id=recorder.trace_id,
            )
            recorder.publish(root_span, trace)
        return Answer(**{**result.model_dump(mode="python"), "trace": trace})


def source_followup(previous: Answer, project_id: str, authorized_projects) -> tuple[dict, ...]:
    """Only prior cited sources in the same authorized session/project; never new facts."""
    previous = Answer.model_validate(previous)
    if project_id != previous.project_id or project_id not in authorized_projects:
        raise ValueError("session project changed")
    cited = (
        {ref.evidence_id for claim in previous.final.published_claims for ref in claim.citations}
        if previous.final
        else None
    )
    return tuple(e for e in previous.evidence if cited is None or e.get("evidence_id") in cited)


@dataclass
class _ObservedAdapter:
    adapter: object
    recorder: Recorder

    def invoke(self, request):
        with self.recorder.span(request.role.lower()):
            return self.adapter.invoke(request)
