"""``Copilot.investigate``: one entry point composing the validated pipeline stages.

query -> Phase 9 routing -> deterministic tools / Phase 8 retrieval
      -> Phase 10C EvidencePackage (investigation route only)
      -> Synthesizer -> deterministic claim validation -> optional Critic
      -> deterministic finalization -> InvestigationResult

Routing, tools, retrieval, evidence execution and claim validation are reused unchanged.
Only the investigation route calls a model; every other route is deterministic.
Each stage is one traced span (MLflow when enabled) with allowlisted metadata only.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from worldbank_copilot.application.guardrails import ScopedTools
from worldbank_copilot.application.observability import Recorder
from worldbank_copilot.application.projection import bounded_report
from worldbank_copilot.copilot.config import CopilotConfig
from worldbank_copilot.copilot.contracts import (
    AttentionSignal,
    Citation,
    Claim,
    CriticStatus,
    EvidenceItem,
    InvestigationResult,
    ModelCall,
    ResultStatus,
    Validation,
)
from worldbank_copilot.investigation.claims import (
    CriticCode,
    CriticOutput,
    Disposition,
    Failure,
    FinalResponse,
    ModelAdapter,
    ModelRequest,
    NodeError,
    SynthesisOutput,
)
from worldbank_copilot.investigation.evidence import EvidenceExecutor, _owned
from worldbank_copilot.investigation.gate import AdmissionContext, AdmissionOutcome, admit
from worldbank_copilot.investigation.planning import PlanningOutcome, template_plan
from worldbank_copilot.investigation.synthesis import (
    CRITIC_INSTRUCTIONS,
    INSTRUCTIONS,
    ApprovedContext,
    build_context,
    finalize,
    parse_output,
    validate_claims,
)
from worldbank_copilot.retrieval.contract import DocumentRetrieval
from worldbank_copilot.routing.models import AccessContext, ExecutionOutcome, Route, RouteResult
from worldbank_copilot.routing.service import RoutingService
from worldbank_copilot.tools.models import (
    ProvenanceClass,
    ToolResult,
    ToolStatus,
    iter_provenance_classes,
)

log = logging.getLogger(__name__)

CRITIC_DISABLED_LIMITATION = (
    "Semantic critic review was disabled by configuration: claims passed deterministic "
    "validation only and are not critic-validated."
)
_STOP_ROUTES = {
    Route.REFUSE: ResultStatus.REFUSE,
    Route.CLARIFY: ResultStatus.CLARIFY,
    Route.SEMANTIC_CLASSIFICATION_REQUIRED: ResultStatus.CLARIFY,
}
_ADMISSION_STOPS = {
    AdmissionOutcome.CLARIFY: ResultStatus.CLARIFY,
    AdmissionOutcome.REFUSE: ResultStatus.REFUSE,
}


@dataclass
class Copilot:
    router: RoutingService
    documents: DocumentRetrieval | None
    config: CopilotConfig
    config_dir: Path
    synthesizer: ModelAdapter | None = None  # None: investigation returns evidence only
    critic: ModelAdapter | None = None
    mlflow_enabled: bool = False

    def investigate(self, query: str, project_id: str) -> InvestigationResult:
        """Answer one question for one authorized project; never raises for request errors."""
        request_id = uuid.uuid4().hex
        recorder = Recorder(self.mlflow_enabled)
        started = time.monotonic()
        with recorder.span("copilot.investigate") as span:
            try:
                result = _Request(self, request_id, query, project_id, recorder, started).run()
            except Exception as exc:  # request boundary: fail closed, never leak internals
                log.exception("investigate failed request_id=%s", request_id)
                result = InvestigationResult(
                    request_id=request_id,
                    query=query,
                    project_id=project_id,
                    status=ResultStatus.FAIL_CLOSED,
                    message=f"Request stopped safely ({type(exc).__name__}).",
                )
            result = result.model_copy(update={"latency_ms": (time.monotonic() - started) * 1000})
            _annotate(span, **_request_attributes(result, self.config))
        return result.model_copy(
            update={"stage_latency_ms": dict(recorder.stages), "trace_id": recorder.trace_id}
        )


class _Request:
    """State for one request; each method is one pipeline stage."""

    def __init__(self, copilot, request_id, query, project_id, recorder, started):
        self.copilot, self.config = copilot, copilot.config
        self.request_id, self.query, self.project_id = request_id, query, project_id
        self.recorder, self.started = recorder, started
        self.access = AccessContext(authorized_projects=(project_id,), active_project_id=project_id)
        self.model_calls: list[ModelCall] = []

    # -- stages -----------------------------------------------------------------------
    def run(self) -> InvestigationResult:
        if self.project_id not in self.config.allowed_projects:
            return self._result(ResultStatus.REFUSE, "Unsupported or unauthorized project.")
        router = self._scoped_router()
        with self.recorder.span("routing") as span:
            routed = router.handle(self.query, self.access, request_id=self.request_id)
            _annotate(
                span,
                route=routed.decision.route.value,
                reason_code=routed.decision.reason_code,
                tools=",".join(routed.executed_tools),
                retrieval_executed=routed.retrieval_executed,
            )
        resolved = routed.understanding.project
        if resolved and resolved.project_id not in (None, self.project_id):
            raise ValueError("router scope mismatch")
        with self.recorder.span("attention") as span:
            signals, signal_notes = self._attention_signals(router, routed)
            _annotate(span, signal_count=len(signals))
        common = dict(routed=routed, attention_signals=signals, limitations=signal_notes)
        route = routed.decision.route
        if route in _STOP_ROUTES:
            return self._result(_STOP_ROUTES[route], routed.decision.detail, **common)
        if route != Route.INVESTIGATION:
            return self._deterministic(routed, common)
        return self._investigation(routed, router, common)

    def _scoped_router(self) -> RoutingService:
        """Request-scoped tools (document date-hint guard) and the request deadline."""
        deadline = self.started + self.config.overall_deadline_seconds
        factory = self.copilot.router.context_factory

        def context(request_id):
            ctx = factory(request_id)
            ctx.deadline = deadline if ctx.deadline is None else min(ctx.deadline, deadline)
            return ctx

        return replace(
            self.copilot.router,
            executor=ScopedTools(self.copilot.router.executor),
            context_factory=context,
        )

    def _attention_signals(self, router, routed: RouteResult):
        """Current Gold attention signals for the side panel; reuses the routed read."""
        result = next((r for r in routed.tool_results if r.tool == "get_attention_signals"), None)
        if result is None:
            result = router.executor.run(
                "get_attention_signals",
                {"project_id": self.project_id, "status": "CURRENT"},
                router.context_factory(self.request_id),
                scope_project_id=self.project_id,
                authorized_projects=(self.project_id,),
            )
        if result.status not in (ToolStatus.OK, ToolStatus.EMPTY):
            return (), (f"Attention signals unavailable ({result.status.value}).",)
        _owned([item.model_dump(mode="json") for item in result.items], self.project_id)
        return tuple(_signal(item) for item in result.items), ()

    def _deterministic(self, routed: RouteResult, common: dict) -> InvestigationResult:
        """Structured/document routes: governed records shown as-is, no model call."""
        items = tuple(
            _tool_evidence(result, item) for result in routed.tool_results for item in result.items
        )
        _owned([item.payload for item in items], self.project_id)
        notes = tuple(
            note
            for result in routed.tool_results
            for note in (*result.caveats, *(m.detail for m in result.mechanical))
        )
        if any(r.status == ToolStatus.AMBIGUOUS_ARGUMENT for r in routed.tool_results):
            status, message = ResultStatus.CLARIFY, "The request needs a more specific scope."
        elif items and routed.outcome == ExecutionOutcome.EXECUTED:
            status, message = ResultStatus.EVIDENCE_ONLY, "Governed records (no model used)."
        else:
            status, message = (
                ResultStatus.INSUFFICIENT_EVIDENCE,
                "UNKNOWN: no qualifying governed evidence was found.",
            )
        common["limitations"] = (*common["limitations"], *notes)
        return self._result(status, message, evidence=items, **common)

    def _investigation(self, routed: RouteResult, router, common: dict) -> InvestigationResult:
        """Phase 10C evidence package, then synthesis/validation/critic/finalization."""
        admission = AdmissionContext(
            self.request_id,
            self.query,
            self.access,
            self.copilot.router.config,
            self.copilot.router.index,
            self.copilot.config_dir,
            self.config.policy(),
            feature_enabled=True,
        )
        admitted = admit(routed, admission)
        if admitted.outcome != AdmissionOutcome.ADMITTED:
            status = _ADMISSION_STOPS.get(admitted.outcome, ResultStatus.FAIL_CLOSED)
            return self._result(
                status, f"Investigation not admitted ({admitted.reason}).", **common
            )
        planned = template_plan(admitted.state, admission)
        if planned.outcome != PlanningOutcome.PLANNED:
            return self._result(
                ResultStatus.CLARIFY,
                "Evidence operations cannot enforce the requested scope.",
                **common,
            )
        with self.recorder.span("evidence_execution") as span:
            report = EvidenceExecutor(
                admission,
                self.copilot.router.executor,
                self.copilot.documents,
                router.context_factory,
                # Without configured pricing, skip only the executor's pricing gate, as
                # the accepted Phase 10C live validation did; operations run normally.
                offline=not self.config.pricing_configured,
            ).execute(planned.state)
            operations = report.package.operations
            _annotate(
                span,
                requirement_count=len(report.package.requirement_summaries),
                structured_operations=sum(o.operation_type == "STRUCTURED" for o in operations),
                document_operations=sum(o.operation_type == "DOCUMENT" for o in operations),
                evidence_count=len(report.package.evidence_index),
                terminal_failure=str(report.terminal_failure or ""),
            )
        if report.terminal_failure is not None:
            return self._result(
                ResultStatus.FAIL_CLOSED,
                f"Evidence execution stopped ({report.terminal_failure}).",
                **common,
            )
        evidence = tuple(_package_evidence(ref) for ref in report.package.evidence_index)
        if self.copilot.synthesizer is None:
            return self._result(
                ResultStatus.EVIDENCE_ONLY if evidence else ResultStatus.INSUFFICIENT_EVIDENCE,
                "Evidence package only (synthesis not configured).",
                evidence=evidence,
                **common,
            )
        context = build_context(bounded_report(report))
        missing = [
            r["objective"] for r in context.requirements if r["required"] and not r["evidence_ids"]
        ]
        if missing:  # the finalizer's coverage rule can never publish here: skip the models
            common["limitations"] = (
                *common["limitations"],
                *(f"No evidence found for required: {objective}" for objective in missing),
            )
            return self._result(
                ResultStatus.INSUFFICIENT_EVIDENCE,
                "UNKNOWN: required evidence is missing, so no answer is synthesized.",
                evidence=evidence,
                validation=Validation(disposition=Disposition.INSUFFICIENT_EVIDENCE.value),
                **common,
            )
        final, critic_status = self._synthesize_and_review(context)
        return self._final_result(final, critic_status, context, evidence, common)

    def _synthesize_and_review(self, context: ApprovedContext):
        """Synthesizer -> deterministic validation -> optional Critic -> finalizer."""
        payload = context.model_dump(mode="json")
        with self.recorder.span("synthesis") as span:
            output, failure = self._call("SYNTHESIZER", payload, SynthesisOutput, span)
        if failure is not None:
            return self._finalize(None, None, context, (failure,)), CriticStatus.NOT_REQUIRED
        with self.recorder.span("deterministic_validation") as span:
            errors = validate_claims(
                output, context, max_claims=self.config.policy().max_claims_per_draft
            )
            _annotate(
                span,
                candidate_claims=len(output.candidate_claims),
                failures=",".join(e.value for e in errors),
            )
        if errors:  # mechanical violations are never sent to, or excused by, the critic
            return self._finalize(None, None, context, errors), CriticStatus.NOT_REQUIRED
        if not output.candidate_claims:  # abstention: nothing for a critic to review
            empty = CriticOutput(findings=())
            return self._finalize(output, empty, context), CriticStatus.NOT_REQUIRED
        if not self.config.models.critic_enabled:
            with self.recorder.span("finalization"):
                final = _finalize_without_critic(output, context)
            return final, CriticStatus.DISABLED
        with self.recorder.span("critic") as span:
            review, failure = self._call(
                "CRITIC",
                {**payload, "candidate_output": output.model_dump(mode="json")},
                CriticOutput,
                span,
            )
        if failure is not None:
            return self._finalize(output, None, context, (failure,)), CriticStatus.FAILED
        final = self._finalize(output, review, context)
        if final.disposition == Disposition.FAIL_CLOSED:
            return final, CriticStatus.FAILED
        supported = all(f.code == CriticCode.SUPPORTED for f in review.findings)
        return final, CriticStatus.SUPPORTED if supported else CriticStatus.REJECTED

    def _finalize(self, output, review, context, failures=()) -> FinalResponse:
        with self.recorder.span("finalization"):
            return finalize(output, review, context, failures=failures)

    def _call(self, role: str, payload: dict, schema, span):
        """One bounded model call; returns (parsed output, None) or (None, Failure)."""
        models = self.config.models
        adapter = self.copilot.synthesizer if role == "SYNTHESIZER" else self.copilot.critic
        if adapter is None:
            raise ValueError(f"{role.lower()} model not configured")
        endpoint = models.synthesizer_endpoint if role == "SYNTHESIZER" else models.critic_endpoint
        remaining = self.started + self.config.overall_deadline_seconds - time.monotonic()
        called, reply, failure, parsed = time.monotonic(), None, None, None
        if remaining <= 0:
            failure = Failure.BUDGET_EXHAUSTED
        else:
            request = ModelRequest(
                role=role,
                system=INSTRUCTIONS if role == "SYNTHESIZER" else CRITIC_INSTRUCTIONS,
                context_json=json.dumps(payload, sort_keys=True, separators=(",", ":")),
                output_schema=schema.model_json_schema(),
                max_output_tokens=models.max_output_tokens,
                timeout_seconds=min(models.timeout_seconds, remaining),
            )
            try:
                reply = adapter.invoke(request)
                parsed = parse_output(reply.text, schema)
            except NodeError as exc:
                failure = exc.category
            except TimeoutError:
                failure = Failure.MODEL_TIMEOUT
        call = ModelCall(
            role=role,
            endpoint=endpoint,
            model_identity=reply.model_identity if reply else None,
            latency_ms=(time.monotonic() - called) * 1000,
            input_tokens=reply.input_tokens if reply else None,
            output_tokens=reply.output_tokens if reply else None,
            outcome=failure.value if failure else "COMPLETED",
        )
        self.model_calls.append(call)
        _annotate(span, **call.model_dump(mode="json", exclude={"role"}))
        return parsed, failure

    # -- result assembly --------------------------------------------------------------
    def _final_result(self, final: FinalResponse, critic_status, context, evidence, common):
        cited = {e["evidence_id"]: e for e in context.evidence}
        claims = tuple(_claim(claim, cited) for claim in final.published_claims)
        if claims:
            status, message = ResultStatus.ANSWER, "Validated, cited claims."
        elif final.disposition == Disposition.FAIL_CLOSED:
            status, message = ResultStatus.FAIL_CLOSED, "No answer published: validation failed."
        else:
            status, message = (
                ResultStatus.INSUFFICIENT_EVIDENCE,
                "UNKNOWN: the evidence does not support a publishable answer.",
            )
        validation = Validation(
            disposition=final.disposition.value,
            mechanical_validity=final.mechanical_validity,
            failures=tuple(f.value for f in final.failures),
            semantic_support=final.semantic_support,
            critic_status=critic_status,
        )
        common["limitations"] = (*common["limitations"], *final.limitations)
        return self._result(
            status, message, claims=claims, evidence=evidence, validation=validation, **common
        )

    def _result(self, status, message, *, routed: RouteResult | None = None, **fields):
        understanding = routed.understanding if routed else None
        intent = understanding.intent if understanding else None
        return InvestigationResult(
            request_id=self.request_id,
            query=self.query,
            project_id=self.project_id,
            route=routed.decision.route.value if routed else None,
            reason_code=routed.decision.reason_code if routed else None,
            intent=intent.intent.value if intent and intent.intent else None,
            temporal_scope=understanding.temporal.model_dump(mode="json")
            if understanding and understanding.temporal
            else None,
            status=status,
            message=message,
            model_calls=tuple(self.model_calls),
            model_capability_note=self.config.models.capability_note if self.model_calls else None,
            **fields,
        )


def _finalize_without_critic(output: SynthesisOutput, context: ApprovedContext) -> FinalResponse:
    """Deterministic finalization when the Critic is disabled by configuration.

    Applies the existing finalizer's non-semantic rules to claims that already passed
    ``validate_claims`` (re-checked here, so validation stays authoritative): abstain on
    insufficient/UNKNOWN evidence or an uncovered required requirement, otherwise publish
    with an explicit limitation. No semantic review is performed or implied.
    """
    limitations = (
        *context.limitations,
        *output.limitations,
        *(("Context omitted evidence.",) if context.omitted_evidence_ids else ()),
        CRITIC_DISABLED_LIMITATION,
    )
    failures = validate_claims(output, context)
    covered = {i for claim in output.candidate_claims for i in claim.requirement_ids}
    if failures:
        disposition = Disposition.FAIL_CLOSED
    elif (
        output.insufficient_evidence
        or not output.candidate_claims
        or not context.evidence
        or any(c.provenance_label == ProvenanceClass.UNKNOWN for c in output.candidate_claims)
        or any(
            r["required"] and (r["requirement_id"] not in covered or not r["evidence_ids"])
            for r in context.requirements
        )
    ):
        disposition = Disposition.INSUFFICIENT_EVIDENCE
    else:
        disposition = Disposition.PUBLISH_WITH_LIMITATIONS
    return FinalResponse(
        project_id=context.project_id,
        package_fingerprint=context.package_fingerprint,
        disposition=disposition,
        published_claims=output.candidate_claims
        if disposition == Disposition.PUBLISH_WITH_LIMITATIONS
        else (),
        limitations=tuple(dict.fromkeys(limitations)),
        failures=failures,
        mechanical_validity="INVALID" if failures else "VALID",
        semantic_support="NOT_ASSESSED",
    )


def _annotate(span, **attributes) -> None:
    """Set allowlisted scalar attributes on an MLflow span (no-op without tracing)."""
    if span is not None:
        span.set_attributes({k: v for k, v in attributes.items() if v is not None})


def _request_attributes(result: InvestigationResult, config: CopilotConfig) -> dict:
    """Request-level trace metadata: identifiers, outcomes and counts, never documents."""
    calls = result.model_calls
    return {
        "request_id": result.request_id,
        "project_id": result.project_id,
        "query": result.query,
        "route": result.route,
        "intent": result.intent,
        "status": result.status.value,
        "abstained": result.status == ResultStatus.INSUFFICIENT_EVIDENCE,
        "fail_closed": result.status == ResultStatus.FAIL_CLOSED,
        "evidence_count": len(result.evidence),
        "claim_count": len(result.claims),
        "citation_count": sum(len(c.citations) for c in result.claims),
        "signal_count": len(result.attention_signals),
        "critic_status": result.validation.critic_status.value,
        "critic_enabled": config.models.critic_enabled,
        "validation_failures": ",".join(result.validation.failures),
        "model_call_count": len(calls),
        "model_endpoints": ",".join(sorted({c.endpoint for c in calls if c.endpoint})),
        "input_tokens": sum(c.input_tokens or 0 for c in calls),
        "output_tokens": sum(c.output_tokens or 0 for c in calls),
        "latency_ms": round(result.latency_ms, 1),
    }


def _claim(claim, cited: dict[str, dict]) -> Claim:
    citations = []
    for ref in claim.citations:
        entry = cited.get(ref.evidence_id, {})
        citation, source = entry.get("citation") or {}, entry.get("source") or {}
        citations.append(
            Citation(
                evidence_id=ref.evidence_id,
                source_identity=ref.source_identity,
                document_id=citation.get("document_id") or source.get("document_id"),
                document_label=citation.get("document_label"),
                pages=tuple(citation.get("pages") or ()),
                table=source.get("table"),
                record_id=source.get("record_id"),
            )
        )
    return Claim(
        claim_id=claim.claim_id,
        text=claim.claim_text,
        claim_type=claim.claim_type.value,
        provenance=claim.provenance_label.value,
        evidence_ids=claim.evidence_ids,
        citations=tuple(citations),
    )


def _package_evidence(ref) -> EvidenceItem:
    data = ref.model_dump(mode="json")
    return EvidenceItem(
        evidence_id=data["evidence_id"],
        provenance=(data["provenance"],),
        source_type=data["source_type"],
        citation=data["citation"],
        source=data["source"],
        payload=data["payload"],
    )


def _tool_evidence(result: ToolResult, item: Any) -> EvidenceItem:
    return EvidenceItem(
        tool=result.tool,
        provenance=tuple(sorted({p.value for p in iter_provenance_classes(item)})),
        payload=item.model_dump(mode="json"),
    )


def _signal(item: Any) -> AttentionSignal:
    data = item.model_dump(mode="json")
    return AttentionSignal(
        signal_id=data["signal_id"],
        title=data["signal_title"],
        description=data["signal_description"],
        category=data["signal_category"],
        severity=data["severity"],
        status=data["signal_status"],
        observed_date=data["observed_date"],
        current_value=data["current_value"],
        comparison_value=data["comparison_value"],
        provenance=data["provenance_class"],
        caveats=tuple(data["caveats"]),
    )
