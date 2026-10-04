"""``Copilot.investigate``: the single online runtime.

question -> router fast path (deterministic structured answers and every refusal)
         -> otherwise the Investigator: objective + governed actions (<= 2 rounds, <= 6 calls)
         -> existing tools / Phase 8 retrieval -> EvidenceReferences -> relevance-aware packing
         -> Synthesizer -> deterministic per-claim integrity -> claim-level Critic
         -> claim-level finalization -> InvestigationResult

Semantic decisions (intent, what to retrieve, support) are made by models; identity,
scope, tool permissions, citations and objective date boundaries are enforced by code.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from worldbank_copilot.application.observability import Recorder
from worldbank_copilot.copilot.config import CopilotConfig
from worldbank_copilot.copilot.contracts import (
    AttentionSignal,
    Citation,
    Claim,
    CriticStatus,
    EvidenceItem,
    InvestigationActivity,
    InvestigationResult,
    ModelCall,
    ResultStatus,
    Validation,
)
from worldbank_copilot.copilot.finalizer import check_integrity, finalize
from worldbank_copilot.copilot.governed import (
    Gathered,
    GovernanceViolation,
    GovernedExecutor,
    anchor,
    approved_context,
    pack,
)
from worldbank_copilot.copilot.investigator import (
    INSTRUCTIONS as INVESTIGATOR_INSTRUCTIONS,
)
from worldbank_copilot.copilot.investigator import (
    InvestigatorDecision,
    govern,
    tool_catalog,
)
from worldbank_copilot.copilot.model_diagnostics import PARSE_REASONS, DiagnosticReason
from worldbank_copilot.copilot.semantic import (
    CRITIC_INSTRUCTIONS,
    SYNTHESIS_INSTRUCTIONS,
    SemanticReview,
    SemanticSynthesis,
    critic_payload,
    enrich,
    project,
)
from worldbank_copilot.investigation.claims import (
    Failure,
    ModelAdapter,
    ModelRequest,
    NodeError,
)
from worldbank_copilot.investigation.evidence import _owned
from worldbank_copilot.investigation.synthesis import parse_output
from worldbank_copilot.retrieval.contract import DocumentRetrieval
from worldbank_copilot.routing.models import AccessContext, ExecutionOutcome, Route, RouteResult
from worldbank_copilot.routing.service import RoutingService
from worldbank_copilot.routing.temporal import parse_temporal
from worldbank_copilot.tools.models import ToolResult, ToolStatus, iter_provenance_classes

log = logging.getLogger(__name__)

PREDICTION_MESSAGE = (
    "This Copilot does not have a validated model for predicting project success or failure. "
    "It can show observed implementation signals and documented risks instead."
)


@dataclass
class Copilot:
    router: RoutingService
    documents: DocumentRetrieval | None
    config: CopilotConfig
    config_dir: Path
    investigator: ModelAdapter | None = None
    synthesizer: ModelAdapter | None = None
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
        self.activity: dict[str, Any] = {}
        self.anchor_handles: dict[str, str] = {}  # round-2 handle -> evidence ID, as shown
        self.investigated = False  # True once the Investigator owns the request

    # -- entry -------------------------------------------------------------------------
    def run(self) -> InvestigationResult:
        if self.project_id not in self.config.allowed_projects:
            return self._result(ResultStatus.REFUSE, "Unsupported or unauthorized project.")
        router = self._fast_path_router()
        with self.recorder.span("routing") as span:
            routed = router.handle(self.query, self.access, request_id=self.request_id)
            _annotate(
                span,
                route=routed.decision.route.value,
                reason_code=routed.decision.reason_code,
                tools=",".join(routed.executed_tools),
            )
        resolved = routed.understanding.project
        if resolved and resolved.project_id not in (None, self.project_id):
            raise ValueError("router scope mismatch")
        with self.recorder.span("attention") as span:
            signals, signal_notes = self._attention_signals(router, routed)
            _annotate(span, signal_count=len(signals))
        common = dict(routed=routed, attention_signals=signals, limitations=signal_notes)
        decision = routed.decision
        if decision.route == Route.REFUSE or decision.decided_at_stage == "input":
            status = ResultStatus.REFUSE if decision.route == Route.REFUSE else ResultStatus.CLARIFY
            return self._result(status, decision.detail, **common)
        if decision.route == Route.STRUCTURED and routed.outcome == ExecutionOutcome.EXECUTED:
            return self._deterministic(routed, common)  # confident fast path, no model call
        return self._investigate(routed, router, common)

    def _fast_path_router(self) -> RoutingService:
        """Request deadline; no document search inside routing (the Investigator owns it)."""
        deadline = self.started + self.config.overall_deadline_seconds
        factory = self.copilot.router.context_factory

        def context(request_id):
            ctx = factory(request_id)
            ctx.deadline = deadline if ctx.deadline is None else min(ctx.deadline, deadline)
            return ctx

        return replace(self.copilot.router, context_factory=context, documents=None)

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
        """Confident structured route: governed records shown as-is, no model call."""
        items = tuple(
            _tool_evidence(result, item) for result in routed.tool_results for item in result.items
        )
        _owned([item.payload for item in items], self.project_id)
        notes = tuple(
            note
            for result in routed.tool_results
            for note in (*result.caveats, *(m.detail for m in result.mechanical))
        )
        status, message = (
            (ResultStatus.EVIDENCE_ONLY, "Governed records (no model used).")
            if items
            else (
                ResultStatus.INSUFFICIENT_EVIDENCE,
                "UNKNOWN: no qualifying governed evidence was found.",
            )
        )
        common["limitations"] = (*common["limitations"], *notes)
        return self._result(status, message, evidence=items, **common)

    # -- investigation -------------------------------------------------------------------
    def _investigate(self, routed: RouteResult, router, common: dict) -> InvestigationResult:
        self.investigated = True
        bounds = self.config.investigation
        executor = GovernedExecutor(
            tools=self.copilot.router.executor,
            documents=self.copilot.documents,
            context_factory=router.context_factory,
            project_id=self.project_id,
            request_id=self.request_id,
        )
        gathered = Gathered()
        decision, rejected, anchor_choice = None, 0, None
        for round_number in range(1, bounds.max_decision_rounds + 1):
            remaining = bounds.max_tool_calls - len(gathered.calls)
            with self.recorder.span(f"investigator_{round_number}") as span:
                decision, failure = self._call(
                    "INVESTIGATOR",
                    INVESTIGATOR_INSTRUCTIONS,
                    self._investigator_payload(round_number, gathered, remaining, decision),
                    InvestigatorDecision,
                    span,
                )
            if failure is not None:
                return self._result(
                    ResultStatus.FAIL_CLOSED,
                    "The investigation could not be planned.",
                    validation=Validation(failures=(failure.value,)),
                    **common,
                )
            self.activity["decision_rounds"] = round_number
            stop = self._disposition(decision, common)
            if stop is not None:
                return stop
            anchor_choice = decision.temporal_anchor or anchor_choice
            if decision.disposition == "ANSWER_NOW" or not decision.actions:
                break
            with self.recorder.span(f"tools_{round_number}") as span:
                for action in decision.actions[:remaining]:
                    call = govern(action, self.project_id)
                    if call is None:
                        rejected += 1
                        continue
                    try:
                        executor.run(call, gathered)
                    except GovernanceViolation as exc:  # objective scope/identity failure
                        return self._result(
                            ResultStatus.FAIL_CLOSED,
                            "No answer published: a governed read violated project scope.",
                            validation=Validation(
                                disposition="FAIL_CLOSED",
                                mechanical_validity="INVALID",
                                failures=(str(exc),),
                            ),
                            **common,
                        )
                rejected += max(0, len(decision.actions) - remaining)
                _annotate(span, tool_calls=len(gathered.calls), evidence=len(gathered.refs))
            if not decision.review_evidence or len(gathered.calls) >= bounds.max_tool_calls:
                break
        self.activity.update(
            tool_calls=len(gathered.calls),
            rejected_actions=rejected,
            evidence_retrieved=len(gathered.refs),
        )
        return self._answer(decision, gathered, anchor_choice, routed, common)

    def _disposition(self, decision, common) -> InvestigationResult | None:
        if decision.disposition == "PREDICTION":
            return self._result(ResultStatus.REFUSE, PREDICTION_MESSAGE, **common)
        if decision.disposition == "OUT_OF_SCOPE":
            return self._result(
                ResultStatus.REFUSE,
                "This question is outside project implementation intelligence.",
                **common,
            )
        if decision.disposition == "CLARIFY":
            return self._result(
                ResultStatus.CLARIFY,
                decision.clarification or "Please rephrase the question.",
                objective=decision.objective,
                **common,
            )
        return None

    def _investigator_payload(self, round_number, gathered, remaining, previous) -> dict:
        payload = {
            "round": round_number,
            "question": self.query,
            "tools": tool_catalog(),
            "remaining_tool_calls": remaining,
        }
        if round_number > 1:
            entries = pack(
                gathered,
                set(gathered.refs),
                max_bytes=self.config.investigation.context_max_bytes // 2,
                max_text_chars=self.config.investigation.evidence_text_chars // 2,
            )
            context = self._context(previous.objective, entries, gathered)
            shown = project(context, objective=previous.objective)
            self.anchor_handles = shown.evidence  # an anchor must resolve as the model saw it
            payload.update(
                objective=previous.objective,
                calls_made=[
                    {"tool": c.tool, "query": c.query, "status": c.status, "evidence": c.evidence}
                    for c in gathered.calls
                ],
                evidence=shown.payload["evidence"],
            )
        return payload

    # -- answer --------------------------------------------------------------------------
    def _answer(self, decision, gathered, anchor_choice, routed, common) -> InvestigationResult:
        bounds = self.config.investigation
        objective = decision.objective
        if not gathered.refs:
            return self._result(
                ResultStatus.INSUFFICIENT_EVIDENCE,
                "UNKNOWN: the investigation found no governed evidence for this question.",
                objective=objective,
                **common,
            )
        keep, periods = set(gathered.refs), None
        if anchor_choice is not None:
            anchor_id = self.anchor_handles.get(anchor_choice.event)
            keep, periods, dropped, usable = (
                anchor(gathered, anchor_id, anchor_choice.relation)
                if anchor_id
                else (keep, None, 0, False)
            )
            if usable:
                self.activity.update(
                    temporal_anchor=anchor_choice.relation, evidence_filtered_by_date=dropped
                )
            else:
                common["limitations"] = (
                    *common["limitations"],
                    "The requested event has no source-stated date; no date filter was applied.",
                )
        entries = pack(
            gathered,
            keep,
            max_bytes=bounds.context_max_bytes,
            max_text_chars=bounds.evidence_text_chars,
        )
        self.activity["evidence_shown"] = len(entries)
        context = self._context(objective, entries, gathered)
        evidence = tuple(_package_evidence(gathered.refs[i]) for i in gathered.refs if i in keep)
        projection = project(context, objective=objective, periods=periods)
        with self.recorder.span("synthesis") as span:
            semantic, failure = self._call(
                "SYNTHESIZER", SYNTHESIS_INSTRUCTIONS, projection.payload, SemanticSynthesis, span
            )
        if failure is not None:
            return self._fail(failure, objective, evidence, common)
        with self.recorder.span("enrichment") as span:
            output, dropped, stats = enrich(semantic, projection, context)
            _annotate(span, **stats.attributes(projection))
        with self.recorder.span("integrity") as span:
            integrity = check_integrity(output, context)
            _annotate(
                span,
                claims_checked=len(output.candidate_claims),
                claims_removed=sum(integrity.removed.values()),
                security_failures=",".join(integrity.security_failures),
            )
        critic_enabled = self.config.models.critic_enabled
        review, critic_status = None, CriticStatus.NOT_REQUIRED
        if integrity.security_failures:
            critic_status = CriticStatus.NOT_REQUIRED
        elif not critic_enabled:
            critic_status = CriticStatus.DISABLED
        elif integrity.claims:
            with self.recorder.span("critic") as span:
                review, failure = self._call(
                    "CRITIC",
                    CRITIC_INSTRUCTIONS,
                    critic_payload(projection, integrity.claims),
                    SemanticReview,
                    span,
                )
            if failure is not None:  # fail closed: no claim is published as reviewed
                return self._fail(failure, objective, evidence, common, critic=True)
            critic_status = CriticStatus.REVIEWED
        with self.recorder.span("finalization") as span:
            final = finalize(
                integrity,
                review,
                critic_enabled=critic_enabled,
                model_insufficient=output.insufficient_evidence,
                limitations=(*context.limitations, *output.limitations),
                removed_before=dropped,
            )
            _annotate(
                span,
                disposition=final.disposition,
                published=len(final.published),
                partially_supported=sum(
                    p.support == "PARTIALLY_SUPPORTED" for p in final.published
                ),
                **{f"removed_{k.lower()}": v for k, v in final.removed.items()},
            )
        cited = {e["evidence_id"]: e for e in context.evidence}
        claims = tuple(_claim(p, cited) for p in final.published)
        if final.disposition == "FAIL_CLOSED":
            status, message = ResultStatus.FAIL_CLOSED, "No answer published: integrity failure."
        elif claims:
            status, message = ResultStatus.ANSWER, "Validated, cited claims."
        else:
            status, message = (
                ResultStatus.INSUFFICIENT_EVIDENCE,
                "UNKNOWN: the evidence does not support a publishable answer.",
            )
        common["limitations"] = (*common["limitations"], *final.limitations)
        return self._result(
            status,
            message,
            claims=claims,
            evidence=evidence,
            objective=objective,
            validation=Validation(
                disposition=final.disposition,
                mechanical_validity="INVALID" if final.failures else "VALID",
                failures=final.failures,
                semantic_support="MODEL_ASSESSED" if review is not None else "NOT_ASSESSED",
                critic_status=critic_status,
                claims_removed=final.removed,
            ),
            **common,
        )

    def _context(self, objective, entries, gathered):
        scope = parse_temporal(self.query).model_dump(mode="json")
        return approved_context(
            request_id=self.request_id,
            project_id=self.project_id,
            question=self.query,
            objective=objective,
            temporal_scope=scope,
            entries=entries,
            gathered=gathered,
        )

    def _fail(self, failure, objective, evidence, common, *, critic=False):
        return self._result(
            ResultStatus.FAIL_CLOSED,
            "No answer published: a model step failed.",
            evidence=evidence,
            objective=objective,
            validation=Validation(
                disposition="FAIL_CLOSED",
                mechanical_validity="INVALID",
                failures=(failure.value,),
                critic_status=CriticStatus.FAILED if critic else CriticStatus.NOT_REQUIRED,
            ),
            **common,
        )

    # -- model calls ---------------------------------------------------------------------
    def _call(self, role: str, system: str, payload: dict, schema, span):
        """One bounded model call; returns (parsed output, None) or (None, Failure)."""
        models = self.config.models
        adapter = {
            "INVESTIGATOR": self.copilot.investigator,
            "SYNTHESIZER": self.copilot.synthesizer,
            "CRITIC": self.copilot.critic,
        }[role]
        if adapter is None:
            raise ValueError(f"{role.lower()} model not configured")
        endpoint = {
            "INVESTIGATOR": models.investigator_endpoint,
            "SYNTHESIZER": models.synthesizer_endpoint,
            "CRITIC": models.critic_endpoint,
        }[role]
        remaining = self.started + self.config.overall_deadline_seconds - time.monotonic()
        called, reply, failure, parsed, reason = time.monotonic(), None, None, None, None
        if remaining <= 0:
            failure = Failure.BUDGET_EXHAUSTED
        else:
            request = ModelRequest(
                # The pinned request contract knows two roles; the Investigator uses the
                # same bounded structured-output transport as the Synthesizer.
                role="SYNTHESIZER" if role == "INVESTIGATOR" else role,
                system=system,
                context_json=json.dumps(payload, sort_keys=True, separators=(",", ":")),
                output_schema=schema.model_json_schema(),
                max_output_tokens=models.max_output_tokens,
                timeout_seconds=min(models.timeout_seconds, remaining),
            )
            try:
                reply = adapter.invoke(request)
            except NodeError as exc:  # envelope/transport: reason from a diagnosed adapter
                failure, reason = exc.category, getattr(adapter, "last_reason", None)
            except TimeoutError:
                failure, reason = Failure.MODEL_TIMEOUT, DiagnosticReason.NO_RESPONSE
            else:
                try:
                    parsed = parse_output(reply.text, schema)
                except NodeError as exc:  # valid envelope, invalid output content
                    failure, reason = exc.category, PARSE_REASONS.get(exc.category)
        call = ModelCall(
            role=role,
            endpoint=endpoint,
            model_identity=reply.model_identity if reply else None,
            latency_ms=(time.monotonic() - called) * 1000,
            input_tokens=reply.input_tokens if reply else None,
            output_tokens=reply.output_tokens if reply else None,
            outcome=failure.value if failure else "COMPLETED",
            diagnostic_reason=reason.value if reason else None,
        )
        self.model_calls.append(call)
        _annotate(span, **call.model_dump(mode="json", exclude={"role"}))
        return parsed, failure

    # -- result assembly -----------------------------------------------------------------
    def _result(self, status, message, *, routed: RouteResult | None = None, **fields):
        understanding = routed.understanding if routed else None
        intent = understanding.intent if understanding else None
        return InvestigationResult(
            request_id=self.request_id,
            query=self.query,
            project_id=self.project_id,
            route="INVESTIGATOR"
            if self.investigated
            else routed.decision.route.value
            if routed
            else None,
            reason_code=routed.decision.reason_code if routed else None,
            intent=intent.intent.value if intent and intent.intent else None,
            temporal_scope=understanding.temporal.model_dump(mode="json")
            if understanding and understanding.temporal
            else None,
            status=status,
            message=message,
            model_calls=tuple(self.model_calls),
            activity=InvestigationActivity(**self.activity) if self.activity else None,
            model_capability_note=self.config.models.capability_note if self.model_calls else None,
            **fields,
        )


def _annotate(span, **attributes) -> None:
    """Set allowlisted scalar attributes on an MLflow span (no-op without tracing)."""
    if span is not None:
        span.set_attributes({k: v for k, v in attributes.items() if v is not None})


def _request_attributes(result: InvestigationResult, config: CopilotConfig) -> dict:
    """Request-level trace metadata: identifiers, outcomes and counts.

    Never user query text, objectives, documents, prompts or model output.
    """
    calls = result.model_calls
    activity = result.activity or InvestigationActivity()
    return {
        "request_id": result.request_id,
        "project_id": result.project_id,
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
        "disposition": result.validation.disposition,
        "validation_failures": ",".join(result.validation.failures),
        "decision_rounds": activity.decision_rounds,
        "tool_calls": activity.tool_calls,
        "model_call_count": len(calls),
        "model_endpoints": ",".join(sorted({c.endpoint for c in calls if c.endpoint})),
        "model_diagnostic_reasons": ",".join(
            c.diagnostic_reason for c in calls if c.diagnostic_reason
        ),
        "input_tokens": sum(c.input_tokens or 0 for c in calls),
        "output_tokens": sum(c.output_tokens or 0 for c in calls),
        "latency_ms": round(result.latency_ms, 1),
    }


def _claim(published, cited: dict[str, dict]) -> Claim:
    claim = published.claim
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
        support=published.support,
        qualifier=published.qualifier,
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
