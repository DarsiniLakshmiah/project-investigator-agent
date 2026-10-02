"""Routing harness: question + access context -> observable routing result (Phase 9B).

    1 input -> 2 project -> [scope refusal: stop, nothing executes]
    3 temporal -> 4 intent -> [refusal / semantic classification required: stop]
    (event anchor resolved against the SOURCE-DATED timeline, scope already checked)
    5 requirements -> 6 route -> 7 execute STRUCTURED / DOCUMENT, or 8 plan INVESTIGATION

Guarantees (tested):
* no tool or retrieval call runs before project scope has been resolved and authorised;
* every executed call is a requirement of the resolved intent, run through the
  ``ToolExecutor`` with the resolved scope (the executor re-checks it);
* there is no STRUCTURED -> DOCUMENT fallback: mechanical failures end as
  INSUFFICIENT_EVIDENCE / DATA_INTEGRITY_ERROR;
* an INVESTIGATION is only planned (dry-run validated); nothing is executed;
* no answer is synthesised; ``semantic_sufficiency`` stays NOT_ASSESSED.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta

from worldbank_copilot.routing.config import RoutingConfig
from worldbank_copilot.routing.entities import EntityIndex, resolve_project, validate_input
from worldbank_copilot.routing.intents import IntentEngine
from worldbank_copilot.routing.models import (
    AccessContext,
    AnchorCandidate,
    ExecutionOutcome,
    IntentDecision,
    IntentStatus,
    InvestigationPlan,
    ProjectStatus,
    QueryUnderstanding,
    RetrievalSpec,
    Route,
    RouteResult,
    RoutingDecision,
    StageTiming,
    TemporalKind,
    TemporalScope,
    TemporalStatus,
)
from worldbank_copilot.routing.requirements import (
    build_investigation_calls,
    build_requirements,
    needs_explicit_time,
    scoped_temporal,
)
from worldbank_copilot.routing.router import decide_intent, decide_route, decide_scope
from worldbank_copilot.routing.temporal import parse_temporal
from worldbank_copilot.tools.base import ToolContext
from worldbank_copilot.tools.executor import ToolExecutor, new_request_id
from worldbank_copilot.tools.models import ToolResult, ToolStatus

log = logging.getLogger(__name__)
ORDINALS = {"first": 0, "original": 0, "second": 1, "third": 2, "fourth": 3}
MECHANICAL = {
    ToolStatus.EMPTY,
    ToolStatus.NOT_FOUND,
    ToolStatus.NOT_COVERED,
    ToolStatus.INSUFFICIENT_EVIDENCE,
    ToolStatus.AMBIGUOUS_ARGUMENT,
}


@dataclass
class RoutingService:
    config: RoutingConfig
    index: EntityIndex
    executor: ToolExecutor
    context_factory: Callable[[str], ToolContext]  # request id -> per-request tool context
    documents: object | None = None  # tools.documents.DocumentSearch (explicit policy)
    # Optional bounded semantic fallback (Phase 9D). None = deterministic 9B.2 behaviour.
    semantic: object | None = None  # routing.semantic.KnnSemanticClassifier-like

    def __post_init__(self) -> None:
        self.engine = IntentEngine(self.config.rules)

    # ------------------------------------------------------------------------------
    def handle(
        self, question: str, access: AccessContext, request_id: str | None = None
    ) -> RouteResult:
        rid = request_id or new_request_id()
        timings: list[StageTiming] = []
        clock = time.perf_counter()

        def lap(stage: str) -> None:
            nonlocal clock
            now = time.perf_counter()
            timings.append(StageTiming(stage=stage, ms=round((now - clock) * 1000, 2)))
            clock = now

        def finish(u, decision, outcome, **kw) -> RouteResult:
            result = RouteResult(
                request_id=rid,
                question=question,
                access=access,
                understanding=u,
                decision=decision,
                outcome=outcome,
                timings=timings,
                versions=self.config.versions,
                **kw,
            )
            log.info(
                "route request_id=%s project=%s route=%s reason=%s outcome=%s tools=%s "
                "retrieval=%s",
                rid,
                u.project.project_id if u.project else None,
                decision.route.value,
                decision.reason_code,
                outcome.value,
                result.executed_tools,
                result.retrieval_executed,
            )
            return result

        inp = validate_input(question, self.config)
        u = QueryUnderstanding(input=inp)
        lap("input")
        if (stop := decide_scope(u)) is not None:
            return finish(u, stop, ExecutionOutcome.NOT_EXECUTED)
        u = u.model_copy(update={"project": resolve_project(inp.normalized, access, self.index)})
        lap("project")
        if (stop := decide_scope(u)) is not None:
            return finish(u, stop, ExecutionOutcome.NOT_EXECUTED)
        assert u.project.status == ProjectStatus.RESOLVED and u.project.project_id
        pid = u.project.project_id
        temporal = parse_temporal(inp.normalized)
        lap("temporal")
        intent = self.engine.decide(inp.normalized, temporal)
        u = u.model_copy(update={"temporal": temporal, "intent": intent})
        lap("intent")
        semantic_decision = None
        if (stop := decide_intent(u)) is not None:
            if stop.route != Route.SEMANTIC_CLASSIFICATION_REQUIRED or self.semantic is None:
                return finish(u, stop, ExecutionOutcome.NOT_EXECUTED)
            # Bounded fallback: scope, authorisation and every refusal are already decided.
            semantic_decision, intent = self._semantic(rid, inp.normalized, pid, temporal, intent)
            u = u.model_copy(update={"intent": intent})
            lap("semantic")
            if semantic_decision.abstain or intent.intent is None:
                return finish(
                    u,
                    RoutingDecision(
                        route=Route.CLARIFY,
                        reason_code="SEMANTIC_ABSTAIN",
                        detail=semantic_decision.reason,
                        decided_at_stage="semantic",
                    ),
                    ExecutionOutcome.NOT_EXECUTED,
                    semantic=semantic_decision,
                )

        tool_results: list[ToolResult] = []
        executed: list[str] = []
        anchors: list[ToolResult] = []
        ctx = self.context_factory(rid)
        if temporal.kind == TemporalKind.EVENT_ANCHORED:
            temporal = self._resolve_anchor(temporal, pid, access, ctx, anchors)
            lap("temporal_anchor")
        temporal = scoped_temporal(intent.intent, temporal, self.config.requirements)
        requirements = build_requirements(
            pid, inp.normalized, intent, temporal, u.project, self.config.requirements
        )
        u = u.model_copy(update={"temporal": temporal, "requirements": requirements})
        lap("requirements")
        decision = decide_route(
            u, explicit_time_missing=needs_explicit_time(intent, temporal, self.config.requirements)
        )
        lap("route")
        if decision.route in (Route.CLARIFY, Route.REFUSE):
            return finish(
                u,
                decision,
                ExecutionOutcome.NOT_EXECUTED,
                tool_results=tool_results,
                anchor_results=anchors,
                semantic=semantic_decision,
                executed_tools=executed,
            )
        if decision.route == Route.INVESTIGATION:
            plan = self._plan(pid, inp.normalized, intent, temporal)
            lap("plan")
            return finish(
                u,
                decision,
                ExecutionOutcome.PLANNED_NOT_EXECUTED,
                investigation_plan=plan,
                tool_results=tool_results,
                anchor_results=anchors,
                semantic=semantic_decision,
                executed_tools=executed,
            )
        retrieval = False
        if decision.route == Route.STRUCTURED:
            for call in requirements.structured:
                tool_results.append(self._run(call.tool, call.arguments, ctx, pid, access))
                executed.append(call.tool)
        else:  # DOCUMENT
            if self.documents is None:
                lap("execute")
                return finish(
                    u,
                    decision,
                    ExecutionOutcome.ERROR,
                    tool_results=tool_results,
                    anchor_results=anchors,
                    semantic=semantic_decision,
                    executed_tools=executed,
                    error="document search is not configured (no rerank policy given)",
                )
            ctx.documents = self.documents
            spec = requirements.document
            tool_results.append(
                self._run(
                    "search_project_documents",
                    {"project_id": pid, "query": spec.query},
                    ctx,
                    pid,
                    access,
                )
            )
            executed.append("search_project_documents")
            retrieval = True
        lap("execute")
        outcome = _outcome(tool_results)
        mechanical = [m for r in tool_results for m in r.mechanical]
        return finish(
            u,
            decision,
            outcome,
            tool_results=tool_results,
            anchor_results=anchors,
            semantic=semantic_decision,
            executed_tools=executed,
            retrieval_executed=retrieval,
            mechanical=mechanical,
        )

    # ------------------------------------------------------------------------------
    def _semantic(self, rid, question, pid, temporal, rules):
        """Ask the fallback for a bounded intent; anything outside the contract abstains."""
        from worldbank_copilot.routing.semantic import SEMANTIC_INTENTS, SemanticQueryContext

        context = SemanticQueryContext(
            request_id=rid,
            question=question,
            project_id=pid,
            temporal_kind=temporal.kind.value,
            rule_hits=tuple(h.rule_id for h in rules.hits),
        )
        decision = self.semantic.classify(context)
        if not decision.abstain and decision.intent not in SEMANTIC_INTENTS:
            decision = decision.model_copy(
                update={
                    "abstain": True,
                    "intent": None,
                    "route": None,
                    "reason": f"out-of-contract intent {decision.intent} rejected",
                }
            )
        if decision.abstain:
            return decision, rules
        intent = rules.model_copy(
            update={
                "status": IntentStatus.RESOLVED,
                "intent": decision.intent,
                "method": "SEMANTIC",
                "decided_by": (f"SEMANTIC:{decision.classifier}",),
                "reason": f"semantic fallback: {decision.reason}",
            }
        )
        return decision, intent

    def _run(
        self, tool: str, arguments: dict, ctx: ToolContext, pid: str, access: AccessContext
    ) -> ToolResult:
        return self.executor.run(
            tool,
            arguments,
            ctx,
            scope_project_id=pid,
            authorized_projects=access.authorized_projects,
        )

    def _resolve_anchor(
        self,
        temporal: TemporalScope,
        pid: str,
        access: AccessContext,
        ctx: ToolContext,
        anchors: list[ToolResult],
    ) -> TemporalScope:
        """Resolve 'since the <event>' against source-stated timeline dates; never guess."""
        expr = next(e for e in temporal.expressions if e.kind == TemporalKind.EVENT_ANCHORED)
        res = self._run(
            "get_project_timeline",
            {"project_id": pid, "event_types": [expr.anchor_event_type]},
            ctx,
            pid,
            access,
        )
        anchors.append(res)
        events = (
            [e for e in res.items if e.event_date is not None]
            if res.status == ToolStatus.OK
            else []
        )
        undated = (len(res.items) - len(events)) if res.status == ToolStatus.OK else 0
        q = expr.anchor_qualifier
        if q and q.isdigit():
            events = [e for e in events if e.event_date.year == int(q)]
        elif q in ORDINALS:
            index = ORDINALS[q]
            events = events[index : index + 1]
        elif q in ("last", "latest", "most recent"):
            events = events[-1:]
        candidates = tuple(
            AnchorCandidate(
                timeline_event_id=e.source.record_id or str(e.event_sequence),
                event_type=e.event_type,
                event_date=e.event_date,
                title=e.event_title,
            )
            for e in events
        )
        if len(events) != 1:
            detail = (
                f"{len(events)} source-dated {expr.anchor_event_type} events match {expr.text!r}"
                + (f" ({undated} undated not considered)" if undated else "")
                + "; the anchor is not guessed"
            )
            return temporal.model_copy(
                update={
                    "status": TemporalStatus.UNRESOLVED,
                    "anchor_candidates": candidates,
                    "detail": detail,
                }
            )
        when = events[0].event_date
        lo = hi = None
        if expr.anchor_direction == "since":
            lo = when
        elif expr.anchor_direction == "after":
            lo = when + timedelta(days=1)
        elif expr.anchor_direction == "before":
            hi = when - timedelta(days=1)
        else:
            hi = when
        return temporal.model_copy(
            update={
                "status": TemporalStatus.RESOLVED,
                "date_from": lo,
                "date_to": hi,
                "anchor_candidates": candidates,
                "detail": f"anchored on {events[0].event_title} "
                f"({when.isoformat()}, source-stated)",
            }
        )

    def _plan(
        self, pid: str, question: str, intent: IntentDecision, temporal: TemporalScope
    ) -> InvestigationPlan:
        calls = build_investigation_calls(pid, intent, self.config.requirements)
        return InvestigationPlan(
            project_id=pid,
            question=question,
            investigation_kind=intent.investigation or "UNSPECIFIED",
            triggered_by=intent.decided_by,
            temporal=temporal,
            structured_calls=tuple(calls),
            document_retrievals=(
                RetrievalSpec(
                    query=question,
                    document_type_hints=intent.document_types,
                    purpose="documentary evidence explaining the established change",
                ),
            ),
            retrieval_profile=getattr(self.documents, "profile_id", None),
        )


def _outcome(results: list[ToolResult]) -> ExecutionOutcome:
    statuses = {r.status for r in results}
    if ToolStatus.DATA_INTEGRITY_ERROR in statuses:
        return ExecutionOutcome.DATA_INTEGRITY_ERROR
    if statuses & {
        ToolStatus.ERROR,
        ToolStatus.TIMEOUT,
        ToolStatus.SCOPE_REFUSED,
        ToolStatus.INVALID_ARGUMENT,
    }:
        return ExecutionOutcome.ERROR
    if statuses & MECHANICAL:
        return ExecutionOutcome.INSUFFICIENT_EVIDENCE
    return ExecutionOutcome.EXECUTED
