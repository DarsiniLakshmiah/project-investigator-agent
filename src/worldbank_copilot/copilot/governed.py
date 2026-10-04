"""Governed evidence execution, objective date anchoring and relevance-aware packing.

Executes Investigator calls only through the existing ToolExecutor (structured tools) and
DocumentRetrieval (Phase 8: project filter, hybrid retrieval, reranking). Results become
existing ``EvidenceReference`` objects via ``assembly.references``. Integrity violations
(ownership, retrieval scope, envelope mismatch) raise ``GovernanceViolation``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from worldbank_copilot.copilot.investigator import SPECS, GovernedCall
from worldbank_copilot.investigation.assembly import references
from worldbank_copilot.investigation.evidence import _owned
from worldbank_copilot.investigation.evidence_models import (
    EvidenceReference,
    OperationRecord,
    OperationStatus,
    OperationType,
)
from worldbank_copilot.investigation.models import BASELINE_ID, fingerprint
from worldbank_copilot.investigation.synthesis import ApprovedContext, source_identity
from worldbank_copilot.retrieval.contract import (
    DocumentRetrieval,
    RetrievalRequest,
    RetrievalResult,
    RetrievalStatus,
)
from worldbank_copilot.routing.models import AnchorCandidate, TemporalKind, TemporalStatus
from worldbank_copilot.tools.executor import ToolExecutor
from worldbank_copilot.tools.models import ToolResult, ToolStatus

log = logging.getLogger(__name__)
OBJECTIVE = "objective"  # the single requirement every claim answers (no templates)
_DATE_KEYS = ("document_date", "event_date", "observed_date")
_TRIMMED_KEYS = ("text", "context_text", "event_description", "signal_description")
_NO_EVIDENCE = {
    ToolStatus.EMPTY,
    ToolStatus.NOT_FOUND,
    ToolStatus.NOT_COVERED,
    ToolStatus.INSUFFICIENT_EVIDENCE,
}


class GovernanceViolation(Exception):
    """Objective integrity failure: the whole response fails closed."""


@dataclass
class CallSummary:
    tool: str
    status: str
    evidence: int
    query: str | None = None  # Investigator-visible only; never traced


@dataclass
class Gathered:
    """Evidence collected so far, in per-call relevance order."""

    refs: dict[str, EvidenceReference] = field(default_factory=dict)
    by_call: list[list[str]] = field(default_factory=list)
    calls: list[CallSummary] = field(default_factory=list)

    def add(self, summary: CallSummary, incoming) -> None:
        ids = []
        for ref in incoming:
            self.refs.setdefault(ref.evidence_id, ref)
            ids.append(ref.evidence_id)
        summary.evidence = len(ids)
        self.calls.append(summary)
        self.by_call.append(list(dict.fromkeys(ids)))


@dataclass
class GovernedExecutor:
    tools: ToolExecutor
    documents: DocumentRetrieval | None
    context_factory: Any
    project_id: str
    request_id: str
    _slots: list[str] = field(default_factory=list)

    def run(self, call: GovernedCall, gathered: Gathered) -> None:
        try:
            if call.is_document:
                record, status = self._document(call)
            else:
                record, status = self._structured(call)
        except GovernanceViolation:
            raise
        except Exception:  # one failed read must not end the investigation
            log.exception("governed %s call failed request_id=%s", call.tool, self.request_id)
            gathered.add(CallSummary(call.tool, OperationStatus.TOOL_ERROR.value, 0), ())
            return
        incoming = references(record)[0] if record is not None else ()
        if any(ref.project_id != self.project_id for ref in incoming):
            raise GovernanceViolation("EVIDENCE_PROJECT_MISMATCH")
        gathered.add(CallSummary(call.tool, status, 0, call.query), incoming)

    def _structured(self, call: GovernedCall):
        ctx = self.context_factory(self.request_id)
        raw = self.tools.run(
            call.tool,
            call.arguments,
            ctx,
            scope_project_id=self.project_id,
            authorized_projects=(self.project_id,),
        )
        self._owned(raw)
        spec = SPECS[call.tool]
        result = ToolResult[spec.item_model].model_validate(raw.model_dump(mode="python"))
        if result.tool_version != spec.version or result.project_id not in (
            None,
            self.project_id,
        ):
            raise GovernanceViolation("TOOL_ENVELOPE_MISMATCH")
        if result.status == ToolStatus.SCOPE_REFUSED:
            raise GovernanceViolation("TOOL_SCOPE_REFUSED")
        status = (
            OperationStatus.OK
            if result.status == ToolStatus.OK
            else OperationStatus.NO_EVIDENCE
            if result.status in _NO_EVIDENCE
            else OperationStatus.TOOL_ERROR
        )
        record = self._record(call, OperationType.STRUCTURED, call.tool, status, result.status)
        return record.model_copy_validated(structured_result=result), result.status.value

    def _document(self, call: GovernedCall):
        if self.documents is None:
            return None, "RETRIEVAL_NOT_CONFIGURED"
        ctx = self.context_factory(self.request_id)
        raw = self.documents.retrieve(
            RetrievalRequest(project_id=self.project_id, query=call.query),
            ctx,
            authorized_projects=(self.project_id,),
        )
        self._owned(raw)
        result = RetrievalResult.model_validate(raw.model_dump(mode="python"))
        if result.profile != BASELINE_ID or result.query_used != call.query:
            raise GovernanceViolation("RETRIEVAL_ENVELOPE_MISMATCH")
        if result.status == RetrievalStatus.SCOPE_REFUSED:
            raise GovernanceViolation("RETRIEVAL_SCOPE_REFUSED")
        if (
            result.status == RetrievalStatus.OK
            and (result.filters_applied or {}).get("project_id") != self.project_id
        ):
            raise GovernanceViolation("RETRIEVAL_FILTER_SCOPE_MISMATCH")
        status = OperationStatus(result.status.value)
        record = self._record(call, OperationType.DOCUMENT, BASELINE_ID, status, result.status)
        return record.model_copy_validated(retrieval_result=result), result.status.value

    def _record(self, call, kind, tool_or_profile, status, reason) -> _Record:
        slot = len(self._slots)  # the bounded tool-call budget slot this call consumes
        self._slots.append(call.tool)
        operation_id = fingerprint(
            "op",
            {
                "request": self.request_id,
                "tool": call.tool,
                "arguments": call.arguments,
                "query": call.query,
            },
        )
        return _Record(
            operation_id=operation_id,
            requirement_ids=(OBJECTIVE,),
            project_id=self.project_id,
            operation_type=kind,
            tool_or_profile=tool_or_profile,
            status=status,
            reason_code=reason.value,
            attempted=True,
            reservation_id=f"tool_call_budget_{slot}",
            reservation_slot=slot,
        )

    def _owned(self, raw) -> None:
        try:
            _owned(raw, self.project_id)
        except Exception as exc:  # EvidenceExecutionError: foreign data in the result
            raise GovernanceViolation("DOWNSTREAM_PROJECT_MISMATCH") from exc


@dataclass(frozen=True)
class _Record:
    operation_id: str
    requirement_ids: tuple[str, ...]
    project_id: str
    operation_type: OperationType
    tool_or_profile: str
    status: OperationStatus
    reason_code: str
    attempted: bool
    reservation_id: str
    reservation_slot: int

    def model_copy_validated(self, **results) -> OperationRecord:
        return OperationRecord(**vars(self), **results)


# -- objective temporal anchoring ----------------------------------------------------
def evidence_date(ref: EvidenceReference) -> date | None:
    for key in _DATE_KEYS:
        value = ref.payload.get(key)
        if value:
            try:
                return date.fromisoformat(str(value)[:10])
            except ValueError:
                return None
    return None


def event_date_of(ref: EvidenceReference | None) -> date | None:
    """The source-stated date of a governed timeline event (never an estimated date)."""
    if ref is None or not ref.payload.get("event_type"):
        return None
    value = ref.payload.get("event_date")
    try:
        return date.fromisoformat(str(value)[:10]) if value else None
    except ValueError:
        return None


def anchor(gathered: Gathered, when: date, relation: str, event_id: str | None = None):
    """Label/filter evidence against an authoritative anchor date.

    Returns (kept evidence ids, period labels, dropped count). Evidence dated on the other
    side of a BEFORE/AFTER boundary is dropped; undated evidence is kept for ordinary facts
    but labeled UNDATED, so it can never establish a before/after relationship.
    """
    periods, kept = {}, set()
    for evidence_id, ref in gathered.refs.items():
        day = evidence_date(ref)
        period = (
            "UNDATED"
            if day is None
            else "EVENT"  # the anchor itself, or dated the same day: neither side
            if day == when or evidence_id == event_id
            else "BEFORE"
            if day < when
            else "AFTER"
        )
        periods[evidence_id] = period
        if relation == "COMPARE" or period in ("UNDATED", "EVENT", relation):
            kept.add(evidence_id)
    return kept, periods, len(gathered.refs) - len(kept)


@dataclass(frozen=True)
class AnchorOutcome:
    """Whether an event-relative relationship can be established, from governed data only."""

    status: str  # NOT_EVENT_RELATIVE | RESOLVED | AMBIGUOUS | NO_SOURCE_DATED_EVENT
    #              | NO_MATCHING_EVENT (the identified event is not in the governed timeline)
    relation: str | None = None  # BEFORE | AFTER | COMPARE
    when: date | None = None
    event_id: str | None = None  # the Investigator-chosen event record, if it was the anchor
    candidates: tuple[str, ...] = ()  # governed "title (date)" labels when ambiguous

    @property
    def resolved(self) -> bool:
        return self.status == "RESOLVED"


_DIRECTIONS = {"before": "BEFORE", "until": "BEFORE", "after": "AFTER", "since": "AFTER"}


def matching_events(events, choice) -> tuple[AnchorCandidate, ...]:
    """Source-dated governed timeline events consistent with what the Investigator identified.

    Exact comparisons only: event type and each date part the question gave. A derived
    candidate date is never used; an undated event never matches.
    """
    wanted = {"year": choice.year, "month": choice.month, "day": choice.day}
    return tuple(
        AnchorCandidate(
            timeline_event_id=e.source.record_id or str(e.event_sequence),
            event_type=e.event_type,
            event_date=e.event_date,
            title=e.event_title,
        )
        for e in events
        if e.event_date is not None
        and choice.event_type in (None, e.event_type)
        and all(v is None or getattr(e.event_date, k) == v for k, v in wanted.items())
    )


def _identifies_event(choice) -> bool:
    return choice is not None and (
        choice.event_type is not None or any((choice.year, choice.month, choice.day))
    )


def resolve_anchor(parsed, choice, chosen_event, resolver, confirm) -> AnchorOutcome:
    """The Investigator identifies the event; governed timeline data confirms and dates it.

    ``parsed`` is the deterministic temporal parse of the question; ``choice`` the
    Investigator's TemporalAnchor (or None); ``chosen_event`` the evidence its handle points
    at; ``confirm`` returns the governed events matching ``choice``; ``resolver`` resolves an
    EVENT_ANCHORED parse. The boundary date always comes from a governed, source-dated
    record, never from model or user text.
    """
    expression = next(
        (e for e in parsed.expressions if e.kind == TemporalKind.EVENT_ANCHORED), None
    )
    relation = (
        choice.relation
        if choice is not None
        else _DIRECTIONS.get(expression.anchor_direction, "COMPARE")
        if expression is not None
        else None
    )
    when = event_date_of(chosen_event)
    if when is not None:
        return AnchorOutcome("RESOLVED", relation, when, event_id=chosen_event.evidence_id)
    if _identifies_event(choice):
        return _from_candidates(relation, confirm(choice), missing="NO_MATCHING_EVENT")
    if parsed.kind != TemporalKind.EVENT_ANCHORED or expression is None:
        # The Investigator named an anchor event, but it carries no source-stated date.
        status = "NOT_EVENT_RELATIVE" if choice is None else "NO_SOURCE_DATED_EVENT"
        return AnchorOutcome(status, relation)
    scope = resolver(parsed)
    if scope.status != TemporalStatus.RESOLVED and len(scope.anchor_candidates) == 1:
        return AnchorOutcome("NO_SOURCE_DATED_EVENT", relation)
    return _from_candidates(relation, scope.anchor_candidates, missing="NO_SOURCE_DATED_EVENT")


def _from_candidates(relation, candidates, *, missing: str) -> AnchorOutcome:
    if len(candidates) == 1:
        return AnchorOutcome("RESOLVED", relation, candidates[0].event_date)
    if len(candidates) > 1:
        return AnchorOutcome(
            "AMBIGUOUS",
            relation,
            candidates=tuple(f"{c.title} ({c.event_date.isoformat()})" for c in candidates),
        )
    return AnchorOutcome(missing, relation)


# -- relevance-aware context packing -------------------------------------------------
def pack(
    gathered: Gathered,
    keep: set[str],
    *,
    max_bytes: int,
    max_text_chars: int,
) -> list[dict]:
    """Round-robin over calls in their returned (ranked) order, trimming long text.

    Every call contributes its best evidence first, so one large result cannot crowd
    out the others; entries stop at the byte budget. Trimming affects only the model
    view: identity, citation and source fields are unchanged.
    """
    queues = [[i for i in ids if i in keep] for ids in gathered.by_call]
    entries, used, seen = [], 0, set()
    while any(queues):
        for queue in queues:
            while queue and queue[0] in seen:
                queue.pop(0)
            if not queue:
                continue
            evidence_id = queue.pop(0)
            seen.add(evidence_id)
            ref = gathered.refs[evidence_id]
            entry = ref.model_dump(mode="json")
            entry["source_identity"] = source_identity(ref)
            entry["payload"] = _trim(entry["payload"], max_text_chars)
            size = len(json.dumps(entry, sort_keys=True).encode())
            if used + size > max_bytes:
                continue
            entries.append(entry)
            used += size
    return entries


def _trim(payload: dict, limit: int) -> dict:
    out = dict(payload)
    for key in _TRIMMED_KEYS:
        value = out.get(key)
        if isinstance(value, str) and len(value) > limit:
            out[key] = value[: limit - 1].rstrip() + "…"
    return out


def approved_context(
    *,
    request_id: str,
    project_id: str,
    question: str,
    objective: str,
    temporal_scope: dict,
    entries: list[dict],
    gathered: Gathered,
) -> ApprovedContext:
    """One objective requirement over the packed evidence; omissions are declared."""
    supplied = [e["evidence_id"] for e in entries]
    omitted = sorted(set(gathered.refs) - set(supplied))
    return ApprovedContext(
        request_id=request_id,
        project_id=project_id,
        plan_id=fingerprint("plan", {"calls": [vars(c) for c in gathered.calls]}),
        package_fingerprint=fingerprint("package", {"evidence": sorted(gathered.refs)}),
        temporal_scope=temporal_scope,
        question=question,
        requirements=(
            {
                "requirement_id": OBJECTIVE,
                "objective": objective,
                "required": True,
                "evidence_ids": supplied,
            },
        ),
        evidence=tuple(entries),
        omitted_evidence_ids=tuple(omitted),
        limitations=(
            "Evidence was selected by the Investigator within a bounded budget.",
            *(
                (f"{len(omitted)} further evidence records were retrieved but not shown.",)
                if omitted
                else ()
            ),
        ),
    )
