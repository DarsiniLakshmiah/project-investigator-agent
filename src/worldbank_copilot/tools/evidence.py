"""Evidence status and provenance contract for Phase 10 (Phase 9F).

Maps a tool envelope to the closed set of evidence states Phase 10 consumes, and checks the
provenance rules the tool layer must keep. Nothing here judges whether evidence supports a
claim: INSUFFICIENT_EVIDENCE and CONFLICTING_EVIDENCE are decided only by Phase 10
synthesis, so no mapping in this module ever produces them.

* OK                      - the tool returned records;
* NO_EVIDENCE             - the query ran and found nothing usable (EMPTY, NOT_FOUND,
                            NOT_COVERED, mechanical INSUFFICIENT_EVIDENCE). For document
                            search this means "not found in the retrieved candidates", never
                            "the documents do not contain it";
* CLARIFICATION_REQUIRED  - an argument matched several candidates and was not guessed;
* SCOPE_REFUSED           - the project scope check refused the call;
* TOOL_ERROR              - the call failed (error, timeout, invalid argument, data
                            integrity, or a provenance-contract violation). Never evidence.
"""

from __future__ import annotations

from collections.abc import Iterator
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from worldbank_copilot.tools.models import Fact, ProvenanceClass, ToolResult, ToolStatus


class EvidenceStatus(StrEnum):
    OK = "OK"
    NO_EVIDENCE = "NO_EVIDENCE"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    SCOPE_REFUSED = "SCOPE_REFUSED"
    TOOL_ERROR = "TOOL_ERROR"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"  # Phase 10 synthesis only
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"  # Phase 10 synthesis only


SYNTHESIS_ONLY = frozenset(
    {EvidenceStatus.INSUFFICIENT_EVIDENCE, EvidenceStatus.CONFLICTING_EVIDENCE}
)

TOOL_STATUS_MAP: dict[ToolStatus, EvidenceStatus] = {
    ToolStatus.OK: EvidenceStatus.OK,
    ToolStatus.EMPTY: EvidenceStatus.NO_EVIDENCE,
    ToolStatus.NOT_FOUND: EvidenceStatus.NO_EVIDENCE,
    ToolStatus.NOT_COVERED: EvidenceStatus.NO_EVIDENCE,
    ToolStatus.INSUFFICIENT_EVIDENCE: EvidenceStatus.NO_EVIDENCE,  # mechanical only
    ToolStatus.AMBIGUOUS_ARGUMENT: EvidenceStatus.CLARIFICATION_REQUIRED,
    ToolStatus.SCOPE_REFUSED: EvidenceStatus.SCOPE_REFUSED,
    ToolStatus.INVALID_ARGUMENT: EvidenceStatus.TOOL_ERROR,
    ToolStatus.DATA_INTEGRITY_ERROR: EvidenceStatus.TOOL_ERROR,
    ToolStatus.TIMEOUT: EvidenceStatus.TOOL_ERROR,
    ToolStatus.ERROR: EvidenceStatus.TOOL_ERROR,
}

# SYSTEM_DERIVED_SIGNAL may only come from Phase 7 rule outputs.
SIGNAL_TOOL = "get_attention_signals"
OVERVIEW_TOOL = "get_project_overview"
OVERVIEW_SIGNAL_FACTS = frozenset(
    {"current_attention_signal_count", "current_watch_signal_count", "current_high_signal_count"}
)


def _classified(obj: Any, label: str) -> Iterator[tuple[str, ProvenanceClass]]:
    """(label, class) for every provenance-classified value inside a tool item."""
    if isinstance(obj, Fact):
        yield obj.name, obj.provenance_class
    elif isinstance(obj, BaseModel):
        cls = getattr(obj, "provenance_class", None)
        if isinstance(cls, ProvenanceClass):
            yield type(obj).__name__, cls
        for name in type(obj).model_fields:
            if name != "provenance_class":
                yield from _classified(getattr(obj, name), name)
    elif isinstance(obj, list | tuple | set):
        for item in obj:
            yield from _classified(item, label)
    elif isinstance(obj, dict):
        for key, item in obj.items():
            yield from _classified(item, str(key))


def provenance_violations(result: ToolResult) -> list[str]:
    """Provenance rules a tool result breaks (empty = compliant)."""
    out = []
    for label, cls in _classified(result.items, "items"):
        if cls == ProvenanceClass.AI_INTERPRETATION:
            out.append(f"{result.tool}: {label} is AI_INTERPRETATION (never produced by tools)")
        elif cls == ProvenanceClass.SYSTEM_DERIVED_SIGNAL and not (
            result.tool == SIGNAL_TOOL
            or (result.tool == OVERVIEW_TOOL and label in OVERVIEW_SIGNAL_FACTS)
        ):
            out.append(
                f"{result.tool}: {label} is SYSTEM_DERIVED_SIGNAL outside the Phase 7 rule outputs"
            )
    return out


def tool_evidence_status(result: ToolResult) -> EvidenceStatus:
    """The evidence state of one tool call. A provenance violation is a TOOL_ERROR."""
    status = TOOL_STATUS_MAP[result.status]
    if status == EvidenceStatus.OK and provenance_violations(result):
        return EvidenceStatus.TOOL_ERROR
    return status
