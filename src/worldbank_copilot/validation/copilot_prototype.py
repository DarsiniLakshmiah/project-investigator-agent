"""Prototype scenario checks: architectural invariants only, never expected answer text.

Each scenario reuses a reviewed Phase 9 routing case (its expected route). The checks
verify scope, routing, model-call discipline, provenance, citations and publication
rules on a real ``InvestigationResult``; the natural-language outcome is reported as-is.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from worldbank_copilot.copilot.contracts import CriticStatus, InvestigationResult, ResultStatus
from worldbank_copilot.investigation.evidence import EvidenceExecutionError, _owned
from worldbank_copilot.tools.models import ProvenanceClass

_DETERMINISTIC = frozenset({ResultStatus.EVIDENCE_ONLY, ResultStatus.INSUFFICIENT_EVIDENCE})
_SYNTHESIZED = frozenset(
    {ResultStatus.ANSWER, ResultStatus.INSUFFICIENT_EVIDENCE, ResultStatus.FAIL_CLOSED}
)
_EVIDENCE_PROVENANCE = {p.value for p in ProvenanceClass} - {"AI_INTERPRETATION"}


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    routing_case: str  # reviewed case in evaluation/routing_cases.yaml
    project_id: str
    question: str
    expected_route: str
    max_model_calls: int
    allowed_statuses: frozenset[ResultStatus]


SCENARIOS = {
    s.scenario_id: s
    for s in (
        Scenario(
            "S1", "r037", "P130544", "What deserves my attention?", "STRUCTURED", 0, _DETERMINISTIC
        ),
        Scenario(
            "S2",
            "r051",
            "P130544",
            "Why did the PDO rating drop to Moderately Unsatisfactory?",
            "INVESTIGATION",
            2,
            _SYNTHESIZED,
        ),
        Scenario(
            "S3",
            "r039",
            "P130544",
            "Show the timeline of restructurings.",
            "STRUCTURED",
            0,
            _DETERMINISTIC,
        ),
        Scenario(
            "S4",
            "r062",
            "P130544",
            "Will the project fail?",
            "REFUSE",
            0,
            frozenset({ResultStatus.REFUSE}),
        ),
    )
}


def check_invariants(
    result: InvestigationResult, scenario: Scenario, *, critic_enabled: bool
) -> dict[str, bool]:
    """Architectural invariants for one scenario result (all must be True)."""
    evidence_ids = {e.evidence_id for e in result.evidence if e.evidence_id}
    provenance_by_id = {e.evidence_id: set(e.provenance) for e in result.evidence}
    roles = Counter(c.role for c in result.model_calls)
    checks = {
        "project_scope": result.project_id == scenario.project_id,
        "route": result.route == scenario.expected_route,
        "status_allowed": result.status in scenario.allowed_statuses,
        "model_call_budget": len(result.model_calls) <= scenario.max_model_calls,
        "one_call_per_role": all(count <= 1 for count in roles.values()),
        "critic_only_if_enabled": critic_enabled or "CRITIC" not in roles,
        "no_cross_project_data": _owned_by(result, scenario.project_id),
        "evidence_provenance_valid": all(
            e.provenance and set(e.provenance) <= _EVIDENCE_PROVENANCE for e in result.evidence
        ),
        "claims_only_in_answer": bool(result.claims) == (result.status == ResultStatus.ANSWER),
        "citations_resolve": all(
            c.citations
            and all(ref.evidence_id in c.evidence_ids for ref in c.citations)
            and set(c.evidence_ids) <= evidence_ids
            for c in result.claims
        ),
        "claim_provenance_matches_evidence": all(
            c.provenance == "AI_INTERPRETATION"
            or all(provenance_by_id.get(i) == {c.provenance} for i in c.evidence_ids)
            for c in result.claims
        ),
        "capability_note_with_models": bool(result.model_calls)
        == (result.model_capability_note is not None),
    }
    if result.status == ResultStatus.ANSWER:
        checks["answer_mechanically_valid"] = result.validation.mechanical_validity == "VALID"
        checks["answer_critic_status"] = result.validation.critic_status == (
            CriticStatus.SUPPORTED if critic_enabled else CriticStatus.DISABLED
        )
    return checks


def run_scenario(copilot, scenario_id: str) -> dict:
    """Run one scenario through ``copilot.investigate`` and report invariants + result."""
    scenario = SCENARIOS[scenario_id]
    result = copilot.investigate(query=scenario.question, project_id=scenario.project_id)
    checks = check_invariants(result, scenario, critic_enabled=copilot.config.models.critic_enabled)
    return {
        "scenario": scenario.scenario_id,
        "routing_case": scenario.routing_case,
        "question": scenario.question,
        "invariants": "PASS" if all(checks.values()) else "FAIL",
        "failed_invariants": sorted(name for name, ok in checks.items() if not ok),
        "result": summarize(result),
    }


def summarize(result: InvestigationResult) -> dict:
    """Compact, display-safe view of a result for notebook output."""
    return {
        "request_id": result.request_id,
        "trace_id": result.trace_id,
        "project_id": result.project_id,
        "route": result.route,
        "intent": result.intent,
        "status": result.status.value,
        "message": result.message,
        "attention_signals": [f"{s.severity}: {s.title}" for s in result.attention_signals],
        "evidence_by_provenance": dict(Counter(p for e in result.evidence for p in e.provenance)),
        "claims": [
            {
                "text": c.text,
                "provenance": c.provenance,
                "citations": [
                    ref.document_label or ref.table or ref.evidence_id for ref in c.citations
                ],
            }
            for c in result.claims
        ],
        "validation": result.validation.model_dump(mode="json"),
        "model_calls": [c.model_dump(mode="json") for c in result.model_calls],
        "limitations": list(result.limitations),
        "latency_ms": round(result.latency_ms, 1),
        "stage_latency_ms": {k: round(v, 1) for k, v in result.stage_latency_ms.items()},
    }


def _owned_by(result: InvestigationResult, project_id: str) -> bool:
    try:
        _owned(
            [
                *(e.model_dump(mode="json") for e in result.evidence),
                *(c.model_dump(mode="json") for c in result.claims),
            ],
            project_id,
        )
    except EvidenceExecutionError:
        return False
    return True
