"""Diagnostic-only explanation of a finalizer disposition (prototype tracing).

``finalizer_branch`` names which ordered condition of the authoritative finalizer applied
(``investigation.synthesis.finalize``, or the critic-disabled finalizer). It never decides
behavior: the result is checked against the disposition actually returned, and any drift is
reported as UNCLASSIFIED instead of a wrong explanation. Attributes are booleans, integers
and fixed codes only: no text, identifiers or handles.
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum

from worldbank_copilot.investigation.claims import (
    CriticCode,
    CriticOutput,
    Disposition,
    FinalResponse,
    SynthesisOutput,
)
from worldbank_copilot.investigation.synthesis import (
    ApprovedContext,
    validate_claims,
    validate_critic,
)
from worldbank_copilot.tools.models import ProvenanceClass


class FinalizerBranch(StrEnum):
    FAIL_CLOSED = "FAIL_CLOSED"
    INSUFFICIENT_FLAG = "INSUFFICIENT_FLAG"
    NO_CLAIMS = "NO_CLAIMS"
    NO_CONTEXT_EVIDENCE = "NO_CONTEXT_EVIDENCE"  # unreachable with claims that pass validation
    UNKNOWN_CLAIM = "UNKNOWN_CLAIM"
    CRITIC_REJECTION = "CRITIC_REJECTION"
    REQUIREMENT_UNCOVERED = "REQUIREMENT_UNCOVERED"
    PUBLISH = "PUBLISH"
    UNCLASSIFIED = "UNCLASSIFIED"  # drift guard: never expected; tests assert it is unused


_DISPOSITION = {
    FinalizerBranch.FAIL_CLOSED: Disposition.FAIL_CLOSED,
    FinalizerBranch.INSUFFICIENT_FLAG: Disposition.INSUFFICIENT_EVIDENCE,
    FinalizerBranch.NO_CLAIMS: Disposition.INSUFFICIENT_EVIDENCE,
    FinalizerBranch.NO_CONTEXT_EVIDENCE: Disposition.INSUFFICIENT_EVIDENCE,
    FinalizerBranch.UNKNOWN_CLAIM: Disposition.INSUFFICIENT_EVIDENCE,
    FinalizerBranch.CRITIC_REJECTION: Disposition.REJECT_UNSUPPORTED,
    FinalizerBranch.REQUIREMENT_UNCOVERED: Disposition.INSUFFICIENT_EVIDENCE,
    FinalizerBranch.PUBLISH: Disposition.PUBLISH_WITH_LIMITATIONS,
}
_PROVENANCE_KEYS = {
    ProvenanceClass.FACT: "claims_fact",
    ProvenanceClass.DOCUMENTED_FINDING: "claims_documented_finding",
    ProvenanceClass.SYSTEM_DERIVED_SIGNAL: "claims_system_derived_signal",
    ProvenanceClass.AI_INTERPRETATION: "claims_ai_interpretation",
    ProvenanceClass.UNKNOWN: "claims_unknown",
}


def classify(
    output: SynthesisOutput | None,
    review: CriticOutput | None,
    context: ApprovedContext,
    failures=(),
    *,
    critic_disabled: bool = False,
) -> FinalizerBranch:
    """The finalizer's conditions, in the finalizer's order."""
    if output is not None:
        failures = (*failures, *validate_claims(output, context))
        if review is not None:
            failures = (*failures, *validate_critic(review, output, context))
    if failures or output is None or (review is None and not critic_disabled):
        return FinalizerBranch.FAIL_CLOSED
    if output.insufficient_evidence:
        return FinalizerBranch.INSUFFICIENT_FLAG
    if not output.candidate_claims:
        return FinalizerBranch.NO_CLAIMS
    if not context.evidence:
        return FinalizerBranch.NO_CONTEXT_EVIDENCE
    if any(c.provenance_label == ProvenanceClass.UNKNOWN for c in output.candidate_claims):
        return FinalizerBranch.UNKNOWN_CLAIM
    if review is not None and any(f.code != CriticCode.SUPPORTED for f in review.findings):
        return FinalizerBranch.CRITIC_REJECTION
    covered = {i for c in output.candidate_claims for i in c.requirement_ids}
    if any(
        r["required"] and (r["requirement_id"] not in covered or not r["evidence_ids"])
        for r in context.requirements
    ):
        return FinalizerBranch.REQUIREMENT_UNCOVERED
    return FinalizerBranch.PUBLISH


def finalization_attributes(
    final: FinalResponse,
    output: SynthesisOutput | None,
    review: CriticOutput | None,
    context: ApprovedContext,
    failures=(),
    *,
    critic_disabled: bool = False,
) -> dict:
    """Allowlisted span attributes explaining ``final.disposition`` (counts and codes)."""
    branch = classify(output, review, context, failures, critic_disabled=critic_disabled)
    if _DISPOSITION.get(branch) != final.disposition:
        branch = FinalizerBranch.UNCLASSIFIED
    supplied = {e["evidence_id"] for e in context.evidence}
    required = [r for r in context.requirements if r["required"]]
    claims = output.candidate_claims if output is not None else ()
    covered = {i for c in claims for i in c.requirement_ids}
    provenance = Counter(c.provenance_label for c in claims)
    attributes = {
        "finalizer_branch": branch.value,
        "required_requirements": len(required),
        "required_requirements_covered": sum(r["requirement_id"] in covered for r in required),
        "required_requirements_without_supplied_handle": sum(
            not supplied & set(r["evidence_ids"]) for r in required
        ),
        **{key: provenance[label] for label, key in _PROVENANCE_KEYS.items()},
    }
    if output is not None:
        attributes["synthesizer_insufficient_evidence"] = output.insufficient_evidence
    if review is not None:
        codes = Counter(f.code for f in review.findings)
        attributes.update({f"critic_{code.value.lower()}": codes[code] for code in CriticCode})
    return attributes
