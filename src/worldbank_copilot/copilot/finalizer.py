"""Claim-level integrity and publication.

Integrity is objective and deterministic: each claim is checked alone with the unchanged
``validate_claims``. Scope/identity/citation violations fail the whole response; any other
integrity failure removes only that claim. Semantic support is the Critic's judgment, per
claim: SUPPORTED keeps, PARTIALLY_SUPPORTED keeps with its qualifier, UNSUPPORTED and
CONTRADICTED remove. If the Critic itself fails (call, parse or schema), it is a secondary
reviewer, not a gate: integrity-valid claims publish as NOT_ASSESSED with a limitation, and
are never shown as critic-approved. A model-written limitation publishes only if the Critic
found it grounded; application limitations always publish. INSUFFICIENT_EVIDENCE only when
no publishable claim remains.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from worldbank_copilot.copilot.semantic import ClaimSupport, SemanticReview
from worldbank_copilot.investigation.claims import CandidateClaim, Failure, SynthesisOutput
from worldbank_copilot.investigation.synthesis import ApprovedContext, validate_claims
from worldbank_copilot.tools.models import ProvenanceClass

# Objective identity/scope failures: the response cannot be trusted at all.
SECURITY = frozenset(
    {
        Failure.PROJECT_ISOLATION_VIOLATION,
        Failure.EVIDENCE_REFERENCE_INVALID,
        Failure.CITATION_VALIDATION_FAILED,
        Failure.TEMPORAL_SCOPE_VIOLATION,
        Failure.REQUIREMENT_REFERENCE_INVALID,
        Failure.SCHEMA_VALIDATION_FAILED,
    }
)
CONTRADICTION_NOTE = "A statement contradicted by the cited evidence was removed."
UNKNOWN_NOTE = "Some requested values are not available in the governed data."
PARTIAL_ANSWER_NOTE = "The evidence answers the question only in part."
CRITIC_DISABLED_NOTE = (
    "Semantic critic review was disabled by configuration: claims passed deterministic "
    "integrity checks only and are not critic-validated."
)
CRITIC_UNAVAILABLE_NOTE = (
    "Secondary semantic review was unavailable; published claims passed deterministic "
    "evidence and citation validation only and are not critic-validated."
)


@dataclass(frozen=True)
class PublishedClaim:
    claim: CandidateClaim
    support: str  # SUPPORTED | PARTIALLY_SUPPORTED | NOT_ASSESSED
    qualifier: str | None = None


@dataclass
class Integrity:
    claims: tuple[CandidateClaim, ...]
    removed: Counter = field(default_factory=Counter)
    security_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class Finalized:
    disposition: str  # PUBLISH_WITH_LIMITATIONS | INSUFFICIENT_EVIDENCE | FAIL_CLOSED
    published: tuple[PublishedClaim, ...]
    removed: dict[str, int]
    failures: tuple[str, ...]
    limitations: tuple[str, ...]
    limitations_withheld: int = 0  # model-written limitations not found grounded


def check_integrity(output: SynthesisOutput, context: ApprovedContext) -> Integrity:
    """Each claim alone against the unchanged validator; never excused by the Critic."""
    kept, removed, security = [], Counter(), []
    for claim in output.candidate_claims:
        single = SynthesisOutput(
            candidate_claims=(claim,),
            insufficient_evidence=False,
            limitations=(),
            summary_claim_ids=(claim.claim_id,),
        )
        errors = validate_claims(single, context)
        if any(e in SECURITY for e in errors):
            security.extend(e.value for e in errors if e in SECURITY)
        elif errors:
            removed[errors[0].value] += 1
        else:
            kept.append(claim)
    return Integrity(tuple(kept), removed, tuple(dict.fromkeys(security)))


def finalize(
    integrity: Integrity,
    review: SemanticReview | None,
    *,
    critic_enabled: bool,
    critic_failed: bool = False,
    model_insufficient: bool,
    limitations: tuple[str, ...],
    model_limitations: tuple[str, ...] = (),
    removed_before: dict[str, int] | None = None,
) -> Finalized:
    removed = Counter(removed_before or {}) + integrity.removed
    if integrity.security_failures:
        return Finalized("FAIL_CLOSED", (), dict(removed), integrity.security_failures, ())
    notes = list(limitations)
    findings, grounded = {}, set()
    if review is not None:
        counts = Counter(f.claim for f in review.findings)
        findings = {f.claim: f for f in review.findings if counts[f.claim] == 1}
        verdicts = Counter(f.limitation for f in review.limitations)
        grounded = {f.limitation for f in review.limitations if f.grounded} - {
            h for h, n in verdicts.items() if n > 1
        }
    kept_limitations = [t for n, t in enumerate(model_limitations, 1) if f"L{n}" in grounded]
    published = []
    for claim in integrity.claims:
        if claim.provenance_label == ProvenanceClass.UNKNOWN:
            removed["UNKNOWN_VALUE"] += 1  # never authoritative; others still publish
            continue
        if not critic_enabled or critic_failed:  # no judgment exists: never "SUPPORTED"
            published.append(PublishedClaim(claim, "NOT_ASSESSED"))
            continue
        finding = findings.get(claim.claim_id)
        if finding is None:
            removed["NOT_REVIEWED"] += 1
        elif finding.support == ClaimSupport.SUPPORTED:
            published.append(PublishedClaim(claim, "SUPPORTED"))
        elif finding.support == ClaimSupport.PARTIALLY_SUPPORTED:
            published.append(PublishedClaim(claim, "PARTIALLY_SUPPORTED", finding.rationale))
        else:
            removed[finding.support.value] += 1
    if published:
        notes.extend(kept_limitations)
    if removed.get("UNKNOWN_VALUE"):
        notes.append(UNKNOWN_NOTE)
    if removed.get(ClaimSupport.CONTRADICTED.value):
        notes.append(CONTRADICTION_NOTE)
    if model_insufficient and published:
        notes.append(PARTIAL_ANSWER_NOTE)
    if not critic_enabled and published:
        notes.append(CRITIC_DISABLED_NOTE)
    elif critic_failed and published:
        notes.append(CRITIC_UNAVAILABLE_NOTE)
    return Finalized(
        "PUBLISH_WITH_LIMITATIONS" if published else "INSUFFICIENT_EVIDENCE",
        tuple(published),
        dict(removed),
        (),
        tuple(dict.fromkeys(notes)),
        len(model_limitations) - (len(kept_limitations) if published else 0),
    )
