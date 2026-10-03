"""Offline regressions: UNKNOWN evidence keeps UNKNOWN/UNCERTAINTY semantics (@4)."""

import hashlib
import json
from pathlib import Path

import pytest

from worldbank_copilot.investigation.claims import (
    CriticOutput,
    Disposition,
    Failure,
    SynthesisOutput,
)
from worldbank_copilot.investigation.evidence_models import EvidenceReference, OperationType
from worldbank_copilot.investigation.models import fingerprint
from worldbank_copilot.investigation.synthesis import (
    CRITIC_INSTRUCTIONS,
    INSTRUCTIONS,
    ApprovedContext,
    finalize,
    source_identity,
    validate_claims,
)
from worldbank_copilot.tools.models import ProvenanceClass, SourceRef
from worldbank_copilot.validation import copilot_e2e
from worldbank_copilot.validation import phase10d_models as h
from worldbank_copilot.validation.phase10d_fixtures import claim, draft, fixture

ROOT = Path(__file__).resolve().parents[2]
LOCK_V3_SHA = "3bf743732a9f8fca8e3d4ce22a7d89720b6c3d2b66b2630d9b7cd9c06423323b"
E2E_BEFORE_V4_SHA = "e64136516075178f746f238cda947a9717e3ccbabbfdcbea2a0d970824e9f5da"
CASES_SHA = "a59f80bceb5df9563e475868f3373dffa107227c9e705f2b1d7f376b26aef685"
INSTRUCTIONS_SHA = "58a8d7dfc71d84ed6d256f01b4263eb526003086b68900165ed503a11adbbe53"
CRITIC_INSTRUCTIONS_SHA = "9fe1d36193d3cf316f20a4dc39a03d3a91392ab09bf4428c52e4c30bb3acbb28"
RELABELS = ("FACT", "DOCUMENTED_FINDING", "SYSTEM_DERIVED_SIGNAL", "AI_INTERPRETATION")
CLAIM_TYPE = {
    "FACT": "ASSERTION",
    "DOCUMENTED_FINDING": "ASSERTION",
    "SYSTEM_DERIVED_SIGNAL": "ASSERTION",
    "AI_INTERPRETATION": "INTERPRETATION",
    "UNKNOWN": "UNCERTAINTY",
}


def entry(kind, record):
    identity = {
        "project_id": "P000001",
        "table": "synthetic.gold",
        "record_id": record,
        "field": "implementation_status",
    }
    ref = EvidenceReference(
        evidence_id=fingerprint("ev", identity),
        project_id="P000001",
        source_type=OperationType.STRUCTURED,
        identity=identity,
        provenance=ProvenanceClass(kind),
        payload={
            "value": None if kind == "UNKNOWN" else "Delay observed",
            "unknown_reason": "Not supplied" if kind == "UNKNOWN" else None,
        },
        source=SourceRef(table="synthetic.gold", record_id=record),
        citation_support="SOURCE_REFERENCE",
    )
    data = ref.model_dump(mode="json")
    data["source_identity"] = source_identity(ref)
    return data


def mixed_context():
    """One requirement linked to FACT and UNKNOWN package evidence."""
    known, unknown = entry("FACT", "record1"), entry("UNKNOWN", "record2")
    data = fixture().model_dump(mode="json")
    data["evidence"] = [known, unknown]
    data["requirements"][0]["evidence_ids"] = [known["evidence_id"], unknown["evidence_id"]]
    return ApprovedContext.model_validate(data), known, unknown


def cites(context, *entries, label):
    return draft(
        context,
        provenance_label=label,
        claim_type=CLAIM_TYPE[label],
        evidence_ids=[e["evidence_id"] for e in entries],
        citations=[
            {"evidence_id": e["evidence_id"], "source_identity": e["source_identity"]}
            for e in entries
        ],
    )


@pytest.mark.parametrize("label", RELABELS)
def test_unknown_evidence_cannot_be_relabeled(label):
    context = fixture("UNKNOWN")
    output = cites(context, context.evidence[0], label=label)
    assert Failure.PROVENANCE_VIOLATION in validate_claims(output, context)


def test_observed_ai_interpretation_shape_is_rejected_and_fails_closed():
    # Shape observed in preserved 10d6 S-unknown output; 10d6 itself is not rescored.
    context = fixture("UNKNOWN")
    output = draft(
        context,
        claim_text="The implementation status is missing.",
        claim_type="INTERPRETATION",
        provenance_label="AI_INTERPRETATION",
    )
    assert validate_claims(output, context) == (Failure.PROVENANCE_VIOLATION,)
    review = CriticOutput.model_validate(
        {
            "findings": [
                {
                    "claim_id": "C1",
                    "code": "SUPPORTED",
                    "evidence_ids": [context.evidence[0]["evidence_id"]],
                    "concise_rationale": "Supported.",
                }
            ]
        }
    )
    final = finalize(output, review, context)
    assert final.disposition == Disposition.FAIL_CLOSED
    assert Failure.PROVENANCE_VIOLATION in final.failures
    assert final.published_claims == ()


def test_authoritative_provenance_comes_from_package_evidence_not_claim():
    context = fixture("UNKNOWN")
    assert context.evidence[0]["provenance"] == "UNKNOWN"
    output = draft(context, claim_type="INTERPRETATION", provenance_label="AI_INTERPRETATION")
    assert Failure.PROVENANCE_VIOLATION in validate_claims(output, context)
    # Same claim over authoritative FACT evidence keeps the existing interpretation behavior.
    fact = fixture("FACT")
    output = draft(fact, claim_type="INTERPRETATION", provenance_label="AI_INTERPRETATION")
    assert validate_claims(output, fact) == ()


def test_valid_unknown_uncertainty_handling_is_preserved():
    context = fixture("UNKNOWN")
    output = draft(
        context,
        claim_text="The status is unknown.",
        claim_type="UNCERTAINTY",
        provenance_label="UNKNOWN",
    )
    assert validate_claims(output, context) == ()
    abstain = SynthesisOutput(
        candidate_claims=(), insufficient_evidence=True, limitations=(), summary_claim_ids=()
    )
    assert validate_claims(abstain, context) == ()
    # UNKNOWN label with the wrong claim type remains a violation, as before.
    wrong_type = draft(context, claim_type="ASSERTION", provenance_label="UNKNOWN")
    assert Failure.PROVENANCE_VIOLATION in validate_claims(wrong_type, context)


@pytest.mark.parametrize("kind", ["FACT", "DOCUMENTED_FINDING", "SYSTEM_DERIVED_SIGNAL"])
def test_non_unknown_evidence_behavior_unchanged(kind):
    context = fixture(kind)
    assert validate_claims(draft(context), context) == ()
    other = next(k for k in ("FACT", "DOCUMENTED_FINDING") if k != kind)
    mislabeled = cites(context, context.evidence[0], label=other)
    assert validate_claims(mislabeled, context) == (Failure.PROVENANCE_VIOLATION,)
    interpretation = cites(context, context.evidence[0], label="AI_INTERPRETATION")
    assert validate_claims(interpretation, context) == ()


@pytest.mark.parametrize("label", [*RELABELS, "UNKNOWN"])
def test_mixed_unknown_and_known_evidence_cannot_bypass(label):
    context, known, unknown = mixed_context()
    for order in ((known, unknown), (unknown, known)):
        output = cites(context, *order, label=label)
        assert Failure.PROVENANCE_VIOLATION in validate_claims(output, context)


def test_mixed_context_separate_claims_remain_valid():
    context, known, unknown = mixed_context()
    facts = claim(
        context,
        evidence_ids=[known["evidence_id"]],
        citations=[
            {"evidence_id": known["evidence_id"], "source_identity": known["source_identity"]}
        ],
    )
    uncertain = claim(
        context,
        claim_id="C2",
        claim_text="The second record's status is unknown.",
        claim_type="UNCERTAINTY",
        provenance_label="UNKNOWN",
        evidence_ids=[unknown["evidence_id"]],
        citations=[
            {"evidence_id": unknown["evidence_id"], "source_identity": unknown["source_identity"]}
        ],
    )
    output = SynthesisOutput(
        candidate_claims=(facts, uncertain),
        insufficient_evidence=False,
        limitations=(),
        summary_claim_ids=("C1", "C2"),
    )
    assert validate_claims(output, context) == ()


def test_capability_lock_v4_lineage_changes_only_validator_sources():
    cases, active = h.prepare(ROOT)
    archived = json.loads((ROOT / h.PREVIOUS_LOCK_FILE).read_text("utf-8"))
    assert h.PREVIOUS_LOCK_FILE == "evaluation/phase10d_model_lock_v3.json"
    assert h.canonical_sha256(ROOT / h.PREVIOUS_LOCK_FILE) == LOCK_V3_SHA
    assert archived["schema"] == "phase10d_capability_lock@3"
    assert active == h.build_lock(ROOT)
    assert active["schema"] == "phase10d_capability_lock@4"
    assert active["supersedes_lock_sha256_lf"] == LOCK_V3_SHA
    changed = {k for k in active.keys() | archived.keys() if active.get(k) != archived.get(k)}
    assert changed == {"schema", "supersedes_lock_sha256_lf", "files_sha256_lf"}
    old, new = archived["files_sha256_lf"], active["files_sha256_lf"]
    assert {name for name in new if new[name] != old.get(name)} == {
        "src/worldbank_copilot/investigation/synthesis.py",
        "src/worldbank_copilot/validation/phase10d_models.py",
        "tests/unit/test_phase10d_transport_compatibility.py",
        "tests/unit/test_phase10d_unknown_provenance.py",
    }
    assert set(old) <= set(new)
    # Frozen experiment, prompts and cases unchanged.
    assert h.canonical_sha256(ROOT / h.CASE_FILE) == CASES_SHA
    assert len(cases) == 11 and sum(c["repeats"] for c in cases) == 19
    assert hashlib.sha256(INSTRUCTIONS.encode()).hexdigest() == INSTRUCTIONS_SHA
    assert hashlib.sha256(CRITIC_INSTRUCTIONS.encode()).hexdigest() == CRITIC_INSTRUCTIONS_SHA


def test_copilot_e2e_lock_changes_only_phase10d_digest():
    archive = ROOT / "evaluation/copilot_e2e_lock_before_10d_v4.json"
    before = json.loads(archive.read_text("utf-8"))
    current = json.loads((ROOT / copilot_e2e.LOCK).read_text("utf-8"))
    assert h.canonical_sha256(archive) == E2E_BEFORE_V4_SHA
    assert before["phase10d_lock_sha256_lf"] == LOCK_V3_SHA
    assert current == copilot_e2e.build_lock(ROOT)
    assert current["phase10d_lock_sha256_lf"] == h.canonical_sha256(ROOT / h.LOCK_FILE)
    changed = {k for k in current.keys() | before.keys() if current.get(k) != before.get(k)}
    assert changed == {"phase10d_lock_sha256_lf"}
