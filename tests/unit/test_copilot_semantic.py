"""Semantic boundary: local handles, deterministic enrichment, unchanged validators."""

import json

import pytest
from pydantic import ValidationError
from tests.unit.test_phase10d_unknown_provenance import entry, mixed_context

from worldbank_copilot.copilot.semantic import (
    SemanticReview,
    SemanticSynthesis,
    critic_payload,
    enrich,
    enrich_review,
    project,
)
from worldbank_copilot.investigation.claims import Failure
from worldbank_copilot.investigation.synthesis import (
    ApprovedContext,
    finalize,
    validate_claims,
    validate_critic,
)
from worldbank_copilot.validation.phase10d_fixtures import fixture


def two_requirement_context(*, orphan=False):
    """R1 -> FACT record1, R2 -> DOCUMENTED_FINDING record2 (+ an unlinked record3)."""
    first, second = entry("FACT", "record1"), entry("DOCUMENTED_FINDING", "record2")
    data = fixture().model_dump(mode="json")
    data["evidence"] = [first, second, *([entry("FACT", "record3")] if orphan else [])]
    data["requirements"] = [
        {
            **data["requirements"][0],
            "requirement_id": "req_a",
            "evidence_ids": [first["evidence_id"]],
        },
        {
            "requirement_id": "req_b",
            "objective": "Second objective",
            "required": True,
            "evidence_ids": [second["evidence_id"]],
        },
    ]
    return ApprovedContext.model_validate(data)


def semantic(*claims, insufficient=False):
    return SemanticSynthesis.model_validate(
        {
            "claims": [
                {"text": text, "evidence": list(handles), "interpretation": interpretation}
                for text, handles, interpretation in claims
            ],
            "insufficient_evidence": insufficient,
            "limitations": [],
        }
    )


def test_handles_are_deterministic_and_grouped_by_requirement():
    context = two_requirement_context()
    first, second = project(context), project(context)
    assert first == second
    assert first.requirements == {"R1": "req_a", "R2": "req_b"}
    assert first.evidence == {
        "E1": context.requirements[0]["evidence_ids"][0],
        "E2": context.requirements[1]["evidence_ids"][0],
    }
    assert [r["evidence"] for r in first.payload["requirements"]] == [["E1"], ["E2"]]
    assert [e["supports"] for e in first.payload["evidence"]] == [["R1"], ["R2"]]


def test_projection_carries_no_canonical_identity_or_scope_object():
    context = two_requirement_context()
    sent = json.dumps(project(context).payload)
    for canonical in (*project(context).evidence.values(), "req_a", "req_b"):
        assert canonical not in sent
    assert {"project_id", "temporal_scope"}.isdisjoint(project(context).payload)
    assert '"source_identity"' not in sent and '"evidence_id"' not in sent


def test_omitted_evidence_gets_no_handle():
    context = two_requirement_context()
    data = context.model_dump(mode="json")
    data["evidence"] = data["evidence"][:1]
    data["omitted_evidence_ids"] = [data["requirements"][1]["evidence_ids"][0]]
    projection = project(ApprovedContext.model_validate(data))
    assert list(projection.evidence) == ["E1"]
    assert projection.payload["requirements"][1] == {
        **projection.payload["requirements"][1],
        "evidence": [],
        "evidence_not_shown": 1,
    }


def test_enrichment_owns_identity_scope_requirements_and_citations():
    context = two_requirement_context()
    projection = project(context)
    output, failure, stats = enrich(
        semantic(("A fact.", ["E1"], False), ("A finding.", ["E2"], False)), projection, context
    )
    assert failure is None and stats.enriched_claims == 2 and stats.handles_resolved == 2
    first, second = output.candidate_claims
    assert (first.claim_id, second.claim_id) == ("C1", "C2")
    assert all(c.project_id == context.project_id for c in output.candidate_claims)
    assert all(
        c.temporal_scope.model_dump(mode="json") == context.temporal_scope
        for c in output.candidate_claims
    )
    assert first.requirement_ids == ("req_a",) and second.requirement_ids == ("req_b",)
    assert (first.provenance_label, first.claim_type) == ("FACT", "ASSERTION")
    assert second.provenance_label == "DOCUMENTED_FINDING"
    refs = {e["evidence_id"]: e for e in context.evidence}
    assert all(
        ref.source_identity == refs[ref.evidence_id]["source_identity"]
        for c in output.candidate_claims
        for ref in c.citations
    )
    assert validate_claims(output, context) == ()  # unchanged validator accepts it


def test_evidence_is_never_attributed_to_an_unrelated_requirement():
    context = two_requirement_context()
    output, failure, _ = enrich(semantic(("A finding.", ["E2"], False)), project(context), context)
    assert failure is None and output.candidate_claims[0].requirement_ids == ("req_b",)
    # The model has no requirement field at all.
    with pytest.raises(ValidationError):
        SemanticSynthesis.model_validate(
            {
                "claims": [
                    {
                        "text": "x",
                        "evidence": ["E2"],
                        "interpretation": False,
                        "requirements": ["R1"],
                    }
                ],
                "insufficient_evidence": False,
                "limitations": [],
            }
        )


def test_evidence_supporting_no_requirement_fails_closed():
    context = two_requirement_context(orphan=True)
    projection = project(context)
    orphan = next(h for h, i in projection.evidence.items() if not projection.supports[i])
    output, failure, _ = enrich(semantic(("Orphan.", [orphan], False)), projection, context)
    assert output is None and failure == Failure.REQUIREMENT_REFERENCE_INVALID


@pytest.mark.parametrize(
    ("handles", "failed"),
    [(["E9"], 1), (["E1", "E1"], 1), (["E1", "E8", "E9"], 2)],
    ids=["unknown", "duplicate", "partly-unknown"],
)
def test_unknown_or_duplicate_handles_fail_closed_without_repair(handles, failed):
    context = two_requirement_context()
    output, failure, stats = enrich(semantic(("x", handles, False)), project(context), context)
    assert output is None and failure == Failure.EVIDENCE_REFERENCE_INVALID
    assert stats.handles_failed == failed and stats.enriched_claims == 0


@pytest.mark.parametrize(
    "raw",
    [
        {"text": "x", "evidence": ["ev_" + "a" * 64], "interpretation": False},
        {"text": "x", "evidence": [], "interpretation": False},
        {"text": "x", "evidence": ["E1"], "interpretation": False, "project_id": "P000001"},
        {"text": "x", "evidence": ["E1"], "interpretation": False, "temporal_scope": {}},
        {"text": "x", "evidence": ["E1"], "interpretation": False, "provenance": "FACT"},
    ],
    ids=["canonical-hash", "no-evidence", "project", "temporal", "provenance"],
)
def test_contract_rejects_hashes_and_system_owned_fields(raw):
    with pytest.raises(ValidationError):
        SemanticSynthesis.model_validate(
            {"claims": [raw], "insufficient_evidence": False, "limitations": []}
        )


def test_evidence_outside_project_fails_closed():
    data = two_requirement_context().model_dump(mode="json")
    data["evidence"][0]["project_id"] = "P999999"
    context = ApprovedContext.model_validate(data)
    output, failure, _ = enrich(semantic(("x", ["E1"], False)), project(context), context)
    assert output is None and failure == Failure.PROJECT_ISOLATION_VIOLATION


def test_assertion_over_mixed_provenance_fails_closed():
    context = two_requirement_context()
    output, failure, _ = enrich(semantic(("x", ["E1", "E2"], False)), project(context), context)
    assert output is None and failure == Failure.PROVENANCE_VIOLATION


def test_interpretation_is_labeled_and_spans_its_requirements():
    context = two_requirement_context()
    output, failure, _ = enrich(semantic(("Why.", ["E1", "E2"], True)), project(context), context)
    claim = output.candidate_claims[0]
    assert failure is None
    assert (claim.provenance_label, claim.claim_type) == ("AI_INTERPRETATION", "INTERPRETATION")
    assert claim.requirement_ids == ("req_a", "req_b")
    assert validate_claims(output, context) == ()


def test_unknown_evidence_never_becomes_authoritative():
    context = fixture("UNKNOWN")
    projection = project(context)
    # Plain claim over UNKNOWN evidence -> UNKNOWN/UNCERTAINTY -> finalizer abstains.
    output, failure, _ = enrich(semantic(("Status missing.", ["E1"], False)), projection, context)
    claim = output.candidate_claims[0]
    assert failure is None and (claim.provenance_label, claim.claim_type) == (
        "UNKNOWN",
        "UNCERTAINTY",
    )
    assert validate_claims(output, context) == ()
    final = finalize(output, enrich_review(review_all(output, "SUPPORTED"), output), context)
    assert final.disposition == "INSUFFICIENT_EVIDENCE" and not final.published_claims
    # Interpretation over UNKNOWN evidence -> unchanged @4 validator rejects it.
    output, failure, _ = enrich(semantic(("Probably late.", ["E1"], True)), projection, context)
    assert failure is None and validate_claims(output, context) == (Failure.PROVENANCE_VIOLATION,)


def test_mixed_unknown_and_known_assertion_cannot_bypass():
    context, *_ = mixed_context()
    output, failure, _ = enrich(semantic(("x", ["E1", "E2"], False)), project(context), context)
    assert output is None and failure == Failure.PROVENANCE_VIOLATION


def test_invalid_enriched_claim_is_still_rejected_by_existing_validator():
    context = two_requirement_context()
    text = "Unlike P999999, the record reports a delay."
    output, failure, _ = enrich(semantic((text, ["E1"], False)), project(context), context)
    assert failure is None
    assert validate_claims(output, context) == (Failure.PROJECT_ISOLATION_VIOLATION,)


def test_abstention_and_limitations_pass_through():
    context = two_requirement_context()
    raw = SemanticSynthesis.model_validate(
        {"claims": [], "insufficient_evidence": True, "limitations": ["Partial history."]}
    )
    output, failure, _ = enrich(raw, project(context), context)
    assert failure is None and output.insufficient_evidence and not output.candidate_claims
    assert output.limitations == ("Partial history.",)


def review_all(output, code):
    return SemanticReview.model_validate(
        {
            "findings": [
                {"claim": c.claim_id, "code": code, "rationale": "r"}
                for c in output.candidate_claims
            ]
        }
    )


def test_critic_works_on_handles_and_review_is_enriched_deterministically():
    context = two_requirement_context()
    projection = project(context)
    output, _, _ = enrich(
        semantic(("A fact.", ["E1"], False), ("A finding.", ["E2"], False)), projection, context
    )
    payload = critic_payload(projection, output)
    assert [c["evidence"] for c in payload["candidate_claims"]] == [["E1"], ["E2"]]
    assert not set(projection.evidence.values()) & set(json.dumps(payload).split('"'))
    review = enrich_review(review_all(output, "SUPPORTED"), output)
    assert review.findings[0].evidence_ids == output.candidate_claims[0].evidence_ids
    assert validate_critic(review, output, context) == ()
    assert finalize(output, review, context).disposition == "PUBLISH_WITH_LIMITATIONS"


def test_critic_finding_for_unknown_claim_fails_existing_check():
    context = two_requirement_context()
    output, _, _ = enrich(semantic(("A fact.", ["E1"], False)), project(context), context)
    review = enrich_review(
        SemanticReview.model_validate(
            {"findings": [{"claim": "C9", "code": "SUPPORTED", "rationale": "r"}]}
        ),
        output,
    )
    assert validate_critic(review, output, context) == (Failure.SCHEMA_VALIDATION_FAILED,)
    assert finalize(output, review, context).disposition == "FAIL_CLOSED"


def shared_context():
    """record1 collected for both R1 and R2; record2 collected for R2 only."""
    data = two_requirement_context().model_dump(mode="json")
    shared = data["requirements"][0]["evidence_ids"][0]
    data["requirements"][1]["evidence_ids"] = [shared, *data["requirements"][1]["evidence_ids"]]
    return ApprovedContext.model_validate(data)


def handle_for(projection, *, shared):
    return next(
        h for h, i in projection.evidence.items() if (len(projection.supports[i]) > 1) == shared
    )


def test_single_requirement_evidence_enriches_normally_beside_shared_evidence():
    context = shared_context()
    projection = project(context)
    handle = handle_for(projection, shared=False)
    output, failure, stats = enrich(semantic(("A finding.", [handle], False)), projection, context)
    assert failure is None and stats.ambiguous_evidence == 0
    assert output.candidate_claims[0].requirement_ids == ("req_b",)
    assert validate_claims(output, context) == ()


@pytest.mark.parametrize(
    ("shared_only", "interpretation"),
    [(True, False), (True, True), (False, True)],
    ids=["shared-assertion", "shared-interpretation", "shared-plus-unshared"],
)
def test_shared_evidence_fails_closed_without_guessing(shared_only, interpretation):
    context = shared_context()
    projection = project(context)
    handles = [handle_for(projection, shared=True)]
    if not shared_only:
        handles.append(handle_for(projection, shared=False))
    output, failure, stats = enrich(semantic(("x", handles, interpretation)), projection, context)
    assert output is None and failure == Failure.REQUIREMENT_REFERENCE_INVALID
    assert stats.ambiguous_evidence == 1 and stats.enriched_claims == 0
    assert isinstance(stats.attributes(projection)["ambiguous_evidence"], int)
