"""Diagnostic finalizer branch: agrees with the authoritative finalizers on every branch."""

import itertools
import json
import re

import pytest
from tests.unit.test_copilot_runtime import (
    INVESTIGATION,
    PROJECT,
    Critic,
    Synthesizer,
    copilot,
    traced,
)
from tests.unit.test_copilot_semantic import semantic, two_requirement_context

from worldbank_copilot.copilot.finalization_trace import (
    _DISPOSITION,
    FinalizerBranch,
    classify,
    finalization_attributes,
)
from worldbank_copilot.copilot.semantic import enrich, project
from worldbank_copilot.copilot.service import _finalize_without_critic
from worldbank_copilot.investigation.claims import (
    CriticOutput,
    Failure,
    ModelReply,
    SynthesisOutput,
)
from worldbank_copilot.investigation.synthesis import ApprovedContext, finalize
from worldbank_copilot.validation.phase10d_fixtures import draft, fixture

ENUM_VALUES = {b.value for b in FinalizerBranch}


def review(output, code="SUPPORTED"):
    return CriticOutput.model_validate(
        {
            "findings": [
                {
                    "claim_id": c.claim_id,
                    "code": code,
                    "evidence_ids": list(c.evidence_ids),
                    "concise_rationale": "r",
                }
                for c in output.candidate_claims
            ]
        }
    )


def with_flag(output, flag=True):
    return SynthesisOutput.model_validate(
        {**output.model_dump(mode="json"), "insufficient_evidence": flag}
    )


def enriched(context, *claims):
    output, failure, _ = enrich(semantic(*claims), project(context), context)
    assert failure is None
    return output


def case(name):
    """(output, review, context, failures, expected branch) for each finalizer branch."""
    two = two_requirement_context()
    both = enriched(two, ("A fact.", ["E1"], False), ("A finding.", ["E2"], False))
    first_only = enriched(two, ("A fact.", ["E1"], False))
    unknown = fixture("UNKNOWN")
    empty = SynthesisOutput(
        candidate_claims=(), insufficient_evidence=False, limitations=(), summary_claim_ids=()
    )
    return {
        "fail-closed-failures": (both, review(both), two, (Failure.MODEL_TIMEOUT,), "FAIL_CLOSED"),
        "fail-closed-invalid-claim": (
            draft(fixture(), evidence_ids=["ev_fabricated"]),
            None,
            fixture(),
            (),
            "FAIL_CLOSED",
        ),
        "fail-closed-no-output": (None, None, two, (Failure.MODEL_OUTPUT_INVALID,), "FAIL_CLOSED"),
        "fail-closed-no-review": (both, None, two, (), "FAIL_CLOSED"),
        "insufficient-flag": (with_flag(both), review(both), two, (), "INSUFFICIENT_FLAG"),
        "no-claims": (empty, CriticOutput(findings=()), two, (), "NO_CLAIMS"),
        "unknown-claim": (
            enriched(unknown, ("Missing.", ["E1"], False)),
            review(enriched(unknown, ("Missing.", ["E1"], False))),
            unknown,
            (),
            "UNKNOWN_CLAIM",
        ),
        "critic-rejection": (both, review(both, "UNSUPPORTED"), two, (), "CRITIC_REJECTION"),
        "requirement-uncovered": (first_only, review(first_only), two, (), "REQUIREMENT_UNCOVERED"),
        "publish": (both, review(both), two, (), "PUBLISH"),
    }[name]


NAMES = [
    "fail-closed-failures",
    "fail-closed-invalid-claim",
    "fail-closed-no-output",
    "fail-closed-no-review",
    "insufficient-flag",
    "no-claims",
    "unknown-claim",
    "critic-rejection",
    "requirement-uncovered",
    "publish",
]


@pytest.mark.parametrize("name", NAMES)
def test_every_branch_matches_the_authoritative_finalizer(name):
    output, rev, context, failures, expected = case(name)
    final = finalize(output, rev, context, failures=failures)
    assert classify(output, rev, context, failures) == expected
    assert _DISPOSITION[FinalizerBranch(expected)] == final.disposition
    attributes = finalization_attributes(final, output, rev, context, failures)
    assert attributes["finalizer_branch"] == expected


def test_no_context_evidence_branch_is_unreachable_once_claims_validate():
    # Claims must cite supplied evidence; with none supplied they fail validation first.
    data = two_requirement_context().model_dump(mode="json")
    output = enriched(two_requirement_context(), ("A fact.", ["E1"], False))
    data["evidence"] = []
    context = ApprovedContext.model_validate(data)
    assert classify(output, review(output), context) == FinalizerBranch.FAIL_CLOSED
    assert finalize(output, review(output), context).disposition == "FAIL_CLOSED"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("insufficient-flag", "INSUFFICIENT_FLAG"),
        ("no-claims", "NO_CLAIMS"),
        ("unknown-claim", "UNKNOWN_CLAIM"),
        ("requirement-uncovered", "REQUIREMENT_UNCOVERED"),
        ("publish", "PUBLISH"),
        ("fail-closed-invalid-claim", "FAIL_CLOSED"),
    ],
)
def test_critic_disabled_finalizer_branches_agree(name, expected):
    output, _, context, _, _ = case(name)
    final = _finalize_without_critic(output, context)
    assert classify(output, None, context, critic_disabled=True) == expected
    attributes = finalization_attributes(final, output, None, context, critic_disabled=True)
    assert attributes["finalizer_branch"] == expected


def test_classification_agrees_across_combinations():
    """Drift guard: every combination of the inputs that drive the finalizer."""
    two = two_requirement_context()
    outputs = {
        "both": enriched(two, ("A fact.", ["E1"], False), ("A finding.", ["E2"], False)),
        "first": enriched(two, ("A fact.", ["E1"], False)),
        "interp": enriched(two, ("Why.", ["E1", "E2"], True)),
    }
    checked = 0
    for (key, output), flag, code, failures in itertools.product(
        outputs.items(),
        (False, True),
        ("SUPPORTED", "UNSUPPORTED", None),
        ((), (Failure.MODEL_TIMEOUT,)),
    ):
        output = with_flag(output, flag)
        rev = None if code is None else review(output, code)
        final = finalize(output, rev, two, failures=failures)
        branch = classify(output, rev, two, failures)
        assert _DISPOSITION[branch] == final.disposition, (key, flag, code, failures)
        assert (
            finalization_attributes(final, output, rev, two, failures)["finalizer_branch"]
            != "UNCLASSIFIED"
        )
        checked += 1
    assert checked == 36


def test_drift_is_reported_as_unclassified_not_misexplained():
    output, rev, context, failures, _ = case("publish")
    wrong = finalize(output, rev, context, failures=(Failure.MODEL_TIMEOUT,))
    assert (
        finalization_attributes(wrong, output, rev, context)["finalizer_branch"] == "UNCLASSIFIED"
    )


def test_structural_counts():
    output, rev, context, failures, _ = case("requirement-uncovered")
    final = finalize(output, rev, context)
    attributes = finalization_attributes(final, output, rev, context)
    assert attributes["required_requirements"] == 2
    assert attributes["required_requirements_covered"] == 1
    assert attributes["required_requirements_without_supplied_handle"] == 0
    assert attributes["claims_fact"] == 1 and attributes["claims_documented_finding"] == 0
    assert attributes["synthesizer_insufficient_evidence"] is False
    assert attributes["critic_supported"] == 1 and attributes["critic_unsupported"] == 0
    data = context.model_dump(mode="json")
    data["evidence"] = data["evidence"][:1]  # R2's evidence omitted from the supplied context
    partial = ApprovedContext.model_validate(data)
    attributes = finalization_attributes(finalize(output, rev, partial), output, rev, partial)
    assert attributes["required_requirements_without_supplied_handle"] == 1


def assert_safe(attributes):
    for key, value in attributes.items():
        assert re.fullmatch(r"[a-z_]+", key), key
        if key == "finalizer_branch":
            assert value in ENUM_VALUES
        else:
            assert type(value) in (int, bool), (key, value)


@pytest.mark.parametrize("name", NAMES)
def test_attributes_are_only_booleans_integers_and_codes(name):
    output, rev, context, failures, _ = case(name)
    assert_safe(
        finalization_attributes(
            finalize(output, rev, context, failures=failures), output, rev, context, failures
        )
    )


def test_end_to_end_span_explains_disposition_without_free_text(monkeypatch):
    spans = traced(monkeypatch)
    app = copilot(Synthesizer(text="SECRET-FINAL-77"), Critic())
    app.mlflow_enabled = True
    result = app.investigate(INVESTIGATION, PROJECT)
    final_span = next(s for s in spans if s.name == "finalization").attributes
    assert_safe(final_span)
    assert final_span["finalizer_branch"] == "PUBLISH" and result.status == "ANSWER"
    assert final_span["required_requirements"] == final_span["required_requirements_covered"]
    text = json.dumps(final_span)
    for leaked in ("SECRET-FINAL-77", "ev_", "req_", '"E1"', '"R1"', INVESTIGATION):
        assert leaked not in text


@pytest.mark.parametrize(
    ("synthesizer", "branch", "status"),
    [
        (lambda: SynthesizerFlag(), "INSUFFICIENT_FLAG", "INSUFFICIENT_EVIDENCE"),
        (lambda: SynthesizerFirstOnly(), "REQUIREMENT_UNCOVERED", "INSUFFICIENT_EVIDENCE"),
    ],
    ids=["model-flag-with-claims", "uncovered-requirement"],
)
def test_end_to_end_insufficient_branches_are_distinguished(
    monkeypatch, synthesizer, branch, status
):
    spans = traced(monkeypatch)
    app = copilot(synthesizer(), Critic())
    app.mlflow_enabled = True
    result = app.investigate(INVESTIGATION, PROJECT)
    assert result.status == status and result.validation.critic_status == "SUPPORTED"
    assert (
        next(s for s in spans if s.name == "finalization").attributes["finalizer_branch"] == branch
    )


class SynthesizerFlag(Synthesizer):
    """Valid claims, but the model also sets insufficient_evidence (as the S2 hypothesis)."""

    def invoke(self, request):
        reply = super().invoke(request)
        data = json.loads(reply.text)
        data["insufficient_evidence"] = True
        return ModelReply(text=json.dumps(data), model_identity=reply.model_identity)


class SynthesizerFirstOnly(Synthesizer):
    """Valid claim for the first required requirement only."""

    def invoke(self, request):
        reply = super().invoke(request)
        data = json.loads(reply.text)
        data["claims"] = data["claims"][:1]
        return ModelReply(text=json.dumps(data), model_identity=reply.model_identity)
