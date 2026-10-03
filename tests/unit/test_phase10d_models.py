"""Model-free 10D contract and capability tests; no external calls."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from worldbank_copilot.common.exceptions import ConfigurationError
from worldbank_copilot.investigation.claims import (
    CriticOutput,
    Disposition,
    Failure,
    ModelReply,
    ModelRequest,
    NodeError,
    SynthesisOutput,
)
from worldbank_copilot.investigation.model_adapter import DatabricksModelAdapter
from worldbank_copilot.investigation.synthesis import (
    build_context,
    finalize,
    parse_output,
    run_nodes,
    validate_claims,
    validate_critic,
)
from worldbank_copilot.routing.bounded_classifier_probe import ChatResponse
from worldbank_copilot.validation import phase10d_models as h
from worldbank_copilot.validation.phase10d_fixtures import claim, draft, fixture

ROOT = Path(__file__).resolve().parents[2]


def review(context, code="SUPPORTED", **changes):
    raw = {
        "findings": [
            {
                "claim_id": "C1",
                "code": code,
                "evidence_ids": [context.evidence[0]["evidence_id"]],
                "concise_rationale": "Supported by the supplied record.",
            }
        ]
    }
    raw["findings"][0].update(changes)
    return CriticOutput.model_validate(raw)


def test_supported_and_unsupported_finalization():
    context = fixture()
    output = draft(context)
    assert not validate_claims(output, context)
    passed = finalize(output, review(context), context)
    assert passed.disposition == Disposition.PUBLISH_WITH_LIMITATIONS
    assert passed.published_claims == (output.candidate_claims[0],)
    failed = finalize(output, review(context, "UNSUPPORTED"), context)
    assert failed.disposition == Disposition.REJECT_UNSUPPORTED
    assert not failed.published_claims
    assert finalize(output, None, context).disposition == Disposition.FAIL_CLOSED


@pytest.mark.parametrize(
    ("changes", "failure"),
    [
        ({"evidence_ids": ["ev_invented"]}, Failure.EVIDENCE_REFERENCE_INVALID),
        ({"project_id": "P999999"}, Failure.PROJECT_ISOLATION_VIOLATION),
        ({"requirement_ids": ["req_missing"]}, Failure.REQUIREMENT_REFERENCE_INVALID),
        ({"citations": []}, Failure.CITATION_VALIDATION_FAILED),
        ({"claim_text": "The project will fail."}, Failure.OVERCLAIMED),
        ({"claim_type": "INTERPRETATION"}, Failure.PROVENANCE_VIOLATION),
        ({"provenance_label": "AI_INTERPRETATION"}, Failure.PROVENANCE_VIOLATION),
        ({"provenance_label": "UNKNOWN"}, Failure.PROVENANCE_VIOLATION),
    ],
)
def test_mechanical_failure_even_if_critic_misses_it(changes, failure):
    context = fixture()
    output = draft(context, **changes)
    assert failure in validate_claims(output, context)
    result = finalize(output, review(context), context)
    assert result.disposition == Disposition.FAIL_CLOSED
    assert not result.published_claims


def test_signal_cannot_be_fact():
    context = fixture("SYSTEM_DERIVED_SIGNAL")
    assert Failure.PROVENANCE_VIOLATION in validate_claims(
        draft(context, provenance_label="FACT"), context
    )
    assert not validate_claims(draft(context), context)


def test_citation_mutation():
    context = fixture()
    output = draft(
        context,
        citations=[
            {"evidence_id": context.evidence[0]["evidence_id"], "source_identity": "changed"}
        ],
    )
    assert Failure.CITATION_VALIDATION_FAILED in validate_claims(output, context)


def test_temporal_mutation():
    context = fixture()
    output = draft(context, temporal_scope={**context.temporal_scope, "date_to": "2099-01-01"})
    assert Failure.TEMPORAL_SCOPE_VIOLATION in validate_claims(output, context)


def test_duplicate_claim_and_summary_ids():
    context = fixture()
    one = claim(context)
    output = SynthesisOutput(
        candidate_claims=(one, one),
        insufficient_evidence=False,
        limitations=(),
        summary_claim_ids=("C1", "C1"),
    )
    assert Failure.SCHEMA_VALIDATION_FAILED in validate_claims(output, context)


@pytest.mark.parametrize("field", ["tools", "project_id", "evidence", "rewrite", "reasoning"])
def test_critic_extra_capability_fields_rejected(field):
    raw = review(fixture()).model_dump(mode="json")
    raw[field] = "forbidden"
    with pytest.raises(NodeError) as exc:
        parse_output(json.dumps(raw), CriticOutput)
    assert exc.value.category == Failure.SCHEMA_VALIDATION_FAILED


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "{} prose",
        "[]",
        '{"findings":[],"findings":[]}',
        '{"findings":[],"extra":1}',
        '{"findings":NaN}',
    ],
)
def test_malformed_strict_output(text):
    with pytest.raises(NodeError):
        parse_output(text, CriticOutput)


def test_output_size_limit():
    with pytest.raises(NodeError) as exc:
        parse_output(" " * 101, CriticOutput, max_bytes=100)
    assert exc.value.category == Failure.OUTPUT_BUDGET_EXCEEDED


def test_critic_missing_claim_and_invented_evidence():
    context = fixture()
    output = draft(context)
    assert validate_critic(CriticOutput(findings=()), output, context)
    assert Failure.EVIDENCE_REFERENCE_INVALID in validate_critic(
        review(context, evidence_ids=["ev_fake"]), output, context
    )


def test_unknown_and_empty_evidence_never_publish():
    context = fixture("UNKNOWN")
    output = draft(context, claim_type="UNCERTAINTY", claim_text="The status is unknown.")
    assert (
        finalize(output, review(context), context).disposition == Disposition.INSUFFICIENT_EVIDENCE
    )
    raw = context.model_dump(mode="json")
    raw["evidence"] = []
    empty = type(context).model_validate(raw)
    abstention = SynthesisOutput(
        candidate_claims=(), insufficient_evidence=True, limitations=(), summary_claim_ids=()
    )
    assert (
        finalize(abstention, CriticOutput(findings=()), empty).disposition
        == Disposition.INSUFFICIENT_EVIDENCE
    )


def evidence_report():
    from tests.unit.test_investigation_evidence import setup

    from worldbank_copilot.investigation.policy import InvestigationPolicy

    executor, state, *_ = setup(
        policy=InvestigationPolicy(
            max_synthesis_context_tokens=100000,
            max_critic_context_tokens=100000,
            aggregate_token_budget=1000000,
        )
    )
    return executor.execute(state)


def test_context_stable_project_scoped_and_whole_references():
    report = evidence_report()
    before = report.model_dump_json()
    context = build_context(report)
    assert context == build_context(report)
    assert all(e["project_id"] == report.package.project_id for e in context.evidence)
    assert len({e["evidence_id"] for e in context.evidence}) == len(context.evidence)
    assert report.model_dump_json() == before
    assert not report.investigation.source_plan.executed
    smaller = build_context(report, byte_limit=len(context.model_dump_json().encode()) // 2)
    assert smaller.omitted_evidence_ids
    for e in smaller.evidence:
        assert e in context.evidence
    with pytest.raises(NodeError) as exc:
        build_context(report, byte_limit=1)
    assert exc.value.category == Failure.CONTEXT_BUDGET_EXCEEDED


class Fake:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def invoke(self, request):
        self.calls.append(request)
        if self.error:
            raise self.error
        data = json.loads(request.context_json)
        if request.role == "CRITIC":
            claims = data["candidate_output"]["candidate_claims"]
            value = {
                "findings": [
                    {
                        "claim_id": c["claim_id"],
                        "code": "SUPPORTED",
                        "evidence_ids": c["evidence_ids"],
                        "concise_rationale": "Supplied source supports the claim.",
                    }
                    for c in claims
                ]
            }
        else:
            refs = {e["evidence_id"]: e for e in data["evidence"]}
            claims = []
            for r in data["requirements"]:
                available = [
                    refs[i]
                    for i in r["evidence_ids"]
                    if i in refs and refs[i]["provenance"] != "UNKNOWN"
                ]
                if not available:
                    continue
                e = available[0]
                claims.append(
                    {
                        "claim_id": f"C{len(claims) + 1}",
                        "claim_text": "The supplied record states this value.",
                        "claim_type": "ASSERTION",
                        "provenance_label": e["provenance"],
                        "evidence_ids": [e["evidence_id"]],
                        "requirement_ids": [r["requirement_id"]],
                        "citations": [
                            {
                                "evidence_id": e["evidence_id"],
                                "source_identity": e["source_identity"],
                            }
                        ],
                        "project_id": data["project_id"],
                        "temporal_scope": data["temporal_scope"],
                        "status": "CANDIDATE",
                    }
                )
            value = {
                "schema_version": "candidate_claims@1",
                "candidate_claims": claims,
                "insufficient_evidence": False,
                "limitations": [],
                "summary_claim_ids": [c["claim_id"] for c in claims],
            }
        return ModelReply(text=json.dumps(value), model_identity="fake")


def test_two_fixed_nodes_no_repair_and_no_source_mutation():
    report = evidence_report()
    before = report.model_dump_json()
    synth, critic = Fake(), Fake()
    result = run_nodes(report, synth, critic, offline=True)
    assert len(synth.calls) == len(critic.calls) == 1
    assert result.budget.extends(report.budget)
    assert sum(r.model_calls for r in result.budget.reservations) == 2
    assert all(r.repair_cycles == 0 for r in result.budget.reservations)
    assert report.model_dump_json() == before
    assert len(result.events) == 2
    assert "context_json" not in type(result.events[0]).model_fields


@pytest.mark.parametrize(
    "error", [TimeoutError(), NodeError(Failure.MODEL_OUTPUT_INVALID), RuntimeError("secret")]
)
def test_node_failure_does_not_bypass_critic_or_retry(error):
    synth, critic = Fake(error), Fake()
    result = run_nodes(evidence_report(), synth, critic, offline=True)
    assert len(synth.calls) == 1 and not critic.calls
    assert result.final.disposition == Disposition.FAIL_CLOSED
    assert "secret" not in result.model_dump_json()


def test_critic_failure_cannot_publish():
    result = run_nodes(evidence_report(), Fake(), Fake(TimeoutError()), offline=True)
    assert result.final.disposition == Disposition.FAIL_CLOSED
    assert not result.final.published_claims


def test_adapter_no_tools_and_reasoning_not_retained():
    transport = Mock()
    transport.post.return_value = ChatResponse(
        200,
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": [
                            {"type": "reasoning", "text": "hidden secret"},
                            {"type": "text", "text": '{"findings":[]}'},
                        ],
                        "reasoning_content": "hidden secret",
                    },
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        },
        {},
        0,
    )
    adapter = DatabricksModelAdapter("candidate", transport=transport)
    request = ModelRequest(
        role="CRITIC",
        system="bounded",
        context_json="{}",
        output_schema=CriticOutput.model_json_schema(),
        max_output_tokens=2000,
        timeout_seconds=60,
    )
    reply = adapter.invoke(request)
    body = transport.post.call_args.args[0]
    assert "tools" not in body and "tool_choice" not in body
    assert body["response_format"]["json_schema"]["strict"]
    assert "hidden secret" not in reply.model_dump_json()
    assert reply.input_tokens == 1


@pytest.mark.parametrize(
    "message", [{"content": "{}", "tool_calls": [{}]}, {"content": "{}", "refusal": "no"}]
)
def test_adapter_rejects_tool_call_and_refusal(message):
    transport = Mock()
    transport.post.return_value = ChatResponse(
        200, {"choices": [{"finish_reason": "stop", "message": message}]}, {}, 0
    )
    request = ModelRequest(
        role="CRITIC",
        system="bounded",
        context_json="{}",
        output_schema={},
        max_output_tokens=20,
        timeout_seconds=1,
    )
    with pytest.raises(NodeError):
        DatabricksModelAdapter("candidate", transport=transport).invoke(request)


def test_lock_and_frozen_phase10c():
    cases, lock = h.prepare(ROOT)
    assert len(cases) == 11 and sum(c["repeats"] for c in cases) == 19
    assert lock["phase10c_lock_sha256_lf"] == h.canonical_sha256(ROOT / h.accepted.LOCK_FILE)
    assert h.accepted.prepare(ROOT)


def test_every_case_constructs_and_has_reason():
    cases = json.loads((ROOT / h.CASE_FILE).read_text("utf-8-sig"))
    for case in cases:
        context, output, data = h.case_input(case)
        assert case["reason"] and data["project_id"] == context.project_id
        if output is not None:
            expected_valid = case["mutation"] in ("NONE", "UNSUPPORTED")
            assert bool(validate_claims(output, context)) != expected_valid


def test_semantic_stability_ignores_prose():
    case = {"role": "SYNTHESIZER", "kind": "FACT", "expectation": "SUPPORTED"}
    context = fixture()
    _, a = h.evaluate(case, draft(context), context, None)
    _, b = h.evaluate(
        case,
        draft(context, claim_text="A delay is observed in this supplied source."),
        context,
        None,
    )
    assert a == b


def test_artifact_no_overwrite_and_receipt_tamper(tmp_path):
    value = {
        "expected_calls": 1,
        "case_results": [],
        "preflight": {"status": "FAIL"},
        "summary": h.summary([], "FAIL", 1),
        "cost_accounting": {
            "actual_billed_cost": "UNAVAILABLE",
            "pricing_version": None,
            "reason": (
                "No trustworthy billing exposed by this capability adapter; "
                "unavailable is not zero."
            ),
        },
    }
    path = tmp_path / "10d1.json"
    writer = h.Writer(path)
    writer.write(value, final=True)
    assert h.read_completed(path)["persistence"]["finalized"]
    with pytest.raises(FileExistsError):
        h.Writer(path)
    path.write_text("{}")
    with pytest.raises(ValueError, match="receipt mismatch"):
        h.read_completed(path)


def test_preflight_failure_persisted_no_model_call(tmp_path, monkeypatch):
    monkeypatch.setattr(
        h.prior,
        "revision_identity",
        lambda *a: SimpleNamespace(runtime_git_status="AMBIGUOUS", model_dump=lambda **k: {}),
    )
    monkeypatch.setattr(h, "prepare", Mock(side_effect=ConfigurationError("CONTENT_MISMATCH")))
    adapter = Mock()
    result = h.run_attempt(
        ROOT,
        tmp_path / "10d1.json",
        commit_sha="a" * 40,
        endpoints=("candidate",),
        adapter_factory=adapter,
        offline=True,
    )
    assert result["summary"]["calls_executed"] == 0
    assert result["preflight"]["status"] == "FAIL"
    adapter.assert_not_called()


def test_semantic_wrapper_rejects_extra_executable_code():
    source = (ROOT / h.NOTEBOOK_FILE).read_text("utf-8")
    assert h.accepted.notebook_identity(source) != h.accepted.notebook_identity(
        source + '\nprint("extra")\n'
    )


def test_json_scalar_coercion_is_rejected():
    context = fixture()
    raw = draft(context).model_dump(mode="json")
    raw["insufficient_evidence"] = "false"
    with pytest.raises(NodeError) as exc:
        parse_output(json.dumps(raw), SynthesisOutput)
    assert exc.value.category == Failure.SCHEMA_VALIDATION_FAILED


def test_interpretation_is_explicit_and_evidence_linked():
    context = fixture()
    output = draft(
        context,
        claim_type="INTERPRETATION",
        provenance_label="AI_INTERPRETATION",
        claim_text="Interpretation: the delay may deserve attention.",
    )
    assert not validate_claims(output, context)
    assert finalize(output, review(context), context).published_claims


def test_mixed_document_structured_context_preserves_source_metadata():
    report = evidence_report()
    context = build_context(report)
    original = {e.evidence_id: e for e in report.package.evidence_index}
    assert {e["source_type"] for e in context.evidence} == {"STRUCTURED", "DOCUMENT"}
    for entry in context.evidence:
        ref = original[entry["evidence_id"]]
        assert entry["citation"] == (ref.citation.model_dump(mode="json") if ref.citation else None)
        assert entry["source"] == (ref.source.model_dump(mode="json") if ref.source else None)


def test_model_budget_exhausted_before_invocation():
    from worldbank_copilot.investigation.evidence_models import EvidenceExecutionReport
    from worldbank_copilot.investigation.policy import Reservation

    report = evidence_report()
    reserved = report.budget.reserve(
        Reservation(reservation_id="preexisting", model_calls=report.budget.policy.max_model_calls)
    )
    raw = report.model_dump(mode="python")
    raw["budget"] = reserved
    report = EvidenceExecutionReport.model_validate(raw)
    synth, critic = Fake(), Fake()
    result = run_nodes(report, synth, critic, offline=True)
    assert result.final.failures == (Failure.BUDGET_EXHAUSTED,)
    assert not synth.calls and not critic.calls


def test_model_output_invalid_prevents_critic():
    adapter = Mock()
    adapter.invoke.return_value = ModelReply(text="not JSON", model_identity="fake")
    critic = Fake()
    result = run_nodes(evidence_report(), adapter, critic, offline=True)
    assert result.final.failures == (Failure.MODEL_OUTPUT_INVALID,)
    assert not critic.calls


class CapabilityFake:
    def __init__(self):
        self.calls = []

    def invoke(self, request):
        self.calls.append(request)
        data = json.loads(request.context_json)
        if request.role == "SYNTHESIZER":
            context = fixture(data["evidence"][0]["provenance"])
            if data["evidence"][0]["provenance"] == "UNKNOWN":
                value = SynthesisOutput(
                    candidate_claims=(),
                    insufficient_evidence=True,
                    limitations=("Requested value is unavailable.",),
                    summary_claim_ids=(),
                )
            else:
                value = draft(context)
        else:
            candidate = data["candidate_output"]["candidate_claims"][0]
            ref = data["evidence"][0]
            code = (
                "PROJECT_SCOPE_VIOLATION"
                if candidate["project_id"] != data["project_id"]
                else "PROVENANCE_INVALID"
                if candidate["provenance_label"] != ref["provenance"]
                else "CITATION_INVALID"
                if candidate["evidence_ids"] != [ref["evidence_id"]]
                else "TEMPORAL_SCOPE_VIOLATION"
                if candidate["temporal_scope"] != data["temporal_scope"]
                else "OVERCLAIMED"
                if "will fail" in candidate["claim_text"]
                else "UNSUPPORTED"
                if "budget doubled" in candidate["claim_text"]
                else "SUPPORTED"
            )
            value = review(fixture(ref["provenance"]), code)
        return ModelReply(text=value.model_dump_json(), model_identity="offline-fake")


def test_full_capability_schedule_offline_and_immutable_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        h.prior,
        "revision_identity",
        lambda *a: SimpleNamespace(runtime_git_status="AMBIGUOUS", model_dump=lambda **k: {}),
    )
    adapter = CapabilityFake()
    result = h.run_attempt(
        ROOT,
        tmp_path / "10d1.json",
        commit_sha="a" * 40,
        endpoints=("candidate",),
        adapter_factory=lambda _: adapter,
        offline=True,
    )
    assert result["validation_environment"] == "OFFLINE_STAND_INS"
    assert result["summary"] == {"calls_executed": 19, "calls_passed": 19, "overall_status": "PASS"}
    assert len(adapter.calls) == 19
    assert result["promotion"] == "NOT_DECIDED"
    assert result["agent_calls"] == result["retrieval_calls"] == result["repair_cycles"] == 0
    with pytest.raises(FileExistsError):
        h.run_attempt(
            ROOT,
            tmp_path / "10d1.json",
            commit_sha="a" * 40,
            endpoints=("candidate",),
            adapter_factory=lambda _: adapter,
            offline=True,
        )
    assert len(adapter.calls) == 19


def test_repeatability_disagreement_fails_the_gate():
    case = {"role": "CRITIC", "expectation": "SUPPORTED"}
    context = fixture()
    _, a = h.evaluate(case, review(context), context, draft(context))
    _, b = h.evaluate(case, review(context, "UNSUPPORTED"), context, draft(context))
    assert a != b


def test_default_adapter_cannot_be_used_in_offline_harness(tmp_path):
    with pytest.raises(ValueError, match="injected"):
        h.run_attempt(
            ROOT,
            tmp_path / "10d1.json",
            commit_sha="a" * 40,
            endpoints=("candidate",),
            offline=True,
        )
    assert not list(tmp_path.iterdir())


def test_existing_live_cost_policy_not_bypassed():
    synth, critic = Fake(), Fake()
    with pytest.raises(NodeError) as exc:
        run_nodes(evidence_report(), synth, critic)
    assert exc.value.category == Failure.INTERNAL_VALIDATION_ERROR
    assert not synth.calls and not critic.calls


def test_foreign_project_identifier_in_claim_text():
    context = fixture()
    output = draft(context, claim_text="P999999 has a delay.")
    assert Failure.PROJECT_ISOLATION_VIOLATION in validate_claims(output, context)


def test_critic_semantic_miss_is_detected_by_capability_oracle():
    context = fixture()
    case = {"role": "CRITIC", "expectation": "UNSUPPORTED"}
    checks, _ = h.evaluate(
        case, review(context), context, draft(context, claim_text="The project budget doubled.")
    )
    assert not checks["critic_detection"]
