"""Offline regressions for the proven Databricks transport incompatibility."""

import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from worldbank_copilot.investigation.claims import (
    CandidateClaim,
    CriticFinding,
    CriticOutput,
    Failure,
    ModelRequest,
    NodeError,
    SynthesisOutput,
)
from worldbank_copilot.investigation.model_adapter import DatabricksModelAdapter, _strict_schema
from worldbank_copilot.investigation.synthesis import (
    CRITIC_INSTRUCTIONS,
    INSTRUCTIONS,
    parse_output,
)
from worldbank_copilot.routing.bounded_classifier_probe import ChatResponse
from worldbank_copilot.validation import phase10d_models as h
from worldbank_copilot.validation.phase10d_fixtures import draft, fixture

ROOT = Path(__file__).resolve().parents[2]
ARCHIVED_SHA = "ec1b9bfc634592a327e6908f4492635b0e35d6bce2e410908733f3eb2dca08f5"
CASES_SHA = "a59f80bceb5df9563e475868f3373dffa107227c9e705f2b1d7f376b26aef685"


def keys(value):
    if isinstance(value, dict):
        yield from value
        for child in value.values():
            yield from keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from keys(child)


@pytest.mark.parametrize("model", [SynthesisOutput, CriticOutput])
def test_transport_copy_strips_only_proven_keywords(model):
    schema = model.model_json_schema()
    before = copy.deepcopy(schema)
    normalized = _strict_schema(schema)
    assert schema == before
    assert {"pattern", "default"}.isdisjoint(keys(normalized))
    assert "pattern" in set(keys(schema))
    assert "$defs" in normalized


def test_nested_lists_preserve_other_constraints_and_strict_objects():
    retained = {
        "minLength": 1,
        "maxLength": 8,
        "minItems": 1,
        "maxItems": 2,
        "format": "date",
        "const": "C1",
        "enum": ["C1"],
        "$ref": "#/$defs/id",
        "prefixItems": [{"pattern": "x", "minLength": 1}],
        "anyOf": [{"pattern": "x", "default": "x", "type": "string"}],
    }
    source = {
        "$defs": {"id": {"pattern": "x", "default": "x", **retained}},
        "properties": {"id": {"pattern": "x", "type": "string"}},
    }
    before = copy.deepcopy(source)
    result = _strict_schema(source)
    assert source == before
    assert {"pattern", "default"}.isdisjoint(keys(result))
    expected = copy.deepcopy(retained)
    expected["prefixItems"] = [{"minLength": 1}]
    expected["anyOf"] = [{"type": "string"}]
    assert result["$defs"]["id"] == expected
    assert result["required"] == ["id"] and result["additionalProperties"] is False


def test_authoritative_synthesis_patterns_are_enforced_after_transport():
    schema = CandidateClaim.model_json_schema()["properties"]
    assert schema["claim_id"]["pattern"] == "^C[0-9]{1,3}$"
    assert schema["project_id"]["pattern"] == "^P[0-9]{6}$"
    value = draft(fixture()).model_dump(mode="json")
    assert parse_output(json.dumps(value), SynthesisOutput).candidate_claims[0].claim_id == "C1"
    for field, invalid in (("claim_id", "claim_1"), ("project_id", "project_1")):
        bad = copy.deepcopy(value)
        bad["candidate_claims"][0][field] = invalid
        with pytest.raises(ValidationError) as error:
            SynthesisOutput.model_validate(bad)
        assert error.value.errors()[0]["type"] == "string_pattern_mismatch"
        with pytest.raises(NodeError) as parsed:
            parse_output(json.dumps(bad), SynthesisOutput)
        assert parsed.value.category == Failure.SCHEMA_VALIDATION_FAILED


def test_authoritative_critic_pattern_and_copied_id():
    assert CriticFinding.model_json_schema()["properties"]["claim_id"]["pattern"] == "^C[0-9]{1,3}$"
    finding = {
        "claim_id": "C1",
        "code": "SUPPORTED",
        "evidence_ids": [],
        "concise_rationale": "Supplied evidence supports this claim.",
    }
    assert CriticOutput.model_validate({"findings": [finding]}).findings[0].claim_id == "C1"
    with pytest.raises(ValidationError):
        CriticOutput.model_validate({"findings": [{**finding, "claim_id": "claim_1"}]})


def test_minimal_identifier_guidance_and_no_reasoning_request():
    for text in (
        "C followed by 1 to 3 digits",
        "C1, C2, C15, C123",
        "claim_1, claim1, C1000",
        "Copy project_id exactly from supplied context",
        "never invent or alter project_id",
    ):
        assert text in INSTRUCTIONS
    assert "Copy claim_id exactly from candidate_output" in CRITIC_INSTRUCTIONS
    assert "never invent, rename, normalize or" in CRITIC_INSTRUCTIONS
    assert "modify a claim_id" in CRITIC_INSTRUCTIONS
    assert "expose or include chain-of-thought" in INSTRUCTIONS


def request():
    return ModelRequest(
        role="SYNTHESIZER",
        system=INSTRUCTIONS,
        context_json="{}",
        output_schema=SynthesisOutput.model_json_schema(),
        max_output_tokens=2000,
        timeout_seconds=60,
    )


@pytest.mark.parametrize(
    "status,error,category",
    [
        (0, "TimeoutError", Failure.MODEL_TIMEOUT),
        (0, "ConnectionError", Failure.MODEL_UNAVAILABLE),
        (400, None, Failure.MODEL_REQUEST_INVALID),
        (401, None, Failure.MODEL_UNAVAILABLE),
        (403, None, Failure.MODEL_UNAVAILABLE),
        (429, None, Failure.MODEL_UNAVAILABLE),
        (500, None, Failure.MODEL_UNAVAILABLE),
    ],
)
def test_sanitized_http_classification_without_response_retention(status, error, category):
    transport = Mock()
    transport.post.return_value = ChatResponse(
        status, {"error": "Bearer secret"}, {"Authorization": "secret"}, 0, error
    )
    with pytest.raises(NodeError) as failed:
        DatabricksModelAdapter("candidate", transport=transport).invoke(request())
    assert failed.value.category == category
    assert str(failed.value) == category.value
    assert {"pattern", "default"}.isdisjoint(
        keys(transport.post.call_args.args[0]["response_format"]["json_schema"]["schema"])
    )
    assert transport.post.call_count == 1


def test_content_array_ignores_reasoning_and_post_generation_parser_remains_authoritative():
    transport = Mock()
    text = draft(fixture()).model_dump_json()
    transport.post.return_value = ChatResponse(
        200,
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": [
                            {"type": "reasoning", "text": "hidden secret"},
                            {"type": "text", "text": text},
                        ],
                        "reasoning_content": "hidden secret",
                    },
                }
            ]
        },
        {},
        0,
    )
    reply = DatabricksModelAdapter("candidate", transport=transport).invoke(request())
    assert reply.text == text and "hidden secret" not in reply.model_dump_json()
    assert parse_output(reply.text, SynthesisOutput).candidate_claims[0].claim_id == "C1"


@pytest.mark.parametrize(
    "body",
    [{}, {"choices": []}, {"choices": [{"finish_reason": "stop", "message": {"content": []}}]}],
)
def test_malformed_success_is_output_invalid(body):
    transport = Mock()
    transport.post.return_value = ChatResponse(200, body, {}, 0)
    with pytest.raises(NodeError) as error:
        DatabricksModelAdapter("candidate", transport=transport).invoke(request())
    assert error.value.category == Failure.MODEL_OUTPUT_INVALID


def test_lock_lineage_and_unchanged_evaluation():
    # Archived v2 -> v3 transition; @4 lineage is verified in test_phase10d_unknown_provenance.
    cases, current = h.prepare(ROOT)
    assert current == h.build_lock(ROOT) == h.build_lock(ROOT)
    v2 = ROOT / "evaluation/phase10d_model_lock_v2.json"
    active = json.loads((ROOT / "evaluation/phase10d_model_lock_v3.json").read_text("utf-8"))
    previous = json.loads(v2.read_text("utf-8"))
    assert active["schema"] == "phase10d_capability_lock@3"
    assert h.canonical_sha256(v2) == ARCHIVED_SHA
    assert active["supersedes_lock_sha256_lf"] == ARCHIVED_SHA
    assert h.canonical_sha256(ROOT / h.CASE_FILE) == CASES_SHA
    assert len(cases) == 11 and sum(c["repeats"] for c in cases) == 19
    for field in (
        "case_set_sha256_lf",
        "max_endpoints",
        "max_calls_per_endpoint",
        "max_output_tokens",
        "timeout_seconds",
        "temperature",
        "retries",
        "required_repeatability",
        "notebook_semantic_identity",
        "phase10c_lock_sha256_lf",
    ):
        assert active[field] == previous[field]
    h.accepted.prepare(ROOT)
    h.prior.prepare_protocol(ROOT, dependency_ok=True)
