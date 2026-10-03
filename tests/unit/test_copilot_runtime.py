"""Offline tests for Copilot.investigate over the real routing/evidence stack.

Model replies are fakes built from the supplied context; they test the harness, never
model quality. No network, endpoint or Databricks access.
"""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from tests.unit.test_phase10c_validation import runtime

from worldbank_copilot.common.exceptions import ConfigurationError
from worldbank_copilot.copilot import (
    Copilot,
    CriticStatus,
    ResultStatus,
    load_copilot_config,
)
from worldbank_copilot.copilot.service import CRITIC_DISABLED_LIMITATION, _finalize_without_critic
from worldbank_copilot.investigation.claims import (
    CriticOutput,
    Failure,
    ModelReply,
    NodeError,
    SynthesisOutput,
)
from worldbank_copilot.investigation.synthesis import finalize
from worldbank_copilot.validation.phase10d_fixtures import draft, fixture

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
PROJECT = "P130544"
INVESTIGATION = "Why did the PDO rating drop to Moderately Unsatisfactory?"  # demo scenario 2
UNCOVERED = "What changed and why?"  # offline package leaves one required requirement empty
TYPES = {"AI_INTERPRETATION": "INTERPRETATION", "UNKNOWN": "UNCERTAINTY"}


def config(**models):
    base = load_copilot_config(CONFIG_DIR, env={})
    return base.model_copy(update={"models": base.models.model_copy(update=models)})


def covering_claims(payload, **changes):
    """One semantic claim per required requirement, citing its first local evidence handle.

    ``changes`` mutate the first claim; a callable value receives the model payload.
    """
    claims = [
        {
            "text": "The supplied record reports this implementation fact.",
            "evidence": [requirement["evidence"][0]],
            "interpretation": False,
        }
        for requirement in payload["requirements"]
        if requirement["required"] and requirement["evidence"]
    ]
    claims[0].update({k: v(payload) if callable(v) else v for k, v in changes.items()})
    return claims


def handles_by_provenance(payload):
    groups = {}
    for entry in payload["evidence"]:
        groups.setdefault(entry["provenance"], []).append(entry["handle"])
    return groups


class Synthesizer:
    """Fake endpoint returning the semantic contract built from the handle projection."""

    def __init__(self, abstain=False, raw=None, error=None, **changes):
        self.abstain, self.raw, self.error, self.changes = abstain, raw, error, changes
        self.requests = []

    def invoke(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        payload = json.loads(request.context_json)
        claims = [] if self.abstain else covering_claims(payload, **self.changes)
        output = {"claims": claims, "insufficient_evidence": self.abstain, "limitations": []}
        return ModelReply(
            text=self.raw or json.dumps(output),
            model_identity="offline-fake",
            input_tokens=10,
            output_tokens=5,
        )


class Critic:
    def __init__(self, code="SUPPORTED", text=None, error=None):
        self.code, self.text, self.error, self.requests = code, text, error, []

    def invoke(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        claims = json.loads(request.context_json)["candidate_claims"]
        findings = [
            {"claim": c["claim"], "code": self.code, "rationale": "Offline fake finding."}
            for c in claims
        ]
        return ModelReply(
            text=self.text or json.dumps({"findings": findings}), model_identity="offline-fake"
        )


def copilot(synthesizer=None, critic=None, **models):
    wired = runtime()
    return Copilot(
        wired.service,
        wired.documents,
        config(**models),
        wired.config_dir,
        synthesizer=synthesizer,
        critic=critic,
    )


def test_supported_answer_is_cited_validated_and_critic_reviewed():
    synthesizer, critic = Synthesizer(), Critic()
    result = copilot(synthesizer, critic).investigate(query=INVESTIGATION, project_id=PROJECT)
    assert result.status == ResultStatus.ANSWER, result.message
    assert result.route == "INVESTIGATION" and result.intent == "CHANGE_INVESTIGATION"
    assert len(synthesizer.requests) == len(critic.requests) == 1
    claim = result.claims[0]
    assert claim.citations and claim.citations[0].evidence_id in claim.evidence_ids
    assert {e.evidence_id for e in result.evidence} >= set(claim.evidence_ids)
    assert result.validation.critic_status == CriticStatus.SUPPORTED
    assert result.validation.semantic_support == "MODEL_ASSESSED"
    assert result.model_capability_note and "12/19" in result.model_capability_note
    assert [c.role for c in result.model_calls] == ["SYNTHESIZER", "CRITIC"]
    assert result.model_calls[0].input_tokens == 10
    assert {
        "routing",
        "attention",
        "evidence_execution",
        "synthesis",
        "enrichment",
        "deterministic_validation",
        "critic",
        "finalization",
    } <= set(result.stage_latency_ms)


def test_abstention_publishes_nothing_and_needs_no_critic():
    critic = Critic()
    result = copilot(Synthesizer(abstain=True), critic).investigate(INVESTIGATION, PROJECT)
    assert result.status == ResultStatus.INSUFFICIENT_EVIDENCE
    assert not result.claims and not critic.requests
    assert result.validation.critic_status == CriticStatus.NOT_REQUIRED


def test_uncovered_required_requirement_abstains_without_model_calls():
    synthesizer, critic = Synthesizer(), Critic()
    result = copilot(synthesizer, critic).investigate(UNCOVERED, PROJECT)
    assert result.status == ResultStatus.INSUFFICIENT_EVIDENCE and not result.claims
    assert result.validation.disposition == "INSUFFICIENT_EVIDENCE"
    assert not synthesizer.requests and not critic.requests and not result.model_calls
    assert any(note.startswith("No evidence found for required") for note in result.limitations)


def mixed_provenance(payload):
    groups = handles_by_provenance(payload)
    assert len(groups) >= 2, "offline package needs two provenance types"
    return [handles[0] for handles in list(groups.values())[:2]]


def duplicate_handle(payload):
    return [payload["evidence"][0]["handle"]] * 2


@pytest.mark.parametrize(
    ("changes", "failure"),
    [
        ({"evidence": ["E99"]}, Failure.EVIDENCE_REFERENCE_INVALID),
        ({"evidence": duplicate_handle}, Failure.EVIDENCE_REFERENCE_INVALID),
        ({"evidence": ["ev_" + "a" * 64]}, Failure.SCHEMA_VALIDATION_FAILED),
        ({"project_id": "P179039"}, Failure.SCHEMA_VALIDATION_FAILED),
        ({"temporal_scope": {"kind": "LATEST"}}, Failure.SCHEMA_VALIDATION_FAILED),
        ({"requirement_ids": ["R1"]}, Failure.SCHEMA_VALIDATION_FAILED),
        ({"text": "Unlike P179039, a delay was observed."}, Failure.PROJECT_ISOLATION_VIOLATION),
        ({"evidence": mixed_provenance}, Failure.PROVENANCE_VIOLATION),
    ],
    ids=[
        "unknown-handle",
        "duplicate-handle",
        "canonical-hash-instead-of-handle",
        "model-supplied-project",
        "model-supplied-temporal-scope",
        "model-supplied-requirement",
        "foreign-project-in-text",
        "mixed-provenance-assertion",
    ],
)
def test_invalid_selection_fails_closed_before_critic(changes, failure):
    critic = Critic()
    result = copilot(Synthesizer(**changes), critic).investigate(INVESTIGATION, PROJECT)
    assert result.status == ResultStatus.FAIL_CLOSED
    assert not result.claims and not critic.requests
    assert result.validation.mechanical_validity == "INVALID"
    assert result.validation.failures == (failure.value,)
    assert [c.role for c in result.model_calls] == ["SYNTHESIZER"]


def test_interpretation_across_provenance_types_is_allowed():
    synthesizer = Synthesizer(evidence=mixed_provenance, interpretation=True)
    result = copilot(synthesizer, Critic()).investigate(INVESTIGATION, PROJECT)
    assert result.status == ResultStatus.ANSWER, result.validation
    assert result.claims[0].provenance == "AI_INTERPRETATION"
    assert result.claims[0].claim_type == "INTERPRETATION"
    assert len(result.claims[0].evidence_ids) == 2


def test_model_never_receives_or_returns_canonical_identities():
    synthesizer, critic = Synthesizer(), Critic()
    result = copilot(synthesizer, critic).investigate(INVESTIGATION, PROJECT)
    assert result.status == ResultStatus.ANSWER
    for request in (*synthesizer.requests, *critic.requests):
        sent = request.context_json
        # Governed evidence content may mention its own project; identity fields may not.
        assert {"project_id", "temporal_scope"}.isdisjoint(json.loads(sent))
        for canonical in {e.evidence_id for e in result.evidence}:
            assert canonical not in sent
        assert "req_" not in sent and '"temporal_scope"' not in sent
        schema = json.dumps(request.output_schema)
        for owned in ("project_id", "temporal_scope", "requirement_ids", "citations"):
            assert owned not in schema
    assert all(c.citations and c.evidence_ids for c in result.claims)


def test_critic_rejection_publishes_nothing():
    result = copilot(Synthesizer(), Critic(code="UNSUPPORTED")).investigate(INVESTIGATION, PROJECT)
    assert result.status == ResultStatus.INSUFFICIENT_EVIDENCE and not result.claims
    assert result.validation.critic_status == CriticStatus.REJECTED
    assert result.validation.disposition == "REJECT_UNSUPPORTED"


@pytest.mark.parametrize(
    "critic",
    [
        Critic(error=NodeError(Failure.MODEL_OUTPUT_INVALID)),
        Critic(error=TimeoutError()),
        Critic(text="not JSON"),
        Critic(text=json.dumps({"findings": []})),  # misses the claim: invalid review
    ],
    ids=["output-invalid", "timeout", "malformed", "incomplete"],
)
def test_enabled_critic_failure_fails_closed(critic):
    result = copilot(Synthesizer(), critic).investigate(INVESTIGATION, PROJECT)
    assert result.status == ResultStatus.FAIL_CLOSED and not result.claims
    assert result.validation.critic_status == CriticStatus.FAILED


def test_disabled_critic_is_recorded_and_never_presented_as_review():
    critic = Critic()
    result = copilot(Synthesizer(), critic, critic_enabled=False).investigate(
        INVESTIGATION, PROJECT
    )
    assert result.status == ResultStatus.ANSWER and not critic.requests
    assert result.validation.critic_status == CriticStatus.DISABLED
    assert result.validation.semantic_support == "NOT_ASSESSED"
    assert result.validation.mechanical_validity == "VALID"
    assert result.validation.disposition == "PUBLISH_WITH_LIMITATIONS"
    assert CRITIC_DISABLED_LIMITATION in result.limitations
    assert [c.role for c in result.model_calls] == ["SYNTHESIZER"]
    assert "critic" not in result.stage_latency_ms


@pytest.mark.parametrize(
    "changes",
    [{"text": "As in P179039, a delay was observed."}, {"evidence": ["E99"]}],
    ids=["foreign-project", "unknown-handle"],
)
def test_disabled_critic_keeps_deterministic_validation_authoritative(changes):
    result = copilot(Synthesizer(**changes), Critic(), critic_enabled=False).investigate(
        INVESTIGATION, PROJECT
    )
    assert result.status == ResultStatus.FAIL_CLOSED and not result.claims


def test_disabled_critic_abstention_publishes_nothing():
    result = copilot(Synthesizer(abstain=True), Critic(), critic_enabled=False).investigate(
        INVESTIGATION, PROJECT
    )
    assert result.status == ResultStatus.INSUFFICIENT_EVIDENCE and not result.claims


@pytest.mark.parametrize(
    ("kind", "changes", "disposition"),
    [
        ("FACT", {}, "PUBLISH_WITH_LIMITATIONS"),
        ("SYSTEM_DERIVED_SIGNAL", {}, "PUBLISH_WITH_LIMITATIONS"),
        ("UNKNOWN", {"claim_type": "UNCERTAINTY"}, "INSUFFICIENT_EVIDENCE"),
        ("FACT", {"evidence_ids": ["ev_fabricated"]}, "FAIL_CLOSED"),
        ("FACT", {"provenance_label": "SYSTEM_DERIVED_SIGNAL"}, "FAIL_CLOSED"),
    ],
)
def test_finalize_without_critic_rules(kind, changes, disposition):
    context = fixture(kind)
    output = draft(context, **changes)
    final = _finalize_without_critic(output, context)
    assert final.disposition == disposition
    assert final.semantic_support == "NOT_ASSESSED"
    assert CRITIC_DISABLED_LIMITATION in final.limitations
    assert bool(final.published_claims) == (disposition == "PUBLISH_WITH_LIMITATIONS")
    assert (final.mechanical_validity == "INVALID") == (disposition == "FAIL_CLOSED")


def test_finalize_without_critic_matches_existing_rules_when_no_semantic_objection():
    """Test oracle only: the existing finalizer given an all-SUPPORTED review."""
    for kind in ("FACT", "SYSTEM_DERIVED_SIGNAL", "DOCUMENTED_FINDING"):
        context = fixture(kind)
        output = draft(context)
        review = CriticOutput.model_validate(
            {
                "findings": [
                    {
                        "claim_id": "C1",
                        "code": "SUPPORTED",
                        "evidence_ids": list(output.candidate_claims[0].evidence_ids),
                        "concise_rationale": "oracle",
                    }
                ]
            }
        )
        existing = finalize(output, review, context)
        assert _finalize_without_critic(output, context).disposition == existing.disposition
    insufficient = SynthesisOutput(
        candidate_claims=draft(fixture()).candidate_claims,
        insufficient_evidence=True,
        limitations=(),
        summary_claim_ids=("C1",),
    )
    final = _finalize_without_critic(insufficient, fixture())
    assert final.disposition == "INSUFFICIENT_EVIDENCE" and not final.published_claims


@pytest.mark.parametrize(
    "synthesizer",
    [
        Synthesizer(error=NodeError(Failure.MODEL_UNAVAILABLE)),
        Synthesizer(error=TimeoutError()),
        Synthesizer(raw="not JSON"),
    ],
    ids=["unavailable", "timeout", "malformed"],
)
def test_synthesizer_failure_fails_closed_without_critic(synthesizer):
    critic = Critic()
    result = copilot(synthesizer, critic).investigate(INVESTIGATION, PROJECT)
    assert result.status == ResultStatus.FAIL_CLOSED and not result.claims
    assert not critic.requests
    assert result.model_calls[0].outcome != "COMPLETED"


def test_retrieval_failure_is_graceful_and_publishes_nothing():
    app = copilot(Synthesizer(), Critic())
    app.documents.retrieve = Mock(side_effect=RuntimeError("index unavailable"))
    result = app.investigate(INVESTIGATION, PROJECT)
    assert app.documents.retrieve.called
    assert result.status in (ResultStatus.FAIL_CLOSED, ResultStatus.INSUFFICIENT_EVIDENCE)
    assert not result.claims and not result.model_calls


def test_router_failure_fails_closed_without_leaking_detail():
    app = copilot(Synthesizer(), Critic())
    app.router.context_factory = Mock(side_effect=RuntimeError("secret internal detail"))
    result = app.investigate(INVESTIGATION, PROJECT)
    assert result.status == ResultStatus.FAIL_CLOSED
    assert "secret" not in result.message and not result.model_calls


@pytest.mark.parametrize(
    "question", ["What deserves my attention?", "Show the timeline of restructurings."]
)
def test_deterministic_routes_never_call_models(question):
    synthesizer, critic = Synthesizer(), Critic()
    result = copilot(synthesizer, critic).investigate(question, PROJECT)
    assert result.status == ResultStatus.EVIDENCE_ONLY and result.route == "STRUCTURED"
    assert result.evidence and not result.claims
    assert not synthesizer.requests and not critic.requests and not result.model_calls
    assert result.model_capability_note is None


def test_attention_signals_are_project_scoped_and_typed():
    result = copilot().investigate("What deserves my attention?", PROJECT)
    assert result.attention_signals
    assert all(s.provenance == "SYSTEM_DERIVED_SIGNAL" for s in result.attention_signals)


def test_prediction_request_is_refused_without_model_calls():
    synthesizer = Synthesizer()
    result = copilot(synthesizer, Critic()).investigate("Will the project fail?", PROJECT)
    assert result.status == ResultStatus.REFUSE and not synthesizer.requests


def test_unauthorized_project_is_refused_before_routing():
    app = copilot(Synthesizer(), Critic())
    app.router.context_factory = Mock(side_effect=AssertionError("must not read"))
    result = app.investigate(INVESTIGATION, "P999999")
    assert result.status == ResultStatus.REFUSE and result.route is None
    app.router.context_factory.assert_not_called()


def test_cross_project_question_is_refused():
    result = copilot(Synthesizer(), Critic()).investigate(
        "What is the current closing date of P179039?", PROJECT
    )
    assert result.status == ResultStatus.REFUSE and not result.evidence


def test_config_loads_defaults_and_env_overrides():
    cfg = load_copilot_config(CONFIG_DIR, env={})
    assert cfg.models.synthesizer_endpoint == "databricks-qwen35-122b-a10b"
    assert cfg.models.critic_enabled is True and PROJECT in cfg.allowed_projects
    cfg = load_copilot_config(
        CONFIG_DIR,
        env={"WBC_COPILOT_CRITIC_ENABLED": "false", "WBC_COPILOT_SYNTHESIZER_ENDPOINT": "other-ep"},
    )
    assert cfg.models.critic_enabled is False
    assert cfg.models.synthesizer_endpoint == "other-ep"
    with pytest.raises(ValueError):
        load_copilot_config(CONFIG_DIR, env={"WBC_COPILOT_CRITIC_ENDPOINT": "bad endpoint/x"})


def test_missing_config_file_is_explicit(tmp_path):
    with pytest.raises(ConfigurationError):
        load_copilot_config(tmp_path, env={})


class FakeSpan:
    def __init__(self, name, log):
        self.name, self.attributes, self.trace_id = name, {}, "trace-1"
        log.append(self)

    def set_attributes(self, attributes):
        self.attributes.update(attributes)


def traced(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    spans = []

    @contextmanager
    def start_span(name):
        yield FakeSpan(name, spans)

    monkeypatch.setitem(__import__("sys").modules, "mlflow", SimpleNamespace(start_span=start_span))
    return spans


SECRET = "SECRET-CLAIM-TEXT-51ab"


def test_trace_has_named_stages_and_allowlisted_metadata(monkeypatch):
    spans = traced(monkeypatch)
    app = copilot(Synthesizer(text=SECRET), Critic())
    app.mlflow_enabled = True
    result = app.investigate(INVESTIGATION, PROJECT)
    assert result.trace_id == "trace-1"
    assert [s.name for s in spans] == [
        "copilot.investigate",
        "routing",
        "attention",
        "evidence_execution",
        "synthesis",
        "enrichment",
        "deterministic_validation",
        "critic",
        "finalization",
    ]
    root = spans[0].attributes
    assert root["project_id"] == PROJECT and root["route"] == "INVESTIGATION"
    assert root["status"] == "ANSWER" and root["critic_status"] == "SUPPORTED"
    assert root["disposition"] == "PUBLISH_WITH_LIMITATIONS"
    assert root["model_call_count"] == 2 and root["input_tokens"] == 10
    assert root["citation_count"] == len(result.claims)
    assert root["model_endpoints"] == "databricks-qwen35-122b-a10b"
    assert spans[3].attributes["evidence_count"] == len(result.evidence)
    assert spans[4].attributes["outcome"] == "COMPLETED"
    enrichment = spans[5].attributes
    assert enrichment["semantic_claims"] == enrichment["enriched_claims"] == len(result.claims)
    assert enrichment["handles_selected"] == enrichment["handles_resolved"] >= 1
    assert enrichment["handles_failed"] == 0 and "failure" not in enrichment
    assert enrichment["local_requirements"] >= 1 and enrichment["local_evidence_handles"] >= 1
    assert result.claims[0].text == SECRET  # published to the user, never to the trace
    serialized = json.dumps([s.attributes for s in spans])
    assert SECRET not in serialized
    keys = {key for s in spans for key in s.attributes}
    values = json.dumps([list(s.attributes.values()) for s in spans])
    for forbidden in ("context_json", "payload", "content", "chunk_text", "claim_text", "system"):
        assert forbidden not in keys and forbidden not in values
    assert all(isinstance(v, (str, int, float, bool)) for s in spans for v in s.attributes.values())


def test_trace_records_handle_failures_structurally(monkeypatch):
    spans = traced(monkeypatch)
    app = copilot(Synthesizer(evidence=["E98", "E99"], text=SECRET), Critic())
    app.mlflow_enabled = True
    app.investigate(INVESTIGATION, PROJECT)
    enrichment = next(s for s in spans if s.name == "enrichment").attributes
    assert enrichment["handles_selected"] >= 2 and enrichment["handles_failed"] == 2
    assert enrichment["failure"] == "EVIDENCE_REFERENCE_INVALID"
    assert "critic" not in [s.name for s in spans]
    assert SECRET not in json.dumps([s.attributes for s in spans])


def test_trace_of_refusal_has_no_model_stages(monkeypatch):
    spans = traced(monkeypatch)
    app = copilot(Synthesizer(), Critic())
    app.mlflow_enabled = True
    app.investigate("Will the project fail?", PROJECT)
    assert [s.name for s in spans] == ["copilot.investigate", "routing", "attention"]
    assert spans[0].attributes["status"] == "REFUSE"
    assert spans[0].attributes["model_call_count"] == 0


def test_pricing_gate_follows_configuration():
    from decimal import Decimal

    assert not config().pricing_configured
    priced = config().model_copy(
        update={"cost_ceiling": Decimal("10"), "pricing_version": "units@1"}
    )
    assert priced.pricing_configured
    assert priced.policy().cost_ceiling == Decimal("10")


def test_configured_output_ceiling_reaches_every_model_request():
    assert config().models.max_output_tokens == 5000
    synthesizer, critic = Synthesizer(), Critic()
    copilot(synthesizer, critic).investigate(INVESTIGATION, PROJECT)
    assert [r.max_output_tokens for r in (*synthesizer.requests, *critic.requests)] == [5000, 5000]


def test_shared_evidence_fails_closed_end_to_end_without_critic(monkeypatch):
    from worldbank_copilot.copilot import service

    original = service.build_context

    def share_first_evidence(report, **kwargs):
        context = original(report, **kwargs)
        data = context.model_dump(mode="json")
        first = data["evidence"][0]["evidence_id"]
        for requirement in data["requirements"]:
            if first not in requirement["evidence_ids"]:
                requirement["evidence_ids"].append(first)
        return type(context).model_validate(data)

    monkeypatch.setattr(service, "build_context", share_first_evidence)
    spans = traced(monkeypatch)
    shared = Synthesizer(
        evidence=lambda p: [next(e["handle"] for e in p["evidence"] if len(e["supports"]) > 1)]
    )
    critic = Critic()
    app = copilot(shared, critic)
    app.mlflow_enabled = True
    result = app.investigate(INVESTIGATION, PROJECT)
    assert result.status == ResultStatus.FAIL_CLOSED
    assert result.validation.failures == (Failure.REQUIREMENT_REFERENCE_INVALID.value,)
    assert not result.claims and sum(len(c.citations) for c in result.claims) == 0
    assert not critic.requests and [c.role for c in result.model_calls] == ["SYNTHESIZER"]
    enrichment = next(s for s in spans if s.name == "enrichment").attributes
    assert type(enrichment["ambiguous_evidence"]) is int and enrichment["ambiguous_evidence"] >= 1
    assert enrichment["failure"] == "REQUIREMENT_REFERENCE_INVALID"
    serialized = json.dumps([s.attributes for s in spans])
    assert "ev_" not in serialized and "req_" not in serialized
    assert "critic" not in [s.name for s in spans]


@pytest.mark.parametrize(
    "question",
    [
        INVESTIGATION + " PLANTEDQUERYMARKER",
        "What deserves my attention? PLANTEDQUERYMARKER",
        "Will the project fail? PLANTEDQUERYMARKER",
    ],
    ids=["investigation", "structured", "refusal"],
)
def test_user_query_text_never_reaches_trace_attributes(monkeypatch, question):
    spans = traced(monkeypatch)
    app = copilot(Synthesizer(), Critic())
    app.mlflow_enabled = True
    result = app.investigate(question, PROJECT)
    assert result.query == question  # still returned to the caller, unchanged
    assert spans and "query" not in spans[0].attributes
    serialized = json.dumps([s.attributes for s in spans])
    assert "PLANTEDQUERYMARKER" not in serialized and question not in serialized
    assert spans[0].attributes["request_id"] == result.request_id
    assert spans[0].attributes["status"] == result.status.value
