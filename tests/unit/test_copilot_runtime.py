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


def covering_claims(context, **changes):
    """One mechanically valid claim per required requirement, over its first evidence.

    ``changes`` mutate the first claim; a callable value receives the context.
    """
    refs = {e["evidence_id"]: e for e in context["evidence"]}
    claims = []
    for requirement in context["requirements"]:
        supplied = [i for i in requirement["evidence_ids"] if i in refs]
        if not requirement["required"] or not supplied:
            continue
        entry = refs[supplied[0]]
        claims.append(
            {
                "claim_id": f"C{len(claims) + 1}",
                "claim_text": "The supplied record reports this implementation fact.",
                "claim_type": TYPES.get(entry["provenance"], "ASSERTION"),
                "provenance_label": entry["provenance"],
                "evidence_ids": [entry["evidence_id"]],
                "requirement_ids": [requirement["requirement_id"]],
                "citations": [
                    {
                        "evidence_id": entry["evidence_id"],
                        "source_identity": entry["source_identity"],
                    }
                ],
                "project_id": context["project_id"],
                "temporal_scope": context["temporal_scope"],
                "status": "CANDIDATE",
            }
        )
    claims[0].update({k: v(context) if callable(v) else v for k, v in changes.items()})
    return claims


class Synthesizer:
    def __init__(self, abstain=False, text=None, error=None, **changes):
        self.abstain, self.text, self.error, self.changes = abstain, text, error, changes
        self.requests = []

    def invoke(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        context = json.loads(request.context_json)
        claims = [] if self.abstain else covering_claims(context, **self.changes)
        output = {
            "schema_version": "candidate_claims@1",
            "candidate_claims": claims,
            "insufficient_evidence": self.abstain,
            "limitations": [],
            "summary_claim_ids": [c["claim_id"] for c in claims],
        }
        return ModelReply(
            text=self.text or json.dumps(output),
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
        claims = json.loads(request.context_json)["candidate_output"]["candidate_claims"]
        findings = [
            {
                "claim_id": c["claim_id"],
                "code": self.code,
                "evidence_ids": c["evidence_ids"],
                "concise_rationale": "Offline fake finding.",
            }
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


@pytest.mark.parametrize(
    ("changes", "failure"),
    [
        ({"evidence_ids": ["ev_fabricated"]}, Failure.EVIDENCE_REFERENCE_INVALID),
        ({"citations": [{"evidence_id": "ev_x", "source_identity": "c"}]}, None),
        ({"project_id": "P179039"}, Failure.PROJECT_ISOLATION_VIOLATION),
        ({"provenance_label": "UNKNOWN", "claim_type": "UNCERTAINTY"}, None),
        (
            {"temporal_scope": lambda c: {**c["temporal_scope"], "date_to": "2099-01-01"}},
            Failure.TEMPORAL_SCOPE_VIOLATION,
        ),
    ],
    ids=["fabricated-evidence", "invalid-citation", "project", "provenance", "temporal"],
)
def test_mechanical_violation_fails_closed_before_critic(changes, failure):
    critic = Critic()
    result = copilot(Synthesizer(**changes), critic).investigate(INVESTIGATION, PROJECT)
    assert result.status == ResultStatus.FAIL_CLOSED
    assert not result.claims and not critic.requests
    assert result.validation.mechanical_validity == "INVALID"
    if failure is not None:
        assert failure.value in result.validation.failures
    else:
        assert result.validation.failures


def test_unknown_evidence_cannot_be_relabeled_as_interpretation():
    critic = Critic()
    app = copilot(Synthesizer(), critic)
    result = app.investigate(INVESTIGATION, PROJECT)
    evidence = {e.evidence_id: e.provenance[0] for e in result.evidence}
    unknown = [i for i, p in evidence.items() if p == "UNKNOWN"]
    if not unknown:
        pytest.skip("offline fixture package carries no UNKNOWN evidence")
    relabel = Synthesizer(
        evidence_ids=unknown[:1],
        provenance_label="AI_INTERPRETATION",
        claim_type="INTERPRETATION",
    )
    result = copilot(relabel, Critic()).investigate(INVESTIGATION, PROJECT)
    assert result.status == ResultStatus.FAIL_CLOSED


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
    [{"project_id": "P179039"}, {"evidence_ids": ["ev_fabricated"]}],
    ids=["project", "fabricated-evidence"],
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
        Synthesizer(text="not JSON"),
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


def test_trace_has_named_stages_and_allowlisted_metadata(monkeypatch):
    spans = traced(monkeypatch)
    app = copilot(Synthesizer(), Critic())
    app.mlflow_enabled = True
    result = app.investigate(INVESTIGATION, PROJECT)
    assert result.trace_id == "trace-1"
    assert [s.name for s in spans] == [
        "copilot.investigate",
        "routing",
        "attention",
        "evidence_execution",
        "synthesis",
        "deterministic_validation",
        "critic",
        "finalization",
    ]
    root = spans[0].attributes
    assert root["project_id"] == PROJECT and root["route"] == "INVESTIGATION"
    assert root["status"] == "ANSWER" and root["critic_status"] == "SUPPORTED"
    assert root["model_call_count"] == 2 and root["input_tokens"] == 10
    assert root["citation_count"] == len(result.claims)
    assert root["model_endpoints"] == "databricks-qwen35-122b-a10b"
    assert spans[3].attributes["evidence_count"] == len(result.evidence)
    assert spans[4].attributes["outcome"] == "COMPLETED"
    serialized = json.dumps([s.attributes for s in spans])
    for forbidden in ("context_json", "payload", "chunk_text", "claim_text", "system"):
        assert forbidden not in serialized
    assert all(isinstance(v, (str, int, float, bool)) for s in spans for v in s.attributes.values())


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
