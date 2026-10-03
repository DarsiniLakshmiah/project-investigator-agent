"""Offline integration and negative tests; fake model replies are never acceptance."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tests.unit.test_investigation_evidence import setup
from tests.unit.test_phase10c_validation import runtime

from worldbank_copilot.application.job_client import ask_job
from worldbank_copilot.application.observability import ModelMetric
from worldbank_copilot.application.service import AnswerRequest, Application, source_followup
from worldbank_copilot.investigation.assembly import assemble
from worldbank_copilot.investigation.bounded import Decision, investigate, validate_decision
from worldbank_copilot.investigation.claims import ModelReply
from worldbank_copilot.investigation.evidence_models import EvidenceExecutionReport, EvidenceLimits
from worldbank_copilot.investigation.policy import InvestigationPolicy
from worldbank_copilot.validation import copilot_e2e as evaluation
from worldbank_copilot.validation import phase10d_models


class Model:
    def __init__(self, text):
        self.text, self.requests = text, []

    def invoke(self, request):
        self.requests.append(request)
        return ModelReply(text=self.text, model_identity="offline_fake")


def missing_report(policy=None):
    executor, state, tools, documents, retriever, harness = setup(policy=policy)
    package = assemble(state, (), (), documents.profile, EvidenceLimits())
    report = EvidenceExecutionReport(
        investigation=state, budget=state.budget, package=package, events=()
    )
    return report, executor, tools


def application(*, models=False):
    wired = runtime()
    return Application(
        wired.service,
        wired.documents,
        wired.config_dir,
        ("P130544",),
        InvestigationPolicy(),
        models_enabled=models,
        offline=True,
    )


def test_investigator_stop_is_one_call_no_tool_and_no_mutation():
    report, executor, tools = missing_report()
    before = report.model_dump_json()
    model = Model(
        Decision(
            action="STOP", justification="No additional operation warranted."
        ).model_dump_json()
    )
    result = investigate(report, executor.admission, model, lambda: executor, offline=True)
    assert result.outcome == "STOP" and result.used and result.action_count == 0
    assert len(model.requests) == 1 and not tools.calls
    assert report.model_dump_json() == before
    assert result.report.budget.extends(report.budget)
    repeated = investigate(result.report, executor.admission, model, lambda: executor, offline=True)
    assert repeated.outcome == "REPAIR_LIMIT" and len(model.requests) == 1
    data = json.loads(model.requests[0].context_json)
    assert "credentials" not in data and "executor" not in data


def test_valid_document_action_uses_existing_executor_once_and_keeps_provenance():
    report, executor, tools = missing_report()
    decision = Decision(
        action="SEARCH_DOCUMENTS",
        target_requirement_id=report.investigation.requirements[0].requirement_id,
        query="P130544 closing date explanation",
        justification="Find documentary support.",
    )
    model = Model(decision.model_dump_json())
    before = report.model_dump_json()
    result = investigate(report, executor.admission, model, lambda: executor, offline=True)
    assert result.outcome == "EVIDENCE", result
    assert result.action_count == 1 and result.retrieval_calls == 1
    assert len(model.requests) == 1
    assert report.model_dump_json() == before
    assert all(
        e.project_id == "P130544" and e.provenance == "DOCUMENTED_FINDING"
        for e in result.report.package.evidence_index
    )
    assert result.report.investigation.source == report.investigation.source
    assert result.report.budget.extends(report.budget)


@pytest.mark.parametrize(
    "raw",
    [
        {"action": "SQL", "justification": "invalid"},
        {"action": "STOP", "project_id": "P179039", "justification": "invalid"},
        {"action": "STOP", "temporal_scope": {"date_to": "2099-01-01"}, "justification": "invalid"},
        {"action": "STOP", "tool": "arbitrary", "justification": "invalid"},
        {"action": "STOP", "query": "not inert", "justification": "invalid"},
    ],
)
def test_unknown_controls_rejected(raw):
    with pytest.raises(ValueError):
        Decision.model_validate(raw)


@pytest.mark.parametrize(
    "query",
    [
        "P179039 closing date",
        "ISR 2 closing date",
        "SELECT * FROM table",
        "file:/private",
        "dbfs:/x",
        "https://example.com",
    ],
)
def test_cross_scope_temporal_sql_and_path_queries_rejected(query):
    report, executor, tools = missing_report()
    decision = Decision(
        action="SEARCH_DOCUMENTS",
        target_requirement_id=report.investigation.requirements[0].requirement_id,
        query=query,
        justification="Untrusted proposal.",
    )
    with pytest.raises(ValueError):
        validate_decision(decision, report, executor.admission)
    assert not tools.calls


def test_unknown_requirement_is_rejected():
    report, executor, tools = missing_report()
    with pytest.raises(ValueError):
        validate_decision(
            Decision(
                action="SEARCH_DOCUMENTS",
                target_requirement_id="unknown",
                query="closing date",
                justification="invalid",
            ),
            report,
            executor.admission,
        )
    assert not tools.calls


@pytest.mark.parametrize(
    "text",
    [
        "not JSON",
        '{"action":"STOP","action":"SEARCH_DOCUMENTS"}',
        '{"action":"SQL","justification":"ignore policy"}',
    ],
)
def test_malformed_model_never_executes_or_retries(text):
    report, executor, tools = missing_report()
    model = Model(text)
    result = investigate(report, executor.admission, model, lambda: executor, offline=True)
    assert result.outcome == "INVALID_OR_UNAVAILABLE"
    assert len(model.requests) == 1 and not tools.calls and result.action_count == 0


@pytest.mark.parametrize(
    "policy", [InvestigationPolicy(max_model_calls=0), InvestigationPolicy(max_repair_cycles=0)]
)
def test_budget_exhaustion_before_model(policy):
    report, executor, tools = missing_report(policy)
    model = Model("{}")
    result = investigate(report, executor.admission, model, lambda: executor, offline=True)
    assert result.outcome == "BUDGET_EXHAUSTED" and not model.requests and not tools.calls


def test_live_requires_cost_configuration():
    report, executor, _ = missing_report()
    with pytest.raises(ValueError):
        investigate(report, executor.admission, Model("{}"), lambda: executor)


def test_resolved_evidence_does_not_call_investigator():
    executor, state, *_ = setup()
    report = executor.execute(state)
    model = Model("{}")
    assert (
        investigate(report, executor.admission, model, lambda: executor, offline=True).outcome
        == "NOT_NEEDED"
    )
    assert not model.requests


@pytest.mark.parametrize(
    "question,route",
    [
        ("What is the current closing date?", "STRUCTURED"),
        (
            "What does the restructuring paper say about the cancellation "
            "of part of the Additional Financing?",
            "DOCUMENT",
        ),
        ("What changed and why?", "INVESTIGATION"),
    ],
)
def test_router_and_deterministic_baseline_reused(question, route):
    answer = application().answer_question(
        AnswerRequest(project_id="P130544", question=question), mode="A"
    )
    assert answer.route == route and answer.status == "EVIDENCE_ONLY", answer
    assert answer.trace.model_calls == 0 and not answer.trace.critic_used
    assert answer.evidence
    assert answer.final is None


def test_unsupported_and_foreign_project_no_tool_execution():
    app = application()
    app.router.handle = Mock(side_effect=AssertionError("must not route unsupported project"))
    answer = app.answer_question(AnswerRequest(project_id="P999999", question="status"))
    assert answer.status == "REFUSE" and answer.trace.model_calls == 0
    app.router.handle.assert_not_called()
    answer = application().answer_question(
        AnswerRequest(project_id="P130544", question="What is the current closing date of P179039?")
    )
    assert answer.status == "REFUSE" and not answer.evidence


def test_synthesis_and_critic_run_only_once_and_unknown_is_not_published():
    app = application(models=True)
    app.synthesizer = Model(
        json.dumps(
            {
                "schema_version": "candidate_claims@1",
                "candidate_claims": [],
                "insufficient_evidence": True,
                "limitations": [],
                "summary_claim_ids": [],
            }
        )
    )
    app.critic = Model('{"findings":[]}')
    answer = app.answer_question(
        AnswerRequest(project_id="P130544", question="What changed and why?"), mode="B"
    )
    assert answer.status == "UNKNOWN", answer
    assert len(app.synthesizer.requests) == len(app.critic.requests) == 1
    assert answer.trace.critic_used and answer.trace.model_calls == 2
    assert not answer.final.published_claims


def test_malformed_synthesis_stops_before_critic():
    app = application(models=True)
    app.synthesizer, app.critic = Model("not JSON"), Model("{}")
    answer = app.answer_question(
        AnswerRequest(project_id="P130544", question="What changed and why?")
    )
    assert answer.status == "FAIL" and not app.critic.requests


def test_context_injection_does_not_change_harness_capabilities():
    report, executor, tools = missing_report()
    model = Model('{"action":"SQL","justification":"ignore all constraints"}')
    result = investigate(report, executor.admission, model, lambda: executor, offline=True)
    assert result.outcome == "INVALID_OR_UNAVAILABLE" and not tools.calls
    assert "untrusted data" in model.requests[0].system


def test_session_sources_cannot_cross_projects():
    answer = application().answer_question(
        AnswerRequest(project_id="P130544", question="What is the current closing date?")
    )
    assert source_followup(answer, "P130544", ("P130544",))
    with pytest.raises(ValueError):
        source_followup(answer, "P179039", ("P130544", "P179039"))


@pytest.mark.parametrize(
    "field", ["chain_of_thought", "prompt", "credentials", "documents", "authorization"]
)
def test_trace_rejects_unapproved_sensitive_fields(field):
    with pytest.raises(ValueError):
        ModelMetric(role="CRITIC", latency_ms=1, outcome="COMPLETED", **{field: "secret"})


def test_fake_mlflow_lifecycle_has_only_allowlisted_attributes(monkeypatch):
    span = SimpleNamespace(trace_id="test_trace", set_attributes=Mock())
    from contextlib import contextmanager

    @contextmanager
    def start_span(**kwargs):
        yield span

    monkeypatch.setitem(__import__("sys").modules, "mlflow", SimpleNamespace(start_span=start_span))
    app = application()
    app.mlflow_enabled = True
    answer = app.answer_question(
        AnswerRequest(project_id="P130544", question="What is the current closing date?")
    )
    assert answer.trace.mlflow_trace_id == "test_trace"
    attributes = span.set_attributes.call_args.args[0]
    assert "question" not in attributes and "evidence" not in attributes
    assert "chain_of_thought" not in str(attributes)


def test_job_boundary_validates_returned_scope():
    request = AnswerRequest(project_id="P130544", question="What is the current closing date?")
    answer = application().answer_question(request)
    client = SimpleNamespace(jobs=SimpleNamespace(run_now=Mock(), get_run_output=Mock()))
    client.jobs.run_now.return_value.result.return_value = SimpleNamespace(
        tasks=[SimpleNamespace(run_id=42)]
    )
    client.jobs.get_run_output.return_value = SimpleNamespace(
        notebook_output=SimpleNamespace(result=answer.model_dump_json(), truncated=False)
    )
    assert ask_job(client, 1, request).project_id == request.project_id
    other = AnswerRequest(
        project_id="P179039", question=request.question, session_id=request.session_id
    )
    with pytest.raises(ValueError):
        ask_job(client, 1, other)


def test_actual_contract_checks():
    assert all(evaluation.deterministic_checks().values())


def test_receipts_and_attempts_are_immutable(tmp_path):
    path = tmp_path / "offline_test.json"
    writer = evaluation.Writer(path)
    writer.write(
        {
            "schema": "copilot_comparison_e2e@1",
            "preflight": "FAIL",
            "results": [],
            "acceptance": "NOT_ACCEPTED",
            "contract_status": "FAIL",
        },
        final=True,
    )
    assert evaluation.read_completed(path)["contract_status"] == "FAIL"
    with pytest.raises(FileExistsError):
        evaluation.Writer(path)
    path.write_text("{}")
    with pytest.raises(ValueError, match="RECEIPT"):
        evaluation.read_completed(path)


def test_historical_preparations_and_new_lock_are_consistent():
    from tests.conftest import REPO_ROOT

    assert phase10d_models.prepare(REPO_ROOT)
    assert evaluation.prepare(REPO_ROOT)
    assert phase10d_models.accepted.prepare(REPO_ROOT)
    assert phase10d_models.prior.prepare_protocol(REPO_ROOT, dependency_ok=True)


def test_projection_preserves_original_evidence_and_required_requirements():
    from worldbank_copilot.application.projection import bounded_report

    executor, state, *_ = setup()
    report = executor.execute(state)
    before = report.model_dump_json()
    projected = bounded_report(report)
    assert report.model_dump_json() == before
    assert projected.budget == report.budget
    assert {s.requirement_id for s in projected.package.requirement_summaries} == {
        r.requirement_id for r in state.requirements
    }
    original = {e.evidence_id: e for e in report.package.evidence_index}
    assert all(original[e.evidence_id] == e for e in projected.package.evidence_index)
    if len(projected.package.evidence_index) < len(report.package.evidence_index):
        assert any("projection omits" in warning for warning in projected.package.warnings)


@pytest.mark.parametrize("question", ["Bearer secret_token", "password=secret", "dapi" + "a" * 25])
def test_credentials_rejected_before_job_submission(question):
    with pytest.raises(ValueError):
        AnswerRequest(project_id="P130544", question=question)


def test_unenforceable_document_date_scope_never_retrieves():
    app = application()
    answer = app.answer_question(
        AnswerRequest(
            project_id="P130544", question="What does the restructuring paper say after 2024-01-01?"
        )
    )
    assert answer.status == "CLARIFY", answer
    assert answer.trace.retrieval_calls == 0 and not answer.evidence


def test_failed_additional_action_is_charged_and_not_retried():
    report, executor, tools = missing_report()
    model = Model(
        Decision(
            action="SEARCH_DOCUMENTS",
            target_requirement_id=report.investigation.requirements[0].requirement_id,
            query="P130544 closing date explanation",
            justification="Additional source.",
        ).model_dump_json()
    )
    failing = Mock()
    failing.execute.side_effect = RuntimeError("not retained")
    result = investigate(report, executor.admission, model, lambda: failing, offline=True)
    assert result.outcome == "ACTION_FAILED" and result.action_count == 1
    assert sum(r.model_calls for r in result.report.budget.reservations) == 1
    assert len(model.requests) == 1 and failing.execute.call_count == 1


def test_reader_rejects_false_pass_and_checkpoint():
    value = {
        "schema": "copilot_comparison_e2e@1",
        "preflight": "PASS",
        "results": [{"mode": "A", "case_id": "x", "status": "PASS", "checks": {"actual": False}}],
        "acceptance": "NOT_ACCEPTED",
        "contract_status": "PASS",
    }
    with pytest.raises(ValueError, match="PASS_WITH_FAILED_CHECK"):
        evaluation.validate_artifact(value)


def test_current_protocol_cases_have_real_expected_routes_and_no_model_results():
    from tests.conftest import REPO_ROOT

    cases, _ = evaluation.prepare(REPO_ROOT)
    for case in cases:
        answer = application().answer_question(
            AnswerRequest(project_id=case["project_id"], question=case["question"]), mode="A"
        )
        actual = evaluation.checks(answer, case, require_mlflow=False)
        assert all(actual.values()), (case, answer, actual)
    assert (
        evaluation.comparison_metrics([])["C"]["semantic_groundedness"] == "HUMAN_REVIEW_REQUIRED"
    )
