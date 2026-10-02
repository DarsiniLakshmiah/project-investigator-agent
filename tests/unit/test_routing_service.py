"""Phase 9B harness: routing outcomes, execution and isolation guarantees end to end
(synthetic governed rows, in-memory reader, spy executor and retriever)."""

import json
from datetime import date
from itertools import permutations

import pytest
from tests.conftest import REPO_CONFIG_DIR
from tests.support.routing_fixtures import ALL, CONFIG, harness, routing_tables
from tests.support.tool_fixtures import IPF, OTHER, PFORR
from tests.unit.test_routing_entities import REFS

from worldbank_copilot.retrieval.rerank_policy import AlwaysRerank
from worldbank_copilot.routing.models import ExecutionOutcome, Route
from worldbank_copilot.tools.models import MechanicalCode, ToolStatus

QUESTIONS_THAT_WOULD_EXECUTE = [
    "What is the current closing date?",
    "How much has been disbursed?",
    "Why was the closing date extended?",
    "Show the PDO rating history",
]


def nothing_ran(h):
    return h.executor.calls == [] and h.retriever.first_stage_calls == [] and h.reads == []


# -- execution ----------------------------------------------------------------------------


def test_structured_executes_exactly_the_required_calls():
    h = harness()
    r = h.ask("What was the implementation progress rating in ISR 2?")
    assert (r.decision.route, r.outcome) == (Route.STRUCTURED, ExecutionOutcome.EXECUTED)
    assert h.executor.calls == [
        (
            "get_rating_history",
            {
                "project_id": IPF,
                "rating_types": ["IP"],
                "isr_sequence_from": 2,
                "isr_sequence_to": 2,
            },
            IPF,
        )
    ]
    assert r.executed_tools == ["get_rating_history"] and not r.retrieval_executed
    assert h.retriever.first_stage_calls == []
    assert r.tool_results[0].status == ToolStatus.OK and r.semantic_sufficiency == "NOT_ASSESSED"


@pytest.mark.parametrize(
    ("question", "tool", "args"),
    [
        ("What is the current closing date?", "get_project_overview", {}),
        (
            "Show the project timeline since 2017",
            "get_project_timeline",
            {"date_from": "2017-01-01"},
        ),
        (
            "How much of loan IBRD-9324-0 was disbursed in ISR 2?",
            "get_financial_status",
            {"loan_number": "IBRD93240", "as_of_isr": 2},
        ),
        (
            "What are the PDO indicators' targets?",
            "get_results_progress",
            {"indicator_type": "PDO", "isr_sequence": "latest"},
        ),
        (
            "What was the risk rating at appraisal?",
            "get_risk_register",
            {"record_type": "FORMAL_RISK_RATING"},
        ),
        ("What deserves my attention?", "get_attention_signals", {"status": "CURRENT"}),
    ],
)
def test_intent_to_requirement_mapping(question, tool, args):
    h = harness()
    r = h.ask(question)
    assert r.decision.route == Route.STRUCTURED, r.decision
    ((name, called, scope),) = h.executor.calls
    assert (name, scope) == (tool, IPF) and called == {"project_id": IPF, **args}


def test_document_route_runs_retrieval_only_after_scope_with_the_supplied_policy():
    h = harness(rerank=AlwaysRerank())
    r = h.ask("Why was the closing date extended?", active=PFORR)
    assert (r.decision.route, r.outcome) == (Route.DOCUMENT, ExecutionOutcome.EXECUTED)
    assert h.retriever.first_stage_calls == [("Why was the closing date extended?", PFORR)]
    assert r.retrieval_executed and r.executed_tools == ["search_project_documents"]
    (item,) = r.tool_results[0].items
    assert item.rerank_decision.policy == "always" and item.reranker == "cross_encoder"
    assert all(e.evidence.project_id == PFORR and not e.relevance_verified for e in item.evidence)


def test_document_route_without_an_explicit_policy_is_an_error_not_a_default():
    h = harness()
    r = h.ask("Why was the closing date extended?", documents=False)
    assert (r.decision.route, r.outcome) == (Route.DOCUMENT, ExecutionOutcome.ERROR)
    assert "not configured" in r.error and nothing_ran(h)


def test_zero_retrieved_evidence_is_mechanically_insufficient():
    h = harness()
    r = h.ask("According to PAD: zebra unicorn quantum?")
    assert r.outcome == ExecutionOutcome.INSUFFICIENT_EVIDENCE
    assert r.mechanical[0].code == MechanicalCode.NO_CHUNKS


# -- isolation ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("active", "foreign", "kind"), [(a, f, k) for a, f in permutations(ALL, 2) for k in REFS[f]]
)
@pytest.mark.parametrize("question", QUESTIONS_THAT_WOULD_EXECUTE)
def test_foreign_reference_executes_nothing(active, foreign, kind, question):
    h = harness()
    r = h.ask(f"{question} ({REFS[foreign][kind]})", active=active)
    assert (r.decision.route, r.decision.reason_code) == (Route.REFUSE, "CROSS_PROJECT")
    assert r.outcome == ExecutionOutcome.NOT_EXECUTED and nothing_ran(h)
    assert r.understanding.intent is None  # stopped before intent / temporal stages


@pytest.mark.parametrize(
    ("question", "active", "authorized", "route", "code"),
    [
        ("What is the current closing date?", None, ALL, Route.CLARIFY, "PROJECT_REQUIRED"),
        ("What is the closing date of P123456?", IPF, ALL, Route.REFUSE, "UNSUPPORTED_PROJECT"),
        ("Disbursement of IBRD-1111-1?", IPF, ALL, Route.REFUSE, "UNSUPPORTED_PROJECT"),
        ("What is the current closing date?", "P000000", ALL, Route.REFUSE, "UNSUPPORTED_PROJECT"),
        (
            "Compare P130544 and P506272 disbursement",
            IPF,
            ALL,
            Route.REFUSE,
            "MULTI_PROJECT_NOT_SUPPORTED",
        ),
        ("What is the current closing date?", IPF, (PFORR,), Route.REFUSE, "NOT_AUTHORIZED"),
        ("", IPF, ALL, Route.REFUSE, "INVALID_REQUEST"),
        ("x" * 2000, IPF, ALL, Route.REFUSE, "INVALID_REQUEST"),
    ],
)
def test_scope_and_input_refusals_execute_nothing(question, active, authorized, route, code):
    h = harness()
    r = h.ask(question, active=active, authorized=authorized)
    assert (r.decision.route, r.decision.reason_code) == (route, code)
    assert r.outcome == ExecutionOutcome.NOT_EXECUTED and nothing_ran(h)


def test_a_mentioned_project_without_active_scope_is_used_only_if_authorised():
    h = harness()
    r = h.ask(
        "What is the current closing date of the Water Security and Resilience Program?",
        active=None,
    )
    assert r.understanding.project.project_id == OTHER
    assert {scope for _, _, scope in h.executor.calls} == {OTHER}
    h = harness()
    r = h.ask("What is the current closing date of P506272?", active=None, authorized=(IPF,))
    assert r.decision.reason_code == "NOT_AUTHORIZED" and nothing_ran(h)


@pytest.mark.parametrize("project", ALL)
def test_every_call_is_scoped_to_the_resolved_project(project):
    h = harness()
    for question in QUESTIONS_THAT_WOULD_EXECUTE + ["What deserves my attention?"]:
        h.ask(question, active=project)
    assert h.executor.calls and {scope for _, _, scope in h.executor.calls} == {project}
    assert {args["project_id"] for _, args, _ in h.executor.calls} == {project}
    assert {pid for _, pid in h.retriever.first_stage_calls} == {project}
    assert {req.project_id for req in h.reads} == {project}


# -- refusals, semantic, investigation ---------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "code"),
    [
        ("Will the project fail?", "PREDICTION_NOT_SUPPORTED"),
        ("Update the closing date to 2030", "READ_ONLY"),
        ("What is the weather in Bengaluru?", "OUT_OF_DOMAIN"),
    ],
)
def test_refusal_intents_execute_nothing(question, code):
    h = harness()
    r = h.ask(question)
    assert (r.decision.route, r.decision.reason_code) == (Route.REFUSE, code) and nothing_ran(h)
    if code == "PREDICTION_NOT_SUPPORTED":
        assert "attention signals" in r.decision.suggestion


def test_semantic_classification_required_is_explicit_and_executes_nothing():
    h = harness()
    r = h.ask("How has the way citizens can register complaints been improved?")
    assert r.decision.route == Route.SEMANTIC_CLASSIFICATION_REQUIRED
    assert r.decision.reason_code == "INTENT_NOT_RESOLVED_BY_RULES" and nothing_ran(h)


@pytest.mark.parametrize(
    ("question", "kind", "tools"),
    [
        (
            "Why did the PDO rating drop?",
            "RATINGS",
            {"get_rating_history", "get_attention_signals"},
        ),
        (
            "Which risks anticipated during appraisal later appeared?",
            "APPRAISAL_RISKS_REALISED",
            {"get_risk_register", "get_rating_history", "get_attention_signals"},
        ),
        (
            "What changed and why?",
            "CHANGE_AND_WHY",
            {"get_project_overview", "get_project_timeline", "get_attention_signals"},
        ),
    ],
)
def test_investigation_is_planned_and_never_executed(question, kind, tools):
    h = harness()
    r = h.ask(question, active=PFORR)
    assert (r.decision.route, r.outcome) == (
        Route.INVESTIGATION,
        ExecutionOutcome.PLANNED_NOT_EXECUTED,
    )
    plan = r.investigation_plan
    assert plan.executed is False and plan.investigation_kind == kind and plan.project_id == PFORR
    assert {c.tool for c in plan.structured_calls} == tools
    assert all(c.validated and c.arguments["project_id"] == PFORR for c in plan.structured_calls)
    assert plan.document_retrievals[0].query == question
    assert nothing_ran(h) and r.tool_results == [] and not r.retrieval_executed


# -- no generic fallback ------------------------------------------------------------------------


def empty_project_tables():
    data = routing_tables()
    for table in data:
        data[table] = [
            r
            for r in data[table]
            if r.get("project_id") != IPF or table in ("gold.project_360", "silver.projects")
        ]
    return data


@pytest.mark.parametrize(
    "question",
    [
        "Show the project timeline",
        "Show the PDO rating history",
        "What deserves my attention?",
        "What are the baseline and target values?",
        "What are the main environmental risks?",
        "What was the IP rating in ISR 9?",
    ],
)
def test_empty_or_missing_structured_data_never_falls_back_to_documents(question):
    h = harness(data=empty_project_tables)
    r = h.ask(question)
    assert r.decision.route == Route.STRUCTURED
    assert r.outcome == ExecutionOutcome.INSUFFICIENT_EVIDENCE
    assert r.retrieval_executed is False and h.retriever.first_stage_calls == []
    assert r.executed_tools and "search_project_documents" not in r.executed_tools
    assert r.mechanical  # the mechanical reason is reported (NO_RECORDS / ISR_NOT_FOUND)


def test_requirements_configuration_has_no_fallback():
    text = (REPO_CONFIG_DIR / "routing" / "requirements.yaml").read_text(encoding="utf-8")
    assert "fallback:" not in text and "fallback_to" not in text
    for intent, req in CONFIG.requirements.intents.items():
        if req.route == "STRUCTURED":
            assert "search_project_documents" not in req.tools, intent


def test_data_integrity_failure_is_reported():
    data = routing_tables()
    data["gold.project_360"] = [r for r in data["gold.project_360"] if r["project_id"] != IPF]
    h = harness(data=data)
    r = h.ask("What is the current closing date?")
    assert r.outcome == ExecutionOutcome.DATA_INTEGRITY_ERROR
    assert h.retriever.first_stage_calls == []


# -- temporal decisions ----------------------------------------------------------------------------


def test_ambiguous_event_anchor_is_clarified_with_candidates_never_guessed():
    h = harness()
    r = h.ask("List the restructurings since the restructuring")
    assert (r.decision.route, r.decision.reason_code) == (Route.CLARIFY, "AMBIGUOUS_TIME")
    assert len(r.decision.clarification_options) == 2
    assert [c.tool for c in r.anchor_results] == ["get_project_timeline"]  # after scope
    assert r.executed_tools == [] and r.tool_results == []


@pytest.mark.parametrize(
    ("anchor", "date_from"),
    [
        ("the 2021 restructuring", date(2021, 5, 20)),
        ("the first restructuring", date(2018, 6, 1)),
        ("the latest restructuring", date(2021, 5, 20)),
    ],
)
def test_resolved_event_anchor_uses_the_source_stated_date(anchor, date_from):
    h = harness()
    r = h.ask(f"Show the timeline since {anchor}")
    assert r.decision.route == Route.STRUCTURED and r.understanding.temporal.date_from == date_from
    assert r.understanding.temporal.anchor_candidates[0].event_date == date_from
    assert r.tool_results[0].filters["date_from"] == date_from.isoformat()


def test_anchor_with_only_undated_or_no_events_is_unresolved():
    r = harness().ask("Show the timeline since effectiveness")
    assert r.decision.reason_code == "AMBIGUOUS_TIME" and "0 source-dated" in r.decision.detail


@pytest.mark.parametrize(
    ("question", "code"),
    [
        ("What changed in disbursement over the last 12 months?", "AMBIGUOUS_TIME"),
        ("What is the overall risk rating?", "TIME_REQUIRED"),
        ("What was the PDO rating in 2021?", "TEMPORAL_NOT_SUPPORTED"),
        ("How much was disbursed in 2020?", "TEMPORAL_NOT_SUPPORTED"),
    ],
)
def test_time_that_cannot_be_applied_exactly_is_clarified(question, code):
    h = harness()
    r = h.ask(question)
    assert (r.decision.route, r.decision.reason_code) == (Route.CLARIFY, code)
    assert r.tool_results == [] and r.executed_tools == [] and h.retriever.first_stage_calls == []


def test_overall_risk_rating_with_explicit_time_proceeds():
    assert (
        harness().ask("What is the overall risk rating in the latest ISR?").decision.route
        == Route.STRUCTURED
    )
    assert (
        harness().ask("What was the overall risk rating at appraisal?").decision.reason_code
        == "INTENT_RISKS"
    )


# -- observability ------------------------------------------------------------------------------


def test_the_decision_is_explainable_from_harness_state():
    h = harness()
    r = h.ask("How much of loan IBRD-9324-0 was disbursed in ISR 2?")
    u = r.understanding
    assert r.request_id == "req-1" and r.access.user_ref == "u-test"
    assert (u.project.project_id, u.project.basis) == (IPF, "ACTIVE_SCOPE")
    assert [m.kind.value for m in u.project.mentions] == ["LOAN_NUMBER"]
    assert u.temporal.kind.value == "ISR_SEQUENCE" and u.temporal.explicit
    assert u.intent.decided_by == ("SUBJECT_FINANCE",) and u.intent.hits
    assert u.requirements.structured[0].tool == "get_financial_status"
    assert r.decision.reason_code == "INTENT_FINANCIAL_STATUS" and r.decision.alternatives_rejected
    assert [t.stage for t in r.timings] == [
        "input",
        "project",
        "temporal",
        "intent",
        "requirements",
        "route",
        "execute",
    ]
    assert r.versions == CONFIG.versions and r.versions["router"] == "9B.2"
    dumped = json.dumps(r.model_dump(mode="json"))
    assert "token" not in dumped.lower() and "password" not in dumped.lower()


def test_stage_timings_stop_where_the_decision_was_made():
    r = harness().ask("What does P179039 say?")
    assert [t.stage for t in r.timings] == ["input", "project"]
    assert r.decision.decided_at_stage == "project"
