"""Application evaluation harness: dataset contract, pure metrics, runner. Offline only."""

import json
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from worldbank_copilot.common import load_project_registry
from worldbank_copilot.copilot.contracts import (
    Citation,
    Claim,
    EvidenceItem,
    InvestigationActivity,
    InvestigationResult,
    ModelCall,
    Validation,
)
from worldbank_copilot.evaluation.app_cases import (
    CASES_FILE,
    CATEGORIES,
    Case,
    FactSpec,
    load_cases,
    resolve_fact,
    select,
)
from worldbank_copilot.evaluation.app_metrics import (
    aggregate,
    citation_metrics,
    context_metrics,
    fact_recall,
    project_isolation,
    render_report,
    retrieval_metrics,
    score_case,
    task_success,
    temporal_citations,
)
from worldbank_copilot.evaluation.app_runner import (
    Capture,
    RunStore,
    load_reviews,
    run_cases,
    select_cases,
)
from worldbank_copilot.investigation.claims import ModelReply
from worldbank_copilot.retrieval.evaluation import EvidenceRef, Question, load_questions

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = {q.id: q for q in load_questions(ROOT / "evaluation/retrieval_questions.yaml")}


@pytest.fixture(scope="module")
def cases():
    registry = load_project_registry(ROOT / "configs")
    return load_cases(
        ROOT / CASES_FILE, project_ids=registry.project_ids, retrieval_questions=QUESTIONS
    )


# -- dataset contract ------------------------------------------------------------------
def test_fifty_unique_cases_five_per_category_on_registered_projects(cases):
    assert len(cases) == 50 and len({c.case_id for c in cases}) == 50
    assert Counter(c.category for c in cases) == {k: 5 for k in CATEGORIES}
    unregistered = [c for c in cases if c.project_id not in ("P130544", "P179039", "P506272")]
    assert [(c.case_id, c.expected.allowed_statuses) for c in unregistered] == [
        ("APP_040", ["REFUSE"])
    ]


def test_golden_facts_are_governed_queries_never_typed_values():
    rows = [json.loads(line) for line in (ROOT / CASES_FILE).read_text("utf-8").splitlines()]
    text = json.dumps(rows)
    for forbidden in ('"value"', '"values"', '"expected_answer"', '"answer"'):
        assert forbidden not in text
    with pytest.raises(ValueError):
        FactSpec.model_validate(
            {"fact_id": "x", "tool": "t", "path": "p", "kind": "date", "value": "2024-07-23"}
        )


def test_prediction_and_isolation_cases_are_strict(cases):
    for case in cases:
        assert case.expected.project_isolation_required is True
        if case.category == "prediction":
            assert case.expected.prediction_must_be_refused
            assert case.expected.allowed_statuses == ["REFUSE"] and case.expected.forbidden_claims
        if case.category == "cross_project":
            assert case.expected.foreign_projects or case.expected.unauthorized_project


def test_invalid_case_contracts_are_rejected():
    base = {
        "case_id": "APP_001",
        "category": "prediction",
        "project_id": "P130544",
        "question": "Will it fail?",
    }
    with pytest.raises(ValueError):  # prediction must require REFUSE
        Case.model_validate(
            {
                **base,
                "expected": {"allowed_statuses": ["ANSWER"], "prediction_must_be_refused": True},
            }
        )
    with pytest.raises(ValueError):  # abstention flag must match the statuses
        Case.model_validate(
            {**base, "expected": {"allowed_statuses": ["ANSWER"], "abstention_acceptable": True}}
        )


def test_golden_resolution_keeps_record_identity():
    items = [
        {
            "event_type": "RESTRUCTURING",
            "event_date": "2024-07-23",
            "source": {"table": "gold.project_timeline", "record_id": "r1"},
        },
        {
            "event_type": "RESTRUCTURING",
            "event_date": None,
            "source": {"table": "t", "record_id": "r2"},
        },
        {
            "event_type": "RESTRUCTURING",
            "event_date": "2021-05-20",
            "source": {"table": "t", "record_id": "r3"},
        },
    ]
    spec = FactSpec(
        fact_id="d",
        tool="get_project_timeline",
        path="event_date",
        kind="date",
        where={"event_date": "2021*"},
    )
    resolved = resolve_fact(spec, items)
    assert resolved["values"] == ["2021-05-20"] and resolved["records"][0]["record_id"] == "r3"
    assert (
        resolve_fact(spec.model_copy(update={"where": {"event_date": "1999*"}}), items)["status"]
        == "UNRESOLVED"
    )
    nested = {
        "ratings": [{"rating_type": "PDO", "rating": {"value": "MU"}}, {"rating": {"value": "S"}}]
    }
    assert select(nested, "ratings.rating.value") == ["MU", "S"]


# -- metric helpers --------------------------------------------------------------------
PID = "P130544"
DOC = f"{PID}-abc"


def question(mode="any", items=2):
    evidence = [
        EvidenceRef(document_id=DOC, pages=[p], contains=f"phrase {p}") for p in range(1, items + 1)
    ]
    return Question(
        id="qx",
        project_id=PID,
        category="c",
        kind="k",
        question="q",
        evidence_mode=mode,
        evidence=evidence,
    )


def row(page, text):
    return {"document_id": DOC, "page_numbers": [page], "chunk_text": text, "project_id": PID}


def evidence(eid, *, source_type="DOCUMENT", day=None, document=DOC, page=1, text="", project=PID):
    payload = {"project_id": project, "text": text, "document_id": document}
    if day:
        payload["document_date"] = day
    return EvidenceItem(
        evidence_id=eid,
        provenance=("DOCUMENTED_FINDING",),
        source_type=source_type,
        citation={"document_id": document, "pages": [page]},
        payload=payload,
    )


def claim(cid, *ids, support="SUPPORTED", relation="NONE", text="A fact."):
    return Claim(
        claim_id=cid,
        text=text,
        claim_type="ASSERTION",
        provenance="DOCUMENTED_FINDING",
        evidence_ids=ids,
        citations=tuple(Citation(evidence_id=i, source_identity="s") for i in ids),
        support=support,
        temporal_relation=relation,
    )


def result(status="ANSWER", claims=(), items=(), **fields):
    return InvestigationResult(
        request_id="r",
        query="q",
        project_id=fields.pop("project_id", PID),
        route="INVESTIGATOR",
        status=status,
        message="m",
        claims=claims,
        evidence=items,
        **fields,
    ).model_dump(mode="json")


def case_with(category="attention", **expected):
    base = {"allowed_statuses": ["ANSWER"]}
    return Case.model_validate(
        {
            "case_id": "APP_001",
            "category": category,
            "project_id": PID,
            "question": "Question?",
            "expected": {**base, **expected},
        }
    )


def test_retrieval_metrics_follow_phase8_relevance():
    rows = [row(9, "noise"), row(1, "it says phrase 1 here"), row(2, "phrase 2")]
    rows += [row(9, "noise")] * 8
    any_mode = retrieval_metrics(rows, [question("any")])
    assert any_mode["recall_at_5"] == 1.0 and any_mode["mrr"] == 0.5
    assert any_mode["precision_at_5"] == 2 / 5 and any_mode["precision_at_10"] == 2 / 10
    assert any_mode["hit_at_5"] == 1.0 and any_mode["ndcg_at_10"] is None  # binary labels only
    all_mode = retrieval_metrics(rows[:2], [question("all")])
    assert all_mode["recall_at_5"] == 0.5
    assert retrieval_metrics(rows, [])["recall_at_10"] is None  # no labels: NOT_APPLICABLE


def test_context_recall_and_precision_detect_packing_loss_and_noise():
    q = [question("all")]
    retrieved = [row(1, "phrase 1"), row(2, "phrase 2"), row(9, "noise")]
    shown = [row(1, "phrase 1"), row(9, "noise")]
    m = context_metrics(retrieved, shown, q)
    assert m["context_recall"] == 0.5 and m["context_precision"] == 0.5
    assert context_metrics(retrieved, None, q)["context_recall"] is None


def test_claim_precision_fact_recall_and_f1_are_separate_units():
    spec = {"fact_id": "d", "tool": "get_project_timeline", "path": "event_date", "kind": "date"}
    case = case_with(required_facts=[spec])
    golden = {
        "d": {**spec, "match": "all", "values": ["2024-07-23", "2021-05-20"], "status": "RESOLVED"}
    }
    res = result(
        claims=(
            claim("C1", "e1", text="Restructured on July 23, 2024."),
            claim("C2", "e1", support="PARTIALLY_SUPPORTED"),
        ),
        items=(evidence("e1"),),
    )
    recall, detail = fact_recall(case, golden, res)
    assert recall == 0.5 and detail == [{"fact_id": "d", "values": 2, "found": 1}]
    scored = score_case(case, {"result": res}, golden, QUESTIONS)
    assert scored["claim_precision_strict"] == 0.5  # PARTIALLY_SUPPORTED is not supported
    assert scored["answer_f1"] == pytest.approx(0.5)
    assert fact_recall(case_with(), {}, res)[0] is None  # no golden facts: NOT_APPLICABLE


def test_citation_metrics_catch_invalid_handles_and_foreign_documents():
    case = case_with()
    good = result(claims=(claim("C1", "e1"),), items=(evidence("e1"),))
    assert citation_metrics(case, good, None)["citation_validity"] == 1.0
    dangling = result(claims=(claim("C1", "e1", "missing"),), items=(evidence("e1"),))
    m = citation_metrics(case, dangling, None)
    assert m["citation_validity"] == 0.5 and m["citation_completeness"] == 0.0
    foreign = result(claims=(claim("C1", "e1"),), items=(evidence("e1", document="P179039-x"),))
    assert citation_metrics(case, foreign, None)["citation_project_consistency"] == 0.0


def test_project_isolation_is_deterministic():
    case = case_with(foreign_projects=["P179039"])
    assert project_isolation(case, result(items=(evidence("e1"),)))
    assert not project_isolation(case, result(items=(evidence("e1", project="P179039"),)))
    assert not project_isolation(
        case, result(claims=(claim("C1", "e1", text="P179039 also"),), items=(evidence("e1"),))
    )
    assert not project_isolation(case, result(project_id="P179039"))


def test_temporal_citations_against_the_governed_anchor_date():
    anchor = date(2024, 7, 23)
    items = {
        e.evidence_id: e.model_dump(mode="json")
        for e in (evidence("b", day="2023-01-01"), evidence("a", day="2025-01-01"), evidence("u"))
    }

    def check(*claims):
        return temporal_citations([c.model_dump(mode="json") for c in claims], items, anchor)

    assert check(claim("C1", "b", relation="BEFORE")) == 1.0
    assert check(claim("C1", "a", relation="BEFORE")) == 0.0
    assert check(claim("C1", "b", "a", relation="ACROSS")) == 1.0
    assert check(claim("C1", "a", relation="ACROSS")) == 0.0
    assert check(claim("C1", "b", "u", relation="ACROSS")) == 0.0  # undated cannot establish
    assert check(claim("C1", "b")) is None  # no temporal claims: NOT_APPLICABLE


def test_refusal_and_false_refusal_scoring():
    predict = case_with(
        "prediction",
        allowed_statuses=["REFUSE"],
        prediction_must_be_refused=True,
        forbidden_claims=[r"(?i)probabilit"],
    )
    assert task_success(predict, result("REFUSE")) == "PASS"
    assert (
        task_success(predict, result(claims=(claim("C1", "e1"),), items=(evidence("e1"),)))
        == "FAIL"
    )
    answerable = case_with(
        acceptable_statuses=["INSUFFICIENT_EVIDENCE"], abstention_acceptable=True
    )
    refused = score_case(answerable, {"result": result("REFUSE")}, {}, QUESTIONS)
    assert (
        refused["false_refusal"]
        and refused["verdict"] == "FAIL"
        and "REFUSAL" in refused["buckets"]
    )
    abstained = score_case(answerable, {"result": result("INSUFFICIENT_EVIDENCE")}, {}, QUESTIONS)
    assert abstained["verdict"] == "PARTIAL" and abstained["unnecessary_abstention"]


def test_verdicts_and_failure_buckets():
    case = case_with()
    crashed = score_case(case, {"result": None, "error": {"type": "ValueError"}}, {}, QUESTIONS)
    assert crashed["verdict"] == "FAIL" and crashed["buckets"] == ["RUNTIME"]
    degraded = result(
        claims=(claim("C1", "e1", support="NOT_ASSESSED"),),
        items=(evidence("e1"),),
        validation=Validation(critic_status="FAILED"),
        model_calls=(ModelCall(role="CRITIC", latency_ms=1, outcome="MODEL_OUTPUT_INVALID"),),
    )
    scored = score_case(case, {"result": degraded}, {}, QUESTIONS)
    assert scored["verdict"] == "PARTIAL" and scored["graceful_degradation"]
    assert scored["runtime_success"] and "CRITIC" in scored["buckets"]
    assert scored["unverified_published"] == 1 and scored["claim_precision_strict"] == 0.0
    leaked = result(items=(evidence("e1", project="P179039"),))
    assert "PROJECT_ISOLATION" in score_case(case, {"result": leaked}, {}, QUESTIONS)["buckets"]
    planning = result(
        "FAIL_CLOSED", activity=InvestigationActivity(planning_failure="SCHEMA_INVALID")
    )
    assert not score_case(case, {"result": planning}, {}, QUESTIONS)["runtime_success"]


def test_aggregate_reports_denominators_and_not_applicable():
    case = case_with()
    ok = score_case(
        case,
        {"result": result(claims=(claim("C1", "e1"),), items=(evidence("e1"),), latency_ms=100)},
        {},
        QUESTIONS,
    )
    crashed = score_case(case, {"result": None, "error": {}}, {}, QUESTIONS)
    summary = aggregate([ok, crashed])
    assert summary["operations"]["runtime_success"] == {"value": 0.5, "n": 2}
    assert summary["retrieval"]["recall_at_10"] == {"value": None, "n": 0}
    assert summary["generation"]["claim_precision_strict"] == {"value": 1.0, "n": 1}
    assert summary["failure_buckets"] == {"RUNTIME": 1}
    report = render_report(summary, [ok, crashed], {"run_id": "t"})
    assert "NOT_APPLICABLE" in report and "## H. Failure analysis" in report


def test_human_reviews_are_bounded():
    assert load_reviews([{"case_id": "APP_001", "relevance": "2", "completeness": ""}]) == {
        "APP_001": {"relevance": 2}
    }
    with pytest.raises(ValueError):
        load_reviews([{"case_id": "APP_001", "relevance": 3}])


# -- runner ------------------------------------------------------------------------------
class FakeAdapter:
    def __init__(self):
        self.last_reason = "R"

    def invoke(self, request):
        return ModelReply(text='{"claims": []}', model_identity="m")


class FakeCopilot:
    def __init__(self, fail=()):
        self.fail, self.calls = set(fail), []
        self.investigator, self.synthesizer, self.critic = FakeAdapter(), FakeAdapter(), None

    def investigate(self, query, project_id):
        self.calls.append(query)
        if query in self.fail:
            raise RuntimeError("boom")
        from types import SimpleNamespace

        request = SimpleNamespace(context_json=json.dumps({"evidence": [{"handle": "E1"}]}))
        self.synthesizer.invoke(request)
        return InvestigationResult(
            request_id="r", query=query, project_id=project_id, status="REFUSE", message="m"
        )


def test_runner_continues_after_a_failure_and_resumes(cases, tmp_path):
    chosen = select_cases(cases, categories=["prediction"])
    app = FakeCopilot(fail={chosen[1].question})
    store = RunStore(tmp_path, "run1")
    run_cases(app, chosen, store, log=lambda *_: None)
    latest = store.latest()
    assert len(latest) == 5 and latest[chosen[1].case_id]["error"]["type"] == "RuntimeError"
    assert latest[chosen[0].case_id]["captured"]["synthesis_evidence"] == [{"handle": "E1"}]
    assert app.synthesizer.last_reason == "R"  # diagnostics pass through the recorder
    run_cases(app, chosen, store, log=lambda *_: None)  # resume: nothing re-run
    assert len(app.calls) == 5
    run_cases(
        app,
        select_cases(chosen, case_ids=[chosen[1].case_id]),
        store,
        rerun=True,
        log=lambda *_: None,
    )
    assert len(store.records()) == 6 and store.latest()[chosen[1].case_id]["attempt"] == 2


def test_capture_wraps_each_adapter_once():
    app = FakeCopilot()
    first = Capture(app)
    inner = app.synthesizer
    Capture(app)
    assert app.synthesizer is inner and app.critic is None and first.calls["critic"] == []
