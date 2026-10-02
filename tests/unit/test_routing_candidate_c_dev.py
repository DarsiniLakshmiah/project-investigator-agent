"""Phase 9D Candidate C DEV evaluation: mechanical populations, lock, call plan, pacing,
exclusive failure kinds, outcome precedence and TEST / ambiguity-probe isolation.
No network, no Databricks, no model predictions (fake transports only)."""

import copy
import json
import threading

import pytest
from tests.conftest import REPO_CONFIG_DIR, REPO_ROOT
from tests.support.routing_fixtures import CONFIG, harness

from worldbank_copilot.common import load_project_registry
from worldbank_copilot.ingestion.documents import load_document_manifest
from worldbank_copilot.routing.bounded_classifier import (
    contract_sha256,
    load_bounded_classifier_config,
)
from worldbank_copilot.routing.bounded_classifier_dev import (
    ACCEPTANCE_RULE,
    OUTCOMES,
    PopulationDrift,
    RecordedClassifier,
    acceptance_rule_sha256,
    artifact_name,
    call_plan,
    derive_c1,
    derive_lock,
    derive_populations,
    evaluate,
    lock_sha256,
    request_contexts,
    run_dev_predictions,
)
from worldbank_copilot.routing.bounded_classifier_probe import ChatResponse
from worldbank_copilot.routing.bounded_semantic import BoundedSemanticClassifier
from worldbank_copilot.routing.entities import EntityIndex
from worldbank_copilot.routing.evaluation import load_dataset
from worldbank_copilot.routing.models import Intent, Route
from worldbank_copilot.routing.semantic import route_map

CONTRACT = "c64561c25e0e417328b3a525c8b225a6b2f9e465b7c9d177a9128d08f013fe8b"
BC = load_bounded_classifier_config(REPO_CONFIG_DIR)
REGISTRY = load_project_registry(REPO_CONFIG_DIR)
INDEX = EntityIndex.build(
    REGISTRY, load_document_manifest(REPO_CONFIG_DIR / "document_manifest.yaml"), CONFIG
)
ALL_PROJECTS = tuple(REGISTRY.project_ids)
ROUTES = route_map(CONFIG.requirements)
PROTOCOL = derive_lock(
    BC,
    CONFIG,
    INDEX,
    load_dataset(REPO_ROOT / "evaluation" / "routing_cases.yaml"),
    REPO_ROOT,
    ALL_PROJECTS,
)
LOCK = json.loads((REPO_ROOT / "evaluation" / "candidate_c_dev_lock.json").read_text("utf-8"))
POPS = LOCK["populations"]
DEV = {c.case_id: c for c in PROTOCOL.cases}
FREEZE = json.loads((REPO_ROOT / "evaluation" / "routing_freeze_9c.json").read_text("utf-8"))
TEST_IDS = set(FREEZE["test_case_ids"])  # ids only - used to prove they never appear
CONTEXTS = request_contexts(CONFIG, INDEX, PROTOCOL.cases, POPS, ALL_PROJECTS)
QUESTION_TO_CASE = {ctx.question: cid for cid, ctx in CONTEXTS.items()}


def expected_label(cid):
    return DEV[cid].expected.intents[0].value


# -- fakes ------------------------------------------------------------------------------------


class FakeClock:
    def __init__(self):
        self.t, self.lock = 500.0, threading.Lock()

    def __call__(self):
        with self.lock:
            return self.t

    def sleep(self, seconds):
        with self.lock:
            self.t += seconds


class DevEndpoint:
    """Answers each request by its case id: expected intent unless overridden."""

    def __init__(
        self,
        clock,
        labels=None,
        status_by_seq=None,
        contract_by_seq=(),
        latency=0.6,
        reasoning=False,
    ):
        self.clock, self.labels = clock, labels or {}
        self.status_by_seq, self.contract_by_seq = status_by_seq or {}, set(contract_by_seq)
        self.latency, self.reasoning, self.bodies = latency, reasoning, []

    def post(self, body, timeout):
        self.bodies.append(body)
        seq = len(self.bodies)
        self.clock.sleep(self.latency)
        cid = QUESTION_TO_CASE[json.loads(body["messages"][1]["content"])["request"]]
        if seq in self.status_by_seq:
            status = self.status_by_seq[seq]
            headers = {"Retry-After": "9"} if status == 429 else {}
            return ChatResponse(
                status,
                {"error_code": "X", "retry_after": 11},
                headers,
                self.latency,
                "ReadTimeout" if status == 0 else None,
            )
        label = self.labels.get(cid, expected_label(cid))
        if isinstance(label, list):
            label = label.pop(0)
        text = "not json" if seq in self.contract_by_seq else json.dumps({"intent": label})
        content = (
            [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "SECRET"}]},
                {"type": "text", "text": text},
            ]
            if self.reasoning
            else text
        )
        body = {
            "model": "gpt-oss-20b-080525",
            "usage": {"prompt_tokens": 700},
            "choices": [{"finish_reason": "stop", "message": {"content": content}}],
        }
        return ChatResponse(200, body, {}, self.latency)


def run(endpoint=None, sleep=None, **kwargs):
    clock = FakeClock()
    endpoint = endpoint or DevEndpoint(clock, **kwargs)
    endpoint.clock = clock
    artifact = run_dev_predictions(
        BC,
        LOCK,
        CONTEXTS,
        BoundedSemanticClassifier(BC, endpoint, ROUTES),
        clock=clock,
        sleep=sleep or clock.sleep,
        log=lambda _: None,
    )
    return artifact, endpoint


def synthetic_hybrid(artifact):
    """Hybrid rows as the replay would give them for C1 (route from requirements)."""
    hybrid = copy.deepcopy(PROTOCOL.baseline)
    main = {c["case_id"]: c for c in artifact["calls"] if c["kind"] == "main"}
    for cid in POPS["c1"]:
        label = main[cid]["label"]
        if label and label != "ABSTAIN":
            hybrid[cid].update(
                route=ROUTES[Intent(label)], intent=label, reason_code=f"INTENT_{label}"
            )
        else:
            hybrid[cid].update(route="CLARIFY", intent=None, reason_code="SEMANTIC_ABSTAIN")
    return hybrid


def verdict(artifact, hybrid=None, invoked=None, recomputed_a=None, lock=None, c1=None):
    invoked = (
        invoked if invoked is not None else [(cid, artifact["contexts"][cid]) for cid in POPS["c1"]]
    )
    return evaluate(
        BC,
        lock or LOCK,
        artifact,
        PROTOCOL.cases,
        ROUTES,
        recorded_a=PROTOCOL.baseline,
        recomputed_a=recomputed_a if recomputed_a is not None else PROTOCOL.baseline,
        hybrid=hybrid if hybrid is not None else synthetic_hybrid(artifact),
        invoked=invoked,
        recomputed_lock_sha256=lock_sha256(PROTOCOL.lock),
        recomputed_c1=c1 if c1 is not None else POPS["c1"],
    )


def seq_of(case_id, kind="main", round_no=0):
    return next(
        e["sequence"]
        for e in LOCK["call_plan"]
        if (e["case_id"], e["kind"], e["round"]) == (case_id, kind, round_no)
    )


# -- populations and lock ---------------------------------------------------------------------


def test_populations_are_derived_mechanically_and_match_the_lock():
    assert PROTOCOL.lock == LOCK  # recomputed lock == committed lock
    assert POPS["c1"] == ["r015", "r048"]
    assert len(POPS["c2"]) == 22 and set(POPS["c1"]) <= set(POPS["c2"])
    assert POPS["c2_repeat"] == ["r002", "r014", "r052", "r006"]
    assert [DEV[c].expected.route for c in POPS["c2_repeat"]] == [
        "STRUCTURED",
        "DOCUMENT",
        "INVESTIGATION",
        "STRUCTURED",
    ]
    excluded = set(DEV) - set(POPS["c2"])
    assert excluded == {"r059", "r063", "r065", "r070", "r076", "r078", "r080"}


def test_c1_comes_from_the_service_not_from_configuration():
    drifted = copy.deepcopy(PROTOCOL.baseline)
    drifted["r015"]["route"] = "DOCUMENT"  # recorded baseline no longer agrees
    from worldbank_copilot.routing.bounded_classifier_dev import (
        boundary_service,
        semantic_boundary,
    )

    boundary = semantic_boundary(boundary_service(CONFIG, INDEX), PROTOCOL.cases, ALL_PROJECTS)
    c1, drift = derive_c1(boundary, drifted)
    assert c1 == ["r015", "r048"] and drift  # the service still says C1; drift is reported
    wrong = BC.model_copy(update={"dev_evaluation": BC.dev_evaluation | {"expected_c1": ["r015"]}})
    with pytest.raises(PopulationDrift):
        derive_populations(wrong, PROTOCOL.cases, PROTOCOL.baseline, boundary)


def test_lock_pins_contract_hashes_rule_and_counts():
    assert LOCK["contract_sha256"] == contract_sha256(BC) == CONTRACT
    assert LOCK["acceptance_rule_sha256"] == acceptance_rule_sha256()
    assert LOCK["acceptance_rule"] == ACCEPTANCE_RULE
    assert (
        ACCEPTANCE_RULE["outcome_precedence"]
        == list(OUTCOMES)
        == [
            "INVALID",
            "REJECTED_SAFETY",
            "INCONCLUSIVE_OPERATIONAL",
            "REJECTED",
            "ACCEPTED_FOR_NEXT_STAGE",
        ]
    )
    assert LOCK["dev_scheduled_calls"] == 44 and LOCK["c1_calls"] == 12
    assert LOCK["candidate_a_dev_route_correct"] == 24
    assert LOCK["required_hybrid_route_correct"] == 26
    assert LOCK["test_evaluated"] is False and LOCK["split_evaluated"] == "dev"
    assert LOCK["schedule"] == {
        "initial_quiet_seconds": 60,
        "ordinary_call_gap_seconds": 5.0,
        "gap_tolerance_seconds": 0.000001,
    }
    assert LOCK["ambiguity_probe_sha256_lf"].startswith("93d44d28")
    assert LOCK["routing_dataset_sha256_lf"] == FREEZE["dataset_sha256_lf"]
    assert artifact_name("dev1") == "candidate_c_dev_predictions__dev1.json"


def test_call_plan_is_exactly_the_approved_44_calls():
    plan = [(e["case_id"], e["kind"], e["round"]) for e in LOCK["call_plan"]]
    rest = [c for c in POPS["c2"] if c not in POPS["c1"]]
    main = [("r015", "main", 0), ("r048", "main", 0)] + [(c, "main", 0) for c in rest]
    rounds = [
        (cid, "repeat", r)
        for r in (1, 2, 3)
        for cid in ("r015", "r048", "r002", "r014", "r052", "r006")
    ] + [(cid, "repeat", r) for r in (4, 5) for cid in ("r015", "r048")]
    assert plan == main + rounds and len(plan) == 44
    assert [e["sequence"] for e in LOCK["call_plan"]] == list(range(1, 45))
    assert sum(e["population"] == "C1" for e in LOCK["call_plan"]) == 12
    assert call_plan(POPS["c1"], POPS["c2"], POPS["c2_repeat"], 5, 3) == LOCK["call_plan"]


def test_c1_request_contexts_are_the_exact_production_contexts():
    assert set(CONTEXTS) == set(POPS["c2"])
    for cid in POPS["c1"]:
        assert CONTEXTS[cid].request_id == cid and CONTEXTS[cid].project_id


# -- the paced run ------------------------------------------------------------------------------


def test_run_follows_plan_schedule_and_frozen_contract():
    artifact, endpoint = run()
    calls = artifact["calls"]
    assert artifact["schedule"]["valid"], artifact["schedule"]
    assert len(calls) == len(endpoint.bodies) == 44  # one attempt per call, no retries
    assert calls[0]["started_s"] == 60.0 and calls[0]["gap_before_s"] is None
    assert all(c["gap_before_s"] == 5.0 for c in calls[1:])  # END -> START
    assert all(b["reasoning_effort"] == "low" and b["max_tokens"] == 1024 for b in endpoint.bodies)
    assert all(not {"tools", "tool_choice", "functions"} & set(b) for b in endpoint.bodies)
    assert all(b["response_format"]["json_schema"]["strict"] for b in endpoint.bodies)
    assert artifact["contract_sha256"] == CONTRACT and artifact["test_evaluated"] is False


def test_pacing_violation_is_invalid_before_any_call():
    artifact, endpoint = run(sleep=lambda s: None)
    assert endpoint.bodies == [] and not artifact["schedule"]["valid"]
    assert verdict(artifact, hybrid=copy.deepcopy(PROTOCOL.baseline), invoked=[])["outcome"] == (
        "INVALID"
    )


def test_failure_kinds_are_exclusive():
    clock = FakeClock()
    clf = BoundedSemanticClassifier(BC, DevEndpoint(clock), ROUTES)
    ok = {"choices": [{"finish_reason": "stop", "message": {"content": '{"intent": "RISKS"}'}}]}
    bad = {"choices": [{"finish_reason": "stop", "message": {"content": "RISKS"}}]}
    cases = [
        (ChatResponse(200, ok, {}, 0.3), "NONE", "RISKS"),
        (ChatResponse(200, bad, {}, 0.3), "CONTRACT", None),
        (ChatResponse(429, {"error_code": "RL"}, {"Retry-After": "3"}, 0.1), "OPERATIONAL", None),
        (ChatResponse(503, {}, {}, 0.1), "OPERATIONAL", None),
        (ChatResponse(0, None, {}, 30.0, "ReadTimeout"), "OPERATIONAL", None),
    ]
    for response, kind, label in cases:
        decision, call = clf.interpret(response)
        assert (call.failure_kind, call.label) == (kind, label)
        assert decision.abstain == (kind != "NONE") and (call.fail_reason is None) == (
            kind == "NONE"
        )
    assert clf.interpret(cases[4][0])[1].fail_reason == "TRANSPORT_ERROR"
    assert clf.interpret(cases[2][0])[1].retry_after_header == "3"


def test_artifact_holds_no_question_text_reasoning_or_test_ids():
    artifact, _ = run(reasoning=True)
    text = json.dumps(artifact)
    assert "SECRET" not in text and "reasoning_content" not in text
    assert not any(ctx.question in text for ctx in CONTEXTS.values())
    assert not any(f'"{tid}"' in text for tid in TEST_IDS)
    assert {c["case_id"] for c in artifact["calls"]} <= set(POPS["c2"])
    assert verdict(artifact)["outcome"] == "ACCEPTED_FOR_NEXT_STAGE"


# -- outcomes and precedence ----------------------------------------------------------------------


def test_all_hard_gates_pass_is_accepted_for_next_stage_only():
    report = verdict(run()[0])
    assert report["outcome"] == "ACCEPTED_FOR_NEXT_STAGE"
    assert report["hybrid"]["hybrid_route_correct"] == 26 and report["hybrid"]["delta"] == 2
    assert [c["effect"] for c in report["hybrid"]["changes"]] == ["correction", "correction"]
    assert all(
        r["intent_correct"] and r["route_correct"] and r["repeat_stable"] for r in report["c1"]
    )
    assert report["c2"]["human_review_required"] is True
    assert (
        "NOT" in ACCEPTANCE_RULE["ACCEPTED_FOR_NEXT_STAGE"]
        or "ONLY" in ACCEPTANCE_RULE["ACCEPTED_FOR_NEXT_STAGE"]
    )


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ({"r015": "DOCUMENT_CONTENT"}, "intent"),  # same DOCUMENT route, wrong intent
        ({"r048": "ABSTAIN"}, "ABSTAIN"),
        ({"r015": "RISKS"}, "wrong executable route"),
    ],
)
def test_c1_quality_failures_are_rejected(labels, message):
    report = verdict(run(labels=labels)[0])
    assert report["outcome"] == "REJECTED"
    assert any(message in f for f in report["findings"]["REJECTED"])


def test_c1_repeat_instability_is_rejected():
    flips = ["DOCUMENT_CONTENT"] + ["DOCUMENT_CONTENT"] * 3 + ["EXPLANATION"] * 2
    report = verdict(run(labels={"r048": flips})[0])
    assert report["outcome"] == "REJECTED"
    assert any("r048: C1 repeats" in f for f in report["findings"]["REJECTED"])


def test_c1_operational_failure_is_inconclusive_and_never_also_rejected():
    report = verdict(run(status_by_seq={seq_of("r015"): 429})[0])
    assert report["outcome"] == "INCONCLUSIVE_OPERATIONAL"
    assert report["findings"]["REJECTED"] == []  # not double-classified
    assert next(r for r in report["c1"] if r["case_id"] == "r015")["evaluable"] is False
    assert report["retry_after"][0] == {"sequence": 1, "header": "9", "body": 11}


def test_c1_repeat_operational_failure_is_inconclusive():
    report = verdict(run(status_by_seq={seq_of("r048", "repeat", 4): 0})[0])
    assert report["outcome"] == "INCONCLUSIVE_OPERATIONAL"
    assert report["findings"]["REJECTED"] == []


def test_operational_rate_above_five_percent_is_inconclusive():
    failing = {seq_of(c): 503 for c in ("r002", "r006", "r012")}  # 3/44 = 0.068
    report = verdict(run(status_by_seq=failing)[0])
    assert report["outcome"] == "INCONCLUSIVE_OPERATIONAL"
    assert report["counts"]["operational_failure_rate"] == round(3 / 44, 4)
    ok = verdict(run(status_by_seq={seq_of("r002"): 503, seq_of("r006"): 503})[0])
    assert ok["outcome"] == "ACCEPTED_FOR_NEXT_STAGE"  # 2/44 = 0.045, non-C1


def test_slow_inference_is_inconclusive_operational():
    assert verdict(run(latency=6.0)[0])["outcome"] == "INCONCLUSIVE_OPERATIONAL"


def test_contract_violation_on_a_valid_response_is_rejected():
    report = verdict(run(contract_by_seq={seq_of("r029")})[0])
    assert report["outcome"] == "REJECTED"
    assert report["findings"]["INCONCLUSIVE_OPERATIONAL"] == []


def test_safety_violations_are_rejected_safety():
    artifact = run()[0]
    leaked = copy.deepcopy(artifact)
    leaked["calls"][0]["tool_keys_in_request"] = ["tools"]
    assert verdict(leaked)["outcome"] == "REJECTED_SAFETY"
    outside = [(cid, artifact["contexts"][cid]) for cid in [*POPS["c1"], "r002"]]
    assert verdict(artifact, invoked=outside)["outcome"] == "REJECTED_SAFETY"
    mutated = synthetic_hybrid(artifact)
    mutated["r048"]["project_id"] = "P000000"
    assert verdict(artifact, hybrid=mutated)["outcome"] == "REJECTED_SAFETY"
    persisted = copy.deepcopy(artifact)
    persisted["calls"][3]["reasoning_content"] = "x"
    assert verdict(persisted)["outcome"] == "REJECTED_SAFETY"


def test_integrity_failures_are_invalid_and_take_precedence():
    artifact = run()[0]
    assert verdict(artifact, c1=["r015"])["outcome"] == "INVALID"
    drift = copy.deepcopy(PROTOCOL.baseline)
    drift["r002"]["route"] = "CLARIFY"
    assert verdict(artifact, recomputed_a=drift)["outcome"] == "INVALID"
    tampered = copy.deepcopy(artifact)
    tampered["lock_sha256"] = "0" * 64
    assert verdict(tampered)["outcome"] == "INVALID"
    test_row = copy.deepcopy(artifact)
    test_row["calls"][5]["case_id"] = sorted(TEST_IDS)[0]
    assert verdict(test_row)["outcome"] == "INVALID"
    both = copy.deepcopy(artifact)  # integrity AND safety -> INVALID wins
    both["lock_sha256"], both["calls"][0]["tool_keys_in_request"] = "0" * 64, ["tools"]
    report = verdict(both)
    assert report["outcome"] == "INVALID" and report["findings"]["REJECTED_SAFETY"]


def test_safety_precedes_operational_which_precedes_rejected():
    artifact = run(status_by_seq={seq_of("r015"): 429}, contract_by_seq={seq_of("r029")})[0]
    report = verdict(artifact)
    assert report["outcome"] == "INCONCLUSIVE_OPERATIONAL" and report["findings"]["REJECTED"]
    leaked = copy.deepcopy(artifact)
    leaked["calls"][2]["tool_keys_in_request"] = ["tools"]
    assert verdict(leaked)["outcome"] == "REJECTED_SAFETY"


def test_non_c1_change_is_rejected():
    artifact = run()[0]
    hybrid = synthetic_hybrid(artifact)
    hybrid["r044"]["route"] = "STRUCTURED"
    report = verdict(artifact, hybrid=hybrid)
    assert report["outcome"] == "REJECTED"
    assert any("r044" in f for f in report["findings"]["REJECTED"])


def test_c2_is_diagnostic_only_and_never_gates():
    wrong = {c: "ATTENTION" for c in POPS["c2"] if c not in POPS["c1"]}
    report = verdict(run(labels=wrong)[0])
    assert report["outcome"] == "ACCEPTED_FOR_NEXT_STAGE"  # C2 never decides
    c2 = report["c2"]
    assert c2["role"].startswith("SHADOW DIAGNOSTIC ONLY") and c2["human_review_required"]
    assert c2["deterministic_correct_broken"] > 0 and c2["patterns_for_review"]
    assert {
        "intent_accuracy",
        "route_accuracy",
        "abstain_rate",
        "non_abstain_precision",
        "coverage",
        "matrix_route",
        "deterministic_errors_corrected",
        "net_route_change_if_replaced",
    } <= set(c2)


def test_repeat_diagnostics_cover_all_repeated_cases():
    report = verdict(
        run(labels={"r002": ["RESULTS_PROGRESS", "RISKS", "RESULTS_PROGRESS", "RESULTS_PROGRESS"]})[
            0
        ]
    )
    rows = {r["case_id"]: r for r in report["repeatability"]}
    assert set(rows) == {"r015", "r048", "r002", "r014", "r052", "r006"}
    assert rows["r002"]["modal_share"] == round(2 / 3, 4) and rows["r002"]["label_changes"] == 1
    assert report["outcome"] == "ACCEPTED_FOR_NEXT_STAGE"  # C2 repeats are diagnostic


# -- the real service invokes the recorded classifier only at the boundary ------------------------


def test_replay_through_the_real_service_invokes_only_c1():
    artifact = run()[0]
    recorder = RecordedClassifier(artifact, ROUTES, "t")
    h = harness()
    h.service.semantic = recorder
    for cid in POPS["c2"]:
        case = DEV[cid]
        result = h.service.handle(case.question, case.access(ALL_PROJECTS), cid)
        if cid in POPS["c1"]:
            assert result.decision.route == Route.DOCUMENT and result.semantic is not None
    assert sorted(cid for cid, _ in recorder.invoked) == POPS["c1"]
    assert all(artifact["contexts"][cid] == key for cid, key in recorder.invoked)


# -- isolation ------------------------------------------------------------------------------------


def test_lock_and_notebook_never_touch_test_or_the_probe():
    lock_text = json.dumps(LOCK)
    assert not any(f'"{tid}"' in lock_text for tid in TEST_IDS)
    nb = (REPO_ROOT / "notebooks" / "08e_candidate_c_dev.py").read_text("utf-8")
    assert 'assert_split_allowed("dev")' in nb
    assert "semantic_ambiguity_probe" not in nb and "load_probe_set" not in nb
    assert "ThreadPool" not in nb and "burst_concurrency" not in nb  # sequential only
    assert nb.index("protocol.lock != lock") < nb.index("run_dev_predictions(bc")
    assert nb.index("artifact_path.exists()") < nb.index("run_dev_predictions(bc")
    assert 'split == "test"' not in nb
    for path in (
        "src/worldbank_copilot/routing/bounded_classifier_dev.py",
        "scripts/candidate_c_dev_9d.py",
    ):
        text = (REPO_ROOT / path).read_text("utf-8")
        assert "semantic_ambiguity_probe" not in text and "load_probe_set" not in text


def test_official_dev_report_is_the_frozen_rejected_verdict():
    report = json.loads((REPO_ROOT / "evaluation" / "candidate_c_dev_9d.json").read_text("utf-8"))
    assert report["outcome"] == "REJECTED" and report["test_evaluated"] is False
    assert report["lock_sha256"] == lock_sha256(LOCK) and report["contract_sha256"] == CONTRACT
    assert report["acceptance_rule_sha256"] == acceptance_rule_sha256()
    for category in ("INVALID", "REJECTED_SAFETY", "INCONCLUSIVE_OPERATIONAL"):
        assert report["findings"][category] == []  # a quality rejection, nothing else
    assert report["hybrid"]["hybrid_route_correct"] == 25
    assert report["hybrid"]["candidate_a_route_correct"] == 24
    c1 = {r["case_id"]: r for r in report["c1"]}
    assert c1["r015"]["predicted"] == "CHANGE_INVESTIGATION" and not c1["r015"]["repeat_stable"]
    assert c1["r015"]["final_hybrid_route"] == "INVESTIGATION"
    assert c1["r048"]["intent_correct"] and c1["r048"]["route_correct"]
    assert c1["r048"]["repeat_stable"]
    assert report["counts"]["operational_failures"] == 0 and report["counts"]["calls_made"] == 44
    text = json.dumps(report) + (REPO_ROOT / "evaluation" / "candidate_c_dev_9d.md").read_text(
        "utf-8"
    )
    assert not any(f'"{tid}"' in text or f" {tid} " in text for tid in TEST_IDS)
    assert not any(ctx.question in text for ctx in CONTEXTS.values())


def test_candidate_c_closure_is_recorded_and_the_probe_was_not_run():
    assert BC.status == "DEV_REJECTED"
    assert BC.dev_evaluation["status"] == "REJECTED"
    assert BC.dev_evaluation["ambiguity_probe_status"] == "NOT_RUN_CANDIDATE_C_DEV_REJECTED"
    dev1 = BC.experiment_history[-1]
    assert (dev1["run"], dev1["status"]) == ("dev1", "REJECTED")
    assert "QUALITY/REPEATABILITY rejection" in dev1["interpretation"]
    assert "ae2bf51ea2c88dced4a72021d4101826606c4d0565de4b20d881b7899cea816e" in dev1["evidence"]


def test_no_prediction_artifact_is_inside_the_repository():
    assert not list(REPO_ROOT.rglob("candidate_c_dev_predictions*.json"))
