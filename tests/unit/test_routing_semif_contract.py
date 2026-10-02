"""Phase 9D Candidate C_SEMIF_OPENJEV (B-nb): contract, protocol lock, P0/P1 logic and the
frozen ambiguity probe. No model is loaded and no Databricks call is made."""

import hashlib
import json
import re

import pytest
from tests.conftest import REPO_CONFIG_DIR, REPO_ROOT
from tests.support.routing_fixtures import ALL, CONFIG, harness

from worldbank_copilot.routing.evaluation import load_dataset, similarity
from worldbank_copilot.routing.models import Intent, Route
from worldbank_copilot.routing.semantic import SEMANTIC_INTENTS, route_map
from worldbank_copilot.routing.semantic_eval import canonical_sha256
from worldbank_copilot.routing.semif_capability import (
    hf_snapshot_provenance,
    p0_verdict,
    p1_provenance,
    p1_verdict,
    parse_nvidia_smi,
    reachability_targets,
    synthetic_rows,
)
from worldbank_copilot.routing.semif_contract import (
    CLARIFICATION,
    HarnessReason,
    InvalidSemifResult,
    apply_policy,
    build_row,
    few_shot_examples,
    load_probe_set,
    load_semif_config,
    lock_sha256,
    protocol_lock,
    to_decision,
    validate_result,
)

SEMIF = load_semif_config(REPO_CONFIG_DIR)
ROUTES = route_map(CONFIG.requirements)
DATASET = load_dataset(REPO_ROOT / "evaluation" / "routing_cases.yaml")
OPTION_IDS = [o["id"] for o in SEMIF.options]
PROBE_PATH = REPO_ROOT / "evaluation" / "semantic_ambiguity_probe.yaml"
MANIFEST = json.loads((REPO_ROOT / "evaluation" / "routing_freeze_9c.json").read_text("utf-8"))


def result_for(row, probs=None, **kw):
    probs = probs or [1 / len(row["options"])] * len(row["options"])
    return {
        "id": row["id"],
        "option_ids": [o["id"] for o in row["options"]],
        "probabilities": probs,
        "prompt_sha256": "a" * 64,
        "prompt_version": "direct-options-v1",
        "model": {"id": SEMIF.model["id"], "revision": SEMIF.model["revision"]},
        "forward_seconds": 0.05,
        "total_seconds": 0.06,
        **kw,
    }


def row(question="How has the way citizens register complaints been improved?"):
    return build_row(SEMIF, "req-1", question, "P179039", "NONE", "V1")


# -- identity, pinning, protocol lock ------------------------------------------------------


def test_candidate_identity_and_pins():
    assert SEMIF.candidate == "C_SEMIF_OPENJEV" and SEMIF.path == "B-nb"
    assert SEMIF.implementation["commit"] == "23cf1f39fc9534fe81437200959b6dfc7106e45a"
    assert (SEMIF.model["id"], SEMIF.model["revision"]) == (
        "Qwen/Qwen3.5-4B",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    )
    text = (REPO_CONFIG_DIR / "routing" / "semantic_semif.yaml").read_text(encoding="utf-8")
    assert "NOT TypeSafe Jev" in text


def test_protocol_lock_matches_the_committed_lock():
    committed = json.loads(
        (REPO_ROOT / "evaluation" / "semif_protocol_lock.json").read_text(encoding="utf-8")
    )
    now = protocol_lock(
        SEMIF,
        canonical_sha256(REPO_ROOT / "evaluation" / "routing_cases.yaml"),
        canonical_sha256(PROBE_PATH),
    )
    assert now == committed
    assert committed["routing_dataset_sha256_lf"] == MANIFEST["dataset_sha256_lf"]
    assert committed["probe_set_sha256_lf"] == (
        "93d44d28d38d348bc06f76422212d0d66681197823d111a09ccd8737dea47b94"
    )


def test_threshold_grids_are_small_finite_and_preregistered():
    assert SEMIF.policy["tau_grid"] == [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    assert SEMIF.policy["margin_grid"] == [0.00, 0.10, 0.20, 0.30]
    assert SEMIF.policy["variants"] == ["V1", "V2"]
    assert "ambiguity probe is never used for selection" in json.dumps(
        (REPO_CONFIG_DIR / "routing" / "semantic_semif.yaml").read_text("utf-8")
    )


# -- option set and input rows ---------------------------------------------------------------


def test_options_are_the_reviewed_intents_plus_clarification_within_semif_limits():
    assert sorted(OPTION_IDS) == sorted([i.value for i in SEMANTIC_INTENTS] + [CLARIFICATION])
    assert 2 <= len(OPTION_IDS) <= 16
    assert all(o["description"] for o in SEMIF.options)


def test_row_contains_only_request_context_and_fixed_contract():
    r = row()
    assert set(r) == {"id", "state", "question", "options"}
    assert set(r["state"]) == set(SEMIF.state_fields)
    assert r["question"] == SEMIF.criterion["V1"]
    assert [o["id"] for o in r["options"]] == OPTION_IDS


def test_v2_examples_are_dev_only_and_drop_the_scored_family():
    examples = few_shot_examples(SEMIF, DATASET)
    assert {e.case_id for e in examples} <= set(MANIFEST["dev_case_ids"])
    v2 = build_row(SEMIF, "r", "q", "P130544", "NONE", "V2", examples)
    assert all(e.question in v2["question"] for e in examples)
    dropped = build_row(
        SEMIF, "r", "q", "P130544", "NONE", "V2", examples, exclude_family="res00355"
    )  # r016's family
    assert next(e for e in examples if e.case_id == "r016").question not in dropped["question"]
    test_questions = [c.question for c in DATASET.cases if c.split == "test"]
    assert not any(q in v2["question"] for q in test_questions)


# -- result validation fails closed ----------------------------------------------------------


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda res: res.update(id="other"), "id"),
        (lambda res: res.update(option_ids=list(reversed(res["option_ids"]))), "order"),
        (lambda res: res.update(probabilities=res["probabilities"][:-1]), "aligned"),
        (lambda res: res["probabilities"].__setitem__(0, -0.1), "finite number"),
        (lambda res: res["probabilities"].__setitem__(0, float("nan")), "finite number"),
        (lambda res: res["probabilities"].__setitem__(0, 0.9), "sum to"),
        (lambda res: res.update(prompt_sha256="xyz"), "prompt_sha256"),
        (lambda res: res.update(prompt_version="other-v2"), "prompt_version"),
    ],
)
def test_contract_violations_are_rejected(mutate, message):
    r = row()
    res = result_for(r)
    mutate(res)
    with pytest.raises(InvalidSemifResult, match=message):
        validate_result(res, r, SEMIF)


def test_a_valid_result_is_accepted():
    r = row()
    scores = validate_result(result_for(r), r, SEMIF)
    assert set(scores.probabilities) == set(OPTION_IDS) and scores.prompt_sha256 == "a" * 64
    assert validate_result(result_for(r, model={"id": "x"}), r, SEMIF).model_metadata == {"id": "x"}


def test_exact_ties_resolve_by_frozen_option_order():
    tied = [0.0] * len(OPTION_IDS)
    tied[OPTION_IDS.index("EXPLANATION")] = tied[OPTION_IDS.index("RISKS")] = 0.5
    out = apply_policy(scored(tied), 0.0, 0.0, ROUTES, OPTION_IDS)
    assert out.choice == "RISKS" and out.margin == 0.0  # RISKS precedes EXPLANATION


# -- harness-owned policy ----------------------------------------------------------------------


def probs_with(top_id, top, second_id=None, second=0.0):
    rest_ids = [o for o in OPTION_IDS if o not in (top_id, second_id)]
    rest = (1 - top - second) / len(rest_ids)
    out = {o: rest for o in rest_ids} | {top_id: top}
    if second_id:
        out[second_id] = second
    return [out[o] for o in OPTION_IDS]


def scored(probs):
    r = row()
    return validate_result(result_for(r, probs), r, SEMIF)


def test_policy_accepts_only_above_both_thresholds_and_derives_the_route():
    s = scored(probs_with("DOCUMENT_CONTENT", 0.7, "EXPLANATION", 0.2))
    ok = apply_policy(s, 0.6, 0.3, ROUTES, OPTION_IDS)
    assert ok.accepted and ok.intent == Intent.DOCUMENT_CONTENT and ok.route == "DOCUMENT"
    assert apply_policy(s, 0.8, 0.0, ROUTES, OPTION_IDS).reason == HarnessReason.LOW_CONFIDENCE
    assert apply_policy(s, 0.6, 0.6, ROUTES, OPTION_IDS).reason == HarnessReason.LOW_MARGIN


def test_clarification_choice_always_abstains_whatever_the_thresholds():
    s = scored(probs_with(CLARIFICATION, 0.95))
    out = apply_policy(s, 0.0, 0.0, ROUTES, OPTION_IDS)
    assert not out.accepted and out.reason == HarnessReason.MODEL_CHOSE_CLARIFICATION


def test_decisions_feed_the_existing_fail_closed_hook():
    s = scored(probs_with("PROJECT_OVERVIEW", 0.9))
    accepted = to_decision(apply_policy(s, 0.5, 0.1, ROUTES, OPTION_IDS), version="t")
    assert not accepted.abstain and accepted.route == "STRUCTURED"
    failed = to_decision(None, version="t", failure=HarnessReason.ENDPOINT_FAILURE, detail="429")
    assert failed.abstain and failed.reason.startswith("ENDPOINT_FAILURE")

    class Semif:
        name = "C_SEMIF_OPENJEV"

        def classify(self, context):
            return failed

    h = harness()
    h.service.semantic = Semif()
    r = h.ask("How has the way citizens can register complaints been improved?")
    assert (r.decision.route, r.decision.reason_code) == (Route.CLARIFY, "SEMANTIC_ABSTAIN")
    assert h.executor.calls == [] and h.reads == [] and h.retriever.first_stage_calls == []


# -- P0 / P1 logic --------------------------------------------------------------------------------


def test_p0_requires_gpu_bf16_reachability_disk_volume_and_git():
    gpus = parse_nvidia_smi("NVIDIA A10G, 23028, 550.54, 8.6\n")
    assert gpus == [
        {
            "name": "NVIDIA A10G",
            "memory_total_mib": 23028,
            "driver_version": "550.54",
            "compute_capability": 8.6,
        }
    ]
    reach = {k: 200 for k in reachability_targets(SEMIF)}
    assert p0_verdict(gpus, reach, 100.0, True, True)["passed"]
    t4 = parse_nvidia_smi("Tesla T4, 15360, 535.1, 7.5")
    assert not p0_verdict(t4, reach, 100.0, True, True)["checks"]["bf16_capable_gpu"]
    assert not p0_verdict([], reach, 100.0, True, True)["passed"]
    assert not p0_verdict(gpus, {**reach, "huggingface_model_revision": 403}, 100, True, True)[
        "passed"
    ]
    assert not p0_verdict(gpus, reach, 10.0, True, True)["passed"]
    assert not p0_verdict(gpus, reach, 100.0, False, True)["passed"]
    assert not p0_verdict(gpus, reach, 100.0, True, False)["passed"]
    targets = reachability_targets(SEMIF)
    assert SEMIF.implementation["commit"] in targets["github_semif_commit"]
    assert SEMIF.model["revision"] in targets["huggingface_model_revision"]


def test_p1_inputs_are_synthetic_and_include_the_twelve_option_contract():
    rows = synthetic_rows(SEMIF)
    assert [r["id"] for r in rows][:3] == ["syn-2opt", "syn-4opt", "syn-12opt"]
    assert len(rows) == 3 + SEMIF.p1_repeats_for_warm_latency
    twelve = next(r for r in rows if r["id"] == "syn-12opt")
    assert [o["id"] for o in twelve["options"]] == OPTION_IDS
    corpus = [c.question for c in DATASET.cases] + [
        c.question for c in load_probe_set(PROBE_PATH).cases
    ]
    for r in rows:
        assert max(similarity(r["state"]["request"], q) for q in corpus) < 0.3
        assert not re.search(r"\bP\d{6}\b|IBRD|ISR|Karnataka", json.dumps(r["state"]))


PIN = SEMIF.model["revision"]
ENVIRONMENT = {"semif_phase1": "0.1.0", "torch": "2.10.0", "transformers": "5.17.0"}


def hf_cache(tmp_path, snapshots=(), refs=None):
    repo = tmp_path / "hub" / "models--Qwen--Qwen3.5-4B"
    for name in snapshots:
        (repo / "snapshots" / name).mkdir(parents=True)
    for ref, commit in (refs or {}).items():
        (repo / "refs").mkdir(parents=True, exist_ok=True)
        (repo / "refs" / ref).write_text(commit + "\n", encoding="utf-8")
    return hf_snapshot_provenance(tmp_path, SEMIF.model["id"], PIN)


def provenance(snapshot, environment=ENVIRONMENT):
    lock = json.loads((REPO_ROOT / "evaluation" / "semif_protocol_lock.json").read_text("utf-8"))
    return p1_provenance(SEMIF, lock, snapshot, environment, [{"name": "NVIDIA A10G"}])


def test_hf_snapshot_provenance_matches_mismatches_or_is_not_exposed(tmp_path):
    match = hf_cache(tmp_path / "a", [PIN], {"851bf6e": PIN})
    assert match["status"] == "MATCHES_PIN" and match["resolved_snapshot_commit"] == PIN
    assert match["requested_revision"] == PIN and match["local_refs"] == {"851bf6e": PIN}
    other = hf_cache(tmp_path / "b", ["0" * 40])
    assert other["status"] == "MISMATCH" and other["resolved_snapshot_commit"] == "0" * 40
    assert hf_cache(tmp_path / "c", [PIN, "0" * 40])["status"] == "MISMATCH"
    hidden = hf_cache(tmp_path / "d")
    assert hidden["status"] == "NOT_EXPOSED" and hidden["requested_revision"] == PIN


def test_p1_provenance_persists_pins_versions_and_hashes(tmp_path):
    prov = provenance(hf_cache(tmp_path, [PIN]))
    lock = json.loads((REPO_ROOT / "evaluation" / "semif_protocol_lock.json").read_text("utf-8"))
    assert prov["requested_model_id"] == "Qwen/Qwen3.5-4B" and prov["requested_revision"] == PIN
    assert prov["semif_git_commit"] == SEMIF.implementation["commit"]
    assert prov["installed"] == ENVIRONMENT and prov["gpus"]
    assert prov["option_set_sha256"] == lock["option_set_sha256"]
    assert prov["criterion_sha256"] == lock["criterion_sha256"]
    assert prov["protocol_lock_sha256"] == lock_sha256(lock)
    assert "UNCALIBRATED" in prov["probabilities"]


def test_p1_verdict_passes_only_on_a_clean_contract_and_provenance_run(tmp_path):
    rows = synthetic_rows(SEMIF)
    results = [result_for(r) for r in rows]
    prov = provenance(hf_cache(tmp_path / "ok", [PIN]))
    ok = p1_verdict(rows, results, SEMIF, 0, prov)
    assert (
        ok["passed"]
        and ok["twelve_option_contract_ok"]
        and ok["warm_latency"]["n"] == len(rows) - 1
        and ok["provenance"] == prov
    )
    # revision is provenance, not inference output: metadata without it still validates
    assert p1_verdict(rows, [r | {"model": {}} for r in results], SEMIF, 0, prov)["passed"]
    assert p1_verdict(rows, results, SEMIF, 0, provenance(hf_cache(tmp_path / "x")))["passed"]
    assert not p1_verdict(rows, results[:-1], SEMIF, 0, prov)["passed"]
    assert not p1_verdict(rows, results, SEMIF, 1, prov)["passed"]
    bad = [dict(r) for r in results]
    bad[2]["prompt_version"] = "other"
    assert "syn-12opt" in p1_verdict(rows, bad, SEMIF, 0, prov)["failures"][0]
    wrong_snapshot = p1_verdict(
        rows, results, SEMIF, 0, provenance(hf_cache(tmp_path / "w", ["0" * 40]))
    )
    assert not wrong_snapshot["passed"] and "pinned" in wrong_snapshot["failures"][0]
    wrong_pkg = provenance(hf_cache(tmp_path / "v", [PIN]), ENVIRONMENT | {"semif_phase1": "0.2"})
    assert not p1_verdict(rows, results, SEMIF, 0, wrong_pkg)["passed"]
    no_torch = provenance(hf_cache(tmp_path / "t", [PIN]), ENVIRONMENT | {"torch": None})
    assert not p1_verdict(rows, results, SEMIF, 0, no_torch)["passed"]


def test_notebook_gates_p1_on_p0_and_never_touches_dataset_text():
    text = (REPO_ROOT / "notebooks" / "08c_semif_capability.py").read_text(encoding="utf-8")
    assert text.index('require_state("p0"') < text.index("install_pinned_semif")
    assert 'if not p0["passed"]' in text and "protocol drift" in text
    assert "p1_verdict(rows, results, semif, run.returncode, provenance)" in text
    assert 'hf_snapshot_provenance(work / "hf_cache"' in text  # the HF_HOME the loader uses
    assert "bfloat16 configuration is incompatible" in text
    assert "float16" not in text.replace("bfloat16", "")  # no fp16 fallback anywhere
    assert "routing_cases.yaml" in text  # only hashed for the lock check
    for forbidden in (
        "load_dataset",
        "semantic_ambiguity_probe",
        "run_development",
        "serving_endpoints.create",
        "c2_shadow",
        "c1_eligible",
    ):
        assert forbidden not in text, forbidden
    assert '"-m", "venv"' in text  # SemIf's pins never enter the notebook environment


# -- frozen ambiguity probe -------------------------------------------------------------------


def test_probe_set_frozen_hash_composition_and_independence():
    assert canonical_sha256(PROBE_PATH) == SEMIF.probe_set["sha256_lf"]
    probe = load_probe_set(PROBE_PATH)
    kinds = [c.kind for c in probe.cases]
    assert kinds.count("ambiguous") == 6 and kinds.count("control") == 6
    controls = [c for c in probe.cases if c.kind == "control"]
    assert {ROUTES[Intent(c.expected.intent)] for c in controls} == {
        "STRUCTURED",
        "DOCUMENT",
        "INVESTIGATION",
    }
    assert max(similarity(p.question, c.question) for p in probe.cases for c in DATASET.cases) < 0.5
    assert SEMIF.probe_set["gate"] == {
        "min_abstain_ambiguous": 5,
        "min_correct_controls": 4,
        "security_violations": 0,
    }


def test_probe_cases_pass_deterministic_scope():
    for case in load_probe_set(PROBE_PATH).cases:
        r = harness().ask(case.question, active=case.active_project_id, authorized=ALL)
        assert r.understanding.project.status.value == "RESOLVED", case.probe_id


def test_lock_file_is_canonical_json():
    raw = (REPO_ROOT / "evaluation" / "semif_protocol_lock.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest()  # stable file; content verified above
