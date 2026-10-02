"""Phase 9E adaptive-rerank diagnostic: policies, feature isolation, counterfactual classes,
Random / Oracle / Pareto, fail-closed drift and nondeterminism, the live-selection rule, the
protocol lock and notebook protections. Synthetic corpus and deterministic stand-ins only:
no Databricks, no AI Search, no real CrossEncoder, no real 9E results."""

import ast
import copy
import hashlib
import itertools
import json
import math
from datetime import date

import pytest
import yaml
from tests.conftest import REPO_CONFIG_DIR, REPO_ROOT
from tests.support.retrieval_golden import OverlapCrossEncoder, ScopedDense, TextEmbeddings

from worldbank_copilot.retrieval import adaptive_eval as ae
from worldbank_copilot.retrieval.adaptive_rerank import (
    FEATURE_FIELDS,
    AdaptivePolicy,
    LivePolicy,
    RecordingDense,
    TriggerFeatures,
    compute_features,
    content_terms,
    is_anchored,
    load_adaptive_config,
    preregistered_policies,
)
from worldbank_copilot.retrieval.config import load_retrieval_settings
from worldbank_copilot.retrieval.evaluation import EvidenceRef, Question, ranking_metrics
from worldbank_copilot.retrieval.query import ProcessedQuery
from worldbank_copilot.retrieval.rerank_policy import NeverRerank
from worldbank_copilot.retrieval.retriever import ChunkStore, Retriever

CFG = load_adaptive_config(REPO_CONFIG_DIR)
RS = load_retrieval_settings(REPO_CONFIG_DIR)
SPEC = CFG.features
LOCK_PATH = REPO_ROOT / "evaluation" / "adaptive_rerank_9e_lock.json"

# -- synthetic corpus -------------------------------------------------------------------------

PROJECTS = ["P100001", "P100002"]
TEXTS = [
    "Procurement delays affected the water treatment plant contract.",
    "The closing date was extended by fifteen months to allow completion.",
    "Metered household connections increased in the latest report.",
    "Disbursement reached 62 percent of the loan after the restructuring.",
    "The operator fee allocation was increased because of exchange rate losses.",
    "Grievances addressed within 30 days rose to 85 percent.",
    "Bulk works progressed in each city; distribution works lag behind.",
    "The PDO rating was downgraded to Moderately Unsatisfactory.",
    "Water treatment capacity doubled after commissioning in 2021.",
    "The restructuring paper cancelled part of the additional financing.",
]


def make_rows():
    rows = []
    for p, project in enumerate(PROJECTS):
        for i, text in enumerate(TEXTS):
            dtype = "ISR" if i % 3 else "RESTRUCTURING_PAPER"
            rows.append(
                {
                    "chunk_id": f"{project}-fixed-{i}",
                    "chunk_strategy": "fixed",
                    "chunk_role": "RETRIEVAL",
                    "chunk_type": "TEXT",
                    "parent_chunk_id": None,
                    "project_id": project,
                    "document_id": f"{project}-doc{i % 4}",
                    "document_type": dtype,
                    "document_label": f"Doc {i % 4}",
                    "document_date": date(2024, 1 + p, 1 + i),
                    "isr_sequence": (i % 4 + 1) if dtype == "ISR" else None,
                    "source_file": f"{project}/f{i % 4}.pdf",
                    "source_hash": f"{p}" * 64,
                    "page_number": i + 1,
                    "page_numbers": [i + 1],
                    "section_title": f"Section {i}",
                    "element_ids": [f"b{i}"],
                    "chunk_text": f"{text} ({project.lower()} note)",
                    "search_text": f"{project} | {dtype} | {text}",
                }
            )
    return rows


ROWS = make_rows()


def q(qid, project, text, evidence=(), mode="any"):
    return Question(
        id=qid,
        project_id=project,
        category="synthetic",
        kind="synthetic",
        question=text,
        answerable=bool(evidence),
        evidence_mode=mode,
        evidence=[EvidenceRef(document_id=d, pages=[pg], contains=c) for d, pg, c in evidence],
    )


QUESTIONS = [
    q(
        "s01",
        "P100001",
        "Why was the closing date extended?",
        [("P100001-doc1", 2, "closing date was extended")],
    ),
    q(
        "s02",
        "P100001",
        "What happened with procurement of the treatment plant?",
        [("P100001-doc0", 1, "procurement delays affected")],
    ),
    q(
        "s03",
        "P100001",
        "How many metered household connections were reported?",
        [("P100001-doc2", 3, "metered household connections")],
    ),
    q(
        "s04",
        "P100001",
        "What does the restructuring paper say about additional financing?",
        [("P100001-doc1", 10, "cancelled part of the additional financing")],
    ),
    q(
        "s05",
        "P100002",
        "How were grievances addressed?",
        [("P100002-doc1", 6, "grievances addressed within 30 days")],
    ),
    q(
        "s06",
        "P100002",
        "Was the PDO rating changed?",
        [("P100002-doc3", 8, "pdo rating was downgraded")],
    ),
    q(
        "s07",
        "P100002",
        "What about the operator fee?",
        [("P100002-doc0", 5, "phrase that exists nowhere in the corpus")],
    ),  # RETRIEVAL_MISS
    q("s08", "P100002", "zebra quantum unicorn"),  # no-answer
    q(
        "s09",
        "P100001",
        "water treatment plant contract capacity",
        [("P100001-doc0", 9, "water treatment capacity doubled")],
    ),  # evidence at rank 2
]


class ScriptedCE:
    """Deterministic CrossEncoder stand-in: keeps retrieval order (ties are stable) except
    it boosts the s09 evidence (-> HELPED) and demotes the s02 evidence (-> HURT)."""

    name = "cross_encoder"

    def score(self, query, texts):
        out = []
        for text in texts:
            low, s = text.lower(), 0.0
            if "capacity doubled" in low and "capacity" in query:
                s = 10.0
            if "procurement delays" in low and "procurement" in query:
                s = -10.0
            out.append(s)
        return out


def retriever(dense=None):
    return Retriever(ChunkStore(ROWS), RS, PROJECTS, TextEmbeddings(), dense or ScopedDense(ROWS))


def synthetic_config(**changes):
    expected = CFG.expected | {"questions": 9, "answerable": 8, "corpus_rows": len(ROWS)}
    return CFG.model_copy(update={"expected": expected, **changes})


def ticking_clock(step=0.001):
    counter = itertools.count()
    return lambda: next(counter) * step


def collect(config=None, reranker=None, **kw):
    config = config or synthetic_config()
    return ae.collect(
        retriever(),
        QUESTIONS,
        reranker or ScriptedCE(),
        config,
        lock_sha256="L" * 64,
        clock=ticking_clock(),
        log=lambda _: None,
        **kw,
    )


def calibrated_config():
    """Drift references set to the synthetic observation (the real ones are Phase 8's)."""
    observed = collect()["drift_check"]["observed"]
    drift = CFG.drift | {"references": observed}
    return synthetic_config(drift=drift)


def analyse(artifact, config):
    return ae.analyse(
        config, {}, artifact, QUESTIONS, lock_sha256="L" * 64, recomputed_lock_sha256="L" * 64
    )


def features(**kw):
    base = dict(
        n_candidates=10,
        top1_agree=True,
        overlap_at_10=1.0,
        distinct_docs_top5=1,
        lexical_coverage_top1=1.0,
        anchored=True,
    )
    return TriggerFeatures(**(base | kw))


def pq(text, isr=(), latest=False, hints=()):
    return ProcessedQuery(text, text.lower(), text.lower(), (), isr, latest, hints)


# -- the 12 frozen adaptive points and the policies -------------------------------------------


def test_exactly_the_preregistered_points():
    names = [p.name for p in preregistered_policies(CFG)]
    assert names == [
        "P2",
        "P3(0.2)",
        "P3(0.3)",
        "P3(0.4)",
        "P3(0.5)",
        "P4(3)",
        "P4(4)",
        "P4(5)",
        "P5(0.3)",
        "P5(0.5)",
        "P5(0.7)",
        "P6",
    ]  # 1 + 4 + 3 + 3 + 1 = 12 adaptive points
    assert len(names) == 12
    assert {p.id: list(p.thresholds) for p in CFG.policies} == {
        "P2": [],
        "P3": [0.2, 0.3, 0.4, 0.5],
        "P4": [3, 4, 5],
        "P5": [0.3, 0.5, 0.7],
        "P6": [],
    }


def test_p2_top1_disagreement():
    p2 = AdaptivePolicy("P2")
    assert p2.trigger(features(top1_agree=False)) and not p2.trigger(features(top1_agree=True))


@pytest.mark.parametrize("tau", [0.2, 0.3, 0.4, 0.5])
def test_p3_every_threshold(tau):
    p = AdaptivePolicy("P3", tau)
    assert p.trigger(features(overlap_at_10=round(tau - 0.1, 1)))
    assert not p.trigger(features(overlap_at_10=tau))  # strict "<"
    assert not p.trigger(features(overlap_at_10=round(tau + 0.1, 1)))


@pytest.mark.parametrize("m", [3, 4, 5])
def test_p4_every_threshold(m):
    p = AdaptivePolicy("P4", m)
    assert p.trigger(features(distinct_docs_top5=m)) and p.trigger(features(distinct_docs_top5=5))
    assert not p.trigger(features(distinct_docs_top5=m - 1))


@pytest.mark.parametrize("tau", [0.3, 0.5, 0.7])
def test_p5_every_threshold(tau):
    p = AdaptivePolicy("P5", tau)
    assert p.trigger(features(lexical_coverage_top1=tau - 0.05))
    assert not p.trigger(features(lexical_coverage_top1=tau))


def test_p5_empty_query_terms_count_as_zero_coverage_and_rerank():
    only_stop = pq("what is the of")
    assert content_terms(only_stop.text, SPEC) == set()
    f = compute_features(
        only_stop,
        ["P100001-fixed-0"],
        ["P100001-fixed-0"],
        ["P100001-fixed-0"],
        {r["chunk_id"]: r for r in ROWS},
        SPEC,
    )
    assert f.lexical_coverage_top1 == 0.0
    assert all(AdaptivePolicy("P5", t).trigger(f) for t in (0.3, 0.5, 0.7))


@pytest.mark.parametrize(
    ("query", "anchored"),
    [
        (pq("What did ISR 23 report", isr=(23,)), True),
        (pq("What does the latest report say", latest=True), True),
        (pq("What does the restructuring paper say", hints=("RESTRUCTURING_PAPER",)), True),
        (pq("Progress in 2021"), True),
        (pq("How many of 62 contracts"), True),
        (pq("Why was the closing date extended"), False),
    ],
)
def test_p6_anchoring(query, anchored):
    assert is_anchored(query, SPEC) is anchored
    f = features(anchored=anchored)
    assert AdaptivePolicy("P6").trigger(f) is (not anchored)


def test_no_candidates_never_rerank():
    empty = features(
        n_candidates=0,
        top1_agree=False,
        overlap_at_10=0.0,
        distinct_docs_top5=5,
        lexical_coverage_top1=0.0,
        anchored=False,
    )
    assert not any(p.trigger(empty) for p in preregistered_policies(CFG))


# -- the policy input boundary ----------------------------------------------------------------

FORBIDDEN = {
    "kind",
    "category",
    "evidence",
    "answerable",
    "answer",
    "labels",
    "matched",
    "class",
    "classes",
    "metrics",
    "relevance",
    "HELPED",
    "HURT",
    "question_id",
}


def test_trigger_features_expose_no_label_or_metadata_field():
    assert set(FEATURE_FIELDS) == {
        "n_candidates",
        "top1_agree",
        "overlap_at_10",
        "distinct_docs_top5",
        "lexical_coverage_top1",
        "anchored",
    }
    assert not FORBIDDEN & set(FEATURE_FIELDS)


def test_policies_reject_anything_but_trigger_features():
    leaky = {**features().__dict__, "answerable": True, "class": "HELPED"}
    for p in preregistered_policies(CFG):
        with pytest.raises(TypeError):
            p.trigger(leaky)
    with pytest.raises(TypeError):
        TriggerFeatures(**leaky)


def test_policy_module_never_touches_labels_or_the_evaluation_layer():
    tree = ast.parse(
        (REPO_ROOT / "src/worldbank_copilot/retrieval/adaptive_rerank.py").read_text("utf-8")
    )
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not FORBIDDEN & (attrs | names)
    assert not {m for m in imports if m and ("evaluation" in m or "adaptive_eval" in m)}


def test_point_decisions_use_features_only():
    artifact = collect()
    stripped = copy.deepcopy(artifact)
    for r in stripped["questions"]:
        r["labels"] = None  # decisions cannot depend on labels
    assert ae.point_decisions(CFG, stripped) == ae.point_decisions(CFG, artifact)


# -- counterfactual classes -------------------------------------------------------------------

EPS = CFG.classification["epsilon"]


def m(mrr, ndcg):
    return {"mrr": mrr, "ndcg_at_5": ndcg}


def test_retrieval_miss_takes_precedence():
    assert ae.classify(m(0, 0), m(1, 1), False, EPS) == ("RETRIEVAL_MISS", False)


@pytest.mark.parametrize(
    ("p0", "p1", "expected"),
    [
        (m(0.5, 0.6), m(1.0, 0.6), ("HELPED", False)),
        (m(0.5, 0.6), m(0.5, 0.9), ("HELPED", False)),  # dRR = 0 -> nDCG tie-break
        (m(1.0, 0.6), m(0.5, 0.6), ("HURT", False)),
        (m(0.5, 0.9), m(0.5, 0.6), ("HURT", False)),
        (m(0.5, 0.6), m(0.5, 0.6), ("NEUTRAL", False)),
        (m(0.5, 0.9), m(1.0, 0.6), ("HELPED", True)),  # signs conflict: class by dRR
        (m(1.0, 0.6), m(0.5, 0.9), ("HURT", True)),
    ],
)
def test_lexicographic_classes_and_signs_conflict(p0, p1, expected):
    assert ae.classify(p0, p1, True, EPS) == expected


def test_metrics_from_matches_equal_phase8_ranking_metrics():
    rows = {r["chunk_id"]: r for r in ROWS}
    question = q(
        "x",
        "P100001",
        "q",
        [
            ("P100001-doc1", 2, "closing date was extended"),
            ("P100001-doc2", 3, "metered household"),
        ],
        mode="all",
    )
    order = [f"P100001-fixed-{i}" for i in (5, 2, 0, 1, 7)]
    expected = ranking_metrics([rows[i] for i in order], question, [5, 10])
    from worldbank_copilot.retrieval.evaluation import matched_items

    got = ae.ranking_metrics_from_matches(
        [matched_items(rows[i], question) for i in order], question, [5, 10]
    )
    assert {k: v for k, v in expected.items() if not k.startswith("leakage")} == got


# -- Random, Oracle, diagnostics, retained gain, Pareto ---------------------------------------


def toy():
    ids = ["a", "b", "c", "d"]
    m0 = {i: {"recall_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0, "ndcg_at_5": 0.0} for i in ids}
    m1 = {
        i: {"recall_at_5": 1.0, "recall_at_10": 1.0, "mrr": v, "ndcg_at_5": v}
        for i, v in zip(ids, (1.0, 0.5, 0.0, 0.25), strict=True)
    }
    return ids, m0, m1


def test_random_expectation_is_exact():
    ids, m0, m1 = toy()
    for k in range(5):
        ref = ae.random_reference(
            k, m0, m1, ids, CFG.model_copy(update={"random": CFG.random | {"draws": 200}})
        )
        assert ref["expected"]["mrr"] == round((k / 4) * (1.0 + 0.5 + 0.0 + 0.25) / 4, 4)


def test_random_draws_are_deterministic_and_seeded_per_k():
    ids, m0, m1 = toy()
    small = CFG.model_copy(update={"random": CFG.random | {"draws": 500}})
    a, b = ae.random_reference(2, m0, m1, ids, small), ae.random_reference(2, m0, m1, ids, small)
    assert a["draws"] == b["draws"]
    assert abs(sum(a["draws"]["mrr"]) / 500 - a["expected"]["mrr"]) < 0.02
    assert CFG.random["seed"] == "9e1" and CFG.random["draws"] == 10000


def test_retained_gain():
    assert ae._retained(0.65, 0.6, 0.7) == 0.5
    assert ae._retained(0.6, 0.6, 0.6) is None


def test_pareto_dominance():
    pts = {
        "A": {"mrr": 0.6, "rerank_rate": 0.0},
        "B": {"mrr": 0.7, "rerank_rate": 1.0},
        "C": {"mrr": 0.65, "rerank_rate": 0.4},
        "D": {"mrr": 0.62, "rerank_rate": 0.5},  # dominated by C
        "E": {"mrr": 0.65, "rerank_rate": 0.4},  # ties C: neither dominates
    }
    assert ae.pareto(pts, list(pts), "mrr", "rerank_rate") == ["A", "B", "C", "E"]


def test_end_to_end_frontier_diagnostics_and_oracle():
    config = calibrated_config()
    report = analyse(collect(config), config)
    assert report["status"] == "VALID", report.get("failures")
    pts = report["points"]
    assert list(pts) == ["P0", "P0@50", "P1"] + [p.name for p in preregistered_policies(CFG)]
    cf = report["counterfactual"]
    assert cf["classes"] == {"RETRIEVAL_MISS": 1, "HELPED": 1, "HURT": 1, "NEUTRAL": 5}
    helped = {r["question_id"] for r in cf["per_question"] if r["class"] == "HELPED"}
    hurt = {r["question_id"] for r in cf["per_question"] if r["class"] == "HURT"}
    decisions = ae.point_decisions(config, collect(config))
    for name, p in pts.items():
        if name == "P0":
            continue
        t = decisions[name]["reranked"] - {"s08"}
        assert p["triggered_answerable"] == len(t)
        assert p["missed_help"] == len(helped - t) and p["hurt_exposure"] == len(t & hurt)
        assert p["futile_reranks"] == len(t & {"s07"})
        for key in (
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "ndcg_at_5",
            "rerank_rate",
            "answerable_rerank_rate",
            "help_capture",
            "unnecessary_rerank_rate",
            "no_answer_reranks",
            "retained_mrr_gain",
            "retained_ndcg_gain",
            "delta_vs_random",
            "composed_p50_ms",
            "composed_p95_ms",
            "per_project",
        ):
            assert key in p
    assert pts["P0@50"]["retained_mrr_gain"] in (0.0, None)
    assert pts["P1"]["rerank_rate"] == round(8 / 9, 4)  # s08 has no candidates
    assert helped == {"s09"} and hurt == {"s02"}
    assert pts["P1"]["help_capture"] == 1.0 and pts["P1"]["hurt_exposure"] == 1
    assert pts["P0@50"]["help_capture"] == 0.0 and pts["P0@50"]["missed_help"] == 1
    oracle = report["oracle"]
    assert oracle["label"] == "USES GROUND TRUTH — NOT DEPLOYABLE" and oracle["deployable"] is False
    assert oracle["mrr"] >= max(pts["P0@50"]["mrr"], pts["P1"]["mrr"])
    assert set(report["counterfactual"]["per_project"]) == set(PROJECTS)

    def keys(obj):
        if isinstance(obj, dict):
            return set(obj) | {k for v in obj.values() for k in keys(v)}
        return {k for v in obj for k in keys(v)} if isinstance(obj, list) else set()

    assert not {k for k in keys(report) if "winner" in k.lower() or "selected" in k.lower()}
    assert "oracle" not in report["pareto"]["mrr_vs_rerank_rate"]


# -- fail-closed behaviour --------------------------------------------------------------------


def test_drift_beyond_tolerance_stops_and_hides_the_frontier():
    report = analyse(collect(), synthetic_config())  # real Phase 8 references
    assert report["status"] == "INVALID_DRIFT" and "points" not in report
    assert "INCOMPARABLE" in report["interpretation"]
    md = ae.render_markdown(report)
    assert "Why no frontier" in md and "| P1 |" not in md


def test_pass_consistency_detects_nondeterminism():
    class NoisyCE(OverlapCrossEncoder):
        calls = 0

        def score(self, query, texts):
            NoisyCE.calls += 1
            return [s + NoisyCE.calls * 1e-3 for s in super().score(query, texts)]

    artifact = collect(calibrated_config(), reranker=NoisyCE())
    assert artifact["status"] == "NONDETERMINISTIC" and not artifact["consistency"]["consistent"]
    report = analyse(artifact, calibrated_config())
    assert report["status"] == "INVALID_NONDETERMINISM" and "points" not in report


def test_pass_consistency_tolerates_identical_passes_only():
    a = {
        "lexical_k50": [["x", 1]],
        "dense_k50": [["x", 1]],
        "fused_k50": [["x", 1]],
        "fused_k10": ["x"],
        "reranked_k50": ["x"],
        "ce_scores": {"x": 1.0},
    }
    b = copy.deepcopy(a)
    b["ce_scores"]["x"] = 1.0 + 5e-7
    assert ae.pass_consistency({"q": [a, b]}, 1e-6)["consistent"]
    b["ce_scores"]["x"] = 1.0 + 2e-6
    assert not ae.pass_consistency({"q": [a, b]}, 1e-6)["consistent"]
    c = copy.deepcopy(a)
    c["fused_k10"] = ["y"]
    assert not ae.pass_consistency({"q": [a, c]}, 1e-6)["consistent"]


def test_artifact_and_lock_integrity_fail_closed():
    config = calibrated_config()
    artifact = collect(config)
    assert (
        ae.analyse(
            config, {}, artifact, QUESTIONS, lock_sha256="L" * 64, recomputed_lock_sha256="X" * 64
        )["status"]
        == "INVALID_ARTIFACT"
    )
    tampered = copy.deepcopy(artifact)
    tampered["questions"][0]["labels"]["class"] = (
        "HELPED" if tampered["questions"][0]["labels"]["class"] != "HELPED" else "HURT"
    )
    assert analyse(tampered, config)["status"] == "INVALID_ARTIFACT"


def test_collection_artifact_holds_ids_only_and_matches_the_retriever_hybrid():
    artifact = collect(calibrated_config())
    assert artifact["status"] == "OK" and artifact["problems"] == []
    text = json.dumps(artifact)
    assert not any(row["chunk_text"] in text for row in ROWS)
    assert not any(question.question in text for question in QUESTIONS)
    first = artifact["questions"][0]
    assert set(first) == {
        "question_id",
        "project_id",
        "answerable",
        "lists",
        "features",
        "labels",
        "timings",
    }
    assert [t["pass"] for t in first["timings"]] == [1, 2, 3]
    no_answer = next(r for r in artifact["questions"] if r["question_id"] == "s08")
    assert no_answer["labels"] is None


INDEX = "cat.silver." + CFG.expected["index_name"]
ENDPOINT = CFG.expected["endpoint"]


def identity(index_rows=12797, endpoint=ENDPOINT, config=None, corpus=None, questions=None):
    config = config or synthetic_config()
    return ae.identity_failures(
        config,
        len(ROWS) if corpus is None else corpus,
        index_rows,
        QUESTIONS if questions is None else questions,
        INDEX,
        endpoint=endpoint,
    )


def test_identity_passes_only_with_the_frozen_row_count_and_endpoint():
    assert CFG.expected["index_rows"] == 12797 and ENDPOINT == "worldbank-gep-ai-search"
    assert identity() == []
    assert identity(corpus=1, questions=QUESTIONS[:3])  # corpus / question checks still apply


def test_missing_index_row_count_fails_closed():
    failures = identity(index_rows=None)
    assert len(failures) == 1 and failures[0].startswith("INDEX_ROW_COUNT_UNAVAILABLE")


@pytest.mark.parametrize("rows", [0, 12796, 12798, 14327])
def test_wrong_index_row_count_fails_closed(rows):
    failures = identity(index_rows=rows)
    assert len(failures) == 1 and failures[0].startswith("INDEX_ROW_COUNT_MISMATCH")


def test_missing_runtime_endpoint_fails_closed():
    failures = identity(endpoint=None)
    assert len(failures) == 1 and failures[0].startswith("ENDPOINT_UNAVAILABLE")


@pytest.mark.parametrize("endpoint", ["worldbank-copilot-vs", "WORLDBANK-GEP-AI-SEARCH", ""])
def test_wrong_runtime_endpoint_fails_closed(endpoint):
    failures = identity(endpoint=endpoint)
    assert len(failures) == 1 and failures[0].startswith("ENDPOINT_MISMATCH")


def test_endpoint_is_a_required_runtime_input():
    with pytest.raises(TypeError):
        ae.identity_failures(synthetic_config(), len(ROWS), 12797, QUESTIONS, INDEX)


def test_collect_notebook_takes_identity_from_the_index_and_checks_it_before_any_query():
    text = (REPO_ROOT / "notebooks" / "07c_adaptive_rerank_collect.py").read_text("utf-8")
    check = text.index("failures = identity_failures(")
    assert text.index('description.get("endpoint_name")') < check
    assert text.index('.get("indexed_row_count")') < check
    assert "endpoint=runtime_endpoint" in text
    stop = text.index('raise RuntimeError(f"STOP: frozen-data identity failed')
    assert check < stop < text.index("artifact = collect(")
    before = text[:stop]
    for query in (".candidates(", ".retrieve(", "first_stage(", "collect(", ".search("):
        assert query not in before.replace("identity_failures(", ""), query
    assert '"endpoint": runtime_endpoint' in text  # artifact records the runtime endpoint


# -- live-selection rule and the live path ----------------------------------------------------


def test_live_selection_uses_rerank_rate_only():
    rates = {
        "P0@50": {"policy_id": "P0@50", "threshold": None, "rerank_rate": 0.5},
        "P1": {"policy_id": "P1", "threshold": None, "rerank_rate": 0.5},
        "P3(0.4)": {"policy_id": "P3", "threshold": 0.4, "rerank_rate": 0.45},
        "P2": {"policy_id": "P2", "threshold": None, "rerank_rate": 0.55},
        "P5(0.3)": {"policy_id": "P5", "threshold": 0.3, "rerank_rate": 0.9},
    }
    chosen = ae.select_live_point(rates)
    assert chosen["point"] == "P2"  # 0.05 tie with P3(0.4): lower policy id wins
    assert chosen["label"].endswith("NOT a production winner")
    rates["P3(0.3)"] = {"policy_id": "P3", "threshold": 0.3, "rerank_rate": 0.45}
    del rates["P2"]
    assert ae.select_live_point(rates)["point"] == "P3(0.3)"  # then the lower threshold


def live_run():
    config = calibrated_config()
    collection = collect(config)
    recorder = RecordingDense(ScopedDense(ROWS))
    live_retriever = retriever(dense=recorder)
    policy = AdaptivePolicy("P3", 0.5)
    selection = {"point": policy.name, "policy_id": "P3", "threshold": 0.5}
    live = ae.run_live(
        live_retriever,
        recorder,
        QUESTIONS,
        [
            ("P0@50", NeverRerank()),
            (policy.name, LivePolicy(policy, live_retriever, recorder, SPEC)),
        ],
        ScriptedCE(),
        config,
        lock_sha256="L" * 64,
        selection=selection,
        clock=ticking_clock(),
        log=lambda _: None,
    )
    return config, collection, live


def test_live_path_matches_offline_decisions_without_a_second_dense_query():
    config, collection, live = live_run()
    assert all(r["dense_calls"] == 1 for r in live["rows"])
    report = ae.compare_live(config, collection, live, QUESTIONS)
    assert all(not p["mismatches"] for p in report["points"].values()), report
    assert set(report["points"]) == {"P0@50", "P3(0.5)"}
    assert {"composed_p50_ms", "live_p50_ms", "difference_p50_ms"} <= set(report["points"]["P0@50"])
    p3 = report["points"]["P3(0.5)"]
    n = len(QUESTIONS)
    assert p3["agreement"] == {
        "rows_checked": 3 * n,
        "decision_mismatches": 0,
        "top5_mismatches": 0,
    }
    reranked = sum(q["rerank"] for q in p3["per_question"])
    assert p3["observed_rerank_rate"] == round(reranked / n, 4)
    assert p3["live_by_decision"]["reranked"]["n"] == reranked
    assert p3["live_by_decision"]["not_reranked"]["n"] == n - reranked
    assert report["points"]["P0@50"]["observed_rerank_rate"] == 0
    assert [s["kind"] for s in p3["passes"].values()] == ["warm-up", "timed", "timed"]
    assert {"relative_difference_p95", "per_question"} <= set(p3)
    for q in p3["per_question"]:
        assert math.isclose(q["difference_ms"], q["live_ms"] - q["composed_ms"], abs_tol=0.11)


def test_live_comparison_reports_every_mismatch_including_warm_up_and_empty_top5():
    config, collection, live = live_run()
    live = copy.deepcopy(live)
    rows = [r for r in live["rows"] if r["point"] == "P3(0.5)"]
    warm = next(r for r in rows if r["pass"] == 1)
    warm["rerank"] = not warm["rerank"]
    timed = next(r for r in rows if r["pass"] == 2)
    timed["top5"] = []
    report = ae.compare_live(config, collection, live, QUESTIONS)
    agreement = report["points"]["P3(0.5)"]["agreement"]
    assert agreement["decision_mismatches"] == 1 and agreement["top5_mismatches"] == 1
    assert any("pass 1: live decision" in m for m in report["points"]["P3(0.5)"]["mismatches"])
    assert any("pass 2: live top-5" in m for m in report["points"]["P3(0.5)"]["mismatches"])
    assert "MISMATCH" in ae.render_live_markdown(report)


# -- protocol lock, production boundary, notebooks --------------------------------------------


def test_lock_equals_the_committed_lock_and_ignores_status():
    production = ae.load_retrieval_production(REPO_CONFIG_DIR)
    lock = ae.build_lock(CFG, REPO_ROOT, production)
    assert lock == json.loads(LOCK_PATH.read_text("utf-8"))
    assert lock["production_null"] is True and lock["policy_definitions_sha256"]
    assert ae.build_lock(CFG.model_copy(update={"status": "RUN"}), REPO_ROOT, production) == lock
    assert lock["scope"] == {
        "production_change": False,
        "holdout_created": False,
        "routing_test_untouched": True,
    }
    assert len(lock["preregistered_points"]) == 12
    assert lock["drift"]["tolerance"] == 0.001


def test_production_retrieval_config_is_unchanged():
    data = yaml.safe_load((REPO_CONFIG_DIR / "retrieval" / "retrieval.yaml").read_text("utf-8"))
    assert data["production"] == {
        "chunk_strategy": None,
        "retrieval": None,
        "reranker": None,
        "candidate_k": None,
        "final_k": 5,
    }
    assert data["min_rerank_score"] is None


@pytest.mark.parametrize("name", ["07c_adaptive_rerank_collect", "07d_adaptive_rerank_live"])
def test_notebooks_are_read_only_and_choose_no_winner(name):
    text = (REPO_ROOT / "notebooks" / f"{name}.py").read_text("utf-8")
    for forbidden in (
        "ensure_vector_index",
        "build_corpus",
        "build_index_source",
        "update_embedding_cache",
        "saveAsTable",
        "insertInto",
        ".write.",
        "create_endpoint",
        "retrieval.yaml",
        "winner =",
        "select_winner",
    ):
        assert forbidden not in text, forbidden
    assert "never overwritten" in text
    first_read = min(text.index(s) for s in ("open_retriever(", "open_vector_index(") if s in text)
    assert text.index("build_lock(") < first_read  # lock verified before any index access


def test_collect_notebook_checks_lock_and_artifact_before_any_query():
    text = (REPO_ROOT / "notebooks" / "07c_adaptive_rerank_collect.py").read_text("utf-8")
    assert text.index("recomputed != lock") < text.index("rp.open_retriever(")
    assert text.index("artifact_path.exists()") < text.index("rp.open_retriever(")
    assert text.index("identity_failures(") < text.index("collect(\n")
    live = (REPO_ROOT / "notebooks" / "07d_adaptive_rerank_live.py").read_text("utf-8")
    assert live.index("adaptive_rerank_9e_live_selection.json") < live.index("run_live(")
    assert "LivePolicy(" in live and "NeverRerank()" in live  # exactly the two live points
    src = (REPO_ROOT / "src/worldbank_copilot/retrieval/adaptive_eval.py").read_text("utf-8")
    run_live_src = src[src.index("def run_live(") : src.index("def never_policy(")]
    assert (
        run_live_src.index("stage = retriever.first_stage(")
        < run_live_src.index("decision = policy.decide(stage)")
        < run_live_src.index("result = retriever.finish(")
    )  # the real production path, in order


def test_frontier_is_recorded_and_no_live_result_exists_yet():
    for name in ("adaptive_rerank_9e.json", "adaptive_rerank_9e.md"):
        assert (REPO_ROOT / "evaluation" / name).exists()
    selection = json.loads((REPO_ROOT / "evaluation" / SELECTION_FILE).read_text("utf-8"))
    assert selection["point"] == "P3(0.4)" and selection["label"] == ae.LIVE_LABEL
    for name in ("adaptive_rerank_9e_live.json", "adaptive_rerank_9e_live.md"):
        assert not (REPO_ROOT / "evaluation" / name).exists()
    assert math.isclose(CFG.drift["references"]["P1"]["mrr"], 0.6827)


# -- 07d pre-run integrity (fail closed before any live query) --------------------------------

SELECTION_FILE = "adaptive_rerank_9e_live_selection.json"
AUTHORIZED = {
    "lock_sha256": "480d0a1ec02e6e54aeb7c4a86d1122c002e1b0607d76ad74a9571a7d231b591e",
    "policy_definitions_sha256": "dafc0bf5c80acf2491099eb95991def149df51211c29eae7ccecafafd6eb1cff",
    "collection_artifact_sha256": "8b6763e349215899b686fff45f49eb44483f65490167b6649e88014599a63d70",  # noqa: E501
    "point": "P3(0.4)",
}


def committed_preflight(**override):
    lock = json.loads(LOCK_PATH.read_text("utf-8"))
    kw = {
        "lock": lock,
        "recomputed_lock": ae.build_lock(
            CFG, REPO_ROOT, ae.load_retrieval_production(REPO_CONFIG_DIR)
        ),
        "frontier": json.loads(
            (REPO_ROOT / "evaluation" / "adaptive_rerank_9e.json").read_text("utf-8")
        ),
        "selection": json.loads((REPO_ROOT / "evaluation" / SELECTION_FILE).read_text("utf-8")),
        "collection_sha256": AUTHORIZED["collection_artifact_sha256"],
        "authorized": dict(AUTHORIZED),
    }
    kw.update(override)
    return kw


def codes(failures):
    return {f.split(":")[0] for f in failures}


def test_preflight_passes_on_the_committed_artifacts():
    assert ae.live_preflight_failures(**committed_preflight()) == []


def test_preflight_fails_closed_on_missing_or_wrong_collection_artifact():
    assert codes(ae.live_preflight_failures(**committed_preflight(collection_sha256=None))) == {
        "COLLECTION_ARTIFACT_UNAVAILABLE"
    }
    assert codes(ae.live_preflight_failures(**committed_preflight(collection_sha256="0" * 64))) == {
        "COLLECTION_ARTIFACT_MISMATCH"
    }


@pytest.mark.parametrize("point", ["P2", "P5(0.3)", "P5(0.5)", "P5(0.7)"])
def test_preflight_rejects_any_other_authorized_point(point):
    kw = committed_preflight()
    kw["authorized"]["point"] = point
    assert codes(ae.live_preflight_failures(**kw)) == {"POINT_NOT_AUTHORIZED"}


def test_preflight_rejects_a_swapped_selection():
    kw = committed_preflight()
    kw["selection"] |= {"point": "P5(0.7)", "policy_id": "P5", "threshold": 0.7}
    assert {"SELECTION_NOT_FROM_FRONTIER", "POINT_NOT_AUTHORIZED"} <= codes(
        ae.live_preflight_failures(**kw)
    )


def test_preflight_rejects_a_frontier_whose_rule_does_not_reproduce_the_selection():
    kw = committed_preflight()
    kw["frontier"]["points"]["P2"]["rerank_rate"] = 0.5  # rule would now pick P2
    assert codes(ae.live_preflight_failures(**kw)) == {"SELECTION_RULE_MISMATCH"}


def test_preflight_rejects_lock_policy_production_and_label_tampering():
    kw = committed_preflight()
    kw["lock"] = copy.deepcopy(kw["lock"])
    kw["lock"]["policy_definitions"]["features"]["overlap_depth"] = 20
    kw["lock"]["production_null"] = False
    found = codes(ae.live_preflight_failures(**kw))
    assert {
        "LOCK_MISMATCH",
        "LOCK_NOT_AUTHORIZED",
        "POLICY_DEFINITIONS_MISMATCH",
        "PRODUCTION_NOT_NULL",
    } <= found
    kw = committed_preflight()
    kw["selection"]["label"] = "production winner"
    assert "SELECTION_LABEL_MISMATCH" in codes(ae.live_preflight_failures(**kw))
    kw = committed_preflight()
    kw["frontier"]["status"] = "INVALID_DRIFT"
    assert codes(ae.live_preflight_failures(**kw)) == {"FRONTIER_MISMATCH"}


def test_file_sha256_hashes_raw_bytes(tmp_path):
    path = tmp_path / "collection.json"
    raw = b'{"a": 1}\r\n'  # raw bytes as written, no newline normalisation
    path.write_bytes(raw)
    assert ae.file_sha256(path) == hashlib.sha256(raw).hexdigest()
    assert ae.file_sha256(tmp_path / "missing.json") is None


def test_read_json_artifact_returns_empty_for_missing_invalid_or_non_object(tmp_path):
    assert ae.read_json_artifact(tmp_path / "missing.json") == {}
    (tmp_path / "bad.json").write_text("{not json", "utf-8")
    assert ae.read_json_artifact(tmp_path / "bad.json") == {}
    (tmp_path / "list.json").write_text("[1, 2]", "utf-8")
    assert ae.read_json_artifact(tmp_path / "list.json") == {}
    (tmp_path / "ok.json").write_text('{"point": "P3(0.4)"}', "utf-8")
    assert ae.read_json_artifact(tmp_path / "ok.json") == {"point": "P3(0.4)"}


def test_preflight_names_missing_frontier_and_selection_without_raising():
    found = codes(ae.live_preflight_failures(**committed_preflight(frontier={}, selection={})))
    assert {"FRONTIER_UNAVAILABLE", "SELECTION_UNAVAILABLE", "POINT_NOT_AUTHORIZED"} <= found


def test_preflight_names_a_malformed_frontier_without_raising():
    kw = committed_preflight()
    del kw["frontier"]["points"]["P2"]["rerank_rate"]
    assert codes(ae.live_preflight_failures(**kw)) == {"SELECTION_RULE_MISMATCH"}


def test_preflight_rejects_a_selection_whose_definition_differs_from_its_name():
    kw = committed_preflight()
    kw["selection"]["threshold"] = 0.5  # name says P3(0.4)
    assert "POINT_DEFINITION_MISMATCH" in codes(ae.live_preflight_failures(**kw))


def test_preflight_rejects_a_production_config_that_is_no_longer_null():
    production = ae.load_retrieval_production(REPO_CONFIG_DIR) | {"reranker": "cross_encoder"}
    kw = committed_preflight(recomputed_lock=ae.build_lock(CFG, REPO_ROOT, production))
    assert codes(ae.live_preflight_failures(**kw)) == {"LOCK_MISMATCH"}


def test_runtime_index_name_fails_closed():
    assert ae.runtime_index_failures(INDEX, INDEX) == []
    assert ae.runtime_index_failures(None, INDEX)[0].startswith("INDEX_NAME_UNAVAILABLE")
    assert ae.runtime_index_failures("", INDEX)[0].startswith("INDEX_NAME_UNAVAILABLE")
    assert ae.runtime_index_failures("cat.silver.other", INDEX)[0].startswith("INDEX_NAME_MISMATCH")


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"corpus": 14326}, "corpus rows"),
        ({"questions": QUESTIONS[:-1]}, "question set differs"),
    ],
)
def test_corpus_and_question_counts_fail_closed(kw, message):
    failures = identity(**kw)
    assert len(failures) == 1 and message in failures[0]


def test_live_notebook_checks_everything_before_any_query():
    text = (REPO_ROOT / "notebooks" / "07d_adaptive_rerank_live.py").read_text("utf-8")
    authorized = text[text.index("AUTHORIZED = {") : text.index("}", text.index("AUTHORIZED = {"))]
    for key, value in AUTHORIZED.items():
        assert f'"{key}": "{value}"' in authorized
    preflight = text.index("failures = live_preflight_failures(")
    stop1 = text.index('raise RuntimeError(f"STOP: live pre-run integrity failed')
    identity = text.index("failures = identity_failures(")
    stop2 = text.index('raise RuntimeError(f"STOP: frozen-data identity failed')
    assert text.index("check_environment(") < preflight < stop1 < text.index("open_vector_index(")
    assert text.index("open_vector_index(") < identity < stop2 < text.index("live = run_live(")
    assert "endpoint=runtime_endpoint" in text and "file_sha256(collection_path)" in text
    assert identity < text.index("runtime_index_failures(runtime_index_name") < stop2
    assert 'read_json_artifact(repo / "evaluation" / "adaptive_rerank_9e.json")' in text
    assert 'live["policy_definitions_sha256"] = lock["policy_definitions_sha256"]' in text
    assert '"cuda_available": torch.cuda.is_available()' in text
    before = "\n".join(ln for ln in text[:stop2].splitlines() if not ln.startswith("# MAGIC"))
    for query in (".search(", "first_stage(", "= run_live(", ".retrieve(", ".candidates("):
        assert query not in before, query
