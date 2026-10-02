"""Phase 9E adaptive-rerank evaluation layer (labels live HERE, never in the policies).

* `collect`       - the Databricks data-collection protocol (07c): per question and pass,
                    the lexical / dense / fused k50 lists, the P0 k10 list, CrossEncoder scores
                    for every fused k50 candidate, TriggerFeatures and timings; pass-1 labels.
                    Checks: hybrid equivalence, pass consistency (ids identical, CE scores
                    within 1e-6), Phase 8 drift. Chunk ids only - no chunk or question text.
* `analyse`       - offline, exact, deterministic simulation of every preregistered point,
                    the counterfactual classes, trigger diagnostics, Random(r), Oracle, Pareto
                    sets and per-project breakdowns. Fail-closed: lock / artifact /
                    nondeterminism / drift problems yield an INVALID report with NO frontier.
* `select_live_point` - the frozen latency-validation rule (rerank rate only).
* `run_live` / `compare_live` - the live end-to-end confirmation (07d) and composed-vs-live.

No automatic winner is chosen anywhere. Production retrieval config is never modified.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

from worldbank_copilot.retrieval.adaptive_rerank import (
    AdaptivePolicy,
    AdaptiveRerankConfig,
    TriggerFeatures,
    compute_features,
    preregistered_policies,
)
from worldbank_copilot.retrieval.evaluation import Question, matched_items, ranking_metrics
from worldbank_copilot.retrieval.lexical import reciprocal_rank_fusion
from worldbank_copilot.retrieval.models import Candidate, ScopeViolation
from worldbank_copilot.retrieval.query import process_query
from worldbank_copilot.retrieval.rerank import NoReranker, Reranker
from worldbank_copilot.retrieval.rerank_policy import NeverRerank

CLASSES = ("RETRIEVAL_MISS", "HELPED", "HURT", "NEUTRAL")
DRIFT_METRICS = ("recall_at_10", "mrr", "ndcg_at_5")
QUALITY = ("recall_at_5", "recall_at_10", "mrr", "ndcg_at_5")
ORACLE_LABEL = "USES GROUND TRUTH — NOT DEPLOYABLE"
LIVE_LABEL = "latency validation point - NOT a production winner"


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _p95(values: Sequence[float]) -> float | None:
    """Nearest-rank p95, as in the Phase 8 RunResult summary."""
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)], 1)


def _p50(values: Sequence[float]) -> float | None:
    return round(statistics.median(values), 1) if values else None


# -- metric helpers (Phase 8 ranking_metrics semantics, from recorded matches) ------------------


def ranking_metrics_from_matches(
    matches: Sequence[set[int]], question: Question, ks: Sequence[int]
) -> dict[str, float]:
    """`evaluation.ranking_metrics` recomputed from per-position matched evidence indexes."""
    n_items = len(question.evidence)
    out: dict[str, float] = {}
    for k in ks:
        covered = set().union(*matches[:k]) if matches[:k] else set()
        if question.evidence_mode == "any":
            out[f"recall_at_{k}"] = 1.0 if covered else 0.0
        else:
            out[f"recall_at_{k}"] = len(covered) / n_items
    first = next((i for i, m in enumerate(matches, 1) if m), None)
    out["mrr"] = 1.0 / first if first else 0.0
    gains = [1.0 if m else 0.0 for m in matches[:5]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(n_items, 5)))
    out["ndcg_at_5"] = dcg / ideal if ideal else 0.0
    out["precision_at_5"] = sum(gains) / 5.0
    return out


def first_relevant_rank(matches: Sequence[set[int]]) -> int | None:
    return next((i for i, m in enumerate(matches, 1) if m), None)


def classify(
    p0: dict[str, float], p1: dict[str, float], in_candidate_set: bool, eps: float
) -> tuple[str, bool]:
    """(class, signs_conflict). RETRIEVAL_MISS first; then lexicographic dRR, dNDCG@5."""
    if not in_candidate_set:
        return "RETRIEVAL_MISS", False
    d_rr, d_ndcg = p1["mrr"] - p0["mrr"], p1["ndcg_at_5"] - p0["ndcg_at_5"]
    conflict = (d_rr > eps and d_ndcg < -eps) or (d_rr < -eps and d_ndcg > eps)
    if d_rr > eps or (abs(d_rr) <= eps and d_ndcg > eps):
        return "HELPED", conflict
    if d_rr < -eps or (abs(d_rr) <= eps and d_ndcg < -eps):
        return "HURT", conflict
    return "NEUTRAL", conflict


# -- collection (07c) ----------------------------------------------------------------------------


def _k50_components(retriever: Any, query: Any, scope: Any, k: int, rrf_k: int):
    lexical = retriever.candidates(query, scope, "lexical", k)
    dense = retriever.candidates(query, scope, "dense", k)
    hits = reciprocal_rank_fusion(
        [[c.chunk_id for c in lexical], [c.chunk_id for c in dense]], rrf_k, limit=k
    )
    fused = [
        Candidate(chunk_id=i, method="hybrid", score=s, rank=r) for r, (i, s) in enumerate(hits, 1)
    ]
    retriever.guard(fused, scope)
    return lexical, dense, retriever.dedupe(fused)


def identity_failures(
    config: AdaptiveRerankConfig,
    corpus_rows: int,
    index_rows: int | None,
    questions: Sequence[Question],
    index_name: str,
    *,
    endpoint: str | None,
) -> list[str]:
    """Frozen-data identity; any failure STOPs before the first query.

    `index_rows` and `endpoint` are the RUNTIME values reported by the index description
    (`status.indexed_row_count`, `endpoint_name`). A missing value fails closed: only the
    exact frozen row count and endpoint may continue.
    """
    exp, failures = config.expected, []
    if corpus_rows != exp["corpus_rows"]:
        failures.append(f"corpus rows {corpus_rows} != {exp['corpus_rows']}")
    if index_rows is None:
        failures.append("INDEX_ROW_COUNT_UNAVAILABLE: the index does not report indexed_row_count")
    elif index_rows != exp["index_rows"]:
        failures.append(f"INDEX_ROW_COUNT_MISMATCH: {index_rows} != {exp['index_rows']}")
    if endpoint is None:
        failures.append("ENDPOINT_UNAVAILABLE: the index does not report its endpoint")
    elif endpoint != exp["endpoint"]:
        failures.append(f"ENDPOINT_MISMATCH: {endpoint} != {exp['endpoint']}")
    if index_name.split(".")[-1] != exp["index_name"]:
        failures.append(f"index {index_name} != {exp['index_name']}")
    if (
        len(questions) != exp["questions"]
        or sum(q.answerable for q in questions) != exp["answerable"]
    ):
        failures.append("question set differs from the frozen 49 / 44")
    return failures


def collect(
    retriever: Any,
    questions: Sequence[Question],
    reranker: Reranker,
    config: AdaptiveRerankConfig,
    *,
    lock_sha256: str,
    environment: dict[str, Any] | None = None,
    index: dict[str, Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """The frozen collection protocol: warm-up + timed passes over every question."""
    stack, spec = config.stack, config.features
    k50, k10, ks = stack["adaptive_candidate_k"], stack["p0_candidate_k"], stack["metrics_k"]
    passes = config.passes["warmup"] + config.passes["timed"]
    records: dict[str, dict[str, Any]] = {}
    per_pass: dict[str, list[dict[str, Any]]] = defaultdict(list)
    problems: list[str] = []
    for pass_no in range(1, passes + 1):
        log(f"pass {pass_no}/{passes}")
        for q in questions:
            query = process_query(q.question, q.project_id, retriever.settings.query)
            scope = retriever.scope_for(
                query, q.project_id, stack["chunk_strategy"], stack["use_type_filters"]
            )
            t0 = clock()
            lexical, dense, fused = _k50_components(retriever, query, scope, k50, stack["rrf_k"])
            t1 = clock()
            fused_ids = [c.chunk_id for c in fused]
            features = compute_features(
                query,
                [c.chunk_id for c in lexical],
                [c.chunk_id for c in dense],
                fused_ids,
                retriever.store.rows,
                spec,
            )
            t2 = clock()
            reranked = retriever.rerank(query, fused, reranker)
            t3 = clock()
            p0 = retriever.dedupe(retriever.candidates(query, scope, "hybrid", k10))
            t4 = clock()
            lists = {
                "lexical_k50": [[c.chunk_id, c.score] for c in lexical],
                "dense_k50": [[c.chunk_id, c.score] for c in dense],
                "fused_k50": [[c.chunk_id, c.score] for c in fused],
                "fused_k10": [c.chunk_id for c in p0],
                "ce_scores": {c.chunk_id: c.rerank_score for c in reranked},
                "reranked_k50": [c.chunk_id for c in reranked],
            }
            per_pass[q.id].append(lists)
            timing = {
                "pass": pass_no,
                "first_stage_k50_ms": round((t1 - t0) * 1000, 3),
                "features_ms": round((t2 - t1) * 1000, 3),
                "rerank_k50_ms": round((t3 - t2) * 1000, 3),
                "p0_k10_ms": round((t4 - t3) * 1000, 3),
            }
            if pass_no == 1:
                check = retriever.dedupe(retriever.candidates(query, scope, "hybrid", k50))
                if [c.chunk_id for c in check] != fused_ids:
                    problems.append(f"{q.id}: fused k50 differs from Retriever hybrid k50")
                records[q.id] = {
                    "question_id": q.id,
                    "project_id": q.project_id,
                    "answerable": q.answerable,
                    "lists": lists,
                    "features": features.__dict__,
                    "labels": _labels(retriever, q, lists, ks, config) if q.answerable else None,
                    "timings": [],
                }
            records[q.id]["timings"].append(timing)
    consistency = pass_consistency(per_pass, config.consistency["ce_score_tolerance"])
    drift = drift_check(records.values(), config)
    status = (
        "INVALID_HYBRID_MISMATCH"
        if problems
        else "NONDETERMINISTIC"
        if not consistency["consistent"]
        else "DRIFT"
        if not drift["within_tolerance"]
        else "OK"
    )
    return {
        "run_id": config.run_ids["collection"],
        "lock_sha256": lock_sha256,
        "status": status,
        "problems": problems,
        "environment": environment or {},
        "index": index or {},
        "passes": dict(config.passes),
        "consistency": consistency,
        "drift_check": drift,
        "questions": [records[q.id] for q in questions],
    }


def _labels(
    retriever: Any,
    q: Question,
    lists: dict[str, Any],
    ks: Sequence[int],
    config: AdaptiveRerankConfig,
) -> dict[str, Any]:
    rows = retriever.store.rows
    fused = [i for i, _ in lists["fused_k50"]]
    ids = set(fused) | set(lists["fused_k10"])
    matched = {i: sorted(matched_items(rows[i], q)) for i in sorted(ids)}
    metrics = {
        name: ranking_metrics([rows[i] for i in order], q, ks)
        for name, order in (
            ("p0_k10", lists["fused_k10"]),
            ("p0_50", fused),
            ("p1", lists["reranked_k50"]),
        )
    }
    if any(v for m in metrics.values() for key, v in m.items() if key.startswith("leakage")):
        raise ScopeViolation(f"{q.id}: cross-project chunk in results")
    metrics = {
        n: {k: v for k, v in m.items() if not k.startswith("leakage")} for n, m in metrics.items()
    }
    return derive_labels(
        q, {i: set(m) for i, m in matched.items()}, fused, lists["reranked_k50"], metrics, config
    ) | {"matched": {i: m for i, m in matched.items() if m}}


def derive_labels(
    q: Question,
    matched: dict[str, set[int]],
    fused: Sequence[str],
    reranked: Sequence[str],
    metrics: dict[str, dict[str, float]],
    config: AdaptiveRerankConfig,
) -> dict[str, Any]:
    in_set = any(matched.get(i) for i in fused)
    cls, conflict = classify(
        metrics["p0_50"], metrics["p1"], in_set, config.classification["epsilon"]
    )
    return {
        "metrics": metrics,
        "first_relevant_rank": {
            "p0_50": first_relevant_rank([matched.get(i, set()) for i in fused]),
            "p1": first_relevant_rank([matched.get(i, set()) for i in reranked]),
        },
        "in_candidate_set": in_set,
        "delta": {
            "rr": metrics["p1"]["mrr"] - metrics["p0_50"]["mrr"],
            "ndcg_at_5": metrics["p1"]["ndcg_at_5"] - metrics["p0_50"]["ndcg_at_5"],
            "recall_at_5": metrics["p1"]["recall_at_5"] - metrics["p0_50"]["recall_at_5"],
            "recall_at_10": metrics["p1"]["recall_at_10"] - metrics["p0_50"]["recall_at_10"],
        },
        "class": cls,
        "signs_conflict": conflict,
    }


def pass_consistency(per_pass: dict[str, list[dict[str, Any]]], tolerance: float) -> dict:
    """Candidate lists identical across passes; CE scores within tolerance. Never averaged."""
    mismatches = []
    for qid, passes in per_pass.items():
        first = passes[0]
        for n, other in enumerate(passes[1:], start=2):
            for key in ("lexical_k50", "dense_k50", "fused_k50"):
                if [i for i, _ in first[key]] != [i for i, _ in other[key]]:
                    mismatches.append(f"{qid} pass {n}: {key} ids differ")
            for key in ("fused_k10", "reranked_k50"):
                if first[key] != other[key]:
                    mismatches.append(f"{qid} pass {n}: {key} differs")
            if set(first["ce_scores"]) != set(other["ce_scores"]) or any(
                abs(first["ce_scores"][i] - other["ce_scores"][i]) > tolerance
                for i in first["ce_scores"]
            ):
                mismatches.append(f"{qid} pass {n}: CrossEncoder scores differ beyond {tolerance}")
    return {"consistent": not mismatches, "tolerance": tolerance, "mismatches": mismatches}


def drift_check(records: Any, config: AdaptiveRerankConfig) -> dict[str, Any]:
    """P0 (k10) and P1 against the Phase 8 references, within the frozen tolerance."""
    labelled = [r["labels"] for r in records if r["labels"] is not None]
    tol, refs = config.drift["tolerance"], config.drift["references"]
    observed, failures = {}, []
    for point, key in (("P0", "p0_k10"), ("P1", "p1")):
        observed[point] = {}
        for metric in DRIFT_METRICS:
            value = (
                statistics.fmean(lab["metrics"][key][metric] for lab in labelled)
                if labelled
                else 0.0
            )
            observed[point][metric] = round(value, 6)
            if abs(value - refs[point][metric]) > tol:
                failures.append(f"{point} {metric} {value:.4f} vs Phase 8 {refs[point][metric]}")
    return {
        "within_tolerance": not failures,
        "tolerance": tol,
        "observed": observed,
        "phase8_reference": refs,
        "failures": failures,
    }


# -- offline analysis -----------------------------------------------------------------------------


def _artifact_failures(
    artifact: dict[str, Any],
    questions: Sequence[Question],
    config: AdaptiveRerankConfig,
    lock_sha: str,
) -> list[str]:
    failures = []
    if artifact.get("lock_sha256") != lock_sha:
        failures.append("artifact was produced under a different protocol lock")
    if artifact.get("run_id") != config.run_ids["collection"]:
        failures.append("artifact run id differs from the locked collection run id")
    if [r["question_id"] for r in artifact.get("questions", [])] != [q.id for q in questions]:
        failures.append("artifact questions differ from the frozen question set")
    for r in artifact.get("questions", []):
        if any(k in json.dumps(r) for k in ('"question":', '"chunk_text"', '"search_text"')):
            failures.append(f"{r['question_id']}: text persisted in the artifact")
    return failures


def recomputed_labels(
    record: dict[str, Any], q: Question, config: AdaptiveRerankConfig
) -> dict[str, Any]:
    matched = {i: set(m) for i, m in record["labels"]["matched"].items()}
    lists, ks = record["lists"], config.stack["metrics_k"]
    fused = [i for i, _ in lists["fused_k50"]]
    metrics = {
        name: ranking_metrics_from_matches([matched.get(i, set()) for i in order], q, ks)
        for name, order in (
            ("p0_k10", lists["fused_k10"]),
            ("p0_50", fused),
            ("p1", lists["reranked_k50"]),
        )
    }
    return derive_labels(q, matched, fused, lists["reranked_k50"], metrics, config)


def _composed_ms(record: dict[str, Any], reranked: bool, point: str) -> float:
    timed = [t for t in record["timings"] if t["pass"] > 1]
    first = statistics.fmean(t["first_stage_k50_ms"] for t in timed)
    feats = statistics.fmean(t["features_ms"] for t in timed)
    rerank = statistics.fmean(t["rerank_k50_ms"] for t in timed)
    if point == "P0":
        return statistics.fmean(t["p0_k10_ms"] for t in timed)
    if point in ("P0@50", "P1"):
        return first + (rerank if point == "P1" and record["lists"]["fused_k50"] else 0.0)
    return first + feats + (rerank if reranked else 0.0)


def _mean(values: Sequence[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def analyse(
    config: AdaptiveRerankConfig,
    lock: dict[str, Any],
    artifact: dict[str, Any],
    questions: Sequence[Question],
    *,
    lock_sha256: str,
    recomputed_lock_sha256: str,
) -> dict[str, Any]:
    """Fail-closed offline analysis. INVALID reports carry the reason and NO frontier."""
    base = {
        "phase": "9E",
        "run_id": config.run_ids["collection"],
        "lock_sha256": lock_sha256,
        "descriptive_only": True,
        "production_change": False,
        "note": "No automatic winner. Every preregistered point is reported.",
    }
    invalid: list[str] = []
    if recomputed_lock_sha256 != lock_sha256:
        invalid.append("recomputed protocol lock differs from the committed lock")
    invalid += _artifact_failures(artifact, questions, config, lock_sha256)
    if invalid:
        return base | {"status": "INVALID_ARTIFACT", "failures": invalid}
    if artifact.get("problems"):
        return base | {"status": "INVALID_HYBRID_MISMATCH", "failures": artifact["problems"]}
    consistency = pass_consistency_from_artifact(artifact)
    if not consistency["consistent"]:
        return base | {"status": "INVALID_NONDETERMINISM", "failures": consistency["mismatches"]}
    by_id = {q.id: q for q in questions}
    for r in artifact["questions"]:
        if r["labels"] is None:
            continue
        again = recomputed_labels(r, by_id[r["question_id"]], config)
        stored = {k: v for k, v in r["labels"].items() if k != "matched"}
        if json.dumps(again, sort_keys=True) != json.dumps(stored, sort_keys=True):
            invalid.append(f"{r['question_id']}: stored labels differ from recomputation")
    if invalid:
        return base | {"status": "INVALID_ARTIFACT", "failures": invalid}
    drift = drift_check(artifact["questions"], config)
    if not drift["within_tolerance"]:
        return base | {
            "status": "INVALID_DRIFT",
            "failures": drift["failures"],
            "drift_check": drift,
            "interpretation": "experiment INCOMPARABLE with Phase 8 pending investigation; "
            "no frontier is computed",
        }
    return base | {"status": "VALID", "drift_check": drift} | frontier(config, artifact, questions)


def pass_consistency_from_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """The collector's own consistency verdict (per-pass lists are compared at collection)."""
    c = artifact.get("consistency") or {}
    return {"consistent": bool(c.get("consistent")), "mismatches": c.get("mismatches", ["missing"])}


def point_decisions(
    config: AdaptiveRerankConfig, artifact: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Every deployable point: which questions it reranks. Features only - no labels."""
    out: dict[str, dict[str, Any]] = {}
    recs = artifact["questions"]
    has = {r["question_id"]: bool(r["lists"]["fused_k50"]) for r in recs}
    out["P0@50"] = {"policy_id": "P0@50", "threshold": None, "reranked": set()}
    out["P1"] = {"policy_id": "P1", "threshold": None, "reranked": {q for q, h in has.items() if h}}
    for policy in preregistered_policies(config):
        reranked = {
            r["question_id"] for r in recs if policy.trigger(TriggerFeatures(**r["features"]))
        }
        out[policy.name] = {
            "policy_id": policy.policy_id,
            "threshold": policy.threshold,
            "reranked": reranked,
        }
    return out


def select_live_point(rates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Frozen latency-validation rule: P2-P6 point with rerank rate (over all questions)
    closest to 0.5; ties -> lower policy id, then lower threshold. Rate ONLY."""
    candidates = [
        (
            round(abs(v["rerank_rate"] - 0.5), 9),  # exact ties reach the frozen tie-breakers
            v["policy_id"],
            -1 if v["threshold"] is None else v["threshold"],
            name,
        )
        for name, v in rates.items()
        if v["policy_id"] in ("P2", "P3", "P4", "P5", "P6")
    ]
    _, policy_id, threshold, name = min(candidates)
    return {
        "point": name,
        "policy_id": policy_id,
        "threshold": None if threshold == -1 else threshold,
        "rerank_rate": rates[name]["rerank_rate"],
        "rule": "closest rerank rate to 0.5; ties: lower policy id, then lower threshold",
        "label": LIVE_LABEL,
    }


def frontier(
    config: AdaptiveRerankConfig, artifact: dict[str, Any], questions: Sequence[Question]
) -> dict[str, Any]:
    recs = {r["question_id"]: r for r in artifact["questions"]}
    qs = {q.id: q for q in questions}
    answerable = [q.id for q in questions if q.answerable]
    all_ids = [q.id for q in questions]
    n_all, n_ans = len(all_ids), len(answerable)
    lab = {qid: recs[qid]["labels"] for qid in answerable}
    cls = {qid: lab[qid]["class"] for qid in answerable}
    groups = {c: {q for q in answerable if cls[q] == c} for c in CLASSES}
    m0 = {q: lab[q]["metrics"]["p0_50"] for q in answerable}
    m1 = {q: lab[q]["metrics"]["p1"] for q in answerable}

    def quality(reranked: set[str], ids: Sequence[str] = answerable) -> dict[str, float | None]:
        return {m: _mean([(m1 if q in reranked else m0)[q][m] for q in ids]) for m in QUALITY}

    p0_50, p1 = quality(set()), quality(set(answerable))
    decisions = point_decisions(config, artifact)
    points: dict[str, dict[str, Any]] = {}
    hist_p0 = {m: _mean([lab[q]["metrics"]["p0_k10"][m] for q in answerable]) for m in QUALITY}
    lat_p0 = [_composed_ms(recs[q], False, "P0") for q in all_ids]
    points["P0"] = {
        "role": "historical Phase 8 NEVER baseline (k10) - not a counterfactual",
        **hist_p0,
        "composed_p50_ms": _p50(lat_p0),
        "composed_p95_ms": _p95(lat_p0),
    }
    rng_cache: dict[int, dict[str, Any]] = {}
    for name, d in decisions.items():
        reranked = d["reranked"]
        t_ans = reranked & set(answerable)
        q = quality(reranked)
        lat = [
            _composed_ms(
                recs[i],
                i in reranked,
                "P0@50" if name == "P0@50" else "P1" if name == "P1" else "adaptive",
            )
            for i in all_ids
        ]
        k = len(t_ans)
        rnd = rng_cache.setdefault(k, random_reference(k, m0, m1, answerable, config))
        h = groups["HELPED"]
        point = {
            "policy_id": d["policy_id"],
            "threshold": d["threshold"],
            **q,
            "triggered": len(reranked),
            "triggered_answerable": k,
            "rerank_rate": round(len(reranked) / n_all, 4),
            "answerable_rerank_rate": round(k / n_ans, 4),
            "help_capture": round(len(t_ans & h) / len(h), 4) if h else None,
            "missed_help": len(h - t_ans),
            "hurt_exposure": len(t_ans & groups["HURT"]),
            "hurt_exposure_share": round(len(t_ans & groups["HURT"]) / len(groups["HURT"]), 4)
            if groups["HURT"]
            else None,
            "unnecessary_rerank_rate": round(
                len(t_ans & (groups["NEUTRAL"] | groups["HURT"])) / k, 4
            )
            if k
            else None,
            "futile_reranks": len(t_ans & groups["RETRIEVAL_MISS"]),
            "no_answer_reranks": len(reranked - set(answerable)),
            "retained_mrr_gain": _retained(q["mrr"], p0_50["mrr"], p1["mrr"]),
            "retained_ndcg_gain": _retained(q["ndcg_at_5"], p0_50["ndcg_at_5"], p1["ndcg_at_5"]),
            "random_expected": rnd["expected"],
            "random_p05": rnd["p05"],
            "random_p95": rnd["p95"],
            "delta_vs_random": {
                m: round(q[m] - rnd["expected"][m], 4) for m in QUALITY if q[m] is not None
            },
            "random_percentile": {
                m: _percentile(rnd["draws"][m], q[m]) for m in ("mrr", "ndcg_at_5")
            },
            "beats_random_mrr_and_ndcg": q["mrr"] > rnd["expected"]["mrr"]
            and q["ndcg_at_5"] > rnd["expected"]["ndcg_at_5"],
            "composed_p50_ms": _p50(lat),
            "composed_p95_ms": _p95(lat),
            "per_project": _per_project(qs, answerable, reranked, m0, m1),
        }
        points[name] = point
    oracle_set = groups["HELPED"]
    oracle_q = quality(oracle_set)
    oracle = {
        "label": ORACLE_LABEL,
        "deployable": False,
        "rerank_rate": round(len(oracle_set) / n_all, 4),
        "answerable_rerank_rate": round(len(oracle_set) / n_ans, 4),
        **oracle_q,
        "retained_mrr_gain": _retained(oracle_q["mrr"], p0_50["mrr"], p1["mrr"]),
        "retained_ndcg_gain": _retained(oracle_q["ndcg_at_5"], p0_50["ndcg_at_5"], p1["ndcg_at_5"]),
        "hurt_avoided": len(groups["HURT"]),
    }
    deployable = [n for n in points if n != "P0"]
    rates = {n: points[n] for n in deployable}
    return {
        "counterfactual": {
            "classes": {c: len(groups[c]) for c in CLASSES},
            "signs_conflict": sorted(q for q in answerable if lab[q]["signs_conflict"]),
            "per_question": [
                {
                    "question_id": q,
                    "project_id": qs[q].project_id,
                    "class": cls[q],
                    "in_candidate_set": lab[q]["in_candidate_set"],
                    "first_relevant_rank": lab[q]["first_relevant_rank"],
                    "delta": lab[q]["delta"],
                    "p0_50": m0[q],
                    "p1": m1[q],
                }
                for q in answerable
            ],
            "per_project": _project_classes(qs, answerable, lab),
        },
        "points": points,
        "random": {
            "label": "Random(r) - same answerable rerank count; not deployable",
            "seed": config.random["seed"],
            "draws": config.random["draws"],
        },
        "oracle": oracle,
        "pareto": {
            f"{a}_vs_{b}": pareto(points, deployable, a, b) for a, b in config.pareto["objectives"]
        },
        "live_validation_point": select_live_point(rates),
    }


def _retained(value: float | None, low: float | None, high: float | None) -> float | None:
    if value is None or low is None or high is None or high == low:
        return None
    return round((value - low) / (high - low), 4)


def random_reference(
    k: int,
    m0: dict[str, dict[str, float]],
    m1: dict[str, dict[str, float]],
    answerable: Sequence[str],
    config: AdaptiveRerankConfig,
) -> dict[str, Any]:
    """Exact expectation P0@50 + (k/n)(P1 - P0@50) and a fixed-seed draw distribution."""
    n = len(answerable)
    ids = sorted(answerable)
    base = {m: sum(m0[q][m] for q in ids) for m in QUALITY}
    gain = {m: {q: m1[q][m] - m0[q][m] for q in ids} for m in QUALITY}
    expected = {m: round((base[m] + (k / n) * sum(gain[m].values())) / n, 4) for m in QUALITY}
    rng = random.Random(f"{config.random['seed']}:{k}")
    draws: dict[str, list[float]] = {m: [] for m in ("mrr", "ndcg_at_5")}
    for _ in range(config.random["draws"]):
        sample = rng.sample(ids, k)
        for m in draws:
            draws[m].append((base[m] + sum(gain[m][q] for q in sample)) / n)
    return {
        "expected": expected,
        "draws": draws,
        "p05": {m: round(sorted(v)[int(0.05 * (len(v) - 1))], 4) for m, v in draws.items()},
        "p95": {m: round(sorted(v)[int(0.95 * (len(v) - 1))], 4) for m, v in draws.items()},
    }


def _percentile(draws: Sequence[float], value: float | None) -> float | None:
    if value is None or not draws:
        return None
    return round(sum(d <= value + 1e-12 for d in draws) / len(draws), 4)


def _per_project(qs, answerable, reranked, m0, m1) -> dict[str, dict[str, Any]]:
    out = {}
    for project in sorted({qs[q].project_id for q in qs}):
        ids = [q for q in answerable if qs[q].project_id == project]
        every = [q for q in qs if qs[q].project_id == project]
        out[project] = {
            "questions": len(ids),
            **{m: _mean([(m1 if q in reranked else m0)[q][m] for q in ids]) for m in QUALITY},
            "rerank_rate": round(sum(q in reranked for q in every) / len(every), 4)
            if every
            else None,
        }
    return out


def _project_classes(qs, answerable, lab) -> dict[str, dict[str, Any]]:
    out = {}
    for project in sorted({qs[q].project_id for q in answerable}):
        ids = [q for q in answerable if qs[q].project_id == project]
        counts = Counter(lab[q]["class"] for q in ids)

        def ranking_miss(key: str, ids=ids) -> float:
            missed = [
                q
                for q in ids
                if lab[q]["in_candidate_set"] and (lab[q]["first_relevant_rank"][key] or 10**9) > 10
            ]
            return round(len(missed) / len(ids), 4)

        out[project] = {
            "questions": len(ids),
            **{c: counts.get(c, 0) for c in CLASSES},
            "candidate_generation_miss_rate": round(counts.get("RETRIEVAL_MISS", 0) / len(ids), 4),
            "ranking_miss_rate_p0_50": ranking_miss("p0_50"),
            "ranking_miss_rate_p1": ranking_miss("p1"),
        }
    return out


def pareto(
    points: dict[str, dict[str, Any]], names: Sequence[str], quality_key: str, cost_key: str
) -> list[str]:
    """Non-dominated points: maximise quality, minimise cost (rerank rate or p95)."""
    cost_field = "rerank_rate" if cost_key == "rerank_rate" else cost_key
    front = []
    for a in names:
        qa, ca = points[a][quality_key], points[a][cost_field]
        dominated = any(
            points[b][quality_key] >= qa
            and points[b][cost_field] <= ca
            and (points[b][quality_key] > qa or points[b][cost_field] < ca)
            for b in names
            if b != a
        )
        if not dominated:
            front.append(a)
    return front


# -- live confirmation (07d) ----------------------------------------------------------------


def file_sha256(path: Path) -> str | None:
    """Raw-byte SHA-256 of a file (the collection artifact as written), or None if absent."""
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def read_json_artifact(path: Path) -> dict[str, Any]:
    """A committed JSON artifact, or {} when it is missing or unreadable, so the preflight
    reports it as a named failure instead of an unhandled exception."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def runtime_index_failures(runtime_name: str | None, index_name: str) -> list[str]:
    """The index the runtime actually opened (its own description) must be the configured one."""
    if not runtime_name:
        return ["INDEX_NAME_UNAVAILABLE: the index does not report its name"]
    if runtime_name != index_name:
        return [f"INDEX_NAME_MISMATCH: {runtime_name} != {index_name}"]
    return []


def live_preflight_failures(
    *,
    lock: dict[str, Any],
    recomputed_lock: dict[str, Any],
    frontier: dict[str, Any],
    selection: dict[str, Any],
    collection_sha256: str | None,
    authorized: dict[str, str],
) -> list[str]:
    """Every 07d pre-run integrity check; any failure STOPs before the first live query.

    `authorized` holds the values the live run was approved for: `lock_sha256`,
    `policy_definitions_sha256`, `collection_artifact_sha256` and `point`. The selection must
    be the committed one, derived by the frozen rule from the committed frontier.
    """
    failures = []
    if not frontier:
        failures.append("FRONTIER_UNAVAILABLE: committed frontier missing or unreadable")
    if not selection:
        failures.append("SELECTION_UNAVAILABLE: committed live selection missing or unreadable")
    sha = lock_sha256(lock)
    if recomputed_lock != lock:
        failures.append("LOCK_MISMATCH: recomputed protocol lock differs from the committed lock")
    if sha != authorized["lock_sha256"]:
        failures.append(f"LOCK_NOT_AUTHORIZED: {sha} != {authorized['lock_sha256']}")
    policy_sha = _sha(lock.get("policy_definitions"))
    if {policy_sha, lock.get("policy_definitions_sha256")} != {
        authorized["policy_definitions_sha256"]
    }:
        failures.append(f"POLICY_DEFINITIONS_MISMATCH: {policy_sha}")
    if lock.get("production_null") is not True:
        failures.append("PRODUCTION_NOT_NULL: production retrieval config is set")
    if collection_sha256 is None:
        failures.append("COLLECTION_ARTIFACT_UNAVAILABLE: collection artifact not found")
    elif collection_sha256 != authorized["collection_artifact_sha256"]:
        failures.append(f"COLLECTION_ARTIFACT_MISMATCH: {collection_sha256}")
    if (
        frontier.get("status") != "VALID"
        or frontier.get("lock_sha256") != sha
        or frontier.get("collection_artifact_sha256") != authorized["collection_artifact_sha256"]
    ):
        failures.append("FRONTIER_MISMATCH: committed frontier is not VALID for this lock/artifact")
    if (
        selection.get("lock_sha256") != sha
        or selection.get("collection_artifact_sha256") != authorized["collection_artifact_sha256"]
    ):
        failures.append("SELECTION_MISMATCH: live selection belongs to another lock/artifact")
    chosen = {
        k: v for k, v in selection.items() if k not in ("lock_sha256", "collection_artifact_sha256")
    }
    committed = frontier.get("live_validation_point")
    if chosen != committed:
        failures.append("SELECTION_NOT_FROM_FRONTIER: selection differs from the frontier's point")
    rates = {n: p for n, p in (frontier.get("points") or {}).items() if n != "P0"}
    try:
        reproduced = select_live_point(rates) if rates else None
    except (KeyError, TypeError, ValueError):  # malformed frontier points
        reproduced = None
    if reproduced is None or reproduced != committed:
        failures.append("SELECTION_RULE_MISMATCH: frozen rule does not reproduce the selection")
    point = selection.get("point")
    if point != authorized["point"] or point not in lock.get("preregistered_points", []):
        failures.append(f"POINT_NOT_AUTHORIZED: {point} != {authorized['point']}")
    elif selection.get("policy_id") != point.split("(")[0] or (
        point != AdaptivePolicy(selection["policy_id"], selection.get("threshold")).name
    ):
        failures.append(f"POINT_DEFINITION_MISMATCH: {point}")
    if selection.get("label") != LIVE_LABEL:
        failures.append("SELECTION_LABEL_MISMATCH: must be a latency validation point")
    return failures


def run_live(
    retriever: Any,
    recorder: Any,
    questions: Sequence[Question],
    points: Sequence[tuple[str, Any]],  # (label, RerankPolicy) - P0@50 uses NeverRerank
    reranker: Reranker,
    config: AdaptiveRerankConfig,
    *,
    lock_sha256: str,
    selection: dict[str, Any],
    clock: Callable[[], float] = time.perf_counter,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Real path: first_stage(k50) -> policy.decide(stage) -> finish(...), timed end to end."""
    stack = config.stack
    passes = config.live["passes"]["warmup"] + config.live["passes"]["timed"]
    rows = []
    for pass_no in range(1, passes + 1):
        log(f"live pass {pass_no}/{passes}")
        for q in questions:
            for label, policy in points:
                recorder.last, before = None, recorder.calls
                t0 = clock()
                stage = retriever.first_stage(
                    q.question,
                    q.project_id,
                    strategy=stack["chunk_strategy"],
                    method=stack["method"],
                    candidate_k=stack["adaptive_candidate_k"],
                    use_type_filters=stack["use_type_filters"],
                )
                decision = policy.decide(stage)
                result = retriever.finish(
                    stage,
                    reranker=reranker if decision.rerank else NoReranker(),
                    final_k=stack["final_k"],
                )
                t1 = clock()
                rows.append(
                    {
                        "pass": pass_no,
                        "point": label,
                        "question_id": q.id,
                        "live_ms": round((t1 - t0) * 1000, 3),
                        "rerank": decision.rerank,
                        "top5": [e.chunk_id for e in result.evidence],
                        "status": result.status,
                        "dense_calls": recorder.calls - before,
                    }
                )
    return {
        "run_id": config.run_ids["live"],
        "lock_sha256": lock_sha256,
        "selection": selection,
        "passes": dict(config.live["passes"]),
        "rows": rows,
    }


def never_policy() -> NeverRerank:
    return NeverRerank()


def compare_live(
    config: AdaptiveRerankConfig,
    collection: dict[str, Any],
    live: dict[str, Any],
    questions: Sequence[Question],
) -> dict[str, Any]:
    """Composed (collection) vs live end-to-end latency, and decision / top-5 agreement."""
    recs = {r["question_id"]: r for r in collection["questions"]}
    sel = live["selection"]
    adaptive = AdaptivePolicy(sel["policy_id"], sel["threshold"])
    final_k = config.stack["final_k"]
    out: dict[str, Any] = {"selection": sel, "points": {}, "label": LIVE_LABEL}
    for label in ("P0@50", sel["point"]):
        is_p0 = label == "P0@50"
        composed, live_ms, mismatches, per_question = [], [], [], []
        decision_mismatches = top5_mismatches = 0
        rows = [x for x in live["rows"] if x["point"] == label]
        for q in questions:
            r = recs[q.id]
            expect_rerank = False if is_p0 else adaptive.trigger(TriggerFeatures(**r["features"]))
            c_ms = _composed_ms(r, expect_rerank, "P0@50" if is_p0 else "adaptive")
            composed.append(c_ms)
            mine = [x for x in rows if x["question_id"] == q.id]
            timed = [x for x in mine if x["pass"] > 1]
            l_ms = statistics.fmean(x["live_ms"] for x in timed)
            live_ms.append(l_ms)
            order = (
                r["lists"]["reranked_k50"]
                if expect_rerank
                else [i for i, _ in r["lists"]["fused_k50"]]
            )
            for x in mine:  # every pass, warm-up included
                if x["rerank"] != expect_rerank:
                    decision_mismatches += 1
                    mismatches.append(
                        f"{q.id} pass {x['pass']}: live decision {x['rerank']} != offline"
                    )
                if x["top5"] != order[:final_k]:
                    top5_mismatches += 1
                    mismatches.append(
                        f"{q.id} pass {x['pass']}: live top-5 differs from the simulated top-5"
                    )
            per_question.append(
                {
                    "question_id": q.id,
                    "rerank": expect_rerank,
                    "composed_ms": round(c_ms, 1),
                    "live_ms": round(l_ms, 1),
                    "difference_ms": round(l_ms - c_ms, 1),
                }
            )
        c50, c95, l50, l95 = _p50(composed), _p95(composed), _p50(live_ms), _p95(live_ms)
        timed_rows = [x for x in rows if x["pass"] > 1]
        by_decision = {
            name: [p["live_ms"] for p in per_question if p["rerank"] is flag]
            for name, flag in (("reranked", True), ("not_reranked", False))
        }
        out["points"][label] = {
            "composed_p50_ms": c50,
            "composed_p95_ms": c95,
            "live_p50_ms": l50,
            "live_p95_ms": l95,
            "difference_p50_ms": round(l50 - c50, 1),
            "difference_p95_ms": round(l95 - c95, 1),
            "relative_difference_p50": round((l50 - c50) / c50, 4) if c50 else None,
            "relative_difference_p95": round((l95 - c95) / c95, 4) if c95 else None,
            "observed_rerank_rate": round(
                sum(x["rerank"] for x in timed_rows) / len(timed_rows), 4
            ),
            "agreement": {
                "rows_checked": len(rows),
                "decision_mismatches": decision_mismatches,
                "top5_mismatches": top5_mismatches,
            },
            "live_by_decision": {
                name: {"n": len(v), "p50_ms": _p50(v), "p95_ms": _p95(v), "mean_ms": _mean(v)}
                for name, v in by_decision.items()
            },
            "passes": {
                str(n): {
                    "kind": "warm-up" if n == 1 else "timed",
                    "p50_ms": _p50([x["live_ms"] for x in rows if x["pass"] == n]),
                    "p95_ms": _p95([x["live_ms"] for x in rows if x["pass"] == n]),
                    "max_ms": max(x["live_ms"] for x in rows if x["pass"] == n),
                }
                for n in sorted({x["pass"] for x in rows})
            },
            "per_question": per_question,
            "mismatches": mismatches,
        }
    return out


# -- protocol lock ---------------------------------------------------------------------------------


def build_lock(
    config: AdaptiveRerankConfig, repo_root: Path, retrieval_production: dict[str, Any]
) -> dict[str, Any]:
    """Everything the 9E analysis depends on. `status` is deliberately excluded."""
    from worldbank_copilot.routing.semantic_eval import canonical_sha256

    root = Path(repo_root)
    hashes = {
        name: canonical_sha256(root / path)
        for name, path in (
            ("questions", config.questions_file),
            ("retrieval_yaml", "configs/retrieval/retrieval.yaml"),
            ("query_yaml", "configs/retrieval/query.yaml"),
            ("chunking_yaml", "configs/retrieval/chunking.yaml"),
            ("embeddings_yaml", "configs/retrieval/embeddings.yaml"),
        )
    }
    policies = {
        "policies": [p.model_dump() for p in config.policies],
        "features": config.features.model_dump(),
    }
    lock = {
        "phase": "9E",
        "run_ids": config.run_ids,
        "artifact_dir": config.artifact_dir,
        "hashes": hashes,
        "production_null": all(
            v is None for k, v in retrieval_production.items() if k != "final_k"
        ),
        "retrieval_production": retrieval_production,
        "stack": config.stack,
        "expected": config.expected,
        "policy_definitions": policies,
        "policy_definitions_sha256": _sha(policies),
        "preregistered_points": [p.name for p in preregistered_policies(config)],
        "classification": config.classification,
        "diagnostic_formulas": {
            "rerank_rate": "|T| / 49",
            "answerable_rerank_rate": "|T & A| / 44",
            "help_capture": "|T & H| / |H|",
            "missed_help": "|H - T|",
            "hurt_exposure": "|T & U| (and / |U|)",
            "unnecessary_rerank_rate": "|T & (N | U)| / |T & A|",
            "futile_reranks": "|T & M|",
            "no_answer_reranks": "|T - A|",
            "retained_gain": "(point - P0@50) / (P1 - P0@50) for MRR and nDCG@5",
            "composed_latency": "first_stage_k50 + features + rerank_k50 if reranked; mean of "
            "timed passes; p50 median, p95 nearest-rank over all 49",
        },
        "random": config.random,
        "oracle": config.oracle,
        "pareto": config.pareto,
        "drift": config.drift,
        "passes": config.passes,
        "consistency": config.consistency,
        "live": config.live,
        "scope": config.scope,
    }
    return json.loads(json.dumps(lock, ensure_ascii=False))  # plain JSON types (as committed)


def lock_sha256(lock: dict[str, Any]) -> str:
    return _sha(lock)


def load_retrieval_production(config_dir: Path) -> dict[str, Any]:
    data = yaml.safe_load((Path(config_dir) / "retrieval" / "retrieval.yaml").read_text("utf-8"))
    return dict(data["production"])


# -- reports ---------------------------------------------------------------------------------------


def render_markdown(report: dict[str, Any]) -> str:
    """Frontier report (question / chunk ids only). INVALID reports show the reason only."""
    head = [
        "# Phase 9E adaptive-rerank diagnostic",
        "",
        f"**Status: {report['status']}** - descriptive diagnostic over the frozen Phase 8 set; "
        "not independent validation; no automatic winner; production config unchanged.",
        "",
    ]
    if report["status"] != "VALID":
        return (
            "\n".join(
                [*head, "## Why no frontier", *[f"- {f}" for f in report.get("failures", [])]]
            )
            + "\n"
        )
    cols = [
        "R@5",
        "R@10",
        "MRR",
        "nDCG@5",
        "rate",
        "help capture",
        "hurt exp.",
        "ret. MRR",
        "ret. nDCG",
        "dMRR vs rnd",
        "dnDCG vs rnd",
        "p50 ms",
        "p95 ms",
    ]
    rows = []
    for name, p in report["points"].items():
        if name == "P0":
            rows.append(
                f"| P0 (k10, historical) | {p['recall_at_5']} | {p['recall_at_10']} | {p['mrr']} "
                f"| {p['ndcg_at_5']} | - | - | - | - | - | - | - | {p['composed_p50_ms']} "
                f"| {p['composed_p95_ms']} |"
            )
            continue
        d = p["delta_vs_random"]
        rows.append(
            f"| {name} | {p['recall_at_5']} | {p['recall_at_10']} | {p['mrr']} | {p['ndcg_at_5']} "
            f"| {p['rerank_rate']} | {p['help_capture']} | {p['hurt_exposure']} "
            f"| {p['retained_mrr_gain']} | {p['retained_ndcg_gain']} | {d.get('mrr')} "
            f"| {d.get('ndcg_at_5')} | {p['composed_p50_ms']} | {p['composed_p95_ms']} |"
        )
    o, cf = report["oracle"], report["counterfactual"]
    return (
        "\n".join(
            [
                *head,
                "## Counterfactual classes (answerable)",
                f"{cf['classes']}; signs_conflict: {cf['signs_conflict'] or 'none'}",
                "",
                "## Per project",
                *[f"- {p}: {v}" for p, v in cf["per_project"].items()],
                "",
                "## Frontier (every preregistered point)",
                "| point | " + " | ".join(cols) + " |",
                "|---" * (len(cols) + 1) + "|",
                *rows,
                "",
                f"## Oracle - {o['label']}",
                f"rate {o['rerank_rate']}, MRR {o['mrr']}, nDCG@5 {o['ndcg_at_5']}, "
                f"retained MRR {o['retained_mrr_gain']}, retained nDCG {o['retained_ndcg_gain']}",
                "",
                "## Pareto sets (deployable points)",
                *[f"- {k}: {v}" for k, v in report["pareto"].items()],
                "",
                "## Live latency validation point (rerank rate rule; NOT a production winner)",
                f"{report['live_validation_point']}",
            ]
        )
        + "\n"
    )


def render_live_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 9E live latency confirmation",
        "",
        f"Selection: {report['selection']} - {report['label']}",
        "",
        "| point | composed p50 | composed p95 | live p50 | live p95 | d p50 | d p95 | mism. |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, p in report["points"].items():
        lines.append(
            f"| {name} | {p['composed_p50_ms']} | {p['composed_p95_ms']} | {p['live_p50_ms']} "
            f"| {p['live_p95_ms']} | {p['difference_p50_ms']} | {p['difference_p95_ms']} "
            f"| {len(p['mismatches'])} |"
        )
    for name, p in report["points"].items():
        a = p["agreement"]
        lines += [
            "",
            f"## {name}",
            "",
            f"- relative difference: p50 {p['relative_difference_p50']}, "
            f"p95 {p['relative_difference_p95']}",
            f"- observed rerank rate (timed passes): {p['observed_rerank_rate']}",
            f"- agreement over {a['rows_checked']} rows: {a['decision_mismatches']} decision, "
            f"{a['top5_mismatches']} top-5 mismatches",
        ]
        for kind, d in p["live_by_decision"].items():
            lines.append(f"- live {kind}: n={d['n']} p50 {d['p50_ms']} p95 {d['p95_ms']}")
        for n, s in p["passes"].items():
            lines.append(f"- pass {n} ({s['kind']}): p50 {s['p50_ms']} p95 {s['p95_ms']}")
        lines += [
            "",
            "| question | rerank | composed ms | live ms | diff ms |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {q['question_id']} | {q['rerank']} | {q['composed_ms']} | {q['live_ms']} "
            f"| {q['difference_ms']} |"
            for q in p["per_question"]
        ]
        lines += [f"- MISMATCH {m}" for m in p["mismatches"]]
    return "\n".join(lines) + "\n"
