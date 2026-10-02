"""Phase 9D development-only evaluation of the semantic fallback.

* Leave-one-family-out (LOFO) over the DEVELOPMENT examples: a question is classified
  with every example of its own family removed, so near-duplicates never vote for
  themselves.
* A configuration grid, a coverage-vs-accuracy table and a pre-registered selection rule
  (configs/routing/semantic.yaml).
* End-to-end DEVELOPMENT routing: the recorded 9B.2 rule result for every dev case, with
  the fallback (LOFO) only where the rules returned SEMANTIC_CLASSIFICATION_REQUIRED.
* ``assert_split_allowed`` refuses the frozen TEST split unless a configuration has been
  frozen AND test evaluation explicitly authorised - this checkpoint does neither.
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from worldbank_copilot.routing.semantic import (
    Example,
    ExampleStore,
    KnnConfig,
    KnnSemanticClassifier,
    SemanticQueryContext,
)

SEMANTIC = "SEMANTIC_CLASSIFICATION_REQUIRED"


def frac(numerator: int, denominator: int) -> str:
    """Always report the denominator: the semantic subset is small."""
    return f"{numerator}/{denominator}"


class FrozenTestEvaluationBlocked(RuntimeError):
    """The frozen TEST split may only be evaluated once, with a frozen configuration."""


def assert_split_allowed(
    split: str, frozen_config: dict | None = None, authorised: bool = False
) -> None:
    if split == "dev":
        return
    if split != "test":
        raise ValueError(f"unknown split {split!r}")
    if not (frozen_config and frozen_config.get("status") == "FROZEN" and authorised):
        raise FrozenTestEvaluationBlocked(
            "frozen TEST evaluation needs a FROZEN semantic configuration and explicit "
            "authorisation (one run, after development selection)"
        )


class SemanticProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int
    cache_namespace: str
    evaluation: str
    grid: dict[str, list[Any]]
    selection: dict[str, float]


def load_protocol(config_dir: Path) -> SemanticProtocol:
    raw = yaml.safe_load((Path(config_dir) / "routing" / "semantic.yaml").read_text("utf-8"))
    return SemanticProtocol.model_validate(raw)


def grid(protocol: SemanticProtocol) -> list[KnnConfig]:
    g = protocol.grid
    return [
        KnnConfig(k=k, weighting=w, min_similarity=s, min_vote_share=v)
        for k, w, s, v in itertools.product(
            g["k"], g["weighting"], g["min_similarity"], g["min_vote_share"]
        )
    ]


@dataclass(frozen=True)
class Prediction:
    example_id: str
    family: str
    gold_intent: str
    gold_route: str
    abstain: bool
    intent: str | None
    route: str | None
    confidence: float
    top_similarity: float
    latency_ms: float


def lofo(classifier: KnnSemanticClassifier, examples: tuple[Example, ...]) -> list[Prediction]:
    out = []
    for e in examples:
        d = classifier.classify(
            SemanticQueryContext(e.example_id, e.question, "DEV", "NONE", ()),
            exclude_families=frozenset({e.family}),
        )
        out.append(
            Prediction(
                e.example_id,
                e.family,
                e.intent.value,
                classifier.route_of[e.intent],
                d.abstain,
                d.intent.value if d.intent else None,
                d.route,
                d.confidence,
                d.neighbours[0].similarity if d.neighbours else 0.0,
                d.latency_ms,
            )
        )
    return out


def summarise(preds: list[Prediction]) -> dict[str, Any]:
    n = len(preds)
    answered = [p for p in preds if not p.abstain]
    route_ok = sum(p.route == p.gold_route for p in answered)
    intent_ok = sum(p.intent == p.gold_intent for p in answered)
    confusion = Counter(f"{p.gold_route} -> {p.route or 'ABSTAIN'}" for p in preds)
    families: dict[str, dict[str, int]] = {}
    for p in preds:
        f = families.setdefault(
            p.family, {"n": 0, "answered": 0, "route_correct": 0, "intent_correct": 0}
        )
        f["n"] += 1
        f["answered"] += not p.abstain
        f["route_correct"] += (not p.abstain) and p.route == p.gold_route
        f["intent_correct"] += (not p.abstain) and p.intent == p.gold_intent
    latencies = sorted(p.latency_ms for p in preds)
    return {
        "n": n,
        "answered": len(answered),
        "coverage": round(len(answered) / n, 3) if n else 0.0,
        "abstention_rate": round(1 - len(answered) / n, 3) if n else 0.0,
        "route_precision": round(route_ok / len(answered), 3) if answered else None,
        "intent_precision": round(intent_ok / len(answered), 3) if answered else None,
        "route_accuracy_all": round(route_ok / n, 3) if n else 0.0,  # abstain counts wrong
        "intent_accuracy_all": round(intent_ok / n, 3) if n else 0.0,
        "route_confusion": dict(sorted(confusion.items())),
        "abstentions": n - len(answered),
        "per_family": dict(sorted(families.items())),
        "classifier_latency_ms": {
            "p50": latencies[len(latencies) // 2] if latencies else None,
            "p95": latencies[int(0.95 * (len(latencies) - 1))] if latencies else None,
        },
    }


def coverage_curve(preds: list[Prediction]) -> list[dict[str, Any]]:
    """Selective accuracy: keep the most confident c predictions (no abstention config)."""
    ranked = sorted(preds, key=lambda p: (-p.confidence, p.example_id))
    rows = []
    for c in range(1, len(ranked) + 1):
        kept = ranked[:c]
        rows.append(
            {
                "kept": c,
                "coverage": round(c / len(ranked), 3),
                "min_confidence": kept[-1].confidence,
                "route_accuracy": round(sum(p.route == p.gold_route for p in kept) / c, 3),
                "intent_accuracy": round(sum(p.intent == p.gold_intent for p in kept) / c, 3),
            }
        )
    return rows


def select(results: list[dict[str, Any]], protocol: SemanticProtocol) -> dict[str, Any] | None:
    rule = protocol.selection
    ok = [
        r
        for r in results
        if r["metrics"]["route_precision"] is not None
        and r["metrics"]["route_precision"] >= rule["min_route_precision"]
        and r["metrics"]["coverage"] >= rule["min_coverage"]
    ]
    if not ok:
        return None
    return sorted(
        ok,
        key=lambda r: (
            -r["metrics"]["coverage"],
            -(r["metrics"]["intent_precision"] or 0),
            r["config"]["min_similarity"] + r["config"]["min_vote_share"],
            r["config"]["k"],
            r["config"]["weighting"] != "uniform",
        ),
    )[0]


def end_to_end_dev(
    dataset: Any,
    baseline: dict[str, dict],
    classifier: KnnSemanticClassifier,
) -> dict[str, Any]:
    """Rules (recorded 9B.2) + fallback (LOFO) over the DEVELOPMENT split only."""
    assert_split_allowed("dev")
    rows = []
    for case in dataset.cases:
        if case.split != "dev":
            continue
        rule = baseline[case.case_id]
        route, via = rule["route"], "rules"
        decision = None
        if route == "SEMANTIC_CLASSIFICATION_REQUIRED":
            decision = classifier.classify(
                SemanticQueryContext(
                    case.case_id,
                    case.question,
                    rule["project_id"] or "",
                    rule["temporal_kind"] or "NONE",
                    (),
                ),
                exclude_families=frozenset({case.family}),
            )
            route, via = (
                ("CLARIFY", "semantic-abstain")
                if decision.abstain
                else (decision.route, "semantic")
            )
        rows.append(
            {
                "case_id": case.case_id,
                "expected": case.expected.route,
                "final": route,
                "via": via,
                "correct": route == case.expected.route,
                "intent": decision.intent.value if decision and decision.intent else None,
                "expected_intents": [i.value for i in case.expected.intents],
            }
        )
    n = len(rows)
    per_route: dict[str, dict[str, int]] = {}
    for r in rows:
        bucket = per_route.setdefault(r["expected"], {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += r["correct"]
    semantic_rows = [r for r in rows if r["via"] != "rules"]
    resolved = sum(r["via"] == "rules" and r["final"] != SEMANTIC for r in rows)
    return {
        "n": n,
        "route_accuracy": frac(sum(r["correct"] for r in rows), n),
        "deterministic_coverage": frac(resolved, n),
        "semantic_invocations": frac(len(semantic_rows), n),
        "semantic_correct": frac(sum(r["correct"] for r in semantic_rows), len(semantic_rows)),
        "semantic_abstained": frac(
            sum(r["via"] == "semantic-abstain" for r in rows), len(semantic_rows)
        ),
        "clarify_rate": frac(sum(r["final"] == "CLARIFY" for r in rows), n),
        "per_route": per_route,
        "rows": rows,
    }


def config_dict(config: KnnConfig) -> dict[str, Any]:
    return asdict(config)


def store_for(dataset: Any) -> ExampleStore:
    store = ExampleStore.from_dataset(dataset)
    store.assert_development_only(dataset)
    return store


def rules_only_dev(dataset: Any, baseline: dict[str, dict]) -> dict[str, Any]:
    """Candidate A on DEVELOPMENT: recorded 9B.2; semantic-needed -> CLARIFY (no fallback)."""
    rows = []
    for case in dataset.cases:
        if case.split != "dev":
            continue
        route = baseline[case.case_id]["route"]
        final = "CLARIFY" if route == "SEMANTIC_CLASSIFICATION_REQUIRED" else route
        rows.append(
            {
                "case_id": case.case_id,
                "expected": case.expected.route,
                "final": final,
                "semantic_needed": route == "SEMANTIC_CLASSIFICATION_REQUIRED",
                "correct": final == case.expected.route,
            }
        )
    n = len(rows)
    return {
        "n": n,
        "route_accuracy": f"{sum(r['correct'] for r in rows)}/{n}",
        "semantic_needed": f"{sum(r['semantic_needed'] for r in rows)}/{n}",
        "rows": rows,
    }


def run_development(
    dataset: Any, routes: dict, protocol: SemanticProtocol, embedder: Any, baseline: dict[str, dict]
) -> dict[str, Any]:
    """The whole development experiment for one embedder (DEVELOPMENT split only)."""
    import statistics

    assert_split_allowed("dev")
    store = store_for(dataset)
    results = []
    for config in grid(protocol):
        clf = KnnSemanticClassifier(embedder, store, config, routes)
        results.append(
            {
                "candidate": "B",
                "embedder": embedder.name,
                "split": "dev",
                "protocol": "leave_one_family_out",
                "config": config_dict(config),
                "name": config.name,
                "metrics": summarise(lofo(clf, store.examples)),
            }
        )
    chosen = select(results, protocol)
    base = KnnSemanticClassifier(embedder, store, KnnConfig(k=3, weighting="similarity"), routes)
    base_preds = lofo(base, store.examples)
    latencies = sorted(p.latency_ms for p in base_preds)
    majority = Counter(routes[e.intent] for e in store.examples).most_common(1)[0]
    return {
        "embedder": embedder.name,
        "split_evaluated": "dev",
        "test_evaluated": False,
        "examples": [e.example_id for e in store.examples],
        "majority_route_baseline": {
            "route": majority[0],
            "accuracy": round(majority[1] / len(store.examples), 3),
        },
        "experiments": results,
        "selected": chosen,
        "base_config": KnnConfig(k=3, weighting="similarity").name,
        "base_lofo": [p.__dict__ for p in base_preds],
        "coverage_curve": coverage_curve(base_preds),
        "rules_only_dev": rules_only_dev(dataset, baseline),
        "hypothetical_fallback_dev": {
            "note": "NOT adopted unless selected; base configuration without abstention",
            **end_to_end_dev(dataset, baseline, base),
        },
        "selected_fallback_dev": end_to_end_dev(
            dataset,
            baseline,
            KnnSemanticClassifier(embedder, store, KnnConfig(**chosen["config"]), routes),
        )
        if chosen
        else None,
        "classifier_latency_ms": {
            "p50": round(statistics.median(latencies), 3),
            "p95": round(latencies[int(0.95 * (len(latencies) - 1))], 3),
            "n": len(latencies),
        },
    }


# -- Databricks development guards (Phase 9D) ---------------------------------------------


class DevelopmentGuardFailed(RuntimeError):
    """A frozen-ground-truth precondition for the development experiment does not hold."""


def canonical_sha256(path: Path) -> str:
    """SHA-256 with CRLF normalised to LF (Windows vs Databricks Git checkouts)."""
    import hashlib

    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def development_guards(
    repo_root: Path, dataset: Any, routing: Any, store: ExampleStore, semantic_decision: dict | None
) -> list[dict[str, Any]]:
    """Every check against the frozen 9C manifest; the caller STOPs if any fails."""
    import json

    manifest = json.loads(
        (Path(repo_root) / "evaluation" / "routing_freeze_9c.json").read_text(encoding="utf-8")
    )
    dev = sorted(c.case_id for c in dataset.cases if c.split == "dev")
    test = sorted(c.case_id for c in dataset.cases if c.split == "test")
    test_questions = {
        " ".join(c.question.lower().split()) for c in dataset.cases if c.split == "test"
    }
    examples = [e.example_id for e in store.examples]
    checks = [
        (
            "dataset hash equals frozen 9C hash",
            canonical_sha256(Path(repo_root) / manifest["dataset"])
            == manifest["dataset_sha256_lf"],
        ),
        (
            "split is exactly the frozen 29 DEV / 51 TEST",
            dev == manifest["dev_case_ids"]
            and test == manifest["test_case_ids"]
            and len(dev) == 29
            and len(test) == 51,
        ),
        (
            "all labels REVIEWED and split FROZEN",
            dataset.review_status == "REVIEWED"
            and dataset.split_status == "FROZEN"
            and {c.label_status for c in dataset.cases} == {"REVIEWED"},
        ),
        (
            "aliases are the reviewed 11",
            {p: list(a) for p, a in routing.aliases.aliases.items()} == manifest["aliases"]
            and routing.aliases.reviewed_by == manifest["aliases_reviewed_by"]
            and sum(len(a) for a in manifest["aliases"].values()) == 11,
        ),
        (
            "router version is 9B.2",
            routing.settings.router_version == manifest["router_version"] == "9B.2",
        ),
        (
            "semantic configuration does not indicate TEST evaluation",
            semantic_decision is not None
            and semantic_decision.get("test_evaluated") is False
            and semantic_decision.get("status") != "FROZEN",
        ),
        ("experiment inputs are a subset of DEV", set(examples) <= set(dev) and bool(examples)),
        (
            "no TEST question is used as an example",
            not ({" ".join(e.question.lower().split()) for e in store.examples} & test_questions),
        ),
    ]
    return [{"guard": name, "passed": bool(ok)} for name, ok in checks]


def require_guards(results: list[dict[str, Any]]) -> None:
    failed = [r["guard"] for r in results if not r["passed"]]
    if failed:
        raise DevelopmentGuardFailed(f"STOP: development guards failed: {failed}")


def probe_embedder(provider: Any, expected_dimension: int, text: str = "routing probe") -> dict:
    """Availability, dimension and single-call latency of the embedding endpoint."""
    import time

    started = time.perf_counter()
    try:
        (vector,) = provider.embed([text])
    except Exception as exc:  # reported, then the notebook STOPs
        error = f"{type(exc).__name__}: {exc}"
        return {"model": provider.model, "available": False, "error": error}
    return {
        "model": provider.model,
        "available": True,
        "dimension": len(vector),
        "dimension_matches": len(vector) == expected_dimension,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "error": None,
    }


def similarity_diagnostic(classifier: KnnSemanticClassifier, protocol: SemanticProtocol) -> dict:
    """Do the pre-registered similarity thresholds discriminate on this embedder's cosines?

    Leave-one-family-out nearest-neighbour similarity per example, and the share of examples
    each registered threshold would keep. If every positive threshold keeps all or none of
    the examples, the similarity grid is inert for this embedder (the notebook STOPs before
    the grid so the protocol is not changed after seeing results).
    """
    tops = []
    for e in classifier.store.examples:
        rows = classifier.scored(e.question, frozenset({e.family}))
        tops.append(rows[0][0] if rows else 0.0)
    tops.sort()
    thresholds = [t for t in protocol.grid["min_similarity"] if t > 0]
    kept = {f"{t:.2f}": round(sum(x >= t for x in tops) / len(tops), 3) for t in thresholds}
    inert = all(v in (0.0, 1.0) for v in kept.values())
    return {
        "top1_similarity": {
            "min": round(tops[0], 4),
            "median": round(tops[len(tops) // 2], 4),
            "max": round(tops[-1], 4),
        },
        "share_kept_by_threshold": kept,
        "similarity_thresholds_inert": inert,
    }
