# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC ## 08_semantic_routing_dev: Phase 9D semantic fallback — DEVELOPMENT split only
# MAGIC Thin entry point. Logic: `worldbank_copilot.routing.semantic` / `semantic_eval`.
# MAGIC
# MAGIC Candidate B (nearest-neighbour) with the validated Qwen embedding endpoint
# MAGIC (configs/retrieval/embeddings.yaml) on the 29 frozen DEVELOPMENT cases, using the
# MAGIC pre-registered protocol in configs/routing/semantic.yaml (leave-one-family-out,
# MAGIC 108-configuration grid, acceptance rule fixed before this run).
# MAGIC
# MAGIC * Step 1 GUARDS: frozen 9C dataset hash, 29/51 split, REVIEWED labels, reviewed
# MAGIC   aliases, router 9B.2, no TEST evaluation recorded, DEV-only inputs. Any failure STOPs.
# MAGIC * Step 2 PROBE: endpoint availability, dimension, latency. Mismatch STOPs.
# MAGIC * Step 3 DIAGNOSTIC: if the registered similarity thresholds are inert on Qwen
# MAGIC   cosines, STOP (the grid is never changed after seeing results).
# MAGIC * Step 4 experiment (Qwen + lexical for comparison), Step 5 MLflow.
# MAGIC
# MAGIC Embeds only the ~22 short development questions (+1 probe text). Routing vectors use
# MAGIC their own cache file (namespace `routing-question`) in the artifact Volume. The
# MAGIC document-chunk embeddings, the AI Search endpoint and every index are never touched.
# MAGIC The frozen TEST split is never evaluated. Writes nothing to Delta.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-databricks.txt -r ../requirements-retrieval.txt -c ../constraints-databricks.txt

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821 (Databricks built-in)

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

# Step 0: dependency health.
from worldbank_copilot.common.dependency_health import check_environment  # noqa: E402

health = check_environment(
    settings.repo_root,  # noqa: F821
    ["requirements-databricks.txt", "requirements-retrieval.txt"],
    settings.config_dir,  # noqa: F821
)
print(health.format())
if not health.ok:
    raise RuntimeError("STOP: notebook environment is inconsistent (see dependency health)")

# COMMAND ----------

# Step 1: hard guards against the frozen 9C ground truth (STOP on any failure).
import yaml  # noqa: E402

from worldbank_copilot.routing.config import load_routing_config  # noqa: E402
from worldbank_copilot.routing.evaluation import load_dataset  # noqa: E402
from worldbank_copilot.routing.semantic_eval import (  # noqa: E402
    assert_split_allowed,
    development_guards,
    require_guards,
    store_for,
)

assert_split_allowed("dev")
repo = settings.repo_root  # noqa: F821
routing = load_routing_config(settings.config_dir)  # noqa: F821
dataset = load_dataset(repo / "evaluation" / "routing_cases.yaml")
decision_9d = yaml.safe_load((repo / "evaluation" / "semantic_config_9d.yaml").read_text("utf-8"))
guards = development_guards(repo, dataset, routing, store_for(dataset), decision_9d)
for g in guards:
    print(f"[{'PASS' if g['passed'] else 'FAIL'}] {g['guard']}")
require_guards(guards)

# COMMAND ----------

# Step 2: Qwen endpoint probe (STOP if unavailable or not the validated capability).
# One real, uncached call through the SAME provider instance Candidate B embeds with;
# the dimension is measured from the returned vector (len), and compared with
# EmbeddingConfig.expected_dimension (validated Phase 8 value: 1024).
from pathlib import Path  # noqa: E402

from worldbank_copilot.retrieval.config import load_retrieval_settings  # noqa: E402
from worldbank_copilot.retrieval.embeddings import embedding_provider  # noqa: E402
from worldbank_copilot.routing.semantic import ProviderEmbedder  # noqa: E402
from worldbank_copilot.routing.semantic_eval import probe_embedder  # noqa: E402

rs = load_retrieval_settings(settings.config_dir)  # noqa: F821
cache_dir = Path(settings.artifact_volume_path) / "routing_question_embeddings"  # noqa: F821
cache_dir.mkdir(parents=True, exist_ok=True)
provider = embedding_provider(rs.embeddings)
qwen_embedder = ProviderEmbedder(provider, cache_path=cache_dir / f"{provider.model}.json")
probe = probe_embedder(qwen_embedder.provider, rs.embeddings.expected_dimension)
print(probe)
if not probe["available"] or not probe["dimension_matches"]:
    raise RuntimeError(f"STOP: embedding endpoint differs from the validated capability: {probe}")

# COMMAND ----------

# Step 3: similarity-threshold diagnostic on Qwen cosines (STOP if the grid is inert).
from worldbank_copilot.routing.semantic import (  # noqa: E402
    KnnConfig,
    KnnSemanticClassifier,
    route_map,
)
from worldbank_copilot.routing.semantic_eval import (  # noqa: E402
    load_protocol,
    similarity_diagnostic,
)

protocol = load_protocol(settings.config_dir)  # noqa: F821
routes = route_map(routing.requirements)
probe_clf = KnnSemanticClassifier(qwen_embedder, store_for(dataset), KnnConfig(), routes)
diagnostic = similarity_diagnostic(probe_clf, protocol)
print(diagnostic)
if diagnostic["similarity_thresholds_inert"]:
    raise RuntimeError(
        "STOP: the pre-registered similarity thresholds do not discriminate on Qwen cosines; "
        "the protocol is not changed after seeing results - report before any modification."
    )

# COMMAND ----------

# Step 4: development experiment - Qwen (and lexical for the side-by-side comparison).
import json  # noqa: E402
import time  # noqa: E402
from datetime import UTC, datetime  # noqa: E402

from worldbank_copilot.routing.semantic import LexicalEmbedder  # noqa: E402
from worldbank_copilot.routing.semantic_eval import canonical_sha256, run_development  # noqa: E402

baseline = {
    r["case_id"]: r
    for r in yaml.safe_load(
        (repo / "evaluation" / "routing_baseline_9B.2.yaml").read_text("utf-8")
    )["results"]
}
started_at = datetime.now(UTC).isoformat()
t0 = time.perf_counter()
qwen = run_development(dataset, routes, protocol, qwen_embedder, baseline)
qwen["wall_clock_s"] = round(time.perf_counter() - t0, 1)
qwen["endpoint_call_latency_ms"] = qwen_embedder.call_latency_ms
lexical = run_development(dataset, routes, protocol, LexicalEmbedder(), baseline)
result = {
    "phase": "9D",
    "split": "dev",
    "test_evaluated": False,
    "started_at": started_at,
    "dataset_sha256_lf": canonical_sha256(repo / "evaluation" / "routing_cases.yaml"),
    "router_version": routing.settings.router_version,
    "guards": guards,
    "probe": probe,
    "similarity_diagnostic": diagnostic,
    "protocol": protocol.model_dump(),
    "runs": {"B-qwen": qwen, "B-lexical": lexical},
    "selected_for_review": (
        {"candidate": "B-qwen", "model": provider.model, "config": qwen["selected"]["config"],
         "metrics": qwen["selected"]["metrics"], "status": "SELECTED_FOR_REVIEW",
         "test_authorised": False}
        if qwen["selected"] else
        {"candidate": "A (rules-only)", "status": "B-qwen REJECTED by the pre-registered rule",
         "test_authorised": False}
    ),
}
out = cache_dir / "semantic_dev_9d_databricks.json"
out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")

for name, run in result["runs"].items():
    print(name, "| majority baseline", run["majority_route_baseline"], "| selected:",
          run["selected"] and run["selected"]["name"])
    print("   rules-only dev:", run["rules_only_dev"]["route_accuracy"],
          "| hypothetical fallback dev:", run["hypothetical_fallback_dev"]["route_accuracy"],
          "semantic correct", run["hypothetical_fallback_dev"]["semantic_correct"])
print("decision:", result["selected_for_review"])
display(spark.createDataFrame([  # noqa: F821
    {"run": name, "config": r["name"], "answered": r["metrics"]["answered"],
     "coverage": r["metrics"]["coverage"], "route_precision": r["metrics"]["route_precision"],
     "intent_precision": r["metrics"]["intent_precision"],
     "route_accuracy_all": r["metrics"]["route_accuracy_all"],
     "intent_accuracy_all": r["metrics"]["intent_accuracy_all"]}
    for name, run in result["runs"].items() for r in run["experiments"]
]))
display(spark.createDataFrame(qwen["coverage_curve"]))  # noqa: F821
display(spark.createDataFrame([  # noqa: F821
    {k: (str(v) if k == "confidence" else v) for k, v in p.items()} for p in qwen["base_lofo"]
]))
print("written:", out)

# COMMAND ----------

# Step 5: MLflow (case ids and aggregate metrics only; no question text).
import mlflow  # noqa: E402

with mlflow.start_run(run_name="9D-dev-candidate-B"):
    mlflow.log_params({
        "phase": "9D", "split": "dev", "candidate": "B", "test_evaluated": False,
        "model": provider.model, "dataset_sha256_lf": result["dataset_sha256_lf"],
        "router_version": result["router_version"], "started_at": started_at,
        "configurations": len(qwen["experiments"]),
        "selected": qwen["selected"] and qwen["selected"]["name"],
    })
    best = qwen["selected"]["metrics"] if qwen["selected"] else {}
    mlflow.log_metrics({
        "qwen_majority_route_accuracy": qwen["majority_route_baseline"]["accuracy"],
        "qwen_classifier_latency_p50_ms": qwen["classifier_latency_ms"]["p50"],
        "qwen_classifier_latency_p95_ms": qwen["classifier_latency_ms"]["p95"],
        "probe_latency_ms": probe["latency_ms"],
        **{f"selected_{k}": v for k, v in best.items() if isinstance(v, int | float)},
    })
    mlflow.log_artifact(str(out))
    print("MLflow run:", mlflow.active_run().info.run_id)
