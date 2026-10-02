# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC ## 08e_candidate_c_dev: Candidate C DEV predictions (Phase 9D, DEV split ONLY)
# MAGIC Thin entry point. Logic: `worldbank_copilot.routing.bounded_classifier_dev` /
# MAGIC `bounded_semantic`. **Compute:** ordinary **Serverless (CPU)**.
# MAGIC
# MAGIC 1. Frozen-input guards and protocol lock: C1 / C2 / repeat populations are re-derived
# MAGIC    mechanically (frozen 9B.2 RoutingService, no data reads) and the recomputed lock must
# MAGIC    equal `evaluation/candidate_c_dev_lock.json`, else STOP (INVALID). Before any call.
# MAGIC 2. The run-specific artifact must not exist yet (never overwritten). Before any call.
# MAGIC 3. The locked 44-call plan with `databricks-gpt-oss-20b` (frozen contract, reasoning
# MAGIC    effort low): 60 s quiet, then strictly sequential calls with a fixed 5.0 s END->START
# MAGIC    gap. No concurrency, burst, retries or backoff; a schedule the pacer cannot honour
# MAGIC    stops before the call (INVALID). Retry-After header/body recorded separately.
# MAGIC 4. Writes the prediction artifact (no question or model text) and STOPs.
# MAGIC
# MAGIC The outcome (C1, hybrid, repeatability, C2 diagnostics) is computed LOCALLY by
# MAGIC `scripts/candidate_c_dev_9d.py`, which replays the recorded predictions through the
# MAGIC real RoutingService. TEST is never evaluated; the ambiguity probe is never opened.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-databricks.txt -c ../constraints-databricks.txt

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821 (Databricks built-in)

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

# Step 1: frozen-input guards + protocol lock (no model call, no data read, DEV only).
import json  # noqa: E402

from worldbank_copilot.ingestion.documents import load_document_manifest  # noqa: E402
from worldbank_copilot.routing.bounded_classifier import (  # noqa: E402
    contract_sha256,
    load_bounded_classifier_config,
)
from worldbank_copilot.routing.bounded_classifier_dev import (  # noqa: E402
    derive_lock,
    lock_sha256,
)
from worldbank_copilot.routing.config import load_routing_config  # noqa: E402
from worldbank_copilot.routing.entities import EntityIndex  # noqa: E402
from worldbank_copilot.routing.evaluation import load_dataset  # noqa: E402
from worldbank_copilot.routing.semantic_eval import assert_split_allowed  # noqa: E402

EXPECTED_CONTRACT = "c64561c25e0e417328b3a525c8b225a6b2f9e465b7c9d177a9128d08f013fe8b"
assert_split_allowed("dev")
repo = settings.repo_root  # noqa: F821
bc = load_bounded_classifier_config(settings.config_dir)  # noqa: F821
routing = load_routing_config(settings.config_dir)  # noqa: F821
manifest = load_document_manifest(settings.config_dir / "document_manifest.yaml")  # noqa: F821
index = EntityIndex.build(registry, manifest, routing)  # noqa: F821
all_projects = tuple(registry.project_ids)  # noqa: F821
if contract_sha256(bc) != EXPECTED_CONTRACT:
    raise RuntimeError("STOP (INVALID): model-facing contract differs from the frozen hash")
protocol = derive_lock(
    bc, routing, index, load_dataset(repo / "evaluation" / "routing_cases.yaml"), repo, all_projects
)
lock = json.loads((repo / "evaluation" / "candidate_c_dev_lock.json").read_text("utf-8"))
if protocol.lock != lock:
    raise RuntimeError("STOP (INVALID): recomputed protocol lock differs from the committed lock")
print("lock", lock_sha256(lock), "| populations", json.dumps(lock["populations"]))
print("calls", lock["dev_scheduled_calls"], "| C1 calls", lock["c1_calls"])

# COMMAND ----------

# Step 2: the run-specific artifact must not exist (checked BEFORE any call).
from pathlib import Path  # noqa: E402

from worldbank_copilot.routing.bounded_classifier_dev import artifact_name  # noqa: E402

require_state("lock", "protocol", step="Step 1")  # noqa: F821
out_dir = Path(settings.artifact_volume_path) / "bounded_classifier_dev"  # noqa: F821
out_dir.mkdir(parents=True, exist_ok=True)
artifact_path = out_dir / artifact_name(lock["run_id"])
if artifact_path.exists():
    raise RuntimeError(f"STOP: {artifact_path.name} already exists - never overwritten")
print("artifact:", artifact_path)

# COMMAND ----------

# Step 3: the locked, paced 44-call DEV plan (the only network stage).
from databricks.sdk import WorkspaceClient  # noqa: E402

from worldbank_copilot.routing.bounded_classifier_dev import (  # noqa: E402
    request_contexts,
    run_dev_predictions,
)
from worldbank_copilot.routing.bounded_classifier_probe import (  # noqa: E402
    DatabricksChatTransport,
)
from worldbank_copilot.routing.bounded_semantic import BoundedSemanticClassifier  # noqa: E402
from worldbank_copilot.routing.semantic import route_map  # noqa: E402

require_state("artifact_path", step="Step 2")  # noqa: F821
contexts = request_contexts(routing, index, protocol.cases, lock["populations"], all_projects)
classifier = BoundedSemanticClassifier(
    bc,
    DatabricksChatTransport(bc.endpoint.preferred, WorkspaceClient().config),
    route_map(routing.requirements),
)
artifact = run_dev_predictions(bc, lock, contexts, classifier)
artifact_path.write_text(json.dumps(artifact, indent=1), encoding="utf-8")

kinds = {}
for call in artifact["calls"]:
    kinds[call["failure_kind"]] = kinds.get(call["failure_kind"], 0) + 1
print(json.dumps({"schedule": artifact["schedule"], "failure_kinds": kinds}, indent=1))
print(f"written: {artifact_path}")
if not artifact["schedule"]["valid"]:
    raise RuntimeError(f"STOP (INVALID): schedule {artifact['schedule']['failures']}")
print(
    "STOP - download the artifact and run scripts/candidate_c_dev_9d.py locally for the "
    "hybrid replay and the outcome. No ambiguity probe, no TEST."
)
