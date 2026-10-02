# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC ## 08d_bounded_classifier_capability: C_DATABRICKS_BOUNDED_CLASSIFIER capability (Phase 9D)
# MAGIC Thin entry point. Logic: `worldbank_copilot.routing.bounded_classifier` / `_probe`.
# MAGIC
# MAGIC **Compute:** ordinary **Serverless (CPU)**. No GPU, no serving endpoint is created or
# MAGIC modified; the Foundation Model API endpoint is only listed and called.
# MAGIC
# MAGIC Capability only, SYNTHETIC requests about a fictional project (no World Bank, routing
# MAGIC dataset, ambiguity-probe or TEST text):
# MAGIC 1. frozen-artefact guard (hashes only); 2. notebook-identity authentication;
# MAGIC 3. read-only discovery: is the configured preferred endpoint (currently
# MAGIC    `databricks-gpt-oss-20b`, `reasoning_effort: low`) present and ready (never substituted);
# MAGIC 4. the pre-registered probe sequence (plain call, parameter acceptance, strict-schema
# MAGIC    classification, unconstrained JSON mode, malformed fixtures, warm latency, burst) -
# MAGIC    one attempt per call, no retries; 5. result JSON to the artifact Volume. Then STOP.
# MAGIC
# MAGIC Gates: endpoint callable, required request configuration accepted, strict structured
# MAGIC output, every reply = one allowed label, fixtures fail closed, no tools in any request,
# MAGIC decisions label-only (route = requirements[intent]), operational failure rate <= 0.05
# MAGIC on required non-burst calls. Diagnostic only: optional parameters
# MAGIC (SUPPORTED/UNSUPPORTED/ERROR), latency, synthetic label agreement, the deliberate burst.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-databricks.txt -c ../constraints-databricks.txt

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821 (Databricks built-in)

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

# Step 1: frozen artefacts are unchanged (hashes only; no case text is loaded).
import json  # noqa: E402

from worldbank_copilot.routing.bounded_classifier import (  # noqa: E402
    load_bounded_classifier_config,
)
from worldbank_copilot.routing.semantic_eval import canonical_sha256  # noqa: E402
from worldbank_copilot.routing.semif_contract import load_semif_config  # noqa: E402

repo = settings.repo_root  # noqa: F821
bc = load_bounded_classifier_config(settings.config_dir)  # noqa: F821
manifest = json.loads((repo / "evaluation" / "routing_freeze_9c.json").read_text("utf-8"))
semif = load_semif_config(settings.config_dir)  # noqa: F821
frozen = {
    "routing_dataset_sha256_lf": canonical_sha256(repo / "evaluation" / "routing_cases.yaml"),
    "ambiguity_probe_sha256_lf": canonical_sha256(repo / semif.probe_set["file"]),
}
if frozen["routing_dataset_sha256_lf"] != manifest["dataset_sha256_lf"]:
    raise RuntimeError("STOP: routing dataset differs from the frozen 9C manifest")
if frozen["ambiguity_probe_sha256_lf"] != semif.probe_set["sha256_lf"]:
    raise RuntimeError("STOP: ambiguity probe differs from its frozen hash")
print(json.dumps(frozen, indent=1), "| SemIf status:", semif.status)

# COMMAND ----------

# Step 2: authentication through the notebook identity (no token is read or printed).
from databricks.sdk import WorkspaceClient  # noqa: E402

w = WorkspaceClient()
auth = {"auth_type": w.config.auth_type, "identity_resolved": False, "error": None}
try:
    auth["identity_resolved"] = bool(w.current_user.me().id)  # identity itself is not recorded
except Exception as exc:  # recorded; the endpoint call below is the decisive check
    auth["error"] = type(exc).__name__
print(auth)

# COMMAND ----------

# Step 3: read-only endpoint discovery. The preferred endpoint is never substituted.
from worldbank_copilot.routing.bounded_classifier_probe import chat_endpoints  # noqa: E402

endpoints = chat_endpoints(
    [e.as_dict() for e in w.serving_endpoints.list()], bc.endpoint.small_model_markers
)
preferred = next((e for e in endpoints if e["name"] == bc.endpoint.preferred), None)
availability = {
    "preferred": bc.endpoint.preferred,
    "listed": preferred is not None,
    "ready": preferred and preferred["ready"],
    "foundation_models": preferred and preferred["foundation_models"],
    "small_model_alternatives": [e["name"] for e in endpoints if e["small_model_candidate"]],
    "chat_endpoints_listed": len(endpoints),
}
print(json.dumps(availability, indent=1))

# COMMAND ----------

# Step 4: pre-registered synthetic probe (one attempt per call, no retries).
from pathlib import Path  # noqa: E402

from worldbank_copilot.routing.bounded_classifier_probe import (  # noqa: E402
    DatabricksChatTransport,
    artifact_name,
    run_capability_probe,
)
from worldbank_copilot.routing.config import load_routing_config  # noqa: E402
from worldbank_copilot.routing.semantic import route_map  # noqa: E402

require_state("availability", "auth", "frozen", step="Steps 1-3")  # noqa: F821
route_of = route_map(load_routing_config(settings.config_dir).requirements)  # noqa: F821
report = run_capability_probe(
    bc, DatabricksChatTransport(bc.endpoint.preferred, w.config), route_of
)
report |= {"availability": availability, "authentication": auth, "frozen_hashes": frozen}

out_dir = Path(settings.artifact_volume_path) / "bounded_classifier_capability"  # noqa: F821
out_dir.mkdir(parents=True, exist_ok=True)
artifact = out_dir / artifact_name(bc.endpoint.preferred)  # endpoint-specific
if artifact.exists():  # earlier endpoint results (e.g. the nano run) are never overwritten
    raise RuntimeError(f"STOP: {artifact.name} already exists - results are never overwritten")
artifact.write_text(json.dumps(report, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=1))
print(f"\nwritten: {artifact}")
print(
    ("CAPABILITY GATES PASSED" if report["passed"] else "CAPABILITY GATES FAILED")
    + ". STOP - report before any DEV, C1, C2, probe or TEST prediction."
)
