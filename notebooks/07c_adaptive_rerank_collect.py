# Databricks notebook source
# MAGIC %md
# MAGIC ## 07c_adaptive_rerank_collect: Phase 9E adaptive-rerank DATA COLLECTION (diagnostic)
# MAGIC Thin entry point. Logic: `worldbank_copilot.retrieval.adaptive_eval.collect`.
# MAGIC
# MAGIC Environment: serverless **environment version 6, ML base** (as notebook 07b).
# MAGIC
# MAGIC Reads the existing governed corpus and the existing AI Search index (read-only).
# MAGIC Creates NO index, NO embedding, NO table; changes NO configuration; chooses NO policy.
# MAGIC
# MAGIC Steps: 0 dependency health → 1 protocol lock + identity (STOP on any mismatch, before
# MAGIC any query) → 2 the frozen collection protocol (1 warm-up + 2 timed passes over the 49
# MAGIC frozen questions: lexical / dense / fused k50, P0 k10, CrossEncoder scores for every
# MAGIC fused k50 candidate, TriggerFeatures, timings) → 3 run-specific JSON artifact to the
# MAGIC Volume (chunk ids only; no chunk or question text). Then STOP: the frontier is computed
# MAGIC locally by `scripts/adaptive_rerank_9e.py`.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-databricks.txt -r ../requirements-retrieval.txt -r ../requirements-reranker.txt -c ../constraints-databricks.txt

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821 (Databricks built-in)

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

# Step 0: dependency health.
from worldbank_copilot.common.dependency_health import check_environment  # noqa: E402

health = check_environment(
    settings.repo_root,  # noqa: F821
    ["requirements-databricks.txt", "requirements-retrieval.txt", "requirements-reranker.txt"],
    settings.config_dir,  # noqa: F821
)
print(health.format())
if not health.ok:
    raise RuntimeError("STOP: notebook environment is inconsistent (see dependency health)")

# COMMAND ----------

# Step 1: protocol lock + frozen-data identity (no query is made before this passes).
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from worldbank_copilot.retrieval import evaluation as ev  # noqa: E402
from worldbank_copilot.retrieval import pipeline as rp  # noqa: E402
from worldbank_copilot.retrieval.adaptive_eval import (  # noqa: E402
    build_lock,
    identity_failures,
    load_retrieval_production,
    lock_sha256,
)
from worldbank_copilot.retrieval.adaptive_rerank import load_adaptive_config  # noqa: E402
from worldbank_copilot.retrieval.config import load_retrieval_settings  # noqa: E402
from worldbank_copilot.retrieval.embeddings import embedding_provider  # noqa: E402

repo = settings.repo_root  # noqa: F821
cfg = load_adaptive_config(settings.config_dir)  # noqa: F821
lock = json.loads((repo / "evaluation" / "adaptive_rerank_9e_lock.json").read_text("utf-8"))
recomputed = build_lock(cfg, repo, load_retrieval_production(settings.config_dir))  # noqa: F821
if recomputed != lock or not lock["production_null"]:
    raise RuntimeError("STOP: protocol lock differs from the committed lock (or production set)")
out_dir = Path(settings.artifact_volume_path) / cfg.artifact_dir  # noqa: F821
artifact_path = out_dir / f"collection__{cfg.run_ids['collection']}.json"
if artifact_path.exists():
    raise RuntimeError(f"STOP: {artifact_path.name} already exists - never overwritten")

rs = load_retrieval_settings(settings.config_dir)  # noqa: F821
if (rs.retrieval.reranker.model, rs.retrieval.reranker.max_length,
        rs.retrieval.reranker.batch_size) != (cfg.stack["reranker_model"],
                                              cfg.stack["reranker_max_length"],
                                              cfg.stack["reranker_batch_size"]):
    raise RuntimeError("STOP: CrossEncoder configuration differs from the locked identity")
provider = embedding_provider(rs.embeddings)
retriever = rp.open_retriever(spark, settings, rs, registry.project_ids, provider)  # noqa: F821
questions = ev.load_questions(repo / cfg.questions_file)
# Runtime identity from the index's own description (read-only; no retrieval query yet).
description = retriever.dense.describe()
index_rows = (description.get("status") or {}).get("indexed_row_count")
runtime_endpoint = description.get("endpoint_name")
index_name = rp.names(settings, rs)["index_name"]  # noqa: F821
failures = identity_failures(cfg, len(retriever.store.rows), index_rows, questions, index_name,
                             endpoint=runtime_endpoint)
print({"lock": lock_sha256(lock), "index": index_name, "index_rows": index_rows,
       "endpoint": runtime_endpoint, "corpus_rows": len(retriever.store.rows),
       "questions": len(questions)})
if failures:
    raise RuntimeError(f"STOP: frozen-data identity failed: {failures}")

# COMMAND ----------

# Step 2: the frozen collection protocol (1 warm-up + 2 timed passes).
import platform  # noqa: E402

import sentence_transformers  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

from worldbank_copilot.retrieval.adaptive_eval import collect  # noqa: E402
from worldbank_copilot.retrieval.rerank import CrossEncoderReranker  # noqa: E402

require_state("retriever", "questions", "lock", step="Step 1")  # noqa: F821
environment = {
    "python": platform.python_version(), "processor": platform.processor(),
    "machine": platform.machine(), "torch": torch.__version__,
    "transformers": transformers.__version__,
    "sentence_transformers": sentence_transformers.__version__,
    "ce_model": cfg.stack["reranker_model"],
}
artifact = collect(
    retriever, questions, CrossEncoderReranker(rs.retrieval.reranker), cfg,
    lock_sha256=lock_sha256(lock), environment=environment,
    index={"endpoint": runtime_endpoint, "name": index_name,
           "row_count": index_rows, "corpus_rows": len(retriever.store.rows)},
)

# COMMAND ----------

# Step 3: write the run-specific artifact, then STOP on any integrity problem.
require_state("artifact", step="Step 2")  # noqa: F821
out_dir.mkdir(parents=True, exist_ok=True)
artifact_path.write_text(json.dumps(artifact, indent=1), encoding="utf-8")
print("written:", artifact_path)
print(json.dumps({"status": artifact["status"], "problems": artifact["problems"],
                  "consistency": artifact["consistency"]["consistent"],
                  "drift": artifact["drift_check"]}, indent=1))
if artifact["status"] != "OK":
    raise RuntimeError(f"STOP: collection status {artifact['status']} - investigate; no frontier")
print("STOP - download the artifact and run scripts/adaptive_rerank_9e.py locally. "
      "No policy is selected and production is unchanged.")
