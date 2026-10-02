# Databricks notebook source
# MAGIC %md
# MAGIC ## 07d_adaptive_rerank_live: Phase 9E LIVE latency confirmation (diagnostic)
# MAGIC Thin entry point. Logic: `worldbank_copilot.retrieval.adaptive_eval.run_live`.
# MAGIC
# MAGIC Run ONLY after the frontier has been computed locally and
# MAGIC `evaluation/adaptive_rerank_9e_live_selection.json` has been committed.
# MAGIC
# MAGIC Exactly two points, over the real production path
# MAGIC `Retriever.first_stage(k50) -> policy.decide(stage) -> Retriever.finish(...)`:
# MAGIC P0@50 (no rerank) and the adaptive point chosen by the frozen latency-validation rule
# MAGIC (rerank rate closest to 0.5 - NOT a production winner). The dense list is captured from
# MAGIC the first stage itself (no second index query). 1 warm-up + 2 timed passes.
# MAGIC Read-only: no index, embedding, table or configuration is created or changed.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-databricks.txt -r ../requirements-retrieval.txt -r ../requirements-reranker.txt -c ../constraints-databricks.txt

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821 (Databricks built-in)

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

# Step 0: dependency health (same environment class as 07c).
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

# Step 1: lock + committed frontier + committed live selection + collection artifact
# (no query is made before this passes). The values the live run is AUTHORIZED for:
AUTHORIZED = {
    "lock_sha256": "480d0a1ec02e6e54aeb7c4a86d1122c002e1b0607d76ad74a9571a7d231b591e",
    "policy_definitions_sha256": "dafc0bf5c80acf2491099eb95991def149df51211c29eae7ccecafafd6eb1cff",
    "collection_artifact_sha256": "8b6763e349215899b686fff45f49eb44483f65490167b6649e88014599a63d70",
    "point": "P3(0.4)",
}
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from worldbank_copilot.retrieval import evaluation as ev  # noqa: E402
from worldbank_copilot.retrieval import pipeline as rp  # noqa: E402
from worldbank_copilot.retrieval.adaptive_eval import (  # noqa: E402
    build_lock,
    file_sha256,
    identity_failures,
    live_preflight_failures,
    load_retrieval_production,
    lock_sha256,
    read_json_artifact,
    runtime_index_failures,
)
from worldbank_copilot.retrieval.adaptive_rerank import load_adaptive_config  # noqa: E402
from worldbank_copilot.retrieval.config import load_retrieval_settings  # noqa: E402
from worldbank_copilot.retrieval.embeddings import embedding_provider  # noqa: E402

repo = settings.repo_root  # noqa: F821
cfg = load_adaptive_config(settings.config_dir)  # noqa: F821
lock = json.loads((repo / "evaluation" / "adaptive_rerank_9e_lock.json").read_text("utf-8"))
frontier = read_json_artifact(repo / "evaluation" / "adaptive_rerank_9e.json")
selection = read_json_artifact(repo / "evaluation" / "adaptive_rerank_9e_live_selection.json")
out_dir = Path(settings.artifact_volume_path) / cfg.artifact_dir  # noqa: F821
collection_path = out_dir / f"collection__{cfg.run_ids['collection']}.json"
failures = live_preflight_failures(
    lock=lock,
    recomputed_lock=build_lock(cfg, repo, load_retrieval_production(settings.config_dir)),  # noqa: F821
    frontier=frontier,
    selection=selection,
    collection_sha256=file_sha256(collection_path),
    authorized=AUTHORIZED,
)
print(
    {
        "lock": lock_sha256(lock),
        "collection": file_sha256(collection_path),
        "selection": selection.get("point"),
        "failures": failures,
    }
)
if failures:
    raise RuntimeError(f"STOP: live pre-run integrity failed: {failures}")
live_path = out_dir / f"live__{cfg.run_ids['live']}.json"
if cfg.run_ids["live"] != "9e1-live" or live_path.exists():
    raise RuntimeError(
        f"STOP: {live_path.name} is not run 9e1-live or already exists - never overwritten"
    )
print("latency validation point (NOT a production winner):", selection)

# COMMAND ----------

# Step 2: frozen-data identity from the index's own description (read-only; no query yet).
from worldbank_copilot.retrieval.retriever import ChunkStore  # noqa: E402

require_state("selection", "lock", step="Step 1")  # noqa: F821
rs = load_retrieval_settings(settings.config_dir)  # noqa: F821
if (
    rs.retrieval.reranker.model,
    rs.retrieval.reranker.max_length,
    rs.retrieval.reranker.batch_size,
) != (
    cfg.stack["reranker_model"],
    cfg.stack["reranker_max_length"],
    cfg.stack["reranker_batch_size"],
):
    raise RuntimeError("STOP: CrossEncoder configuration differs from the locked identity")
provider = embedding_provider(rs.embeddings)
store = ChunkStore.from_table(spark, rp.names(settings, rs)["chunks"])  # noqa: F821
dense_index = rp.open_vector_index(settings, rs)  # noqa: F821
questions = ev.load_questions(repo / cfg.questions_file)
description = dense_index.describe()
index_rows = (description.get("status") or {}).get("indexed_row_count")
runtime_endpoint = description.get("endpoint_name")
index_name = rp.names(settings, rs)["index_name"]  # noqa: F821
runtime_index_name = description.get("name")
failures = identity_failures(
    cfg, len(store.rows), index_rows, questions, index_name, endpoint=runtime_endpoint
) + runtime_index_failures(runtime_index_name, index_name)
print(
    {
        "index": index_name,
        "runtime_index": runtime_index_name,
        "index_rows": index_rows,
        "endpoint": runtime_endpoint,
        "corpus_rows": len(store.rows),
        "questions": len(questions),
    }
)
if failures:
    raise RuntimeError(f"STOP: frozen-data identity failed: {failures}")

# COMMAND ----------

# Step 3: live end-to-end runs over the real production path (exactly two points).
import platform  # noqa: E402

import sentence_transformers  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

from worldbank_copilot.retrieval.adaptive_eval import run_live  # noqa: E402
from worldbank_copilot.retrieval.adaptive_rerank import (  # noqa: E402
    AdaptivePolicy,
    LivePolicy,
    RecordingDense,
)
from worldbank_copilot.retrieval.rerank import CrossEncoderReranker  # noqa: E402
from worldbank_copilot.retrieval.rerank_policy import NeverRerank  # noqa: E402
from worldbank_copilot.retrieval.retriever import Retriever  # noqa: E402

require_state("store", "dense_index", "questions", step="Step 2")  # noqa: F821
recorder = RecordingDense(dense_index)
retriever = Retriever(store, rs, registry.project_ids, embeddings=provider, dense=recorder)  # noqa: F821
adaptive = LivePolicy(
    AdaptivePolicy(selection["policy_id"], selection["threshold"]),
    retriever,
    recorder,
    cfg.features,
)
live = run_live(
    retriever,
    recorder,
    questions,
    [("P0@50", NeverRerank()), (selection["point"], adaptive)],
    CrossEncoderReranker(rs.retrieval.reranker),
    cfg,
    lock_sha256=lock_sha256(lock),
    selection=selection,
)
live["environment"] = {
    "python": platform.python_version(),
    "processor": platform.processor(),
    "machine": platform.machine(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "transformers": transformers.__version__,
    "sentence_transformers": sentence_transformers.__version__,
    "ce_model": cfg.stack["reranker_model"],
}
live["policy_definitions_sha256"] = lock["policy_definitions_sha256"]
live["index"] = {
    "endpoint": runtime_endpoint,
    "name": index_name,
    "runtime_name": runtime_index_name,
    "row_count": index_rows,
    "corpus_rows": len(store.rows),
}
live["collection_artifact_sha256"] = file_sha256(collection_path)

# COMMAND ----------

# Step 4: write the run-specific artifact, then STOP.
require_state("live", step="Step 3")  # noqa: F821
out_dir.mkdir(parents=True, exist_ok=True)
live_path.write_text(json.dumps(live, indent=1), encoding="utf-8")
print("written:", live_path)
print(
    {
        "rows": len(live["rows"]),
        "statuses": sorted({r["status"] for r in live["rows"]}),
        "dense_calls_per_row": sorted({r["dense_calls"] for r in live["rows"]}),
    }
)
print(
    "STOP - run scripts/adaptive_rerank_9e_live.py locally for composed vs live latency. "
    "No policy is promoted and production is unchanged."
)
