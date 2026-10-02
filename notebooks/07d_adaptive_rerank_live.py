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

# Step 1: lock + committed live selection (before any query).
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from worldbank_copilot.retrieval import evaluation as ev  # noqa: E402
from worldbank_copilot.retrieval import pipeline as rp  # noqa: E402
from worldbank_copilot.retrieval.adaptive_eval import (  # noqa: E402
    build_lock,
    load_retrieval_production,
    lock_sha256,
)
from worldbank_copilot.retrieval.adaptive_rerank import load_adaptive_config  # noqa: E402
from worldbank_copilot.retrieval.config import load_retrieval_settings  # noqa: E402
from worldbank_copilot.retrieval.embeddings import embedding_provider  # noqa: E402

repo = settings.repo_root  # noqa: F821
cfg = load_adaptive_config(settings.config_dir)  # noqa: F821
lock = json.loads((repo / "evaluation" / "adaptive_rerank_9e_lock.json").read_text("utf-8"))
if build_lock(cfg, repo, load_retrieval_production(settings.config_dir)) != lock:  # noqa: F821
    raise RuntimeError("STOP: protocol lock differs from the committed lock")
selection = json.loads(
    (repo / "evaluation" / "adaptive_rerank_9e_live_selection.json").read_text("utf-8")
)
if selection.get("lock_sha256") != lock_sha256(lock) or selection.get("policy_id") not in (
    "P2", "P3", "P4", "P5", "P6"
):
    raise RuntimeError("STOP: live selection is missing, for another lock, or not a P2-P6 point")
out_dir = Path(settings.artifact_volume_path) / cfg.artifact_dir  # noqa: F821
live_path = out_dir / f"live__{cfg.run_ids['live']}.json"
if live_path.exists():
    raise RuntimeError(f"STOP: {live_path.name} already exists - never overwritten")
print("latency validation point (NOT a production winner):", selection)

# COMMAND ----------

# Step 2: live end-to-end runs over the real production path.
from worldbank_copilot.retrieval.adaptive_eval import run_live  # noqa: E402
from worldbank_copilot.retrieval.adaptive_rerank import (  # noqa: E402
    AdaptivePolicy,
    LivePolicy,
    RecordingDense,
)
from worldbank_copilot.retrieval.rerank import CrossEncoderReranker  # noqa: E402
from worldbank_copilot.retrieval.rerank_policy import NeverRerank  # noqa: E402
from worldbank_copilot.retrieval.retriever import ChunkStore, Retriever  # noqa: E402

require_state("selection", "lock", step="Step 1")  # noqa: F821
rs = load_retrieval_settings(settings.config_dir)  # noqa: F821
provider = embedding_provider(rs.embeddings)
store = ChunkStore.from_table(spark, rp.names(settings, rs)["chunks"])  # noqa: F821
recorder = RecordingDense(rp.open_vector_index(settings, rs))  # noqa: F821
retriever = Retriever(store, rs, registry.project_ids, embeddings=provider, dense=recorder)  # noqa: F821
questions = ev.load_questions(repo / cfg.questions_file)
adaptive = LivePolicy(
    AdaptivePolicy(selection["policy_id"], selection["threshold"]), retriever, recorder,
    cfg.features,
)
live = run_live(
    retriever, recorder, questions, [("P0@50", NeverRerank()), (selection["point"], adaptive)],
    CrossEncoderReranker(rs.retrieval.reranker), cfg,
    lock_sha256=lock_sha256(lock), selection=selection,
)
out_dir.mkdir(parents=True, exist_ok=True)
live_path.write_text(json.dumps(live, indent=1), encoding="utf-8")
print("written:", live_path)
print("STOP - run scripts/adaptive_rerank_9e_live.py locally for composed vs live latency.")
