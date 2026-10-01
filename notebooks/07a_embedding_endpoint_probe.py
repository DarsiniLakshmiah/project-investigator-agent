# Databricks notebook source
# MAGIC %md
# MAGIC ## 07a_embedding_endpoint_probe: Load probe for a candidate embedding endpoint (Phase 8)
# MAGIC Read-only experiment. It sends at most 3 requests (1 input, then two small batches with
# MAGIC a controlled pause, one attempt each, no retries) to the endpoint configured under
# MAGIC `probe` in `configs/retrieval/embedding_candidates.yaml`
# MAGIC (`databricks-qwen3-embedding-0-6b`).
# MAGIC
# MAGIC It writes nothing: no embedding cache, no index source, no Vector Search index, and
# MAGIC it does not change the production embedding configuration (`embeddings.yaml`).

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

# Step 1: representative inputs (read-only): 9 real retrieval texts in a fixed order.
from worldbank_copilot.retrieval import pipeline as rp  # noqa: E402
from worldbank_copilot.retrieval.config import load_retrieval_settings  # noqa: E402
from worldbank_copilot.retrieval.endpoint_probe import (  # noqa: E402
    load_probe_config,
    result_schema,
    run_probe,
)

rs = load_retrieval_settings(settings.config_dir)  # noqa: F821
probe_config = load_probe_config(settings.config_dir)  # noqa: F821
samples = [
    r["search_text"]
    for r in spark.sql(  # noqa: F821
        f"SELECT search_text FROM {rp.names(settings, rs)['chunks']} "  # noqa: F821
        "WHERE chunk_strategy = 'structure' AND chunk_role = 'RETRIEVAL' "
        f"ORDER BY text_sha256 LIMIT {1 + 2 * probe_config.batch_size}"
    ).collect()
]
print(probe_config)
print(f"{len(samples)} sample texts, {min(map(len, samples))}-{max(map(len, samples))} chars")

# COMMAND ----------

# Step 2: load probe (1 input -> batch -> pause -> batch). No retries, nothing persisted.
report = run_probe(probe_config, samples, log=print)
print(report.format())
# Explicit schema: NULL-only columns (error_code, retry_after, error) are typed.
display(spark.createDataFrame(report.records(), schema=result_schema()))  # noqa: F821
print("VERDICT:", report.verdict, "| dimension:", report.dimension)
