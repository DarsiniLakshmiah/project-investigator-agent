# Databricks notebook source
# ruff: noqa: E501
# MAGIC %md
# MAGIC ## Phase 10C evidence capability validation
# MAGIC USER RUN ONLY after reviewing and committing the implementation AND harness.
# MAGIC Use serverless environment 6 ML, as for 07e. Set the exact reviewed full commit SHA.
# MAGIC Cost is not an acceptance criterion; infrastructure billing is unavailable, not zero.
# MAGIC Four cases, seven logical operations, three frozen baseline document retrievals.
# MAGIC No models/agents, warm-up, tuning, case retries or manufactured failures.
# MAGIC First attempt: run_id=10c1. Preserve every failed file; later attempts need a new ID.
# MAGIC A PASS requires the final JSON AND validated SHA/byte completion receipt.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-databricks.txt -r ../requirements-retrieval.txt -r ../requirements-reranker.txt -c ../constraints-databricks.txt

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from worldbank_copilot.validation.phase10c_evidence import run_databricks_validation  # noqa: E402

dbutils.widgets.text("commit_sha", "", "Exact reviewed full committed source SHA")  # noqa: F821
dbutils.widgets.text("run_id", "10c1", "New immutable attempt ID")  # noqa: F821
artifact = run_databricks_validation(
    spark,  # noqa: F821
    settings,  # noqa: F821
    commit_sha=dbutils.widgets.get("commit_sha").strip(),  # noqa: F821
    run_id=dbutils.widgets.get("run_id").strip(),  # noqa: F821
)
print(artifact["summary"])
print("STOP - preserve the artifact and receipt for review; do not rerun this attempt.")
if artifact["summary"]["overall_status"] != "PASS":
    raise RuntimeError("10C capability validation failed; inspect preserved attempt")
