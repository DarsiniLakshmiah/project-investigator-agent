# Databricks notebook source
# MAGIC %md
# MAGIC ## Phase 9F-C: bounded contract acceptance (corrected attempt 9f2)
# MAGIC Run only after review and after the user commits/pushes the frozen harness.
# MAGIC Use serverless environment 6 ML, as for 07b/07d. No warm-up, tuning or suite/case retries.
# MAGIC Inherited transport retry settings remain unchanged.
# MAGIC Set commit_sha to the FULL SHA of the Git folder revision containing this harness.
# MAGIC Output: <artifact Volume>/phase9_contract/phase9_contract_validation__9f2.json.
# MAGIC The reported 9f1 SHA precheck failed before artifact reservation and before C01.
# MAGIC 9f2 records that first attempt and preserves any existing 9f1 artifact.
# MAGIC Existing output (including failed preflight/interrupted run) stops execution.
# MAGIC Execute C01-C10 once each. Any failure: preserve the artifact and STOP for diagnosis.
# MAGIC Never edit cases/expectations, tune, patch-and-rerun, or delete the first artifact.
# MAGIC No live run was performed during local implementation.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-databricks.txt -r ../requirements-retrieval.txt -r ../requirements-reranker.txt -c ../constraints-databricks.txt

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from worldbank_copilot.validation.phase9_contract import run_databricks_validation

dbutils.widgets.text("commit_sha", "", "Reviewed Git folder commit (full SHA)")  # noqa: F821
artifact = run_databricks_validation(
    spark, settings, commit_sha=dbutils.widgets.get("commit_sha").strip(), run_id="9f2"  # noqa: F821
)
print(artifact["summary"])
print("STOP - preserve the first-run artifact; review before any further action.")
if artifact["summary"]["overall_status"] != "PASS":
    raise RuntimeError("9F-C FAIL: see preserved artifact; do not rerun")
