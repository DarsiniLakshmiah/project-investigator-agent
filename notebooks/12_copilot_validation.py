# Databricks notebook source
# MAGIC %md
# MAGIC ## Comparative A/B/C and final E2E contract validation
# MAGIC User-run after reviewing, committing, synchronizing and configuring the runtime.
# MAGIC A contract PASS does not establish model quality or select an architecture.
# COMMAND ----------
# MAGIC %pip install -q -r ../requirements-databricks.txt -r ../requirements-retrieval.txt -r ../requirements-reranker.txt -r ../requirements-phase10d.txt -r ../requirements-copilot-runtime.txt -c ../constraints-databricks.txt
# COMMAND ----------
dbutils.library.restartPython()
# COMMAND ----------
# MAGIC %run ./_bootstrap
# COMMAND ----------
from worldbank_copilot.validation.copilot_e2e import run_databricks

dbutils.widgets.text("commit_sha", "", "Exact reviewed full committed SHA")
dbutils.widgets.text("run_id", "", "NEW immutable 10fN or e2eN attempt ID")
artifact = run_databricks(spark, settings, commit_sha=dbutils.widgets.get("commit_sha").strip(),
                         run_id=dbutils.widgets.get("run_id").strip())
print({"preflight":artifact["preflight"],"contract_status":artifact["contract_status"],
       "acceptance":artifact["acceptance"],"quality_review":artifact["quality_review"]})
if artifact["contract_status"] != "PASS":
    raise RuntimeError("Preserve failed attempt and receipts; inspect before any new run")
