# Databricks notebook source
# MAGIC %md
# MAGIC ## User-run proposal-only Investigator capability probe
# MAGIC Synthetic empty-evidence fixtures; real model calls; NO proposed tool executes.
# MAGIC Preserve every attempt. Contract validity is not model quality acceptance.
# COMMAND ----------
# MAGIC %pip install -q -r ../requirements-databricks.txt -r ../requirements-retrieval.txt -r ../requirements-reranker.txt -r ../requirements-phase10d.txt -r ../requirements-copilot-runtime.txt -c ../constraints-databricks.txt
# COMMAND ----------
dbutils.library.restartPython()
# COMMAND ----------
# MAGIC %run ./_bootstrap
# COMMAND ----------
from worldbank_copilot.validation.copilot_e2e import run_investigator_probe

dbutils.widgets.text("commit_sha", "", "Exact reviewed full committed SHA")
dbutils.widgets.text("run_id", "", "NEW immutable 10eN ID")
artifact = run_investigator_probe(spark,settings,commit_sha=dbutils.widgets.get("commit_sha").strip(),run_id=dbutils.widgets.get("run_id").strip())
print({"contract_status":artifact["contract_status"],"acceptance":artifact["acceptance"]})
if artifact["contract_status"] != "PASS":
    raise RuntimeError("Preserve failed capability attempt and inspect its receipts")
