# Databricks notebook source
# MAGIC %md
# MAGIC ## Phase 10D bounded synthesis/critic capability validation
# MAGIC User-run only after reviewing, committing and synchronizing the implementation.
# MAGIC Synthetic fixtures only; no retrieval, SQL, Investigator or repair loop.
# MAGIC No model is promoted by this run. Preserve artifacts and completion receipts.

# COMMAND ----------
# MAGIC %pip install -q -r ../requirements-databricks.txt -r ../requirements-phase10d.txt -c ../constraints-databricks.txt

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %run ./_bootstrap

# COMMAND ----------
from worldbank_copilot.validation.phase10d_models import run_databricks_validation

dbutils.widgets.text("commit_sha", "", "Exact reviewed full committed SHA")
dbutils.widgets.text("run_id", "10d1", "New immutable attempt ID")
dbutils.widgets.text("endpoints", "", "One or two approved chat endpoint names, comma separated")
artifact = run_databricks_validation(
    settings,
    commit_sha=dbutils.widgets.get("commit_sha").strip(),
    run_id=dbutils.widgets.get("run_id").strip(),
    endpoints=dbutils.widgets.get("endpoints").strip(),
)
print(artifact["summary"])
if artifact["summary"]["overall_status"] != "PASS":
    raise RuntimeError("10D capability validation failed; preserve the attempt and receipt")
