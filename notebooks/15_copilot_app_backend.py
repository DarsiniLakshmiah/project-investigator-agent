# Databricks notebook source
# MAGIC %md
# MAGIC ## Copilot App backend: one `copilot.investigate()` request per job run
# MAGIC Thin job entry point for the Databricks App (`copilot_app/`). Job parameters:
# MAGIC `project_id`, `question`. Returns the `InvestigationResult` JSON via notebook exit.
# MAGIC Same install set as notebook 14 (runtime MLflow is used, never installed).

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-databricks.txt -r ../requirements-retrieval.txt -r ../requirements-reranker.txt -r ../requirements-phase10d.txt -c ../constraints-databricks.txt

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from worldbank_copilot.copilot.databricks import build_copilot

dbutils.widgets.text("project_id", "P130544", "Project")  # noqa: F821
dbutils.widgets.text("question", "", "Question")  # noqa: F821
copilot = build_copilot(spark, settings)  # noqa: F821
result = copilot.investigate(
    query=dbutils.widgets.get("question"),  # noqa: F821
    project_id=dbutils.widgets.get("project_id").strip(),  # noqa: F821
)
dbutils.notebook.exit(result.model_dump_json())  # noqa: F821
