# Databricks notebook source
# MAGIC %md
# MAGIC ## One request backend for the Databricks App
# MAGIC User-run/job only. Configure authorized projects, endpoints and pricing in the job environment.
# COMMAND ----------
# MAGIC %pip install -q -r ../requirements-databricks.txt -r ../requirements-retrieval.txt -r ../requirements-reranker.txt -r ../requirements-phase10d.txt -r ../requirements-copilot-runtime.txt -c ../constraints-databricks.txt
# COMMAND ----------
dbutils.library.restartPython()
# COMMAND ----------
# MAGIC %run ./_bootstrap
# COMMAND ----------
from worldbank_copilot.application.runtime import build_databricks_application
from worldbank_copilot.application.service import AnswerRequest

dbutils.widgets.text("request_json", "", "Bounded request JSON; no endpoints or grants")
request = AnswerRequest.model_validate_json(dbutils.widgets.get("request_json"))
application = build_databricks_application(spark, settings, models_enabled=True)
answer = application.answer_question(request)
dbutils.notebook.exit(answer.model_dump_json())
