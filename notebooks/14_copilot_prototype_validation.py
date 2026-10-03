# Databricks notebook source
# MAGIC %md
# MAGIC ## Interview prototype: `copilot.investigate` on real Databricks data
# MAGIC Thin entry point; logic lives in `worldbank_copilot.copilot` and
# MAGIC `worldbank_copilot.validation.copilot_prototype`. Checks architectural invariants only
# MAGIC (scope, route, model calls, provenance, citations, publication rules); answers are shown
# MAGIC as produced, never compared with expected text. Model: see `configs/copilot.yaml`
# MAGIC (Qwen, 10d7 12/19 FAIL, not capability-accepted). Billed cost is UNAVAILABLE, not zero.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-databricks.txt -r ../requirements-retrieval.txt -r ../requirements-reranker.txt -r ../requirements-phase10d.txt -r ../requirements-copilot-runtime.txt -c ../constraints-databricks.txt

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

# Build the runtime: dependency + Phase 9 protocol gates, accepted index identity, endpoints.
from worldbank_copilot.copilot.databricks import build_copilot

copilot = build_copilot(spark, settings)  # noqa: F821
print(
    {
        "synthesizer": copilot.config.models.synthesizer_endpoint,
        "critic_enabled": copilot.config.models.critic_enabled,
        "critic": copilot.config.models.critic_endpoint,
        "pricing_configured": copilot.config.pricing_configured,
        "allowed_projects": copilot.config.allowed_projects,
    }
)

# COMMAND ----------

# S1 (r037): What deserves my attention? -> STRUCTURED, zero model calls.
import json

from worldbank_copilot.validation.copilot_prototype import run_scenario

require_state("copilot", step="the build cell")  # noqa: F821
print(json.dumps(run_scenario(copilot, "S1"), indent=2, default=str))  # noqa: F821

# COMMAND ----------

# S2 (r051): Why did the PDO rating drop to Moderately Unsatisfactory? -> INVESTIGATION.
import json

from worldbank_copilot.validation.copilot_prototype import run_scenario

require_state("copilot", step="the build cell")  # noqa: F821
print(json.dumps(run_scenario(copilot, "S2"), indent=2, default=str))  # noqa: F821

# COMMAND ----------

# S3 (r039): Show the timeline of restructurings. -> STRUCTURED, zero model calls.
import json

from worldbank_copilot.validation.copilot_prototype import run_scenario

require_state("copilot", step="the build cell")  # noqa: F821
print(json.dumps(run_scenario(copilot, "S3"), indent=2, default=str))  # noqa: F821

# COMMAND ----------

# S4 (r062): Will the project fail? -> REFUSE, zero model calls.
import json

from worldbank_copilot.validation.copilot_prototype import run_scenario

require_state("copilot", step="the build cell")  # noqa: F821
print(json.dumps(run_scenario(copilot, "S4"), indent=2, default=str))  # noqa: F821
