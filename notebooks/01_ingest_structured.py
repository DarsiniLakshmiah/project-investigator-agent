# Databricks notebook source
# MAGIC %md
# MAGIC ## 01_ingest_structured: Ingest structured sources into Bronze
# MAGIC Thin entry point. Logic lives in `worldbank_copilot.ingestion` (contracts, readers,
# MAGIC document inventory, data-quality report) and `worldbank_copilot.transformations.bronze`.
# MAGIC
# MAGIC **Status:** ingestion + data-quality report implemented (Phase 2). Writing Bronze to
# MAGIC Delta is not implemented yet: it waits on the Databricks runtime decision.

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from worldbank_copilot.ingestion.data_quality import build_data_quality_report  # noqa: E402
from worldbank_copilot.ingestion.pipeline import ingest_bronze  # noqa: E402
from worldbank_copilot.ingestion.validation_report import format_text_report  # noqa: E402

result = ingest_bronze(settings, registry)  # noqa: F821 (defined by _bootstrap)
report = build_data_quality_report(result, registry)  # noqa: F821
print(format_text_report(result, report, registry))  # noqa: F821

# COMMAND ----------

raise NotImplementedError(
    "Delta BronzeWriter not implemented yet (pending Databricks runtime decision). "
    "Bronze tables are available in `result.all_tables()`."
)
