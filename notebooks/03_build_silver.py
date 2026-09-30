# Databricks notebook source
# MAGIC %md
# MAGIC ## 03_build_silver: Build normalized Silver tables
# MAGIC Thin entry point. Logic lives in `worldbank_copilot.transformations.silver`
# MAGIC (transforms), `silver_quality` (validation) and `silver_lineage` (field lineage).
# MAGIC
# MAGIC **Status:** structured Silver implemented (Phase 3); document-derived Silver tables
# MAGIC (ISR snapshots, results, events, risks) are built by `04_extract_document_facts`.
# MAGIC Writing to Delta is not implemented yet: it waits on the Databricks runtime decision.

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from worldbank_copilot.ingestion.pipeline import ingest_bronze  # noqa: E402
from worldbank_copilot.transformations.silver import build_silver  # noqa: E402
from worldbank_copilot.transformations.silver_quality import (  # noqa: E402
    build_silver_quality_report,
)
from worldbank_copilot.transformations.silver_report import format_silver_report  # noqa: E402

bronze = ingest_bronze(settings, registry)  # noqa: F821 (defined by _bootstrap)
silver = build_silver(bronze, registry)  # noqa: F821
report = build_silver_quality_report(silver, registry)  # noqa: F821
print(format_silver_report(silver, report, registry))  # noqa: F821

# COMMAND ----------

raise NotImplementedError(
    "Delta SilverWriter not implemented yet (pending Databricks runtime decision). "
    "Silver tables are available in `silver.tables`."
)
