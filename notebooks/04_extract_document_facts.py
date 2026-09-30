# Databricks notebook source
# MAGIC %md
# MAGIC ## 04_extract_document_facts: Structured extraction from parsed documents
# MAGIC Thin entry point. Logic lives in `worldbank_copilot.extraction` (ISR snapshots,
# MAGIC results, appraisal risks, formal events, closing-date reconciliation, indicator
# MAGIC identity, cross-source checks) and `extraction.pipeline` (cache, report, outputs).
# MAGIC
# MAGIC **Status:** implemented (Phase 5). Reads the Phase 4 parsed cache (run
# MAGIC `02_parse_documents` first); Docling is not re-run. Outputs are written under
# MAGIC `data.local_output_root/silver_documents`; a Delta writer is not implemented yet
# MAGIC (pending the Databricks runtime decision). Requires `pypdfium2` (ships with the
# MAGIC `documents` extra) for the logged PDF text-layer fallback.

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from worldbank_copilot.extraction.pipeline import (  # noqa: E402
    OUTPUT_DIR,
    build_quality_report,
    run_extraction,
    summarize,
    write_outputs,
)
from worldbank_copilot.ingestion.pipeline import ingest_bronze  # noqa: E402
from worldbank_copilot.transformations.silver import build_silver  # noqa: E402

silver = build_silver(ingest_bronze(settings, registry), registry)  # noqa: F821 (from _bootstrap)
run = run_extraction(
    settings,  # noqa: F821
    registry,  # noqa: F821
    silver.tables["silver_loans"].rows,
    silver.tables["silver_projects"].rows,
    progress=print,
)
report = build_quality_report(run)
written = write_outputs(run, report, settings.local_output_root / OUTPUT_DIR)  # noqa: F821
summary = summarize(run)
print(summary["row_counts"], report.counts())
