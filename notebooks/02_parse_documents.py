# Databricks notebook source
# MAGIC %md
# MAGIC ## 02_parse_documents: Parse project PDFs with Docling
# MAGIC Thin entry point. Logic lives in `worldbank_copilot.parsing` (Docling adapter,
# MAGIC metadata handlers, manifest reconciliation, cache-aware pipeline, quality report).
# MAGIC
# MAGIC **Status:** implemented (Phase 4). Parsed JSON is written under
# MAGIC `data.local_output_root/parsed`; a Delta/volume writer for parsed documents is not
# MAGIC implemented yet (pending the Databricks runtime decision). Requires the `documents`
# MAGIC extra (Docling) on the cluster.

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

from worldbank_copilot.parsing.docling_parser import DoclingDocumentParser  # noqa: E402
from worldbank_copilot.parsing.pipeline import run_parsing  # noqa: E402
from worldbank_copilot.parsing.report import (  # noqa: E402
    build_parsing_report,
    format_parsing_report,
)

run = run_parsing(settings, registry, DoclingDocumentParser(), progress=print)  # noqa: F821
report = build_parsing_report(run, registry)  # noqa: F821
print(format_parsing_report(run, report, registry))  # noqa: F821
