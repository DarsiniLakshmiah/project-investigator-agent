# Databricks notebook source
# MAGIC %md
# MAGIC ## 05_platformize_databricks: Governed Delta foundation (Phase 6)
# MAGIC Thin entry point. All logic lives in `worldbank_copilot.lakehouse` (contracts, identities,
# MAGIC row builders, source-hash verification, reconciliation, Unity Catalog + Delta store).
# MAGIC
# MAGIC Flow: validate catalog → validate/create schemas and Volumes → verify every source hash
# MAGIC against `configs/source_snapshot.json` → rebuild Bronze, structured Silver and
# MAGIC document-derived Silver with the Phase 2-5 code → validate contracts → reconcile with
# MAGIC `configs/reconciliation/expected_profiles.json` → MERGE into Delta → read back and
# MAGIC reconcile → run again to prove idempotency → validation queries.
# MAGIC
# MAGIC **Prerequisites** (see README "Phase 6 in Databricks"): this repository opened as a
# MAGIC Databricks Git folder; a cluster/serverless runtime with Python ≥ 3.11 and Delta MERGE
# MAGIC `WHEN NOT MATCHED BY SOURCE` (DBR 15.4 LTS or later); the source files copied to
# MAGIC `/Volumes/<catalog>/<bronze>/<source_volume>/data/` and the Phase 4 parsed cache to
# MAGIC `/Volumes/<catalog>/<silver>/<artifact_volume>/parsed/`.
# MAGIC
# MAGIC The run stops on any source-hash mismatch, contract violation or reconciliation
# MAGIC difference. No Gold tables are created.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-databricks.txt

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821 (Databricks built-in)

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

# Step 1: Unity Catalog objects (catalog validated; schemas and Volumes created only if missing).
from worldbank_copilot.lakehouse.pipeline import (  # noqa: E402
    format_report,
    run_platform,
    setup_unity_catalog,
)
from worldbank_copilot.lakehouse.spark_store import SparkDeltaStore  # noqa: E402
from worldbank_copilot.lakehouse.validation_sql import load_queries  # noqa: E402

store = SparkDeltaStore(
    spark,  # noqa: F821 (Databricks built-in)
    settings.require("databricks.catalog"),  # noqa: F821 (from _bootstrap)
    {"bronze": settings.databricks.bronze_schema, "silver": settings.databricks.silver_schema},  # noqa: F821
)
for status in setup_unity_catalog(store, settings):  # noqa: F821
    print(f"{status.object_type:<7} {status.name:<55} {status.status}")
print("source data root :", settings.data_root)  # noqa: F821
print("artefact root    :", settings.local_output_root)  # noqa: F821

# COMMAND ----------

# Step 2: verify sources, rebuild, validate, reconcile, MERGE, read back and reconcile again.
first = run_platform(settings, registry, spark=spark, progress=print)  # noqa: F821
print(format_report(first))

# COMMAND ----------

# Step 3: idempotency — the same snapshot again must change no Delta rows.
second = run_platform(settings, registry, spark=spark, progress=print)  # noqa: F821
print(format_report(second))
assert second.idempotent_write, "second run changed Delta rows: investigate before continuing"
assert {n: p["fingerprint"] for n, p in first.profiles.items()} == {
    n: p["fingerprint"] for n, p in second.profiles.items()
}

# COMMAND ----------

# Step 4: validation queries (read-only checks of the persisted data; not Gold analytics).
for name, query in load_queries(settings).items():  # noqa: F821
    print(f"-- {name}")
    display(spark.sql(query))  # noqa: F821
