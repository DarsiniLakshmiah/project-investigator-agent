# Databricks notebook source
# MAGIC %md
# MAGIC ## 06_build_gold_intelligence: Deterministic Gold intelligence layer (Phase 7)
# MAGIC Thin entry point. All logic lives in `worldbank_copilot.intelligence` (native Spark
# MAGIC transformations, rule configuration, contracts, invariants) and uses the Phase 6
# MAGIC Unity Catalog / Delta store.
# MAGIC
# MAGIC Flow: validate the governed Silver tables against their Phase 6 contracts → build Gold
# MAGIC with Spark (timeline, result progress, risk register, attention signals, project 360)
# MAGIC → validate Gold contracts and invariants (any ERROR stops before writing) → create/validate
# MAGIC `worldbank_copilot.gold` → MERGE → read back and reconcile → run again to prove
# MAGIC idempotency → validation queries.
# MAGIC
# MAGIC Attention signals are deterministic observations with evidence, **not predictions**.
# MAGIC Rules and thresholds: `configs/intelligence/attention_rules.yaml`.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-databricks.txt

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821 (Databricks built-in)

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

# Step 1: build, validate, persist, read back and reconcile.
from worldbank_copilot.intelligence.pipeline import format_report, run_gold  # noqa: E402
from worldbank_copilot.lakehouse.validation_sql import load_queries  # noqa: E402

first = run_gold(spark, settings, progress=print)  # noqa: F821
print(format_report(first))

# COMMAND ----------

# Step 2: idempotency — unchanged Silver must change no Gold rows.
second = run_gold(spark, settings, progress=print)  # noqa: F821
print(format_report(second))
assert second.idempotent_write, "second run changed Gold rows: investigate before continuing"
assert {n: p["fingerprint"] for n, p in first.profiles.items()} == {
    n: p["fingerprint"] for n, p in second.profiles.items()
}

# COMMAND ----------

# Step 3: read-only validation queries (sql/phase7_validation.sql).
queries = load_queries(settings, settings.repo_root / "sql" / "phase7_validation.sql")  # noqa: F821
for name, query in queries.items():
    print(f"-- {name}")
    display(spark.sql(query))  # noqa: F821
