"""gold.result_progress: one row per Silver result observation, with safe calculations.

Identity: ``canonical_indicator_id`` is the Silver ``indicator_key`` (exact normalized
name, or a reviewed alias from configs/indicator_aliases.yaml). Near-identical pairs in
silver.indicator_match_candidates are never joined; an indicator that belongs to a
PENDING_REVIEW pair is flagged ``PENDING_ALIAS_REVIEW`` and stays its own series.

Calculations are made only when they are mathematically meaningful, otherwise they are
NULL and ``calculation_status`` says why (in order of precedence):
EXTRACTION_NOT_EXACT, DLI_LAYOUT_NOT_EVALUATED (Disbursement-Linked Indicator tables:
their columns carry DLI achievement/allocation semantics and showed column misalignment,
so they are not read as results-framework progress), QUALITATIVE_UNIT (Yes/No, Text),
MISSING_CURRENT, MISSING_BASELINE,
MISSING_TARGET, NON_NUMERIC_VALUE, ZERO_TARGET_DISTANCE (target == baseline), OK.

Direction is taken from the indicator itself (baseline -> target); no "higher is better"
assumption is made. Progress = (current - baseline) / (target - baseline) * 100, rounded
to 2 decimals.
"""

from __future__ import annotations

from typing import Any

from worldbank_copilot.intelligence.frames import DECIMAL_SQL, F, number, printed_date

QUALITATIVE_UNITS = ("Yes/No", "Text")


def _pending_keys(candidates: Any) -> Any:
    f = F()
    pending = candidates.where(f.col("review_status") == "PENDING_REVIEW")
    keys = pending.select(f.col("indicator_key_a").alias("k")).unionByName(
        pending.select(f.col("indicator_key_b").alias("k"))
    )
    return keys.distinct().withColumn("pending", f.lit(True))


def build_result_progress(silver: dict[str, Any]) -> Any:
    from pyspark.sql import Window

    f = F()
    results = silver["project_results"]
    pending = _pending_keys(silver["indicator_match_candidates"])
    frame = (
        results.join(pending, results["indicator_key"] == pending["k"], "left")
        .withColumn("baseline_number", number("baseline_value"))
        .withColumn("current_number", number("current_value"))
        .withColumn("target_number", number("target_value"))
    )
    status = (
        f.when(f.col("status") != "EXACT", "EXTRACTION_NOT_EXACT")
        .when(f.col("layout") == "DLI", "DLI_LAYOUT_NOT_EVALUATED")
        .when(f.col("unit").isin(*QUALITATIVE_UNITS), "QUALITATIVE_UNIT")
        .when(f.col("current_value").isNull(), "MISSING_CURRENT")
        .when(f.col("baseline_value").isNull(), "MISSING_BASELINE")
        .when(f.col("target_value").isNull(), "MISSING_TARGET")
        .when(
            f.col("current_number").isNull()
            | f.col("baseline_number").isNull()
            | f.col("target_number").isNull(),
            "NON_NUMERIC_VALUE",
        )
        .when(f.col("target_number") == f.col("baseline_number"), "ZERO_TARGET_DISTANCE")
        .otherwise("OK")
    )
    ok = f.col("calculation_status") == "OK"
    distance = f.col("target_number") - f.col("baseline_number")
    progress = f.round((f.col("current_number") - f.col("baseline_number")) * 100 / distance, 2)
    comparable_values = (
        (f.col("status") == "EXACT")
        & (f.col("layout") != "DLI")
        & ~f.col("unit").isin(*QUALITATIVE_UNITS)
    )
    series = Window.partitionBy("project_id", "indicator_key").orderBy("isr_sequence")
    frame = (
        frame.withColumn("calculation_status", status)
        .withColumn("progress_percentage", f.when(ok, progress).cast(DECIMAL_SQL))
        .withColumn(
            "target_gap",
            f.when(
                comparable_values
                & f.col("current_number").isNotNull()
                & f.col("target_number").isNotNull(),
                f.col("target_number") - f.col("current_number"),
            ).cast(DECIMAL_SQL),
        )
        .withColumn("previous_isr_sequence", f.lag("isr_sequence").over(series))
        .withColumn("_prev_status", f.lag("calculation_status").over(series))
        .withColumn("previous_number", f.lag("current_number").over(series))
        .withColumn("previous_target_number", f.lag("target_number").over(series))
        .withColumn("_prev_target_value", f.lag("target_value").over(series))
    )
    # NULL-safe: the first observation of a series has no previous values, and a NULL
    # condition must never fall through to a directional result.
    both_ok = f.coalesce(ok & (f.col("_prev_status") == "OK"), f.lit(False))
    change = f.col("current_number") - f.col("previous_number")
    same_target = f.coalesce(
        f.col("previous_target_number") == f.col("target_number"), f.lit(False)
    )
    comparable = both_ok & same_target & change.isNotNull()
    frame = (
        frame.withColumn("absolute_change", f.when(both_ok, change).cast(DECIMAL_SQL))
        .withColumn(
            "percentage_change",
            f.when(
                both_ok & (f.col("previous_number") != 0),
                f.round(change * 100 / f.col("previous_number"), 2),
            ).cast(DECIMAL_SQL),
        )
        # A missing target in one ISR (extraction gap) is not a change: both must be printed.
        .withColumn(
            "target_changed",
            f.coalesce(
                f.col("_prev_target_value").isNotNull()
                & f.col("target_value").isNotNull()
                & (f.col("_prev_target_value") != f.col("target_value")),
                f.lit(False),
            ),
        )
        .withColumn(
            "trend_direction",
            f.when(~comparable, "NOT_COMPARABLE")
            .when(change == 0, "UNCHANGED")
            .when(f.signum(change) == f.signum(distance), "TOWARD_TARGET")
            .otherwise("AWAY_FROM_TARGET"),
        )
        .withColumn(
            "target_status",
            f.when(~ok, "NOT_EVALUABLE")
            .when(f.col("progress_percentage") >= 100, "MET")
            .otherwise("NOT_MET"),
        )
        .withColumn(
            "identity_review_status",
            f.when(f.col("pending"), "PENDING_ALIAS_REVIEW")
            .when(f.col("identity_basis") == "ALIAS", "REVIEWED_ALIAS")
            .otherwise("EXACT_IDENTITY"),
        )
    )
    return frame.select(
        "project_id",
        f.col("indicator_key").alias("canonical_indicator_id"),
        f.col("indicator_name_raw").alias("indicator_name"),
        "indicator_type",
        "unit",
        "layout",
        "identity_basis",
        "identity_review_status",
        "isr_sequence",
        f.col("observation_date").alias("reporting_date"),
        "baseline_value",
        "current_value",
        "target_value",
        "baseline_number",
        "current_number",
        "target_number",
        "target_date",
        printed_date("target_date").alias("target_date_parsed"),
        "previous_isr_sequence",
        "previous_number",
        "previous_target_number",
        "absolute_change",
        "percentage_change",
        "progress_percentage",
        "target_gap",
        "trend_direction",
        "target_status",
        "calculation_status",
        "target_changed",
        f.col("status").alias("extraction_status"),
        f.col("evidence_table_id").alias("table_id"),
        f.lit("silver.project_results").alias("source_table"),
        f.col("record_id").alias("source_record_id"),
        "document_id",
        f.col("evidence_page_number").alias("page_number"),
        f.col("evidence_section").alias("section"),
        "extraction_method",
        f.lit("DOCUMENTED_FINDING").alias("provenance_class"),
    )
