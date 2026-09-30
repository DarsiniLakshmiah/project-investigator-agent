"""gold.project_360: one current-state record per project (Spark).

Sources: silver.projects (identity, status, dates), silver.project_financial_summary
(loan-snapshot finance, authoritative for loan-level values), silver.project_enrichment
(original closing date), silver.isr_snapshots (latest / previous ISR by sequence),
silver.project_events, silver.appraisal_risks, silver.project_results and the signals.

Fields intentionally not provided (Silver cannot support them reliably): a single
"total commitment" merging workbook and loan figures (both are kept separately in
Silver; Gold uses the loan snapshot), and an ISR-reported project status (not printed
in a parseable form; silver.projects.project_status is used).
"""

from __future__ import annotations

from typing import Any

from worldbank_copilot.intelligence.frames import F
from worldbank_copilot.intelligence.risks import rank_expression
from worldbank_copilot.intelligence.rules import RatingScales


def _change(current: str, previous: str) -> Any:
    f = F()
    return (
        f.when(f.col(current).isNull() | f.col(previous).isNull(), "NOT_COMPARABLE")
        .when(f.col(current) > f.col(previous), "UPGRADE")
        .when(f.col(current) < f.col(previous), "DOWNGRADE")
        .otherwise("UNCHANGED")
    )


def build_project_360(silver: dict[str, Any], signals: Any, scales: RatingScales) -> Any:
    from pyspark.sql import Window

    f = F()
    projects = silver["projects"]
    fin = silver["project_financial_summary"].select(
        "project_id",
        f.col("original_principal_total_usd").alias("original_principal_usd"),
        f.col("cancelled_total_usd").alias("cancelled_usd"),
        f.col("net_principal_after_cancellation_usd").alias("net_principal_usd"),
        f.col("disbursed_total_usd").alias("disbursed_usd"),
        f.col("undisbursed_total_usd").alias("undisbursed_usd"),
        f.col("disbursement_vs_net_principal_pct").alias("disbursement_pct_of_net_principal"),
        f.col("snapshot_date").alias("financial_snapshot_date"),
        "loan_count",
        "loans_with_valuation_caveats",
        f.col("record_id").alias("fin_id"),
    )
    original = (
        silver["project_enrichment"]
        .where(
            (f.col("target_table") == "silver_projects")
            & (f.col("target_field") == "original_closing_date")
        )
        .select(
            "project_id",
            f.col("value").alias("original_closing_date"),
            f.col("status").alias("original_closing_date_status"),
            f.col("record_id").alias("enrichment_id"),
        )
    )

    order = Window.partitionBy("project_id").orderBy(f.col("isr_sequence").desc())
    isr = (
        silver["isr_snapshots"]
        .withColumn("_n", f.row_number().over(order))
        .withColumn("do_rank", rank_expression("pdo_rating", scales.performance))
        .withColumn(
            "ip_rank", rank_expression("implementation_progress_rating", scales.performance)
        )
    )
    latest = isr.where(f.col("_n") == 1).select(
        "project_id",
        f.col("isr_sequence").alias("latest_isr_sequence"),
        f.col("canonical_report_date").alias("latest_isr_date"),
        f.col("canonical_date_basis").alias("latest_isr_date_basis"),
        f.col("pdo_rating").alias("latest_do_rating"),
        f.col("implementation_progress_rating").alias("latest_ip_rating"),
        f.col("overall_risk_rating").alias("latest_overall_risk_rating"),
        f.col("do_rank").alias("_do"),
        f.col("ip_rank").alias("_ip"),
        f.col("record_id").alias("latest_isr_id"),
    )
    previous = isr.where(f.col("_n") == 2).select(
        "project_id",
        f.col("pdo_rating").alias("previous_do_rating"),
        f.col("implementation_progress_rating").alias("previous_ip_rating"),
        f.col("do_rank").alias("_pdo"),
        f.col("ip_rank").alias("_pip"),
        f.col("record_id").alias("previous_isr_id"),
    )
    counts_isr = isr.groupBy("project_id").agg(f.count(f.lit(1)).alias("number_of_isrs"))

    events = (
        silver["project_events"]
        .groupBy("project_id")
        .agg(
            f.count(f.lit(1)).alias("number_of_project_events"),
            f.sum(
                ((f.col("event_type") == "RESTRUCTURING") & f.col("event_date").isNotNull()).cast(
                    "int"
                )
            ).alias("number_of_restructurings"),
            f.sum(
                ((f.col("event_type") == "RESTRUCTURING") & f.col("event_date").isNull()).cast(
                    "int"
                )
            ).alias("number_of_restructuring_papers"),
        )
    )
    risks = (
        silver["appraisal_risks"]
        .groupBy("project_id")
        .agg(
            f.sum((f.col("framing") == "FORMAL_RISK_RATING").cast("int")).alias(
                "formal_risk_count"
            ),
            f.sum((f.col("framing") == "ASSESSMENT_FINDING").cast("int")).alias(
                "assessment_finding_count"
            ),
        )
    )
    results = (
        silver["project_results"]
        .groupBy("project_id")
        .agg(
            f.countDistinct("indicator_key").alias("number_of_result_indicators"),
            f.max("observation_date").alias("latest_results_reporting_date"),
        )
    )
    current = (
        signals.where(f.col("signal_status") == "CURRENT")
        .groupBy("project_id")
        .agg(
            f.count(f.lit(1)).alias("current_attention_signal_count"),
            f.sum((f.col("severity") == "WATCH").cast("int")).alias("current_watch_signal_count"),
            f.sum((f.col("severity") == "HIGH").cast("int")).alias("current_high_signal_count"),
        )
    )

    frame = projects.select(
        "project_id",
        "project_name",
        f.col("financing_instrument").alias("instrument"),
        "project_status",
        "approval_date",
        f.col("effective_date").alias("effectiveness_date"),
        "current_closing_date",
        f.col("record_id").alias("project_record_id"),
    )
    for other in (fin, original, latest, previous, counts_isr, events, risks, results, current):
        frame = frame.join(other, "project_id", "left")
    zero = (
        "number_of_isrs",
        "number_of_project_events",
        "number_of_restructurings",
        "number_of_restructuring_papers",
        "formal_risk_count",
        "assessment_finding_count",
        "number_of_result_indicators",
        "current_attention_signal_count",
        "current_watch_signal_count",
        "current_high_signal_count",
    )
    for column in zero:
        frame = frame.withColumn(column, f.coalesce(f.col(column), f.lit(0)).cast("bigint"))
    ids = f.array_sort(
        f.filter(
            f.array(
                "project_record_id", "fin_id", "enrichment_id", "latest_isr_id", "previous_isr_id"
            ),
            lambda x: x.isNotNull(),
        )
    )
    return (
        frame.withColumn(
            "days_extended",
            f.when(
                f.col("original_closing_date").isNotNull(),
                f.datediff("current_closing_date", "original_closing_date"),
            ),
        )
        .withColumn("do_rating_change", _change("_do", "_pdo"))
        .withColumn("ip_rating_change", _change("_ip", "_pip"))
        .withColumn(
            "loans_with_valuation_caveats",
            f.coalesce("loans_with_valuation_caveats", f.array().cast("array<string>")),
        )
        .withColumn("source_record_ids", ids)
    )
