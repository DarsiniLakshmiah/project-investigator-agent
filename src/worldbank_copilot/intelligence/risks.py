"""gold.risk_register: documented risks and assessment findings (Spark).

* FORMAL_RISK_RATING and ASSESSMENT_FINDING rows come one-to-one from
  silver.appraisal_risks; the framing is kept, never collapsed.
* ISR_SORT_RATING rows are the SORT ratings of each project's latest ISR (by sequence):
  the current formal risk view. Earlier ISR SORT rows stay in silver.isr_sort_ratings.
* ``resolution_status`` is always NOT_STATED: the documents do not state resolution and
  none is inferred.
"""

from __future__ import annotations

from typing import Any

from worldbank_copilot.intelligence.frames import F
from worldbank_copilot.intelligence.rules import RatingScales

TITLE_LENGTH = 120


def rank_expression(column: str, scale: dict[str, int]) -> Any:
    f = F()
    expr = f.lit(None).cast("bigint")
    for label, rank in scale.items():
        expr = f.when(f.col(column) == label, f.lit(rank)).otherwise(expr)
    return expr


def _appraisal(risks: Any, scales: RatingScales) -> Any:
    f = F()
    title = f.when(f.col("framing") == "FORMAL_RISK_RATING", f.col("risk_category")).otherwise(
        f.coalesce(
            f.substring(f.col("risk_description"), 1, TITLE_LENGTH),
            f.col("risk_category"),
            f.lit("(untitled finding)"),
        )
    )
    return risks.select(
        "project_id",
        f.col("risk_id").alias("risk_or_finding_id"),
        f.col("framing").alias("record_type"),
        f.col("risk_category").alias("category"),
        title.alias("title"),
        f.col("risk_description").alias("description"),
        "activity",
        f.col("risk_rating_raw").alias("rating_raw"),
        f.col("risk_rating").alias("rating"),
        rank_expression("risk_rating", scales.risk).alias("rating_rank"),
        f.lit(None).cast("string").alias("rating_at_approval"),
        f.lit(None).cast("string").alias("previous_rating"),
        f.col("identified_date").alias("observed_date"),
        f.lit(None).cast("bigint").alias("isr_sequence"),
        "source_document_type",
        "mitigation_text",
        f.lit("NOT_STATED").alias("resolution_status"),
        f.col("status").alias("extraction_status"),
        f.lit("silver.appraisal_risks").alias("source_table"),
        f.col("record_id").alias("source_record_id"),
        f.col("evidence_document_id").alias("document_id"),
        f.col("evidence_page_number").alias("page_number"),
        f.col("evidence_section").alias("section"),
        f.col("evidence_extraction_method").alias("extraction_method"),
        f.lit("DOCUMENTED_FINDING").alias("provenance_class"),
    )


def _latest_isr_sort(sort_ratings: Any, isr: Any, scales: RatingScales) -> Any:
    f = F()
    latest = isr.groupBy("project_id").agg(f.max("isr_sequence").alias("_latest"))
    rows = sort_ratings.join(latest, "project_id").where(f.col("isr_sequence") == f.col("_latest"))
    return rows.select(
        "project_id",
        f.col("record_id").alias("risk_or_finding_id"),
        f.lit("ISR_SORT_RATING").alias("record_type"),
        f.col("risk_category").alias("category"),
        f.col("risk_category").alias("title"),
        f.lit(None).cast("string").alias("description"),
        f.lit(None).cast("string").alias("activity"),
        f.col("current_rating_raw").alias("rating_raw"),
        f.col("current_rating").alias("rating"),
        rank_expression("current_rating", scales.risk).alias("rating_rank"),
        f.col("rating_at_approval").alias("rating_at_approval"),
        f.col("previous_rating").alias("previous_rating"),
        f.col("canonical_report_date").alias("observed_date"),
        "isr_sequence",
        f.lit("ISR").alias("source_document_type"),
        f.lit(None).cast("string").alias("mitigation_text"),
        f.lit("NOT_STATED").alias("resolution_status"),
        f.col("current_rating_status").alias("extraction_status"),
        f.lit("silver.isr_sort_ratings").alias("source_table"),
        f.col("record_id").alias("source_record_id"),
        f.col("evidence_document_id").alias("document_id"),
        f.col("evidence_page_number").alias("page_number"),
        f.col("evidence_section").alias("section"),
        f.col("evidence_extraction_method").alias("extraction_method"),
        f.lit("DOCUMENTED_FINDING").alias("provenance_class"),
    )


def build_risk_register(silver: dict[str, Any], scales: RatingScales) -> Any:
    return _appraisal(silver["appraisal_risks"], scales).unionByName(
        _latest_isr_sort(silver["isr_sort_ratings"], silver["isr_snapshots"], scales)
    )
