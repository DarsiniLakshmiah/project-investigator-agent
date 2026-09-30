"""gold.attention_signals: deterministic, evidence-backed signals (native Spark).

One builder per rule in configs/intelligence/attention_rules.yaml (``BUILDERS``). A
configured, enabled rule without a builder, or a builder without a configured rule,
fails before anything is built. Thresholds come only from the configuration.

Signals are observations, not predictions: descriptions state what the data shows.
Each signal keeps its primary Silver/Gold record (``source_table``,
``source_record_id``), all supporting record ids, the document location and the
extraction method, so it can be traced back to the source page.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from worldbank_copilot.common.exceptions import ConfigurationError
from worldbank_copilot.intelligence.frames import DECIMAL_SQL, F
from worldbank_copilot.intelligence.risks import rank_expression
from worldbank_copilot.intelligence.rules import RatingScales, Rule, RuleSet

SIGNAL_COLUMNS = (
    "signal_id",
    "project_id",
    "rule_id",
    "rule_version",
    "signal_category",
    "signal_type",
    "subject",
    "signal_title",
    "signal_description",
    "severity",
    "signal_status",
    "observed_date",
    "effective_sequence",
    "current_value",
    "comparison_value",
    "threshold",
    "value_unit",
    "rule_description",
    "supporting_record_ids",
    "evidence_status",
    "caveats",
    "source_table",
    "source_record_id",
    "document_id",
    "page_number",
    "section",
    "extraction_method",
    "provenance_class",
)


def text(column: Any) -> Any:
    """Decimal/number -> text without trailing zeros ('58.970000' -> '58.97')."""
    f = F()
    s = column.cast("string")
    return f.when(
        s.contains("."), f.regexp_replace(f.regexp_replace(s, "0+$", ""), r"\.$", "")
    ).otherwise(s)


def shown(column: Any) -> Any:
    """Value for a description: text, or 'not stated' when NULL (concat would be NULL)."""
    f = F()
    return f.coalesce(column.cast("string"), f.lit("not stated"))


def _null(kind: str = "string") -> Any:
    return F().lit(None).cast(kind)


def emit(frame: Any, rule: Rule, **cols: Any) -> Any:
    """Standard signal columns; missing optional columns are NULL / empty."""
    f = F()
    defaults = {
        "observed_date": _null("date"),
        "effective_sequence": _null("bigint"),
        "current_value": _null(),
        "comparison_value": _null(),
        "threshold": _null(),
        "value_unit": _null(),
        "evidence_status": _null(),
        "caveats": f.array().cast("array<string>"),
        "document_id": _null(),
        "page_number": _null("bigint"),
        "section": _null(),
        "extraction_method": _null(),
    }
    defaults.update(cols)
    fixed = {
        "rule_id": f.lit(rule.rule_id),
        "rule_version": f.lit(rule.rule_version),
        "signal_category": f.lit(rule.category),
        "signal_type": f.lit(rule.rule_id),
        "signal_title": f.lit(rule.title),
        "rule_description": f.lit(" ".join(rule.description.split())),
        "provenance_class": f.lit("SYSTEM_DERIVED_SIGNAL"),
    }
    out = frame.select(
        *[
            (fixed.get(n) if n in fixed else defaults.get(n, f.col(n))).alias(n)
            for n in SIGNAL_COLUMNS
            if n != "signal_id"
        ]
    )
    key = f.concat_ws(
        "|",
        "rule_id",
        "project_id",
        "subject",
        f.coalesce(f.col("effective_sequence").cast("string"), f.lit("-")),
    )
    return out.withColumn("signal_id", f.substring(f.sha2(key, 256), 1, 32)).select(*SIGNAL_COLUMNS)


def _latest_isr(isr: Any) -> Any:
    return isr.groupBy("project_id").agg(F().max("isr_sequence").alias("latest_isr_sequence"))


def _status(sequence: str = "effective_sequence") -> Any:
    f = F()
    return f.when(f.col(sequence) == f.col("latest_isr_sequence"), "CURRENT").otherwise(
        "HISTORICAL"
    )


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


def schedule_closing_date_extended(rule: Rule, s: dict[str, Any], **_: Any) -> Any:
    f = F()
    watch, high = (
        int(rule.threshold("watch_extension_months")),
        int(rule.threshold("high_extension_months")),
    )
    original = (
        s["project_enrichment"]
        .where(
            (f.col("target_table") == "silver_projects")
            & (f.col("target_field") == "original_closing_date")
            & f.col("value").isNotNull()
        )
        .select(
            "project_id",
            f.col("value").alias("original"),
            f.col("loan_number").alias("first_loan"),
            f.col("record_id").alias("enrichment_id"),
            "status",
            "evidence_document_id",
            "evidence_page_number",
            "evidence_section",
            "evidence_extraction_method",
        )
    )
    changes = (
        s["project_events"]
        .where(f.col("event_type") == "CLOSING_DATE_CHANGE")
        .groupBy("project_id")
        .agg(
            f.max("event_date").alias("last_change_date"),
            f.array_sort(f.collect_set("record_id")).alias("change_ids"),
        )
    )
    projects = s["projects"].select(
        "project_id", "current_closing_date", f.col("record_id").alias("project_record_id")
    )
    frame = (
        projects.join(original, "project_id")
        .join(changes, "project_id", "left")
        .where(f.col("current_closing_date") > f.col("original"))
    )
    days = f.datediff("current_closing_date", "original")
    severity = (
        f.when(f.col("current_closing_date") >= f.add_months("original", high), "HIGH")
        .when(f.col("current_closing_date") >= f.add_months("original", watch), "WATCH")
        .otherwise("INFO")
    )
    return emit(
        frame,
        rule,
        subject=f.lit("Project closing date"),
        severity=severity,
        signal_status=f.lit("CURRENT"),
        observed_date=f.col("last_change_date"),
        current_value=f.col("current_closing_date").cast("string"),
        comparison_value=f.col("original").cast("string"),
        threshold=f.lit(f"WATCH >= {watch} months, HIGH >= {high} months"),
        value_unit=f.lit("date"),
        signal_description=f.concat(
            f.lit("The current closing date ("),
            f.col("current_closing_date").cast("string"),
            f.lit(") is "),
            days.cast("string"),
            f.lit(" days after the original closing date ("),
            f.col("original").cast("string"),
            f.lit(", first loan "),
            f.col("first_loan"),
            f.lit(")."),
        ),
        supporting_record_ids=f.array_sort(
            f.array_distinct(
                f.concat(
                    f.array("project_record_id", "enrichment_id"),
                    f.coalesce(f.col("change_ids"), f.array().cast("array<string>")),
                )
            )
        ),
        evidence_status=f.col("status"),
        source_table=f.lit("silver.project_enrichment"),
        source_record_id=f.col("enrichment_id"),
        document_id=f.col("evidence_document_id"),
        page_number=f.col("evidence_page_number"),
        section=f.col("evidence_section"),
        extraction_method=f.col("evidence_extraction_method"),
    )


def schedule_repeated_changes(rule: Rule, s: dict[str, Any], **_: Any) -> Any:
    f = F()
    minimum = int(rule.threshold("watch_change_count"))
    events = s["project_events"].where(
        (f.col("event_type") == "CLOSING_DATE_CHANGE")
        & f.col("old_closing_date").isNotNull()
        & f.col("new_closing_date").isNotNull()
    )
    change = f.concat_ws(
        " ",
        f.col("loan_number"),
        f.col("old_closing_date").cast("string"),
        f.lit("->"),
        f.col("new_closing_date").cast("string"),
    )
    # Ordering with a unique tie-breaker keeps max_by deterministic.
    order = f.struct(
        f.coalesce(f.col("event_date"), f.lit("1900-01-01").cast("date")), f.col("record_id")
    )
    frame = (
        events.groupBy("project_id")
        .agg(
            f.count_distinct(
                f.col("loan_number"), f.col("old_closing_date"), f.col("new_closing_date")
            ).alias("n"),
            f.array_join(f.array_sort(f.collect_set(change)), "; ").alias("changes"),
            f.array_sort(f.collect_set("record_id")).alias("ids"),
            f.max("event_date").alias("last_date"),
            f.max_by("record_id", order).alias("primary_id"),
            f.max_by("status", order).alias("primary_status"),
            f.max_by("evidence_document_id", order).alias("doc"),
            f.max_by("evidence_page_number", order).alias("page"),
            f.max_by("evidence_section", order).alias("section_"),
            f.max_by("evidence_extraction_method", order).alias("method"),
        )
        .where(f.col("n") >= minimum)
    )
    return emit(
        frame,
        rule,
        subject=f.lit("Closing-date changes"),
        severity=f.lit("WATCH"),
        signal_status=f.lit("CURRENT"),
        observed_date=f.col("last_date"),
        current_value=f.col("n").cast("string"),
        threshold=f.lit(f">= {minimum} changes"),
        value_unit=f.lit("distinct changes"),
        signal_description=f.concat(
            f.col("n").cast("string"),
            f.lit(" distinct closing-date changes are documented: "),
            f.col("changes"),
            f.lit("."),
        ),
        supporting_record_ids=f.col("ids"),
        evidence_status=f.col("primary_status"),
        source_table=f.lit("silver.project_events"),
        source_record_id=f.col("primary_id"),
        document_id=f.col("doc"),
        page_number=f.col("page"),
        section=f.col("section_"),
        extraction_method=f.col("method"),
    )


# ---------------------------------------------------------------------------
# Implementation ratings (ISR sequence order, ordinal ranks only)
# ---------------------------------------------------------------------------

RATING_SUBJECTS = (
    ("PDO rating", "pdo_rating"),
    ("Implementation progress rating", "implementation_progress_rating"),
)


def rating_history(isr: Any, scales: RatingScales) -> Any:
    """One row per ISR x rating (PDO / IP) with rank and the preceding ISR's rating."""
    from pyspark.sql import Window

    f = F()
    parts = []
    for subject, column in RATING_SUBJECTS:
        parts.append(
            isr.select(
                "project_id",
                "isr_sequence",
                "canonical_report_date",
                "document_id",
                f.col("record_id").alias("isr_record_id"),
                f.lit(subject).alias("subject"),
                f.col(column).alias("rating"),
                f.col(f"{column}_status").alias("rating_status"),
                f.col(f"{column}_page_number").alias("rating_page"),
                f.col(f"{column}_extraction_method").alias("rating_method"),
                rank_expression(column, scales.performance).alias("rank"),
            )
        )
    long = parts[0].unionByName(parts[1])
    w = Window.partitionBy("project_id", "subject").orderBy("isr_sequence")
    return (
        long.withColumn("prev_rank", f.lag("rank").over(w))
        .withColumn("prev_rating", f.lag("rating").over(w))
        .withColumn("prev_sequence", f.lag("isr_sequence").over(w))
        .withColumn("prev_record_id", f.lag("isr_record_id").over(w))
        .join(_latest_isr(isr), "project_id")
    )


def _rating_change(frame: Any, rule: Rule, severity: Any, verb: str) -> Any:
    f = F()
    return emit(
        frame,
        rule,
        subject=f.col("subject"),
        severity=severity,
        signal_status=_status("isr_sequence"),
        observed_date=f.col("canonical_report_date"),
        effective_sequence=f.col("isr_sequence").cast("bigint"),
        current_value=f.col("rating"),
        comparison_value=f.col("prev_rating"),
        value_unit=f.lit("rating"),
        signal_description=f.concat(
            f.lit("The "),
            f.col("subject"),
            f.lit(f" {verb} from "),
            f.col("prev_rating"),
            f.lit(" (ISR "),
            f.col("prev_sequence").cast("string"),
            f.lit(") to "),
            shown(f.col("rating")),
            f.lit(" (ISR "),
            f.col("isr_sequence").cast("string"),
            f.lit(")."),
        ),
        supporting_record_ids=f.array_sort(f.array("isr_record_id", "prev_record_id")),
        evidence_status=f.col("rating_status"),
        source_table=f.lit("silver.isr_snapshots"),
        source_record_id=f.col("isr_record_id"),
        document_id=f.col("document_id"),
        page_number=f.col("rating_page"),
        extraction_method=f.col("rating_method"),
    )


def rating_downgrade(rule: Rule, s: dict[str, Any], scales: RatingScales, **_: Any) -> Any:
    f = F()
    low = scales.performance_below_satisfactory_max_rank
    frame = rating_history(s["isr_snapshots"], scales).where(f.col("rank") < f.col("prev_rank"))
    severity = f.when(f.col("rank") <= low, "HIGH").otherwise("WATCH")
    return _rating_change(frame, rule, severity, "was downgraded")


def rating_recovery(rule: Rule, s: dict[str, Any], scales: RatingScales, **_: Any) -> Any:
    f = F()
    low = scales.performance_below_satisfactory_max_rank
    frame = rating_history(s["isr_snapshots"], scales).where(
        (f.col("prev_rank") <= low) & (f.col("rank") > low)
    )
    return _rating_change(frame, rule, f.lit("INFO"), "recovered")


def runs(frame: Any, partition: list[str], flag: Any) -> Any:
    """Group consecutive rows (by isr_sequence) where ``flag`` is true into runs."""
    from pyspark.sql import Window

    f = F()
    all_rows = Window.partitionBy(*partition).orderBy("isr_sequence")
    flagged = Window.partitionBy(*partition, "_flag").orderBy("isr_sequence")
    return (
        frame.withColumn("_flag", flag)
        .withColumn("_run", f.row_number().over(all_rows) - f.row_number().over(flagged))
        .where(f.col("_flag"))
    )


def rating_below_satisfactory_persistent(
    rule: Rule, s: dict[str, Any], scales: RatingScales, **_: Any
) -> Any:
    f = F()
    low = scales.performance_below_satisfactory_max_rank
    minimum, high = (
        int(rule.threshold("min_consecutive_isrs")),
        int(rule.threshold("high_consecutive_isrs")),
    )
    history = rating_history(s["isr_snapshots"], scales).where(f.col("rank").isNotNull())
    grouped = (
        runs(history, ["project_id", "subject"], f.col("rank") <= low)
        .groupBy("project_id", "subject", "_run", "latest_isr_sequence")
        .agg(
            f.count(f.lit(1)).alias("n"),
            f.min("isr_sequence").alias("first_seq"),
            f.max("isr_sequence").alias("last_seq"),
            f.max("canonical_report_date").alias("last_date"),
            f.array_join(f.array_sort(f.collect_set("rating")), ", ").alias("ratings"),
            f.array_sort(f.collect_list("isr_record_id")).alias("ids"),
            f.max_by("isr_record_id", "isr_sequence").alias("primary_id"),
            f.max_by("document_id", "isr_sequence").alias("doc"),
            f.max_by("rating_page", "isr_sequence").alias("page"),
            f.max_by("rating_method", "isr_sequence").alias("method"),
            f.max_by("rating_status", "isr_sequence").alias("primary_status"),
        )
        .where(f.col("n") >= minimum)
    )
    return emit(
        grouped,
        rule,
        subject=f.col("subject"),
        severity=f.when(f.col("n") >= high, "HIGH").otherwise("WATCH"),
        signal_status=_status("last_seq"),
        observed_date=f.col("last_date"),
        effective_sequence=f.col("last_seq").cast("bigint"),
        current_value=f.col("n").cast("string"),
        threshold=f.lit(f"WATCH >= {minimum} ISRs, HIGH >= {high} ISRs"),
        value_unit=f.lit("consecutive ISRs"),
        signal_description=f.concat(
            f.lit("The "),
            f.col("subject"),
            f.lit(" was below satisfactory ("),
            f.col("ratings"),
            f.lit(") for "),
            f.col("n").cast("string"),
            f.lit(" consecutive ISRs (sequence "),
            f.col("first_seq").cast("string"),
            f.lit(" to "),
            f.col("last_seq").cast("string"),
            f.lit(")."),
        ),
        supporting_record_ids=f.col("ids"),
        evidence_status=f.col("primary_status"),
        source_table=f.lit("silver.isr_snapshots"),
        source_record_id=f.col("primary_id"),
        document_id=f.col("doc"),
        page_number=f.col("page"),
        extraction_method=f.col("method"),
    )


# ---------------------------------------------------------------------------
# Results (gold.result_progress; single indicator identity only)
# ---------------------------------------------------------------------------


def _results_with_latest(progress: Any, s: dict[str, Any]) -> Any:
    return progress.join(_latest_isr(s["isr_snapshots"]), "project_id")


def _result_signal(frame: Any, rule: Rule, **cols: Any) -> Any:
    f = F()
    caveat = f.when(
        f.col("identity_review_status") == "PENDING_ALIAS_REVIEW",
        f.lit(
            "Indicator belongs to a pending alias candidate pair; evaluated as its own series only."
        ),
    )
    base = dict(
        subject=f.col("indicator_name"),
        signal_status=_status("isr_sequence"),
        observed_date=f.col("reporting_date"),
        effective_sequence=f.col("isr_sequence"),
        value_unit=f.col("unit"),
        evidence_status=f.col("extraction_status"),
        caveats=f.filter(f.array(caveat), lambda x: x.isNotNull()),
        supporting_record_ids=f.array("source_record_id"),
        source_table=f.lit("silver.project_results"),
        source_record_id=f.col("source_record_id"),
        document_id=f.col("document_id"),
        page_number=f.col("page_number"),
        section=f.col("section"),
        extraction_method=f.col("extraction_method"),
    )
    base.update(cols)
    return emit(frame, rule, **base)


def result_target_date_passed(rule: Rule, s: dict[str, Any], progress: Any, **_: Any) -> Any:
    from pyspark.sql import Window

    f = F()
    high = rule.threshold("high_progress_below_percent")
    latest = Window.partitionBy("project_id", "canonical_indicator_id").orderBy(
        f.col("isr_sequence").desc()
    )
    effective = s["projects"].select("project_id", "effective_date")
    # Plausibility guard: a target date before project effectiveness cannot be the
    # indicator's target for this operation (counted in gold.quality_observations).
    frame = (
        _results_with_latest(progress, s)
        .join(effective, "project_id")
        .withColumn("_n", f.row_number().over(latest))
        .where(
            (f.col("_n") == 1)
            & (f.col("calculation_status") == "OK")
            & f.col("target_date_parsed").isNotNull()
            & (f.col("target_date_parsed") >= f.col("effective_date"))
            & (f.col("reporting_date") > f.col("target_date_parsed"))
            & (f.col("progress_percentage") < 100)
        )
    )
    return _result_signal(
        frame,
        rule,
        severity=f.when(f.col("progress_percentage") < f.lit(high), "HIGH").otherwise("WATCH"),
        current_value=f.concat(text(f.col("progress_percentage")), f.lit("% of target change")),
        comparison_value=f.concat(
            f.lit("target "),
            shown(f.col("target_value")),
            f.lit(" by "),
            shown(f.col("target_date")),
        ),
        threshold=f.lit(f"target not met after target date; HIGH below {text_literal(high)}%"),
        signal_description=f.concat(
            f.lit("In ISR "),
            f.col("isr_sequence").cast("string"),
            f.lit(" ("),
            f.col("reporting_date").cast("string"),
            f.lit("), '"),
            f.col("indicator_name"),
            f.lit("' is "),
            shown(f.col("current_value")),
            f.lit(" against a target of "),
            shown(f.col("target_value")),
            f.lit(" due "),
            shown(f.col("target_date")),
            f.lit(" (baseline "),
            shown(f.col("baseline_value")),
            f.lit("; "),
            text(f.col("progress_percentage")),
            f.lit("% of the targeted change)."),
        ),
    )


def text_literal(value: Any) -> str:
    return format(value.normalize(), "f") if hasattr(value, "normalize") else str(value)


def _with_previous_record(progress: Any) -> Any:
    from pyspark.sql import Window

    f = F()
    w = Window.partitionBy("project_id", "canonical_indicator_id").orderBy("isr_sequence")
    return progress.withColumn("previous_record_id", f.lag("source_record_id").over(w))


def result_moved_away(rule: Rule, s: dict[str, Any], progress: Any, **_: Any) -> Any:
    f = F()
    frame = _results_with_latest(_with_previous_record(progress), s).where(
        f.col("trend_direction") == "AWAY_FROM_TARGET"
    )
    return _result_signal(
        frame,
        rule,
        severity=f.lit("WATCH"),
        current_value=f.col("current_value"),
        comparison_value=f.concat(
            f.lit("previous "),
            shown(text(f.col("previous_number"))),
            f.lit(" (ISR "),
            shown(f.col("previous_isr_sequence")),
            f.lit("); target "),
            shown(f.col("target_value")),
        ),
        signal_description=f.concat(
            f.lit("'"),
            f.col("indicator_name"),
            f.lit("' moved from "),
            shown(text(f.col("previous_number"))),
            f.lit(" (ISR "),
            shown(f.col("previous_isr_sequence")),
            f.lit(") to "),
            shown(f.col("current_value")),
            f.lit(" (ISR "),
            f.col("isr_sequence").cast("string"),
            f.lit("), away from its target of "),
            shown(f.col("target_value")),
            f.lit("."),
        ),
        supporting_record_ids=f.array_sort(f.array("source_record_id", "previous_record_id")),
    )


def result_target_changed(rule: Rule, s: dict[str, Any], progress: Any, **_: Any) -> Any:
    f = F()
    frame = _results_with_latest(_with_previous_record(progress), s).where(f.col("target_changed"))
    previous = f.coalesce(text(f.col("previous_target_number")), f.lit("(non-numeric)"))
    return _result_signal(
        frame,
        rule,
        severity=f.lit("INFO"),
        current_value=f.col("target_value"),
        comparison_value=previous,
        signal_description=f.concat(
            f.lit("The target of '"),
            f.col("indicator_name"),
            f.lit("' changed from "),
            previous,
            f.lit(" (ISR "),
            shown(f.col("previous_isr_sequence")),
            f.lit(") to "),
            shown(f.col("target_value")),
            f.lit(" (ISR "),
            f.col("isr_sequence").cast("string"),
            f.lit(")."),
        ),
        supporting_record_ids=f.array_sort(f.array("source_record_id", "previous_record_id")),
    )


def result_no_change(rule: Rule, s: dict[str, Any], progress: Any, **_: Any) -> Any:
    from pyspark.sql import Window

    f = F()
    minimum = int(rule.threshold("min_consecutive_observations"))
    months = int(rule.threshold("min_span_months"))
    w = Window.partitionBy("project_id", "canonical_indicator_id").orderBy("isr_sequence")
    starts = (f.col("trend_direction") != "UNCHANGED").cast("int")
    frame = _results_with_latest(progress, s).withColumn(
        "_run", f.sum(starts).over(w.rowsBetween(Window.unboundedPreceding, Window.currentRow))
    )
    grouped = (
        frame.groupBy("project_id", "canonical_indicator_id", "_run", "latest_isr_sequence")
        .agg(
            f.count(f.lit(1)).alias("n"),
            f.min((f.col("calculation_status") == "OK").cast("int")).alias("all_ok"),
            f.min("isr_sequence").alias("first_seq"),
            f.max("isr_sequence").alias("last_seq"),
            f.min("reporting_date").alias("first_date"),
            f.max("reporting_date").alias("last_date"),
            f.array_sort(f.collect_list("source_record_id")).alias("ids"),
            *[
                f.max_by(c, "isr_sequence").alias(c)
                for c in (
                    "indicator_name",
                    "current_value",
                    "target_value",
                    "target_status",
                    "unit",
                    "extraction_status",
                    "identity_review_status",
                    "source_record_id",
                    "document_id",
                    "page_number",
                    "section",
                    "extraction_method",
                )
            ],
        )
        .where(
            (f.col("n") >= minimum)
            & (f.col("all_ok") == 1)
            & (f.col("target_status") == "NOT_MET")
            & (f.add_months("first_date", months) <= f.col("last_date"))
        )
    )
    return _result_signal(
        grouped.withColumnRenamed("last_seq", "isr_sequence").withColumnRenamed(
            "last_date", "reporting_date"
        ),
        rule,
        severity=f.lit("INFO"),
        current_value=f.col("current_value"),
        comparison_value=f.concat(f.lit("target "), shown(f.col("target_value"))),
        threshold=f.lit(f">= {minimum} observations over >= {months} months"),
        signal_description=f.concat(
            f.lit("'"),
            f.col("indicator_name"),
            f.lit("' reported "),
            shown(f.col("current_value")),
            f.lit(" in "),
            f.col("n").cast("string"),
            f.lit(" consecutive ISRs (sequence "),
            f.col("first_seq").cast("string"),
            f.lit(" to "),
            f.col("isr_sequence").cast("string"),
            f.lit(", "),
            f.col("first_date").cast("string"),
            f.lit(" to "),
            f.col("reporting_date").cast("string"),
            f.lit("); target "),
            shown(f.col("target_value")),
            f.lit("."),
        ),
        supporting_record_ids=f.col("ids"),
    )


# ---------------------------------------------------------------------------
# Financial execution
# ---------------------------------------------------------------------------


def finance_disbursement_lag(rule: Rule, s: dict[str, Any], **_: Any) -> Any:
    f = F()
    watch, high = rule.threshold("watch_gap_points"), rule.threshold("high_gap_points")
    instruments = rule.applies_to_instruments or []
    projects = s["projects"].select(
        "project_id", "financing_instrument", "effective_date", "current_closing_date"
    )
    fin = s["project_financial_summary"]
    frame = fin.join(projects, "project_id").where(
        f.col("financing_instrument").isin(*instruments)
        & f.col("effective_date").isNotNull()
        & (f.col("current_closing_date") > f.col("effective_date"))
    )
    elapsed = f.round(
        f.datediff("snapshot_date", "effective_date").cast(DECIMAL_SQL)
        * 100
        / f.datediff("current_closing_date", "effective_date"),
        2,
    )
    frame = (
        frame.withColumn("elapsed_pct", elapsed.cast(DECIMAL_SQL))
        .withColumn(
            "gap",
            (f.col("elapsed_pct") - f.col("disbursement_vs_net_principal_pct")).cast(DECIMAL_SQL),
        )
        .where(f.col("gap") >= f.lit(watch))
    )
    caveats = f.filter(
        f.array(
            f.when(
                f.size("loans_with_valuation_caveats") > 0,
                f.concat(
                    f.lit("Loan snapshot valuation caveat (FX) for "),
                    f.array_join("loans_with_valuation_caveats", ", "),
                    f.lit("."),
                ),
            ),
            f.when(
                f.col("loan_count") > 1,
                f.lit(
                    "Several loans with different effectiveness dates; elapsed time is "
                    "measured from "
                    "project effectiveness."
                ),
            ),
        ),
        lambda x: x.isNotNull(),
    )
    return emit(
        frame,
        rule,
        subject=f.lit("Disbursement vs elapsed time"),
        severity=f.when(f.col("gap") >= f.lit(high), "HIGH").otherwise("WATCH"),
        signal_status=f.lit("CURRENT"),
        observed_date=f.col("snapshot_date"),
        current_value=f.concat(
            text(f.col("disbursement_vs_net_principal_pct")), f.lit("% of net principal disbursed")
        ),
        comparison_value=f.concat(
            text(f.col("elapsed_pct")), f.lit("% of implementation period elapsed")
        ),
        threshold=f.lit(
            f"WATCH gap >= {text_literal(watch)} points, HIGH gap >= {text_literal(high)} points"
        ),
        value_unit=f.lit("percent"),
        signal_description=f.concat(
            f.lit("At the loan snapshot of "),
            f.col("snapshot_date").cast("string"),
            f.lit(", "),
            text(f.col("disbursement_vs_net_principal_pct")),
            f.lit("% of net principal was disbursed while "),
            text(f.col("elapsed_pct")),
            f.lit("% of the period from effectiveness ("),
            f.col("effective_date").cast("string"),
            f.lit(") to the current closing date ("),
            f.col("current_closing_date").cast("string"),
            f.lit(") had elapsed (gap "),
            text(f.col("gap")),
            f.lit(" points)."),
        ),
        caveats=caveats,
        supporting_record_ids=f.array("record_id"),
        source_table=f.lit("silver.project_financial_summary"),
        source_record_id=f.col("record_id"),
    )


# ---------------------------------------------------------------------------
# Restructuring / scope changes
# ---------------------------------------------------------------------------


def change_restructuring(rule: Rule, s: dict[str, Any], **_: Any) -> Any:
    f = F()
    watch = int(rule.threshold("watch_restructuring_count"))
    events = s["project_events"].where(f.col("event_type") == "RESTRUCTURING")
    dated = f.col("event_date").isNotNull()
    latest = f.struct(f.col("event_date"), f.col("record_id"))  # undated papers sort first
    frame = (
        events.groupBy("project_id")
        .agg(
            f.sum(dated.cast("int")).alias("n"),
            f.sum((~dated).cast("int")).alias("papers"),
            f.array_join(
                f.array_sort(f.collect_list(f.when(dated, f.col("event_date").cast("string")))),
                ", ",
            ).alias("dates"),
            f.max("event_date").alias("last_date"),
            f.array_sort(f.collect_list("record_id")).alias("ids"),
            f.max_by("record_id", latest).alias("primary_id"),
            f.max_by("status", latest).alias("primary_status"),
            f.max_by("evidence_document_id", latest).alias("doc"),
            f.max_by("evidence_page_number", latest).alias("page"),
            f.max_by("evidence_section", latest).alias("section_"),
            f.max_by("evidence_extraction_method", latest).alias("method"),
        )
        .where(f.col("n") >= 1)
    )
    return emit(
        frame,
        rule,
        subject=f.lit("Restructurings"),
        severity=f.when(f.col("n") >= watch, "WATCH").otherwise("INFO"),
        signal_status=f.lit("CURRENT"),
        observed_date=f.col("last_date"),
        current_value=f.col("n").cast("string"),
        threshold=f.lit(f"WATCH >= {watch}"),
        value_unit=f.lit("restructurings"),
        signal_description=f.concat(
            f.lit("The project has undergone "),
            f.col("n").cast("string"),
            f.lit(" restructurings with a formal approval date ("),
            f.col("dates"),
            f.lit("); "),
            f.col("papers").cast("string"),
            f.lit(" undated restructuring papers describe them."),
        ),
        supporting_record_ids=f.col("ids"),
        evidence_status=f.col("primary_status"),
        source_table=f.lit("silver.project_events"),
        source_record_id=f.col("primary_id"),
        document_id=f.col("doc"),
        page_number=f.col("page"),
        section=f.col("section_"),
        extraction_method=f.col("method"),
    )


def _event_signal(frame: Any, rule: Rule, **cols: Any) -> Any:
    f = F()
    base = dict(
        signal_status=f.lit("CURRENT"),
        observed_date=f.col("event_date"),
        supporting_record_ids=f.array("record_id"),
        evidence_status=f.col("status"),
        source_table=f.lit("silver.project_events"),
        source_record_id=f.col("record_id"),
        document_id=f.col("evidence_document_id"),
        page_number=f.col("evidence_page_number"),
        section=f.col("evidence_section"),
        extraction_method=f.col("evidence_extraction_method"),
    )
    base.update(cols)
    return emit(frame, rule, **base)


def change_additional_financing(rule: Rule, s: dict[str, Any], **_: Any) -> Any:
    f = F()
    frame = s["project_events"].where(f.col("event_type") == "ADDITIONAL_FINANCING")
    amount = f.concat_ws(
        " ", text(f.col("additional_financing_amount")), f.col("additional_financing_currency")
    )
    return _event_signal(
        frame,
        rule,
        subject=f.concat(
            f.lit("Additional financing "),
            f.coalesce(f.col("event_date").cast("string"), f.lit("")),
        ),
        severity=f.lit("INFO"),
        current_value=amount,
        value_unit=f.lit("amount"),
        signal_description=f.concat(
            f.lit("Additional financing of "),
            amount,
            f.lit(" is documented ("),
            f.coalesce(f.col("event_date_basis"), f.lit("undated")),
            f.lit(": "),
            f.coalesce(f.col("event_date").cast("string"), f.lit("no date")),
            f.lit(")."),
        ),
    )


def change_cancellation(rule: Rule, s: dict[str, Any], **_: Any) -> Any:
    f = F()
    fin = s["project_financial_summary"].select(
        "project_id", "cancelled_total_usd", f.col("record_id").alias("fin_id")
    )
    frame = (
        s["project_events"]
        .where(f.col("event_type") == "CANCELLATION")
        .join(fin, "project_id")
        .where(f.col("cancelled_total_usd") > 0)
    )
    document_amount = f.concat_ws(" ", text(f.col("cancelled_amount")), f.col("cancelled_currency"))
    snapshot_amount = f.concat(
        text(f.col("cancelled_total_usd")), f.lit(" USD cancelled (loan snapshot)")
    )
    caveat = f.when(
        f.col("cancelled_currency") != "USD",
        f.lit(
            "Document amount and loan-snapshot amount are in different currencies; not compared."
        ),
    )
    return _event_signal(
        frame,
        rule,
        subject=f.concat(f.lit("Cancellation "), f.col("loan_number")),
        severity=f.lit("INFO"),
        current_value=document_amount,
        comparison_value=snapshot_amount,
        value_unit=f.lit("amount"),
        signal_description=f.concat(
            f.lit("A cancellation of "),
            document_amount,
            f.lit(" on "),
            f.col("loan_number"),
            f.lit(" is documented (value date "),
            f.coalesce(f.col("event_date").cast("string"), f.lit("not stated")),
            f.lit("); the loan snapshot reports "),
            snapshot_amount,
            f.lit(". Stated reason: "),
            f.coalesce(f.col("reason_text"), f.lit("not stated")),
            f.lit("."),
        ),
        caveats=f.filter(f.array(caveat), lambda x: x.isNotNull()),
        supporting_record_ids=f.array_sort(f.array("record_id", "fin_id")),
    )


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------


def risk_overall_elevated(rule: Rule, s: dict[str, Any], scales: RatingScales, **_: Any) -> Any:
    f = F()
    substantial = scales.risk["Substantial"]
    isr = s["isr_snapshots"].withColumn("rank", rank_expression("overall_risk_rating", scales.risk))
    history = isr.join(_latest_isr(s["isr_snapshots"]), "project_id")
    grouped = (
        runs(history, ["project_id"], f.col("rank") >= substantial)
        .groupBy("project_id", "_run", "latest_isr_sequence")
        .agg(
            f.count(f.lit(1)).alias("n"),
            f.min("isr_sequence").alias("first_seq"),
            f.max("isr_sequence").alias("last_seq"),
            f.array_sort(f.collect_list("record_id")).alias("ids"),
            *[
                f.max_by(c, "isr_sequence").alias(a)
                for c, a in (
                    ("record_id", "primary_id"),
                    ("overall_risk_rating", "rating"),
                    ("rank", "rank_"),
                    ("canonical_report_date", "last_date"),
                    ("document_id", "doc"),
                    ("overall_risk_rating_page_number", "page"),
                    ("overall_risk_rating_extraction_method", "method"),
                    ("overall_risk_rating_status", "primary_status"),
                )
            ],
        )
        .where(f.col("last_seq") == f.col("latest_isr_sequence"))
    )
    return emit(
        grouped,
        rule,
        subject=f.lit("Overall risk rating"),
        severity=f.when(f.col("rank_") >= scales.risk["High"], "HIGH").otherwise("WATCH"),
        signal_status=f.lit("CURRENT"),
        observed_date=f.col("last_date"),
        effective_sequence=f.col("last_seq").cast("bigint"),
        current_value=f.col("rating"),
        threshold=f.lit("WATCH Substantial, HIGH High"),
        value_unit=f.lit("rating"),
        signal_description=f.concat(
            f.lit("The overall risk rating in the latest ISR (sequence "),
            f.col("last_seq").cast("string"),
            f.lit(") is "),
            f.col("rating"),
            f.lit("; it has been Substantial or higher for "),
            f.col("n").cast("string"),
            f.lit(" consecutive ISRs (since sequence "),
            f.col("first_seq").cast("string"),
            f.lit(")."),
        ),
        supporting_record_ids=f.col("ids"),
        evidence_status=f.col("primary_status"),
        source_table=f.lit("silver.isr_snapshots"),
        source_record_id=f.col("primary_id"),
        document_id=f.col("doc"),
        page_number=f.col("page"),
        extraction_method=f.col("method"),
    )


BUILDERS: dict[str, Callable[..., Any]] = {
    "SCHEDULE_CLOSING_DATE_EXTENDED": schedule_closing_date_extended,
    "SCHEDULE_REPEATED_CLOSING_DATE_CHANGES": schedule_repeated_changes,
    "RATING_DOWNGRADE": rating_downgrade,
    "RATING_BELOW_SATISFACTORY_PERSISTENT": rating_below_satisfactory_persistent,
    "RATING_RECOVERY": rating_recovery,
    "RESULT_TARGET_DATE_PASSED_NOT_MET": result_target_date_passed,
    "RESULT_MOVED_AWAY_FROM_TARGET": result_moved_away,
    "RESULT_NO_CHANGE": result_no_change,
    "RESULT_TARGET_CHANGED": result_target_changed,
    "FINANCE_DISBURSEMENT_LAG": finance_disbursement_lag,
    "CHANGE_RESTRUCTURING": change_restructuring,
    "CHANGE_ADDITIONAL_FINANCING": change_additional_financing,
    "CHANGE_CANCELLATION": change_cancellation,
    "RISK_OVERALL_RATING_ELEVATED": risk_overall_elevated,
}


def check_rule_coverage(rules: RuleSet) -> None:
    configured = {r.rule_id for r in rules.rules}
    missing = sorted(set(rules.enabled) - set(BUILDERS))
    orphan = sorted(set(BUILDERS) - configured)
    if missing or orphan:
        raise ConfigurationError(
            f"rules without builders {missing}; builders without configured rules {orphan}"
        )


def build_signals(
    silver: dict[str, Any], progress: Any, rules: RuleSet, scales: RatingScales
) -> Any:
    check_rule_coverage(rules)
    frames = [
        BUILDERS[rule_id](rule, silver, scales=scales, progress=progress)
        for rule_id, rule in sorted(rules.enabled.items())
    ]
    signals = frames[0]
    for frame in frames[1:]:
        signals = signals.unionByName(frame)
    return signals
