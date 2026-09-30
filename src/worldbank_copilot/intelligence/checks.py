"""Gold invariants and quality checks (computed in Spark; only counts reach the driver).

ERROR checks are invariants that must hold before anything is written (a non-zero count
blocks the write). WARNING / INFO checks describe known conditions that stay queryable
in gold.quality_observations. Invariants are independent of any previous run: they
compare Gold with Silver and with the rule configuration, never with remembered counts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from worldbank_copilot.intelligence.frames import F
from worldbank_copilot.intelligence.rules import RuleSet

COMMENTS_LABEL = "comments on achieving targets"
PREDICTIVE_WORDING = r"(?i)\b(will fail|failure probability|probability of|likely to fail|predict)"


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    severity: str  # ERROR | WARNING | INFO
    table_name: str
    count: int
    message: str
    details: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.severity == "ERROR":
            return "FAIL" if self.count else "PASS"
        return "OBSERVED" if self.count else "PASS"


def _count(frame: Any) -> int:
    return frame.count()


def _ids(silver: dict[str, Any]) -> Any:
    """(table, record_id) of every Silver input record."""
    f = F()
    frames = [
        frame.select(f.lit(f"silver.{name}").alias("t"), f.col("record_id").alias("id"))
        for name, frame in silver.items()
    ]
    out = frames[0]
    for frame in frames[1:]:
        out = out.unionByName(frame)
    return out


def _grouped(frame: Any, *columns: str) -> dict[str, int]:
    rows = frame.groupBy(*columns).count().collect()
    return {
        "|".join(str(r[c]) for c in columns): r["count"]
        for r in sorted(rows, key=lambda r: tuple(str(r[c]) for c in columns))
    }


def run_checks(silver: dict[str, Any], gold: dict[str, Any], rules: RuleSet) -> list[CheckResult]:
    f = F()
    ids = _ids(silver)
    p360, timeline, progress = (
        gold["project_360"],
        gold["project_timeline"],
        gold["result_progress"],
    )
    risks, signals = gold["risk_register"], gold["attention_signals"]
    projects = silver["projects"].select("project_id")
    out: list[CheckResult] = []

    def add(check_id, severity, table, count, message, **details):
        out.append(CheckResult(check_id, severity, table, int(count), message, details))

    # --- project_360 ---------------------------------------------------------------
    add(
        "P360_ONE_ROW_PER_PROJECT",
        "ERROR",
        "gold.project_360",
        _count(projects.join(p360, "project_id", "left_anti"))
        + _count(p360.join(projects, "project_id", "left_anti"))
        + (_count(p360) - p360.select("project_id").distinct().count()),
        "Every Silver project has exactly one project_360 row and no other row exists.",
    )

    # --- timeline -------------------------------------------------------------------
    add(
        "TIMELINE_PROJECTS_RESOLVE",
        "ERROR",
        "gold.project_timeline",
        _count(timeline.select("project_id").join(projects, "project_id", "left_anti")),
        "Every timeline project_id exists in silver.projects.",
    )
    add(
        "TIMELINE_SOURCES_RESOLVE",
        "ERROR",
        "gold.project_timeline",
        _count(
            timeline.join(
                ids,
                (timeline["source_table"] == ids["t"])
                & (timeline["source_record_id"] == ids["id"]),
                "left_anti",
            )
        ),
        "Every timeline row resolves to its Silver source record.",
    )
    events = silver["project_events"].select(
        f.col("record_id").alias("rid"), f.col("event_date").alias("silver_date")
    )
    from_events = timeline.where(f.col("source_table") == "silver.project_events").join(
        events, f.col("source_record_id") == f.col("rid")
    )
    add(
        "TIMELINE_NO_DERIVED_DATE_PROMOTED",
        "ERROR",
        "gold.project_timeline",
        _count(from_events.where(~f.col("event_date").eqNullSafe(f.col("silver_date"))))
        + _count(
            timeline.where(
                (f.col("event_date_status") == "DERIVED_CANDIDATE")
                & f.col("event_date").isNotNull()
            )
        ),
        "event_date equals the source-stated Silver date; candidate dates are never promoted.",
    )
    from pyspark.sql import Window

    isr_rows = timeline.where(f.col("event_type") == "ISR_REPORT")
    w = Window.partitionBy("project_id").orderBy("event_sequence")
    add(
        "TIMELINE_ISR_SEQUENCE_ORDER",
        "ERROR",
        "gold.project_timeline",
        _count(
            isr_rows.withColumn("_prev", f.lag("isr_sequence").over(w)).where(
                f.col("_prev") > f.col("isr_sequence")
            )
        ),
        "ISRs appear in the timeline in ISR sequence order.",
    )
    add(
        "TIMELINE_DATES_PLAUSIBLE",
        "ERROR",
        "gold.project_timeline",
        _count(
            timeline.where(
                (f.col("event_date") < f.lit("1944-07-01").cast("date"))
                | (f.col("event_date") > f.lit("2100-12-31").cast("date"))
            )
        ),
        "No impossible event dates.",
    )

    # --- result_progress ------------------------------------------------------------
    results = silver["project_results"].select(
        f.col("record_id").alias("rid"), f.col("indicator_key").alias("silver_key")
    )
    add(
        "RESULTS_ROWS_RESOLVE",
        "ERROR",
        "gold.result_progress",
        _count(progress.join(results, f.col("source_record_id") == f.col("rid"), "left_anti"))
        + _count(results.join(progress, f.col("source_record_id") == f.col("rid"), "left_anti")),
        "One result_progress row per silver.project_results row, and no other.",
    )
    add(
        "RESULTS_IDENTITY_PRESERVED",
        "ERROR",
        "gold.result_progress",
        _count(
            progress.join(results, f.col("source_record_id") == f.col("rid")).where(
                f.col("canonical_indicator_id") != f.col("silver_key")
            )
        ),
        "canonical_indicator_id is the Silver indicator_key (no merging in Gold).",
    )
    pending = silver["indicator_match_candidates"].where(f.col("review_status") == "PENDING_REVIEW")
    series = progress.select("canonical_indicator_id").distinct()
    add(
        "RESULTS_PENDING_ALIASES_SEPARATE",
        "ERROR",
        "gold.result_progress",
        _count(pending.where(f.col("indicator_key_a") == f.col("indicator_key_b")))
        + _count(
            pending.join(
                series, f.col("indicator_key_a") == f.col("canonical_indicator_id"), "left_anti"
            )
        )
        + _count(
            pending.join(
                series, f.col("indicator_key_b") == f.col("canonical_indicator_id"), "left_anti"
            )
        ),
        "Both indicators of every pending alias pair remain separate series in Gold.",
    )
    add(
        "RESULTS_PROGRESS_ONLY_WHEN_VALID",
        "ERROR",
        "gold.result_progress",
        _count(
            progress.where(
                (f.col("progress_percentage").isNotNull() & (f.col("calculation_status") != "OK"))
                | (f.col("progress_percentage").isNull() & (f.col("calculation_status") == "OK"))
            )
        ),
        "progress_percentage is set exactly when calculation_status is OK.",
    )

    # --- risk_register --------------------------------------------------------------
    appraisal = risks.where(f.col("record_type") != "ISR_SORT_RATING")
    silver_risks = silver["appraisal_risks"].select(
        f.col("record_id").alias("rid"), f.col("framing")
    )
    add(
        "RISKS_RESOLVE_AND_KEEP_FRAMING",
        "ERROR",
        "gold.risk_register",
        _count(appraisal.join(silver_risks, f.col("source_record_id") == f.col("rid"), "left_anti"))
        + _count(
            silver_risks.join(appraisal, f.col("source_record_id") == f.col("rid"), "left_anti")
        )
        + _count(
            appraisal.join(silver_risks, f.col("source_record_id") == f.col("rid")).where(
                f.col("record_type") != f.col("framing")
            )
        )
        + _count(
            risks.join(
                ids,
                (risks["source_table"] == ids["t"]) & (risks["source_record_id"] == ids["id"]),
                "left_anti",
            )
        ),
        "Every risk/finding resolves to Silver and keeps its framing.",
    )

    # --- attention_signals ----------------------------------------------------------
    enabled = rules.enabled
    versions = [f"{r.rule_id}@v{r.version}" for r in enabled.values()]
    add(
        "SIGNALS_VALID_RULES",
        "ERROR",
        "gold.attention_signals",
        _count(
            signals.where(~f.col("rule_id").isin(*enabled) | ~f.col("rule_version").isin(*versions))
        ),
        "Every signal references an enabled rule at its configured version.",
    )
    add(
        "SIGNALS_HAVE_EVIDENCE",
        "ERROR",
        "gold.attention_signals",
        _count(
            signals.where(
                (f.size("supporting_record_ids") == 0)
                | ~f.array_contains("supporting_record_ids", f.col("source_record_id"))
            )
        ),
        "Every signal has supporting records including its primary source record.",
    )
    supporting = signals.select(f.explode("supporting_record_ids").alias("sid")).distinct()
    all_ids = ids.select(f.col("id").alias("sid")).distinct()
    add(
        "SIGNALS_EVIDENCE_RESOLVES",
        "ERROR",
        "gold.attention_signals",
        _count(supporting.join(all_ids, "sid", "left_anti"))
        + _count(
            signals.join(
                ids,
                (signals["source_table"] == ids["t"]) & (signals["source_record_id"] == ids["id"]),
                "left_anti",
            )
        ),
        "Every primary and supporting record id resolves to a Silver record.",
    )
    add(
        "SIGNALS_NEUTRAL_WORDING",
        "ERROR",
        "gold.attention_signals",
        _count(signals.where(f.col("signal_description").rlike(PREDICTIVE_WORDING))),
        "No predictive or probabilistic wording in signal descriptions.",
    )
    ai = sum(
        _count(frame.where(f.col("provenance_class") == "AI_INTERPRETATION"))
        for frame in (timeline, progress, risks, signals)
    )
    add(
        "NO_AI_INTERPRETATION",
        "ERROR",
        "gold.*",
        ai,
        "No Gold row is classified AI_INTERPRETATION in Phase 7.",
    )

    # --- observations (queryable, non-blocking) ---------------------------------------
    not_ok = progress.where(f.col("calculation_status") != "OK")
    add(
        "RESULTS_NOT_EVALUABLE",
        "INFO",
        "gold.result_progress",
        _count(not_ok),
        "Result observations without a safe progress calculation (reason per status).",
        by_status=_grouped(not_ok, "calculation_status"),
    )
    pend = progress.where(f.col("identity_review_status") == "PENDING_ALIAS_REVIEW")
    add(
        "RESULTS_PENDING_ALIAS_REVIEW",
        "INFO",
        "gold.result_progress",
        _count(pend),
        "Observations of indicators in a pending alias pair (kept as separate series).",
        pending_pairs=_count(pending),
    )
    label = silver["project_results"].where(f.lower(f.trim(f.col("comments"))) == COMMENTS_LABEL)
    add(
        "SILVER_COMMENTS_LABEL_EXTRACTION",
        "WARNING",
        "silver.project_results",
        _count(label),
        "Silver 'comments' holds the label 'Comments on achieving targets' instead of the "
        "comment text; comments are not used by any Gold calculation.",
        by_project_isr=_grouped(label, "project_id", "isr_sequence"),
    )
    anomalies = timeline.where(f.col("date_sequence_anomaly"))
    add(
        "ISR_SEQUENCE_DATE_ANOMALY",
        "WARNING",
        "gold.project_timeline",
        _count(anomalies),
        "ISR dates out of sequence order; kept as printed, ordered by ISR sequence.",
        isrs=_grouped(anomalies, "project_id", "isr_sequence"),
    )
    before = progress.join(
        silver["projects"].select("project_id", "effective_date"), "project_id"
    ).where(f.col("target_date_parsed") < f.col("effective_date"))
    add(
        "RESULTS_TARGET_DATE_BEFORE_EFFECTIVENESS",
        "WARNING",
        "gold.result_progress",
        _count(before),
        "Printed target dates earlier than project effectiveness; kept as "
        "printed, excluded from RESULT_TARGET_DATE_PASSED_NOT_MET.",
        by_project=_grouped(before, "project_id"),
    )
    negative = progress.where(f.col("progress_percentage") < 0)
    add(
        "RESULTS_NEGATIVE_PROGRESS",
        "INFO",
        "gold.result_progress",
        _count(negative),
        "Current value on the opposite side of the baseline from the target (negative "
        "progress) as printed; worth reviewing against the source (e.g. baseline definitions).",
        by_project=_grouped(negative, "project_id"),
    )
    add(
        "TIMELINE_CANDIDATE_DATES",
        "INFO",
        "gold.project_timeline",
        _count(timeline.where(f.col("event_date_status") == "DERIVED_CANDIDATE")),
        "Undated events carried with derived candidate dates (event_date stays NULL).",
    )
    trusted = ("EXACT", "NORMALIZED", "DERIVED_FROM_EXPLICIT_SOURCE")
    weak = signals.where(
        f.col("evidence_status").isNotNull() & ~f.col("evidence_status").isin(*trusted)
    )
    add(
        "SIGNALS_ON_NON_EXACT_EVIDENCE",
        "WARNING",
        "gold.attention_signals",
        _count(weak),
        "Signals whose primary evidence record is not EXACT (see evidence_status).",
        by_rule=_grouped(weak, "rule_id", "evidence_status"),
    )
    dq = silver["data_quality_observations"]
    add(
        "SILVER_QUALITY_CARRIED",
        "INFO",
        "silver.data_quality_observations",
        _count(dq),
        "Silver quality observations remain queryable next to Gold.",
        by_severity=_grouped(dq, "severity"),
    )
    return out


def quality_frame(spark: Any, results: list[CheckResult]) -> Any:
    """Check results as a DataFrame (built from literals: no Python worker needed)."""
    f = F()
    frames = [
        spark.range(1).select(
            f.lit(r.check_id).alias("check_id"),
            f.lit(r.severity).alias("severity"),
            f.lit(r.table_name).alias("table_name"),
            f.lit(None).cast("string").alias("project_id"),
            f.lit(r.count).cast("bigint").alias("observation_count"),
            f.lit(f"[{r.status}] {r.message}").alias("message"),
            f.lit(json.dumps(r.details, sort_keys=True)).alias("details_json"),
        )
        for r in results
    ]
    out = frames[0]
    for frame in frames[1:]:
        out = out.unionByName(frame)
    return out


def blocking(results: list[CheckResult]) -> list[CheckResult]:
    return [r for r in results if r.severity == "ERROR" and r.count]
