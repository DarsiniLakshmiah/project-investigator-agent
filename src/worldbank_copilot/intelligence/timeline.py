"""gold.project_timeline: canonical chronological project history (Spark).

Sources: silver.projects (approval, effectiveness, current closing), silver.project_
enrichment (original closing), silver.isr_snapshots (ISR reporting points) and
silver.project_events (formal events).

Dates:
* ``event_date`` is only ever a date stated by a source. An undated restructuring paper
  keeps ``event_date = NULL``; its derived ``candidate_event_date`` is carried in its own
  columns with status DERIVED_CANDIDATE and never promoted.
* ``ordering_date`` exists only to order the timeline. For ISRs it is the smallest
  canonical date of that ISR and every later ISR, so ISR sequence order is kept when a
  printed date is out of order (P179039 ISR 5); such ISRs get
  ``date_sequence_anomaly = true`` and their ``event_date`` stays as printed.
"""

from __future__ import annotations

from typing import Any

from worldbank_copilot.intelligence.frames import F

_TYPE_ORDER = {
    "APPROVAL": 1,
    "EFFECTIVENESS": 2,
    "ORIGINAL_CLOSING_DATE": 3,
    "ISR_REPORT": 6,
    "CURRENT_CLOSING_DATE": 9,
}


def _common(frame: Any, **cols: Any) -> Any:
    f = F()
    defaults = {
        "event_subtype": f.lit(None).cast("string"),
        "event_description": f.lit(None).cast("string"),
        "event_date_basis": f.lit(None).cast("string"),
        "candidate_event_date": f.lit(None).cast("date"),
        "candidate_date_basis": f.lit(None).cast("string"),
        "candidate_date_status": f.lit(None).cast("string"),
        "isr_sequence": f.lit(None).cast("bigint"),
        "loan_number": f.lit(None).cast("string"),
        "extraction_status": f.lit(None).cast("string"),
        "document_id": f.lit(None).cast("string"),
        "page_number": f.lit(None).cast("bigint"),
        "section": f.lit(None).cast("string"),
        "extraction_method": f.lit(None).cast("string"),
    }
    defaults.update(cols)
    names = [
        "project_id",
        "event_type",
        "event_title",
        "event_date",
        "event_date_status",
        "source_table",
        "source_record_id",
        "provenance_class",
        *sorted(k for k in defaults if k not in {"project_id"}),
    ]
    seen = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return frame.select(
        *[defaults[n].alias(n) if n in defaults else f.col(n).alias(n) for n in seen]
    )


def _project_milestones(projects: Any) -> Any:
    f = F()
    out = []
    for event_type, column, status, title in (
        ("APPROVAL", "approval_date", "SOURCE_STATED", "Board approval"),
        ("EFFECTIVENESS", "effective_date", "SOURCE_STATED", "Project effectiveness"),
        ("CURRENT_CLOSING_DATE", "current_closing_date", "SCHEDULED", "Current closing date"),
    ):
        frame = projects.where(f.col(column).isNotNull()).select(
            "project_id",
            f.col("record_id").alias("source_record_id"),
            f.col(column).alias("event_date"),
        )
        out.append(
            _common(
                frame.withColumn("event_type", f.lit(event_type))
                .withColumn("event_title", f.lit(title))
                .withColumn("event_date_status", f.lit(status))
                .withColumn("source_table", f.lit("silver.projects"))
                .withColumn("provenance_class", f.lit("FACT")),
                event_subtype=f.lit("PROJECT"),
                event_date_basis=f.lit(
                    f"silver.projects.{column} (Projects & Operations workbook)"
                ),
            )
        )
    return out


def _original_closing(enrichment: Any) -> Any:
    f = F()
    frame = enrichment.where(
        (f.col("target_table") == "silver_projects")
        & (f.col("target_field") == "original_closing_date")
        & f.col("value").isNotNull()
    )
    return _common(
        frame.select(
            "project_id",
            "loan_number",
            "status",
            "rule",
            "value",
            "record_id",
            "evidence_document_id",
            "evidence_page_number",
            "evidence_section",
            "evidence_extraction_method",
        )
        .withColumn("event_type", f.lit("ORIGINAL_CLOSING_DATE"))
        .withColumn("event_title", f.lit("Original closing date"))
        .withColumn("event_date", f.col("value"))
        .withColumn("event_date_status", f.lit("SOURCE_STATED"))
        .withColumn("source_table", f.lit("silver.project_enrichment"))
        .withColumn("source_record_id", f.col("record_id"))
        .withColumn("provenance_class", f.lit("DOCUMENTED_FINDING")),
        event_subtype=f.col("loan_number"),
        event_description=f.concat(
            f.lit("Original closing of the first loan "),
            f.col("loan_number"),
            f.lit(": "),
            f.col("rule"),
        ),
        event_date_basis=f.lit("explicit document sources agree (silver.project_enrichment)"),
        loan_number=f.col("loan_number"),
        extraction_status=f.col("status"),
        document_id=f.col("evidence_document_id"),
        page_number=f.col("evidence_page_number"),
        section=f.col("evidence_section"),
        extraction_method=f.col("evidence_extraction_method"),
    )


def _isr_points(isr: Any) -> Any:
    f = F()
    description = f.concat_ws(
        "; ",
        f.concat(f.lit("PDO rating: "), f.coalesce(f.col("pdo_rating"), f.lit("not found"))),
        f.concat(
            f.lit("Implementation progress: "),
            f.coalesce(f.col("implementation_progress_rating"), f.lit("not found")),
        ),
        f.concat(
            f.lit("Overall risk: "), f.coalesce(f.col("overall_risk_rating"), f.lit("not found"))
        ),
    )
    return _common(
        isr.select("*")
        .withColumn("event_type", f.lit("ISR_REPORT"))
        .withColumn(
            "event_title", f.concat(f.lit("ISR sequence "), f.col("isr_sequence").cast("string"))
        )
        .withColumn("event_date", f.col("canonical_report_date"))
        .withColumn("event_date_status", f.lit("SOURCE_STATED"))
        .withColumn("source_table", f.lit("silver.isr_snapshots"))
        .withColumn("source_record_id", f.col("record_id"))
        .withColumn("provenance_class", f.lit("DOCUMENTED_FINDING")),
        event_subtype=f.col("isr_number"),
        event_description=description,
        event_date_basis=f.concat(f.lit("ISR "), f.col("canonical_date_basis")),
        isr_sequence=f.col("isr_sequence").cast("bigint"),
        # The canonical date comes from document metadata (header / archive date); no page
        # or extraction method is claimed for it.
        document_id=f.col("document_id"),
    )


def _event_description() -> Any:
    f = F()
    closing = f.concat_ws(
        " ",
        f.col("loan_number"),
        f.lit("closing date"),
        f.col("old_closing_date").cast("string"),
        f.lit("->"),
        f.col("new_closing_date").cast("string"),
    )
    cancellation = f.concat_ws(
        " ",
        f.col("loan_number"),
        f.lit("cancelled"),
        f.col("cancelled_amount").cast("string"),
        f.col("cancelled_currency"),
    )
    financing = f.concat_ws(
        " ",
        f.lit("Additional financing"),
        f.col("additional_financing_amount").cast("string"),
        f.col("additional_financing_currency"),
    )
    return (
        f.when(f.col("event_type") == "CLOSING_DATE_CHANGE", closing)
        .when(f.col("event_type") == "CANCELLATION", cancellation)
        .when(f.col("event_type") == "ADDITIONAL_FINANCING", financing)
        .otherwise(f.col("change_description"))
    )


def _formal_events(events: Any) -> Any:
    f = F()
    status = (
        f.when(f.col("event_date").isNotNull(), "SOURCE_STATED")
        .when(f.col("candidate_event_date").isNotNull(), "DERIVED_CANDIDATE")
        .otherwise("UNDATED")
    )
    title = f.initcap(f.regexp_replace(f.col("event_type"), "_", " "))
    return _common(
        events.select("*")
        .withColumn("event_title", title)
        .withColumn("event_date_status", status)
        .withColumn("source_table", f.lit("silver.project_events"))
        .withColumn("source_record_id", f.col("record_id"))
        .withColumn("provenance_class", f.lit("DOCUMENTED_FINDING")),
        event_subtype=f.col("loan_number"),
        event_description=_event_description(),
        event_date_basis=f.col("event_date_basis"),
        candidate_event_date=f.col("candidate_event_date"),
        candidate_date_basis=f.col("candidate_date_basis"),
        candidate_date_status=f.col("candidate_date_status"),
        loan_number=f.col("loan_number"),
        extraction_status=f.col("status"),
        document_id=f.col("evidence_document_id"),
        page_number=f.col("evidence_page_number"),
        section=f.col("evidence_section"),
        extraction_method=f.col("evidence_extraction_method"),
    )


def build_timeline(silver: dict[str, Any]) -> Any:
    from pyspark.sql import Window

    f = F()
    parts = [
        *_project_milestones(silver["projects"]),
        _original_closing(silver["project_enrichment"]),
        _isr_points(silver["isr_snapshots"]),
        _formal_events(silver["project_events"]),
    ]
    timeline = parts[0]
    for part in parts[1:]:
        timeline = timeline.unionByName(part)

    # ISR ordering: suffix minimum of canonical dates keeps sequence order.
    later = (
        Window.partitionBy("project_id", "event_type")
        .orderBy(f.col("isr_sequence").desc())
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    nxt = Window.partitionBy("project_id", "event_type").orderBy("isr_sequence")
    is_isr = f.col("event_type") == "ISR_REPORT"
    timeline = (
        timeline.withColumn("_suffix_min", f.when(is_isr, f.min("event_date").over(later)))
        .withColumn("_next_date", f.when(is_isr, f.lead("event_date").over(nxt)))
        .withColumn(
            "date_sequence_anomaly",
            f.coalesce(is_isr & (f.col("event_date") > f.col("_next_date")), f.lit(False)),
        )
        .withColumn(
            "ordering_date",
            f.coalesce(
                f.when(is_isr, f.col("_suffix_min")),
                f.col("event_date"),
                f.col("candidate_event_date"),
            ),
        )
        .withColumn(
            "ordering_basis",
            f.when(is_isr & (f.col("_suffix_min") != f.col("event_date")), "ISR_SEQUENCE_ADJUSTED")
            .when(f.col("event_date").isNotNull(), "EVENT_DATE")
            .when(f.col("candidate_event_date").isNotNull(), "CANDIDATE_DATE")
            .otherwise("NONE"),
        )
    )
    priority = f.coalesce(
        *[f.when(f.col("event_type") == k, f.lit(v)) for k, v in _TYPE_ORDER.items()], f.lit(5)
    )
    order = Window.partitionBy("project_id").orderBy(
        f.col("ordering_date").asc_nulls_last(),
        priority,
        f.col("isr_sequence").asc_nulls_last(),
        "event_type",
        "source_record_id",
    )
    timeline = timeline.withColumn("event_sequence", f.row_number().over(order))
    timeline = timeline.withColumn(
        "timeline_event_id",
        f.substring(
            f.sha2(
                f.concat_ws("|", "project_id", "source_table", "source_record_id", "event_type"),
                256,
            ),
            1,
            32,
        ),
    )
    return timeline.drop("_suffix_min", "_next_date")
