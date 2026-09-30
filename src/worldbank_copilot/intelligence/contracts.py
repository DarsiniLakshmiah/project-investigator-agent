"""Explicit contracts for the Gold tables (Phase 7).

Built with the Phase 6 contract mechanism (``lakehouse.contracts``), so Gold rows get
the same identity (``record_id``), content hash (``record_hash``) and operational
columns as Bronze/Silver. Column descriptions become Delta column comments and state
where each value comes from.

Provenance classes (Claude.md §10): FACT (structured data), DOCUMENTED_FINDING
(source-document evidence), SYSTEM_DERIVED_SIGNAL (documented deterministic rule),
AI_INTERPRETATION (never produced in Phase 7), UNKNOWN.
"""

from __future__ import annotations

from worldbank_copilot.extraction.provenance import ExtractionMethod
from worldbank_copilot.intelligence.rules import CATEGORIES, SEVERITIES
from worldbank_copilot.lakehouse.contracts import (
    Column,
    DType,
    Role,
    TableContract,
    build_contract,
)

PROVENANCE_CLASSES = (
    "FACT",
    "DOCUMENTED_FINDING",
    "SYSTEM_DERIVED_SIGNAL",
    "AI_INTERPRETATION",
    "UNKNOWN",
)
METHODS = tuple(m.value for m in ExtractionMethod)
S, B, D, DEC, BOOL = DType.STRING, DType.BIGINT, DType.DATE, DType.DECIMAL, DType.BOOLEAN
ARR = DType.ARRAY_STRING


def c(
    name: str,
    dtype: DType,
    nullable: bool = True,
    role: Role = Role.DATA,
    vocab: tuple[str, ...] | None = None,
    doc: str | None = None,
) -> Column:
    return Column(name, dtype, nullable, role, vocab, doc)


def provenance(classes: tuple[str, ...] = PROVENANCE_CLASSES) -> list[Column]:
    """Primary evidence of a Gold row: Silver record, document location, method."""
    P = Role.PROVENANCE
    return [
        c("source_table", S, False, P, doc="Silver table of the primary source record."),
        c("source_record_id", S, False, P, doc="record_id of the primary Silver record."),
        c("document_id", S, True, P, doc="Source document (NULL for structured sources)."),
        c("page_number", B, True, P),
        c("section", S, True, P),
        c("extraction_method", S, True, P, METHODS),
        c("provenance_class", S, False, P, classes),
    ]


DATE_STATUSES = ("SOURCE_STATED", "SCHEDULED", "DERIVED_CANDIDATE", "UNDATED")
ORDERING_BASES = ("EVENT_DATE", "ISR_SEQUENCE_ADJUSTED", "CANDIDATE_DATE", "NONE")
RATING_CHANGES = ("UPGRADE", "DOWNGRADE", "UNCHANGED", "NOT_COMPARABLE")
TRENDS = ("TOWARD_TARGET", "AWAY_FROM_TARGET", "UNCHANGED", "NOT_COMPARABLE")
TARGET_STATUSES = ("MET", "NOT_MET", "NOT_EVALUABLE")
CALCULATION_STATUSES = (
    "OK",
    "EXTRACTION_NOT_EXACT",
    "DLI_LAYOUT_NOT_EVALUATED",
    "QUALITATIVE_UNIT",
    "MISSING_BASELINE",
    "MISSING_CURRENT",
    "MISSING_TARGET",
    "NON_NUMERIC_VALUE",
    "ZERO_TARGET_DISTANCE",
)
IDENTITY_REVIEW = ("EXACT_IDENTITY", "REVIEWED_ALIAS", "PENDING_ALIAS_REVIEW")
RISK_RECORD_TYPES = ("FORMAL_RISK_RATING", "ASSESSMENT_FINDING", "ISR_SORT_RATING")
SIGNAL_STATUSES = ("CURRENT", "HISTORICAL")


def project_360_contract() -> TableContract:
    F, Dc = "FACT (silver.projects)", "DOCUMENTED_FINDING (silver.isr_snapshots)"
    fin = "FACT (silver.project_financial_summary, loan snapshot)"
    body = [
        c("project_id", S, False, Role.KEY),
        c("project_name", S, doc=F),
        c("instrument", S, doc=F),
        c("project_status", S, doc=F),
        c("approval_date", D, doc=F),
        c("effectiveness_date", D, doc=F),
        c(
            "original_closing_date",
            D,
            doc="DOCUMENTED_FINDING (silver.project_enrichment: "
            "original closing of the first loan, agreed by explicit sources).",
        ),
        c("original_closing_date_status", S, doc="Enrichment status (never promoted)."),
        c("current_closing_date", D, doc=F),
        c("days_extended", B, doc="current_closing_date - original_closing_date (days)."),
        c("original_principal_usd", DEC, doc=fin),
        c("cancelled_usd", DEC, doc=fin),
        c("net_principal_usd", DEC, doc=fin),
        c("disbursed_usd", DEC, doc=fin),
        c("undisbursed_usd", DEC, doc=fin),
        c("disbursement_pct_of_net_principal", DEC, doc=fin),
        c("financial_snapshot_date", D, doc=fin),
        c("loan_count", B, doc=fin),
        c("loans_with_valuation_caveats", ARR, doc=fin),
        c("latest_isr_sequence", B, doc=Dc),
        c("latest_isr_date", D, doc=Dc),
        c("latest_isr_date_basis", S, doc=Dc),
        c("latest_do_rating", S, doc="PDO rating of the latest ISR (by sequence). " + Dc),
        c("latest_ip_rating", S, doc="Implementation progress rating, latest ISR. " + Dc),
        c("latest_overall_risk_rating", S, doc=Dc),
        c("previous_do_rating", S, doc="PDO rating of the preceding ISR (by sequence)."),
        c("previous_ip_rating", S, doc="IP rating of the preceding ISR (by sequence)."),
        c("do_rating_change", S, False, vocab=RATING_CHANGES),
        c("ip_rating_change", S, False, vocab=RATING_CHANGES),
        c("number_of_isrs", B, False),
        c("number_of_project_events", B, False),
        c(
            "number_of_restructurings",
            B,
            False,
            doc="Restructurings with a formal approval date (ISR restructuring history).",
        ),
        c(
            "number_of_restructuring_papers",
            B,
            False,
            doc="Undated restructuring papers (describe the dated restructurings).",
        ),
        c("formal_risk_count", B, False),
        c("assessment_finding_count", B, False),
        c("number_of_result_indicators", B, False),
        c("latest_results_reporting_date", D),
        c("current_attention_signal_count", B, False, doc="SYSTEM_DERIVED_SIGNAL"),
        c("current_watch_signal_count", B, False, doc="SYSTEM_DERIVED_SIGNAL"),
        c("current_high_signal_count", B, False, doc="SYSTEM_DERIVED_SIGNAL"),
        c(
            "source_record_ids",
            ARR,
            False,
            Role.PROVENANCE,
            doc="Silver record_ids the row was built from.",
        ),
    ]
    return build_contract(
        "project_360",
        "gold",
        "gold.project_360",
        body,
        ("project_id",),
        "One current-state record per project.",
    )


def timeline_contract() -> TableContract:
    body = [
        c("timeline_event_id", S, False, Role.KEY),
        c("project_id", S, False),
        c("event_type", S, False),
        c("event_subtype", S),
        c("event_title", S, False),
        c("event_description", S),
        c("event_date", D, doc="Only a date stated by a source; never a derived candidate."),
        c("event_date_status", S, False, vocab=DATE_STATUSES),
        c("event_date_basis", S),
        c("candidate_event_date", D, doc="Derived candidate (not an official date)."),
        c("candidate_date_basis", S),
        c("candidate_date_status", S),
        c("ordering_date", D, doc="Date used only to order the timeline."),
        c("ordering_basis", S, False, vocab=ORDERING_BASES),
        c("event_sequence", B, False, doc="1..n per project in timeline order."),
        c("isr_sequence", B),
        c("loan_number", S),
        c(
            "date_sequence_anomaly",
            BOOL,
            False,
            doc="ISR whose date is later than the next ISR's date (kept as printed).",
        ),
        c("extraction_status", S),
        *provenance(("FACT", "DOCUMENTED_FINDING")),
    ]
    return build_contract(
        "project_timeline",
        "gold",
        "gold.project_timeline",
        body,
        ("project_id", "source_table", "source_record_id", "event_type"),
        "Canonical chronological project history.",
        profile_groups=[("project_id",), ("event_type",), ("event_date_status",)],
    )


def result_progress_contract() -> TableContract:
    body = [
        c("project_id", S, False),
        c(
            "canonical_indicator_id",
            S,
            False,
            doc="Silver indicator_key: exact normalized name or reviewed alias only.",
        ),
        c("indicator_name", S, False),
        c("indicator_type", S),
        c("unit", S),
        c("layout", S),
        c("identity_basis", S),
        c("identity_review_status", S, False, vocab=IDENTITY_REVIEW),
        c("isr_sequence", B),
        c("reporting_date", D),
        c("baseline_value", S),
        c("current_value", S),
        c("target_value", S),
        c("baseline_number", DEC),
        c("current_number", DEC),
        c("target_number", DEC),
        c("target_date", S),
        c("target_date_parsed", D),
        c("previous_isr_sequence", B, doc="Previous observation of the same indicator."),
        c("previous_number", DEC),
        c("previous_target_number", DEC),
        c("absolute_change", DEC),
        c("percentage_change", DEC),
        c(
            "progress_percentage",
            DEC,
            doc="(current-baseline)/(target-baseline)*100; NULL unless calculation_status=OK.",
        ),
        c("target_gap", DEC, doc="target - current (in the indicator's unit)."),
        c("trend_direction", S, False, vocab=TRENDS),
        c("target_status", S, False, vocab=TARGET_STATUSES),
        c("calculation_status", S, False, vocab=CALCULATION_STATUSES),
        c("target_changed", BOOL, False),
        c("extraction_status", S, False),
        c("table_id", S, role=Role.PROVENANCE),
        *provenance(("DOCUMENTED_FINDING",)),
    ]
    return build_contract(
        "result_progress",
        "gold",
        "gold.result_progress",
        body,
        ("source_record_id",),
        "Results-framework observations with safe progress calculations.",
        profile_groups=[
            ("project_id",),
            ("calculation_status",),
            ("identity_review_status",),
            ("trend_direction",),
        ],
    )


def risk_register_contract() -> TableContract:
    body = [
        c("project_id", S, False),
        c("risk_or_finding_id", S, False),
        c("record_type", S, False, vocab=RISK_RECORD_TYPES),
        c("category", S),
        c("title", S, False),
        c("description", S),
        c("activity", S),
        c("rating_raw", S),
        c("rating", S),
        c("rating_rank", B),
        c("rating_at_approval", S),
        c("previous_rating", S),
        c("observed_date", D),
        c("isr_sequence", B),
        c("source_document_type", S),
        c("mitigation_text", S),
        c(
            "resolution_status",
            S,
            False,
            vocab=("NOT_STATED",),
            doc="Documents do not state resolution; never inferred.",
        ),
        c("extraction_status", S),
        *provenance(("DOCUMENTED_FINDING",)),
    ]
    return build_contract(
        "risk_register",
        "gold",
        "gold.risk_register",
        body,
        ("source_table", "source_record_id"),
        "Documented risks and assessment findings (framing preserved).",
        profile_groups=[("project_id", "record_type")],
    )


def attention_signals_contract() -> TableContract:
    body = [
        c("signal_id", S, False, Role.KEY),
        c("project_id", S, False),
        c("rule_id", S, False),
        c("rule_version", S, False),
        c("signal_category", S, False, vocab=CATEGORIES),
        c("signal_type", S, False),
        c("subject", S, False, doc="What the signal is about (rating, indicator, ...)."),
        c("signal_title", S, False),
        c("signal_description", S, False),
        c("severity", S, False, vocab=SEVERITIES),
        c("signal_status", S, False, vocab=SIGNAL_STATUSES),
        c("observed_date", D),
        c("effective_sequence", B),
        c("current_value", S),
        c("comparison_value", S),
        c("threshold", S),
        c("value_unit", S),
        c("rule_description", S, False),
        c("supporting_record_ids", ARR, False, Role.PROVENANCE),
        c("evidence_status", S, doc="Extraction status of the primary source record."),
        c("caveats", ARR, False),
        *provenance(("SYSTEM_DERIVED_SIGNAL",)),
    ]
    return build_contract(
        "attention_signals",
        "gold",
        "gold.attention_signals",
        body,
        ("rule_id", "project_id", "subject", "effective_sequence"),
        "Deterministic, evidence-backed implementation signals. Not predictions.",
        profile_groups=[("project_id",), ("rule_id",), ("severity",), ("signal_status",)],
    )


def quality_contract() -> TableContract:
    body = [
        c("check_id", S, False),
        c("severity", S, False, vocab=("ERROR", "WARNING", "INFO")),
        c("table_name", S, False),
        c("project_id", S),
        c("observation_count", B, False),
        c("message", S, False),
        c("details_json", S),
    ]
    return build_contract(
        "quality_observations",
        "gold",
        "gold.quality_observations",
        body,
        ("check_id", "table_name", "project_id"),
        "Gold invariants and quality checks (ERROR blocks the write).",
        profile_groups=[("severity",)],
    )


GOLD_CONTRACTS = {
    "project_360": project_360_contract,
    "project_timeline": timeline_contract,
    "result_progress": result_progress_contract,
    "risk_register": risk_register_contract,
    "attention_signals": attention_signals_contract,
    "quality_observations": quality_contract,
}


GOLD_LOCK_FILE = "gold_contracts.lock.json"


def contracts_lock() -> dict:
    return {f"gold.{name}": fn().to_dict() for name, fn in GOLD_CONTRACTS.items()}


def write_lock(config_dir) -> None:
    """Re-record the frozen Gold contracts (only for a reviewed schema change)."""
    import json
    from pathlib import Path

    path = Path(config_dir) / GOLD_LOCK_FILE
    text = json.dumps(contracts_lock(), indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")
