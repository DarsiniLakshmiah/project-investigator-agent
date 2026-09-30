"""Document-derived Silver schemas (Phase 5).

All four datasets keep full provenance (``EvidenceRef``) and deterministic
statuses. Values that could not be established are NULL with a status and an
issue, never guessed.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from worldbank_copilot.extraction.provenance import (
    EvidenceRef,
    ExtractionIssue,
    ExtractionMethod,
    ExtractionStatus,
)


class _Row(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractedRating(_Row):
    raw_rating: str | None
    normalized_rating: str | None
    status: ExtractionStatus
    evidence: EvidenceRef | None = None


class SortRating(_Row):
    risk_category: str
    rating_at_approval: ExtractedRating | None = None
    previous_rating: ExtractedRating | None = None
    current_rating: ExtractedRating | None = None


class LoanDisbursement(_Row):
    """Per-loan financial line as printed in an ISR (US$ millions unless stated)."""

    loan_number: str  # Statement notation, e.g. IBRD86010
    loan_status: str | None = None
    currency: str | None = None
    original_musd: Decimal | None = None
    revised_musd: Decimal | None = None
    cancelled_musd: Decimal | None = None
    disbursed_musd: Decimal | None = None
    undisbursed_musd: Decimal | None = None
    historical_disbursed_musd: Decimal | None = Field(
        default=None, description="'Historical Disbursed' column as printed (newer ISRs)."
    )
    disbursed_pct_reported: Decimal | None = None
    status: ExtractionStatus
    evidence: EvidenceRef


class LoanKeyDates(_Row):
    loan_number: str
    approval_date: date | None = None
    signing_date: date | None = None
    effectiveness_date: date | None = None
    original_closing_date: date | None = None
    revised_closing_date: date | None = None
    status: ExtractionStatus
    evidence: EvidenceRef


class NarrativeSection(_Row):
    title: str
    text: str
    start_page: int
    end_page: int
    evidence: EvidenceRef


class IsrSnapshot(_Row):
    """silver.isr_snapshots: one row per ISR report."""

    project_id: str
    document_id: str
    isr_sequence: int | None
    isr_number: str | None
    archive_date: date | None
    header_date: date | None
    canonical_report_date: date | None
    canonical_date_basis: str | None = Field(
        description="'header_date' when a header/report date was identified, else "
        "'archive_date' (policy: IMPLEMENTATION_PLAN.md Phase 5)."
    )
    date_difference_days: int | None = Field(description="header_date - archive_date")
    project_status: str | None = None
    pdo_rating: ExtractedRating | None = None
    implementation_progress_rating: ExtractedRating | None = None
    overall_risk_rating: ExtractedRating | None = None
    previous_pdo_rating: ExtractedRating | None = None
    previous_implementation_rating: ExtractedRating | None = None
    previous_overall_risk_rating: ExtractedRating | None = None
    sort_ratings: list[SortRating] = Field(default_factory=list)
    loan_disbursements: list[LoanDisbursement] = Field(default_factory=list)
    loan_key_dates: list[LoanKeyDates] = Field(default_factory=list)
    commitment_amount_musd: Decimal | None = Field(
        default=None, description="Sum of per-loan 'Revised' amounts (DERIVED), US$ millions."
    )
    disbursed_amount_musd: Decimal | None = Field(
        default=None, description="Sum of per-loan 'Disbursed' amounts (DERIVED), US$ millions."
    )
    disbursement_pct_reported: Decimal | None = Field(
        default=None, description="Explicit '% Disbursed' only when the ISR reports one loan."
    )
    key_issues_text: str | None = None
    key_decisions_text: str | None = None
    implementation_status_text: str | None = None
    narrative_sections: list[NarrativeSection] = Field(default_factory=list)
    restructuring_history: list[EvidenceRef] = Field(default_factory=list)
    source_document: str
    source_pages: list[int]
    source_refs: list[EvidenceRef]
    quality_issues: list[ExtractionIssue] = Field(default_factory=list)


class ResultObservation(_Row):
    """silver.project_results: project + indicator + observation (ISR)."""

    project_id: str
    indicator_key: str
    indicator_id_source: str | None = None  # e.g. IN00867688 when printed
    indicator_name_raw: str
    indicator_name_normalized: str
    indicator_type: str | None = None  # PDO | INTERMEDIATE | DLI
    indicator_tags: list[str] = Field(
        default_factory=list, description="Tags printed after the unit, e.g. DLI, CRI."
    )
    reported_status: str | None = Field(
        default=None, description="DLI status as printed (e.g. 'Not Due', 'Achieved')."
    )
    result_area: str | None = None
    unit: str | None = None
    baseline_value: str | None = None
    baseline_date: str | None = None
    previous_value: str | None = None
    previous_date: str | None = None
    current_value: str | None = None
    current_date: str | None = None
    target_value: str | None = None
    target_date: str | None = None
    comments: str | None = None
    isr_sequence: int | None = None
    observation_date: date | None = None
    source_document: str
    source_page: int
    source_table: str | None = None
    layout: str  # BLOCK | WIDE
    extraction_method: ExtractionMethod
    status: ExtractionStatus
    source_ref: EvidenceRef
    name_source_ref: EvidenceRef | None = Field(
        default=None, description="Set when the name was completed from the PDF text layer."
    )
    identity_basis: str | None = Field(
        default=None, description="EXACT_NAME | SOURCE_INDICATOR_ID | ALIAS (see identity rules)"
    )
    quality_issues: list[ExtractionIssue] = Field(default_factory=list)


class AppraisalRisk(_Row):
    """silver.appraisal_risks: what an appraisal-stage source identified."""

    project_id: str
    risk_id: str
    framing: str  # FORMAL_RISK_RATING | ASSESSMENT_FINDING
    risk_category: str | None
    risk_description: str | None = None
    activity: str | None = None
    risk_rating: ExtractedRating | None = None
    mitigation_text: str | None = None
    rating_justification: str | None = None
    identified_date: date | None = None
    source_document: str
    source_document_type: str | None
    source_page: int
    source_section: str | None
    source_text: str | None
    extraction_method: ExtractionMethod
    status: ExtractionStatus
    source_refs: list[EvidenceRef]
    quality_issues: list[ExtractionIssue] = Field(default_factory=list)


class ProjectEvent(_Row):
    """silver.project_events: formal events stated in authoritative documents."""

    project_id: str
    event_id: str
    event_type: str
    event_date: date | None = None
    event_date_basis: str | None = None
    # Derived, never authoritative: a deterministic candidate for an undated paper.
    # event_date stays NULL; see events.restructuring_date_candidates.
    candidate_event_date: date | None = None
    candidate_date_basis: str | None = None
    candidate_date_status: ExtractionStatus | None = None
    loan_number: str | None = None
    old_closing_date: date | None = None
    new_closing_date: date | None = None
    additional_financing_amount: Decimal | None = None
    additional_financing_currency: str | None = None
    cancelled_amount: Decimal | None = None
    cancelled_currency: str | None = None
    results_framework_changed: bool | None = None
    components_changed: bool | None = None
    fund_reallocation: bool | None = None
    change_flags: dict[str, bool] = Field(default_factory=dict)
    reason_text: str | None = None
    change_description: str | None = None
    source_document: str
    source_page: int
    source_section: str | None
    source_text: str | None
    extraction_method: ExtractionMethod
    status: ExtractionStatus
    source_refs: list[EvidenceRef]
    quality_issues: list[ExtractionIssue] = Field(default_factory=list)


class ProjectEnrichment(_Row):
    """Explicit, reviewable enrichment of a Silver field from document evidence."""

    project_id: str
    target_table: str
    target_field: str
    loan_number: str | None = None
    value: date | None
    status: ExtractionStatus
    rule: str
    evidence: list[EvidenceRef]
    candidates: list[dict] = Field(default_factory=list)
