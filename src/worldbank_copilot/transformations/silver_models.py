"""Silver row schemas (typed) and provenance models.

Provenance design
-----------------
* **Row level:** every Silver row has ``source_refs``, a list of ``SourceRef``
  pointing at the exact Bronze records it was built from (Bronze table, source
  file, sheet, source row, ingestion run, and a readable record key).
  Derived rows (e.g. the project financial summary) reference the Bronze records
  of every input row.
* **Field level:** ``silver_lineage.FIELD_LINEAGE`` states, for each Silver field,
  which Bronze field(s) and source column(s) it comes from and how it was
  transformed. Row refs plus field lineage answer "where did this number come
  from?" without copying Bronze metadata into every column.
* **Issues:** values that could not be normalised are ``None`` in Silver and
  listed in the row's ``quality_issues`` with the raw value and reason.

Money is ``Decimal`` (serialised as a string in JSON); dates are ``date``.
Amount fields carry the unit the source column states (``_usd`` = "US$" columns).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bronze_table: str
    source_file: str
    source_sheet: str | None = None
    source_row: int | None = None
    ingestion_run_id: str
    record_key: str | None = None


class IssueKind(StrEnum):
    MALFORMED = "MALFORMED"  # value present but not normalisable
    STRUCTURE = "STRUCTURE"  # source structure inconsistent (e.g. theme hierarchy)
    CONFLICT = "CONFLICT"  # several source rows disagree on one value


class ValueIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str
    kind: IssueKind
    raw_value: str | None = None
    reason: str
    source_ref: SourceRef | None = None
    project_id: str | None = None


class SilverRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_refs: list[SourceRef] = Field(default_factory=list)
    quality_issues: list[ValueIssue] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


class SilverProject(SilverRow):
    """One row per project. Workbook = authoritative for project metadata."""

    project_id: str
    project_name: str | None
    country: str | None
    country_code: str | None
    country_code_source: str | None = Field(
        description="Bronze table the country code came from; the workbook has no code column."
    )
    region: str | None
    state: str | None = Field(
        default=None, description="Always NULL: no structured source provides a state."
    )
    financing_instrument: str | None
    financing_type: str | None
    project_status: str | None
    last_stage_reached: str | None
    approval_date: date | None
    effective_date: date | None
    current_closing_date: date | None = Field(
        description="Workbook 'Project Closing Date' (the only closing date it provides)."
    )
    original_closing_date: date | None = Field(
        default=None, description="Always NULL in Phase 3: not in structured sources."
    )
    public_disclosure_date: date | None
    borrower: str | None
    implementing_agency: str | None
    primary_sector: str | None = Field(
        description="Sector with the strictly highest percentage; NULL on ties or no sectors."
    )
    sectors: list[str]
    major_sectors: list[str]
    themes: list[str] = Field(description="Distinct level-1 themes in source order.")
    project_development_objective: str | None = Field(description="HTML removed.")
    workbook_ibrd_commitment_usd: Decimal | None
    workbook_ida_commitment_usd: Decimal | None
    workbook_grant_amount_usd: Decimal | None
    workbook_total_commitment_usd: Decimal | None
    current_project_cost_usd: Decimal | None
    environmental_assessment_category: str | None
    environmental_and_social_risk: str | None
    source_last_updated: date | None
    source_file: str


class SilverProjectSector(SilverRow):
    project_id: str
    major_sector: str | None
    sector: str | None
    sector_percent: Decimal | None
    taxonomy_label: str | None


class SilverProjectTheme(SilverRow):
    """One row per distinct node of the theme hierarchy (levels 1-3)."""

    project_id: str
    level: int
    theme_name: str
    parent_theme_name: str | None
    level_1_theme: str
    theme_path: list[str]
    percentage: Decimal | None
    taxonomy_label: str | None


# ---------------------------------------------------------------------------
# Loans and financial summary
# ---------------------------------------------------------------------------


class SilverLoan(SilverRow):
    """One row per loan. Loan snapshot = authoritative for loan-level finance."""

    project_id: str
    raw_loan_number: str
    normalized_loan_number: str | None
    loan_base_number: str | None
    loan_suffix: str | None
    lender: str | None
    loan_type: str | None
    loan_status: str | None
    borrower: str | None
    country_code: str | None
    currency_of_commitment: str | None = Field(
        description="As provided; blank in the current snapshot and never inferred."
    )
    original_principal_usd: Decimal | None
    cancelled_amount_usd: Decimal | None
    disbursed_amount_usd: Decimal | None
    undisbursed_amount_usd: Decimal | None
    repaid_to_ibrd_usd: Decimal | None
    due_to_ibrd_usd: Decimal | None
    exchange_adjustment_usd: Decimal | None
    borrowers_obligation_usd: Decimal | None
    loans_held_usd: Decimal | None
    principal_components_total_usd: Decimal | None = Field(
        description="disbursed + undisbursed + cancelled (reported for comparison only)."
    )
    principal_components_difference_usd: Decimal | None = Field(
        description="original_principal - principal_components_total. Not assumed to be 0."
    )
    board_approval_date: date | None
    agreement_signing_date: date | None
    effective_date: date | None = Field(description="Source 'Effective Date (Most Recent)'.")
    closing_date: date | None = Field(description="Source 'Closed Date (Most Recent)'.")
    last_disbursement_date: date | None
    first_repayment_date: date | None
    last_repayment_date: date | None
    snapshot_date: date | None = Field(description="Source 'End of Period'.")
    valuation_caveats: list[str]
    source_file: str


class SilverProjectFinancialSummary(SilverRow):
    """One row per project per snapshot date, derived only from silver loans."""

    project_id: str
    snapshot_date: date | None
    loan_count: int
    raw_loan_numbers: list[str]
    original_principal_total_usd: Decimal | None
    cancelled_total_usd: Decimal | None
    disbursed_total_usd: Decimal | None
    undisbursed_total_usd: Decimal | None
    net_principal_after_cancellation_usd: Decimal | None = Field(
        description="original_principal_total - cancelled_total."
    )
    disbursement_vs_original_principal_pct: Decimal | None = Field(
        description="disbursed_total / original_principal_total * 100 (2 dp, half-even). "
        "Financial only; not physical progress."
    )
    disbursement_vs_net_principal_pct: Decimal | None = Field(
        description="disbursed_total / net_principal_after_cancellation * 100 (2 dp, "
        "half-even). Financial only; not physical progress."
    )
    workbook_ibrd_commitment_usd: Decimal | None = Field(
        description="Copied from silver projects for side-by-side comparison; not overwritten."
    )
    loan_principal_minus_workbook_commitment_usd: Decimal | None
    commitment_sources_agree: bool | None
    loans_with_valuation_caveats: list[str]


# ---------------------------------------------------------------------------
# Procurement
# ---------------------------------------------------------------------------


class AwardAmountBasis(StrEnum):
    SINGLE_SUPPLIER_ROW = "SINGLE_SUPPLIER_ROW"
    MULTI_SUPPLIER_UNRESOLVED = "MULTI_SUPPLIER_UNRESOLVED"


class SilverProcurementAward(SilverRow):
    """One row per contract award (project_id + WB contract number)."""

    project_id: str
    contract_id: str
    contract_description: str | None
    procurement_category: str | None
    procurement_method: str | None
    review_type: str | None
    project_global_practice: str | None
    borrower_contract_reference: str | None
    contract_signing_date: date | None
    fiscal_year: int | None
    supplier_count: int
    supplier_names: list[str]
    amount_basis: AwardAmountBasis
    contract_amount_usd: Decimal | None = Field(
        description="Set only for single-supplier awards; NULL when the amount is ambiguous."
    )
    amount_lower_bound_usd: Decimal | None = Field(
        description="Largest supplier-row amount (if amounts repeat the award value)."
    )
    amount_upper_bound_usd: Decimal | None = Field(
        description="Sum of supplier-row amounts (if amounts are per-supplier shares)."
    )
    source_file: str


class SilverProcurementSupplier(SilverRow):
    """One row per contract-supplier relationship (one Bronze procurement row)."""

    project_id: str
    contract_id: str
    supplier_name: str | None
    supplier_id: str | None
    supplier_country: str | None
    supplier_country_code: str | None
    supplier_row_amount_usd: Decimal | None = Field(
        description="Amount on this supplier's source row; meaning unresolved when the "
        "award has several suppliers. Do not sum across suppliers."
    )


class SilverProcurementCoverage(SilverRow):
    project_id: str
    dataset: str
    covered_by_dataset: bool
    coverage_status: str
    source_row_count: int
    award_count: int | None = Field(
        description="NULL when the project is NOT_COVERED_BY_THIS_DATASET: unknown, not zero."
    )
    supplier_relationship_count: int | None
    interpretation: str
