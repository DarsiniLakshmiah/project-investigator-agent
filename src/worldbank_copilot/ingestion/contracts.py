"""Source contracts: what each structured source is expected to contain.

A contract maps actual source headers to internal Bronze field names, marks
which columns are required, and declares each column's kind (and date formats)
so later checks never have to guess. Header names are matched after whitespace
normalisation and case-folding (see ``tabular.normalize_column_name``), which
absorbs harmless quirks such as ``"Project Development Objective "``.

Contracts are derived from the files inspected on 2026-09-26 (IMPLEMENTATION_PLAN.md §5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from worldbank_copilot.common.dates import ISO_DATE, ISO_DATETIME_Z, US_SLASH


class FieldKind(StrEnum):
    TEXT = "text"
    IDENTIFIER = "identifier"
    DATE = "date"
    AMOUNT = "amount"  # monetary value
    NUMBER = "number"  # other numeric value (percentages, coordinates, years)


@dataclass(frozen=True)
class ColumnSpec:
    field: str
    source_names: tuple[str, ...]
    required: bool = False
    kind: FieldKind = FieldKind.TEXT
    date_formats: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind is FieldKind.DATE and not self.date_formats:
            raise ValueError(f"Date column {self.field!r} must declare date_formats")


@dataclass(frozen=True)
class SourceContract:
    """Expected structure of one tabular source (a CSV file or a workbook sheet)."""

    name: str
    bronze_table: str
    columns: tuple[ColumnSpec, ...]
    project_id_field: str
    sheet_name: str | None = None
    # Optional second header row of API field names directly below the header.
    secondary_header_names: frozenset[str] = frozenset()
    max_header_scan_rows: int = 10
    # Concepts CLAUDE.md expects that this source does not provide at all.
    known_gaps: tuple[str, ...] = ()
    # Fields holding free text that may contain markup (preserved as-is in Bronze).
    rich_text_fields: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        fields = [c.field for c in self.columns]
        if len(fields) != len(set(fields)):
            raise ValueError(f"Contract {self.name!r} declares duplicate fields")
        if self.project_id_field not in fields:
            raise ValueError(f"Contract {self.name!r}: project_id_field not among columns")

    @property
    def required_fields(self) -> list[str]:
        return [c.field for c in self.columns if c.required]

    def fields_of_kind(self, kind: FieldKind) -> list[ColumnSpec]:
        return [c for c in self.columns if c.kind is kind]


def _col(field: str, *names: str, required=False, kind=FieldKind.TEXT, dates=()) -> ColumnSpec:
    return ColumnSpec(field, names, required, kind, tuple(dates))


_WB_DATES = (ISO_DATE, ISO_DATETIME_Z)
_CSV_DATES = (US_SLASH,)
_ID, _DATE, _AMT, _NUM = FieldKind.IDENTIFIER, FieldKind.DATE, FieldKind.AMOUNT, FieldKind.NUMBER

# --------------------------------------------------------------------------
# Projects & Operations workbook (all.xlsx)
# --------------------------------------------------------------------------

PROJECTS_API_FIELD_NAMES = frozenset(
    {
        "id", "regionname", "countryshortname", "status", "last_stage_reached_name",
        "project_name", "pdo", "impagency", "public_disclosure_date", "boardapprovaldate",
        "loan_effective_date", "closingdate", "curr_project_cost", "curr_ibrd_commitment",
        "idacommamt", "grantamt", "curr_total_commitment", "borrower", "lendinginstr",
        "envassesmentcategorycode", "esrc_ovrl_risk_rate", "supplementprojectflg",
        "cons_serv_reqd_ind", "proj_last_upd_date", "projectfinancialtype",
    }
)  # fmt: skip

PROJECTS = SourceContract(
    name="projects_workbook.world_bank_projects",
    bronze_table="bronze_projects_raw",
    sheet_name="World Bank Projects",
    project_id_field="source_project_id",
    secondary_header_names=PROJECTS_API_FIELD_NAMES,
    known_gaps=("state", "original_closing_date"),
    rich_text_fields=("project_development_objective",),
    columns=(
        _col("source_project_id", "Project ID", required=True, kind=_ID),
        _col("region", "Region"),
        _col("country", "Country", required=True),
        _col("project_status", "Project Status", required=True),
        _col("last_stage_reached_name", "Last Stage Reached Name"),
        _col("project_name", "Project Name", required=True),
        _col("project_development_objective", "Project Development Objective", required=True),
        _col("implementing_agency", "Implementing Agency", required=True),
        _col("public_disclosure_date", "Public Disclosure Date", kind=_DATE, dates=_WB_DATES),
        _col("board_approval_date", "Board Approval Date", required=True, kind=_DATE,
             dates=_WB_DATES),
        _col("loan_effective_date", "Loan Effective Date", kind=_DATE, dates=_WB_DATES),
        _col("project_closing_date", "Project Closing Date", required=True, kind=_DATE,
             dates=_WB_DATES),
        _col("current_project_cost", "Current Project Cost", kind=_AMT),
        _col("ibrd_commitment", "IBRD Commitment", required=True, kind=_AMT),
        _col("ida_commitment", "IDA Commitment", kind=_AMT),
        _col("grant_amount", "Grant Amount", kind=_AMT),
        _col("total_ibrd_ida_grant_commitment", "Total IBRD, IDA and Grant Commitment",
             kind=_AMT),
        _col("borrower", "Borrower"),
        _col("lending_instrument", "Lending Instrument", required=True),
        _col("environmental_assessment_category", "Environmental Assessment Category"),
        _col("environmental_and_social_risk", "Environmental and Social Risk"),
        _col("associated_project", "Associated Project"),
        _col("consultant_services_required", "Consultant Services Required"),
        _col("last_update_date", "Last Update Date", kind=_DATE, dates=_WB_DATES),
        _col("financing_type", "Financing Type"),
    ),
)  # fmt: skip

THEMES = SourceContract(
    name="projects_workbook.themes",
    bronze_table="bronze_themes_raw",
    sheet_name="Themes",
    project_id_field="source_project_id",
    columns=(
        _col("source_project_id", "Project ID", required=True, kind=_ID),
        _col("level_1", "Level 1", required=True),
        _col("percentage_1", "Percentage 1", required=True, kind=_NUM),
        _col("level_2", "Level 2"),
        _col("percentage_2", "Percentage 2", kind=_NUM),
        _col("level_3", "Level 3"),
        _col("percentage_3", "Percentage 3", kind=_NUM),
    ),
)

SECTORS = SourceContract(
    name="projects_workbook.sectors",
    bronze_table="bronze_sectors_raw",
    sheet_name="Sectors",
    project_id_field="source_project_id",
    columns=(
        _col("source_project_id", "Project ID", required=True, kind=_ID),
        _col("major_sector", "Major Sector", required=True),
        _col("sector", "Sector", required=True),
        _col("sector_percent", "Sector Percent", required=True, kind=_NUM),
    ),
)

GEO_LOCATIONS = SourceContract(
    name="projects_workbook.geo_locations",
    bronze_table="bronze_geo_locations_raw",
    sheet_name="GEO Locations",
    project_id_field="source_project_id",
    columns=(
        _col("source_project_id", "Project ID", required=True, kind=_ID),
        _col("geo_loc_id", "GEO Loc ID", required=True, kind=_ID),
        _col("place_id", "Place ID", kind=_ID),
        _col("wbg_country_key", "WBG Country Key"),
        _col("geo_loc_name", "GEO Loc Name", required=True),
        _col("geo_latitude", "GEO Latitude Number", required=True, kind=_NUM),
        _col("geo_longitude", "GEO Longitude Number", required=True, kind=_NUM),
        _col("admin_unit1_name", "Admin Unit1 Name"),
        _col("admin_unit2_name", "Admin Unit2 Name"),
    ),
)

FINANCERS = SourceContract(
    name="projects_workbook.financers",
    bronze_table="bronze_financers_raw",
    sheet_name="Financers",
    project_id_field="source_project_id",
    columns=(
        # The Financers sheet labels its key "Project", not "Project ID".
        _col("source_project_id", "Project", "Project ID", required=True, kind=_ID),
        _col("financer_name", "Name", required=True),
        _col("current_amount", "Current Amount", kind=_AMT),
        _col("amount_usd", "Amount (USD)", required=True, kind=_AMT),
        _col("financer_id", "Financer ID", required=True, kind=_ID),
        _col("currency", "Currency", required=True),
        _col("project_financial_type", "Project Financial Type"),
    ),
)

WORKBOOK_CONTRACTS = (PROJECTS, THEMES, SECTORS, GEO_LOCATIONS, FINANCERS)

# --------------------------------------------------------------------------
# IBRD Statement of Loans and Guarantees snapshot (CSV)
# --------------------------------------------------------------------------

LOANS = SourceContract(
    name="ibrd_statement_of_loans",
    bronze_table="bronze_loans_raw",
    project_id_field="source_project_id",
    known_gaps=("current_principal", "original_closing_date"),
    columns=(
        _col("end_of_period", "End of Period", required=True, kind=_DATE, dates=_CSV_DATES),
        _col("raw_loan_number", "Loan Number", required=True, kind=_ID),
        _col("region", "Region"),
        _col("country_code", "Country / Economy Code"),
        _col("country", "Country / Economy"),
        _col("borrower", "Borrower"),
        _col("guarantor_country_code", "Guarantor Country / Economy Code"),
        _col("guarantor", "Guarantor"),
        _col("loan_type", "Loan Type"),
        _col("loan_status", "Loan Status", required=True),
        _col("interest_rate", "Interest Rate", kind=_NUM),
        _col("currency_of_commitment", "Currency of Commitment"),
        _col("source_project_id", "Project ID", required=True, kind=_ID),
        _col("project_name", "Project Name"),
        _col("original_principal_amount_usd", "Original Principal Amount (US$)", required=True,
             kind=_AMT),
        _col("cancelled_amount_usd", "Cancelled Amount (US$)", required=True, kind=_AMT),
        _col("undisbursed_amount_usd", "Undisbursed Amount (US$)", required=True, kind=_AMT),
        _col("disbursed_amount_usd", "Disbursed Amount (US$)", required=True, kind=_AMT),
        _col("repaid_to_ibrd_usd", "Repaid to IBRD (US$)", kind=_AMT),
        _col("due_to_ibrd_usd", "Due to IBRD (US$)", kind=_AMT),
        _col("exchange_adjustment_usd", "Exchange Adjustment (US$)", kind=_AMT),
        _col("borrowers_obligation_usd", "Borrower's Obligation (US$)", kind=_AMT),
        _col("sold_3rd_party_usd", "Sold 3rd Party (US$)", kind=_AMT),
        _col("repaid_3rd_party_usd", "Repaid 3rd Party (US$)", kind=_AMT),
        _col("due_3rd_party_usd", "Due 3rd Party (US$)", kind=_AMT),
        _col("loans_held_usd", "Loans Held (US$)", kind=_AMT),
        _col("first_repayment_date", "First Repayment Date", kind=_DATE, dates=_CSV_DATES),
        _col("last_repayment_date", "Last Repayment Date", kind=_DATE, dates=_CSV_DATES),
        _col("agreement_signing_date", "Agreement Signing Date", required=True, kind=_DATE,
             dates=_CSV_DATES),
        _col("board_approval_date", "Board Approval Date", required=True, kind=_DATE,
             dates=_CSV_DATES),
        _col("effective_date_most_recent", "Effective Date (Most Recent)", required=True,
             kind=_DATE, dates=_CSV_DATES),
        _col("closed_date_most_recent", "Closed Date (Most Recent)", required=True, kind=_DATE,
             dates=_CSV_DATES),
        _col("last_disbursement_date", "Last Disbursement Date", kind=_DATE, dates=_CSV_DATES),
        _col("board_approval_fiscal_year", "Board approval - Fiscal year", kind=_NUM),
        _col("board_approval_calendar_year", "Board approval - Calendar year", kind=_NUM),
    ),
)  # fmt: skip

# --------------------------------------------------------------------------
# India IPF contract awards (CSV)
# --------------------------------------------------------------------------

PROCUREMENT = SourceContract(
    name="ipf_contract_awards_india",
    bronze_table="bronze_procurement_raw",
    project_id_field="source_project_id",
    columns=(
        _col("as_of_date", "As of Date", kind=_DATE, dates=_CSV_DATES),
        _col("fiscal_year", "Fiscal Year", required=True, kind=_NUM),
        _col("region", "Region"),
        _col("borrower_country", "Borrower Country / Economy"),
        _col("borrower_country_code", "Borrower Country / Economy Code"),
        _col("source_project_id", "Project ID", required=True, kind=_ID),
        _col("project_name", "Project Name"),
        _col("procurement_category", "Procurement Category", required=True),
        _col("procurement_method", "Procurement Method", required=True),
        _col("project_global_practice", "Project Global Practice"),
        _col("wb_contract_number", "WB Contract Number", required=True, kind=_ID),
        _col("contract_description", "Contract Description"),
        _col("contract_signing_date", "Contract Signing Date", required=True, kind=_DATE,
             dates=_CSV_DATES),
        _col("supplier", "Supplier", required=True),
        _col("supplier_country", "Supplier Country / Economy"),
        _col("supplier_country_code", "Supplier Country / Economy Code"),
        _col("supplier_contract_amount_usd", "Supplier Contract Amount (USD)", required=True,
             kind=_AMT),
        _col("borrower_contract_reference_number", "Borrower Contract Reference Number",
             kind=_ID),
        _col("contract_signed_calendar_year", "Contract signed - Calendar year", kind=_NUM),
        _col("review_type", "Review type"),
        _col("supplier_id", "Supplier ID", kind=_ID),
    ),
)  # fmt: skip

ALL_TABULAR_CONTRACTS = (*WORKBOOK_CONTRACTS, LOANS, PROCUREMENT)
