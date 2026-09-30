"""Hand-built Bronze tables for focused Silver tests."""

from __future__ import annotations

from worldbank_copilot.ingestion.contracts import LOANS, PROCUREMENT, PROJECTS, SourceContract
from worldbank_copilot.transformations.bronze import BronzeTable, SourceMetadata


def bronze_table(name: str, contract: SourceContract | None, rows: list[dict]) -> BronzeTable:
    records = []
    for i, values in enumerate(rows, start=2):
        record = {
            "_source_file": "src.csv",
            "_source_sheet": None,
            "_source_row": i,
            "_ingested_at": "2026-09-26T00:00:00+00:00",
            "_ingestion_run_id": "run-1",
            "project_id": values.get("project_id", "P130544"),
        }
        if contract is not None:
            record.update({c.field: "" for c in contract.columns})
            record["source_project_id"] = record["project_id"]
        record.update(values)
        records.append(record)
    metadata = SourceMetadata(
        source_name=name, source_file="src.csv", ingested_at="t", ingestion_run_id="run-1"
    )
    return BronzeTable(name, records, metadata)


def loan(pid: str, number: str, original: str, **values) -> dict:
    return {
        "project_id": pid,
        "raw_loan_number": number,
        "normalized_loan_number": None,
        "original_principal_amount_usd": original,
        "disbursed_amount_usd": original,
        "undisbursed_amount_usd": "0",
        "cancelled_amount_usd": "0",
        "exchange_adjustment_usd": "0",
        "end_of_period": "08/31/2026",
        "board_approval_date": "03/31/2016",
        "agreement_signing_date": "05/24/2016",
        "effective_date_most_recent": "08/22/2016",
        "closed_date_most_recent": "11/22/2024",
        "country_code": "IN",
        **values,
    }


def loans_table(rows: list[dict]) -> BronzeTable:
    return bronze_table("bronze_loans_raw", LOANS, rows)


def award_row(pid: str, number: str, supplier: str, amount: str, **values) -> dict:
    return {
        "project_id": pid,
        "wb_contract_number": number,
        "supplier": supplier,
        "supplier_id": supplier.replace(" ", "")[:8],
        "supplier_contract_amount_usd": amount,
        "contract_description": "Support Organisations",
        "procurement_category": "Consultant Services",
        "procurement_method": "Quality And Cost-Based Selection",
        "contract_signing_date": "06/05/2021",
        "fiscal_year": "2021",
        "review_type": "Post",
        **values,
    }


def procurement_table(rows: list[dict]) -> BronzeTable:
    return bronze_table("bronze_procurement_raw", PROCUREMENT, rows)


def project_row(pid: str = "P130544", **values) -> dict:
    return {
        "project_id": pid,
        "project_name": "Name",
        "country": "India",
        "project_status": "Active",
        "lending_instrument": "Investment Project Financing",
        "board_approval_date": "2016-03-31T00:00:00Z",
        "project_closing_date": "2027-09-30",
        "ibrd_commitment": "100000000.0",
        **values,
    }


def projects_table(rows: list[dict]) -> BronzeTable:
    return bronze_table("bronze_projects_raw", PROJECTS, rows)
