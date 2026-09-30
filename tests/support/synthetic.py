"""Small synthetic source files that reproduce the real files' structural quirks.

Values are invented for tests only; they are never used outside the test suite.
"""

from __future__ import annotations

import csv
from pathlib import Path

import openpyxl
import yaml

from worldbank_copilot.ingestion.contracts import LOANS, PROCUREMENT

TITLE = "World Bank Projects, data as of 09/26/2026 13:01:01 EST"

PROJECT_HEADERS = [
    "Project ID", "Region", "Country", "Project Status", "Last Stage Reached Name",
    "Project Name", "Project Development Objective ", "Implementing Agency",
    "Public Disclosure Date", "Board Approval Date", "Loan Effective Date",
    "Project Closing Date", "Current Project Cost", "IBRD Commitment", "IDA Commitment",
    "Grant Amount", "Total IBRD, IDA and Grant Commitment", "Borrower", "Lending Instrument",
    "Environmental Assessment Category", "Environmental and Social Risk", "Associated Project",
    "Consultant Services Required", "Last Update Date", "Financing Type",
]  # fmt: skip
PROJECT_API = [
    "id", "regionname", "countryshortname", "status", "last_stage_reached_name",
    "project_name", "pdo", "impagency", "public_disclosure_date", "boardapprovaldate",
    "loan_effective_date", "closingdate", "curr_project_cost", "curr_ibrd_commitment",
    "idacommamt", "grantamt", "curr_total_commitment", "borrower", "lendinginstr",
    "envassesmentcategorycode", "esrc_ovrl_risk_rate", "supplementprojectflg",
    "cons_serv_reqd_ind", "proj_last_upd_date", "projectfinancialtype",
]  # fmt: skip


def project_row(pid, name, instrument, ibrd, pdo="Objective.", cost=153000000.0):
    return [
        pid, "South Asia", "India", "Active", "Board Approved", name, pdo, "Agency",
        "2012-07-16", "2016-03-31T00:00:00Z", "2016-08-22", "2027-09-30", cost, ibrd, 0.0,
        0.0, ibrd, "India", instrument, "B", "Not Applicable", "N", "TBD", "2024-04-04", "IBRD",
    ]  # fmt: skip


def default_workbook_sheets() -> dict[str, list[list]]:
    return {
        "World Bank Projects": [
            [TITLE] + [None] * 24,
            PROJECT_HEADERS,
            PROJECT_API,
            project_row("P000001", "Out of scope", "Specific Investment Loan", 1.0),
            project_row("P130544", "IN Karnataka Urban Water Supply Modernization Project",
                        "Investment Project Financing", 100000000.0, pdo="<p>Objective.</p>"),
            project_row("P179039", "Karnataka Sustainable Rural Water Supply Program",
                        "Program-for-Results Financing", 363000000.0),
            project_row("P506272", "Karnataka Water Security and Resilience Program",
                        "Program-for-Results Financing", 426000000.0, cost=""),
        ],
        "Themes": [
            [TITLE] + [None] * 6,
            ["Project ID", "Level 1", "Percentage 1", "Level 2", "Percentage 2", "Level 3",
             "Percentage 3"],
            ["P000001", "Theme", "50", "", "", "", ""],
            ["P130544", "FY17 - Urban", "100", "", "", "", ""],
            ["P130544", "> FY17 - Urban", "100", " Water  Institutions", "100", "", ""],
        ],
        "Sectors": [
            [TITLE, None, None, None],
            ["Project ID", "Major Sector", "Sector", "Sector Percent"],
            ["P130544", "Water", "Water Supply", 95.0],
            ["P179039", "Water", "Water Supply ", 44.0],
        ],
        "GEO Locations": [
            [TITLE] + [None] * 8,
            ["Project ID", "GEO Loc ID", "Place ID", "WBG Country Key", "GEO Loc Name",
             "GEO Latitude Number", "GEO Longitude Number", "Admin Unit1 Name",
             "Admin Unit2 Name"],
            ["P130544", "1269920", "PLA0055045", "IN", "Hubli", "15.34776", "75.13378", "", ""],
        ],
        "Financers": [
            [TITLE] + [None] * 6,
            ["Project", "Name", "Current Amount", "Amount (USD)", "Financer ID", "Currency",
             "Project Financial Type"],
            ["P130544", "Borrower/Recipient", "100000000", "53000000", "BORR", "USD",
             "Local Contributor"],
            ["P130544", "IBRD", "100000000", "100000000", "IBRD", "USD", "World Bank Group"],
            ["P179039", "IBRD", "363000000", "363000000", "IBRD", "USD  ", "World Bank Group"],
        ],
    }  # fmt: skip


def write_workbook(path: Path, sheets: dict[str, list[list]]) -> Path:
    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    for name, rows in sheets.items():
        sheet = workbook.create_sheet(name)
        for row in rows:
            sheet.append(row)
    workbook.save(path)
    return path


def _headers(contract) -> list[str]:
    return [c.source_names[0] for c in contract.columns]


def loan_row(pid, loan, original, cancelled="0", end_of_period="08/31/2026", **overrides):
    row = {h: "" for h in _headers(LOANS)}
    row.update({
        "End of Period": end_of_period, "Loan Number": loan, "Project ID": pid,
        "Loan Status": "Disbursing", "Original Principal Amount (US$)": original,
        "Cancelled Amount (US$)": cancelled, "Undisbursed Amount (US$)": "0",
        "Disbursed Amount (US$)": original, "Agreement Signing Date": "05/24/2016",
        "Board Approval Date": "03/31/2016", "Effective Date (Most Recent)": "08/22/2016",
        "Closed Date (Most Recent)": "11/22/2024", "Currency of Commitment": "",
        "Country / Economy Code": "IN",
    })  # fmt: skip
    row.update(overrides)
    return row


def default_loan_rows() -> list[dict]:
    return [
        loan_row("P000001", "IBRD11110", "5"),
        loan_row("P130544", "IBRD86010", "100000000"),
        # Reconciles exactly: 125,062,500.52 disbursed + 0 undisbursed + 24,937,499.48 cancelled.
        loan_row(
            "P130544",
            "IBRD93240",
            "150000000",
            cancelled="24937499.48",
            **{"Disbursed Amount (US$)": "125062500.52"},
        ),
        loan_row("P179039", "IBRD94960", "363000000"),
        loan_row("P506272", "IBRD98350", "426000000"),
    ]


def contract_row(pid, number, supplier, amount="1000.000000", **overrides):
    row = {h: "" for h in _headers(PROCUREMENT)}
    row.update({
        "As of Date": "09/26/2026", "Fiscal Year": "2021", "Project ID": pid,
        "Procurement Category": "Consultant Services", "Procurement Method": "QCBS",
        "WB Contract Number": number, "Contract Signing Date": "06/05/2021",
        "Supplier": supplier, "Supplier Contract Amount (USD)": amount, "Review type": "Post",
    })  # fmt: skip
    row.update(overrides)
    return row


def default_contract_rows() -> list[dict]:
    return [
        contract_row("P000001", "1", "A"),
        contract_row("P130544", "1657297", "SUPPLIER ONE", "35729.020000"),
        contract_row("P130544", "1657297", "SUPPLIER TWO", "35729.020000"),
        contract_row("P130544", "1610810", "L AND T", "162413524.890000"),
    ]


def write_csv(path: Path, headers: list[str], rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    return path


DOCUMENTS = {
    "P130544": {
        "Disclosable-Version-of-the-ISR-IN-X-P130544-Sequence-No-01.pdf": b"isr-1",
        "ISR-Disclosable-P130544-04-10-2017-1491820303083.pdf": b"isr-2",
        "RAD696381150.pdf": b"loan-agreement",
    },
    "P179039": {"P179039-3c28639e.pdf": b"isr-p179039-1"},
    "P506272": {},
}

MANIFEST = {
    "documents": {
        "P130544": [
            {"filename": "ISR-Disclosable-P130544-04-10-2017-1491820303083.pdf",
             "size_bytes": 5, "document_type": "ISR", "isr_sequence": 2,
             "document_date": "2017-04-10", "date_basis": "isr_archived_date",
             "evidence": "test"},
            {"filename": "RAD696381150.pdf", "size_bytes": 14, "document_type": "LOAN_AGREEMENT",
             "loan_number_raw": "8601-IN", "evidence": "test"},
        ],
        "P179039": [
            {"filename": "P179039-3c28639e.pdf", "size_bytes": 13, "document_type": "ISR",
             "isr_sequence": 1, "document_date": "2023-06-16",
             "date_basis": "isr_archived_date", "evidence": "test"},
        ],
    }
}  # fmt: skip

EXPECTED_ISR = {"P130544": 2, "P179039": 1, "P506272": 0}


def build_source_tree(root: Path, repo_config_dir: Path) -> Path:
    """Create data/ and configs/ under ``root``; returns the config directory."""
    data = root / "data"
    data.mkdir()
    write_workbook(data / "all.xlsx", default_workbook_sheets())
    write_csv(
        data / "ibrd_statement_of_loans_and_guarantees_latest_available_snapshot_x.csv",
        _headers(LOANS),
        default_loan_rows(),
    )
    write_csv(data / "contract_awards_in_investment_project_financing_india_projects_x.csv",
              _headers(PROCUREMENT), default_contract_rows())  # fmt: skip
    for pid, files in DOCUMENTS.items():
        folder = data / pid
        folder.mkdir()
        for name, content in files.items():
            (folder / name).write_bytes(content)

    config = root / "configs"
    (config / "environments").mkdir(parents=True)
    base = yaml.safe_load(
        (repo_config_dir / "environments" / "base.yaml").read_text(encoding="utf-8")
    )
    base["data"]["local_data_root"] = str(data)
    base["data"]["local_output_root"] = str(root / "out")
    (config / "environments" / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    for env in ("local", "databricks"):
        (config / "environments" / f"{env}.yaml").write_text("{}\n", encoding="utf-8")

    projects = yaml.safe_load((repo_config_dir / "projects.yaml").read_text(encoding="utf-8"))
    for project in projects["projects"]:
        project["expected_isr_count"] = EXPECTED_ISR[project["project_id"]]
    (config / "projects.yaml").write_text(yaml.safe_dump(projects), encoding="utf-8")
    (config / "document_manifest.yaml").write_text(yaml.safe_dump(MANIFEST), encoding="utf-8")
    return config
