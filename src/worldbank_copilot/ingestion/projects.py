"""Bronze ingestion of the World Bank Projects & Operations workbook.

Each relevant sheet becomes its own Bronze table (no merging in Bronze):
World Bank Projects, Themes, Sectors, GEO Locations, Financers.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path

import openpyxl

from worldbank_copilot.ingestion.contracts import WORKBOOK_CONTRACTS, SourceContract
from worldbank_copilot.ingestion.tabular import iter_worksheet_rows, read_tabular_source
from worldbank_copilot.transformations.bronze import BronzeTable


def ingest_workbook(
    path: Path | str,
    project_ids: Iterable[str],
    *,
    source_file: str,
    ingested_at: str,
    run_id: str,
    contracts: Iterable[SourceContract] = WORKBOOK_CONTRACTS,
) -> dict[str, BronzeTable]:
    """Read every contracted sheet; returns ``{bronze_table_name: table}``."""
    ids = list(project_ids)
    with warnings.catch_warnings():
        # The export has no default cell style; openpyxl warns but reads correctly.
        warnings.filterwarnings("ignore", message="Workbook contains no default style")
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        tables: dict[str, BronzeTable] = {}
        for contract in contracts:
            if contract.sheet_name is None:
                raise ValueError(f"Workbook contract {contract.name!r} has no sheet_name")
            tables[contract.bronze_table] = read_tabular_source(
                iter_worksheet_rows(workbook, contract.sheet_name),
                contract,
                ids,
                source_file=source_file,
                source_sheet=contract.sheet_name,
                ingested_at=ingested_at,
                run_id=run_id,
            )
        return tables
    finally:
        workbook.close()
