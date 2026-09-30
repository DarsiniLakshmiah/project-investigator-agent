"""Bronze ingestion of the IBRD Statement of Loans and Guarantees snapshot.

One Bronze record per loan row; a project may have several loans
(P130544 -> IBRD86010, IBRD93240). Nothing is aggregated here. Financial
values, currency and the snapshot date (End of Period) are kept exactly as
provided; no currency or USD-equivalent logic is applied.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from worldbank_copilot.common.identifiers import normalize_loan_number
from worldbank_copilot.ingestion.contracts import LOANS
from worldbank_copilot.ingestion.tabular import iter_csv_rows, read_tabular_source
from worldbank_copilot.transformations.bronze import BronzeTable


def ingest_loans(
    path: Path | str,
    project_ids: Iterable[str],
    *,
    source_file: str,
    ingested_at: str,
    run_id: str,
) -> BronzeTable:
    table = read_tabular_source(
        iter_csv_rows(path),
        LOANS,
        project_ids,
        source_file=source_file,
        ingested_at=ingested_at,
        run_id=run_id,
    )
    for record in table.records:
        record["normalized_loan_number"] = normalize_loan_number(record["raw_loan_number"])
    position = table.fields.index("raw_loan_number") + 1
    table.fields.insert(position, "normalized_loan_number")
    table.metadata.notes.append(
        "normalized_loan_number is derived deterministically from raw_loan_number "
        "(worldbank_copilot.common.identifiers); null when the notation is unrecognised."
    )
    return table


def loans_by_project(table: BronzeTable) -> dict[str, list[str]]:
    """``{project_id: [raw_loan_number, ...]}``, preserving source order."""
    grouped: dict[str, list[str]] = {}
    for record in table.records:
        grouped.setdefault(record["project_id"], []).append(record["raw_loan_number"])
    return grouped
