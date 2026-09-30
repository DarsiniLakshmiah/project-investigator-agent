"""Bronze ingestion of the India IPF contract-awards dataset, and its coverage.

The dataset covers Investment Project Financing only. Zero rows for a project
therefore mean different things depending on coverage, and they never mean
"no procurement activity":

* ``RECORDS_PRESENT``: the dataset has rows for the project.
* ``NO_RECORDS_IN_DATASET``: the project is covered by the dataset, but the
  dataset contains no rows for it.
* ``NOT_COVERED_BY_THIS_DATASET``: the project is outside the dataset's
  coverage (e.g. Program-for-Results), so the dataset says nothing about its
  procurement.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from worldbank_copilot.common.project_registry import ProjectRegistry
from worldbank_copilot.ingestion.contracts import PROCUREMENT
from worldbank_copilot.ingestion.tabular import iter_csv_rows, read_tabular_source
from worldbank_copilot.transformations.bronze import BronzeTable

PROCUREMENT_DATASET = "ipf_contract_awards_india"


class ProcurementCoverageStatus(StrEnum):
    RECORDS_PRESENT = "RECORDS_PRESENT"
    NO_RECORDS_IN_DATASET = "NO_RECORDS_IN_DATASET"
    NOT_COVERED_BY_THIS_DATASET = "NOT_COVERED_BY_THIS_DATASET"


class ProcurementCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str
    dataset: str
    covered_by_dataset: bool
    coverage_status: ProcurementCoverageStatus
    row_count: int
    interpretation: str


_INTERPRETATION = {
    ProcurementCoverageStatus.RECORDS_PRESENT: (
        "The dataset contains contract-award records for this project."
    ),
    ProcurementCoverageStatus.NO_RECORDS_IN_DATASET: (
        "The project is within the dataset's coverage but the dataset contains no "
        "records for it. This is not evidence that no procurement took place."
    ),
    ProcurementCoverageStatus.NOT_COVERED_BY_THIS_DATASET: (
        "The project is outside this dataset's coverage; the dataset provides no "
        "information about its procurement. Zero rows must not be read as no "
        "procurement activity."
    ),
}


def ingest_procurement(
    path: Path | str,
    project_ids: Iterable[str],
    *,
    source_file: str,
    ingested_at: str,
    run_id: str,
) -> BronzeTable:
    return read_tabular_source(
        iter_csv_rows(path),
        PROCUREMENT,
        project_ids,
        source_file=source_file,
        ingested_at=ingested_at,
        run_id=run_id,
    )


def assess_procurement_coverage(
    table: BronzeTable, registry: ProjectRegistry, dataset: str = PROCUREMENT_DATASET
) -> list[ProcurementCoverage]:
    """Coverage status per registered project, driven by registry metadata."""
    counts = table.count_by_project()
    results = []
    for project in registry.projects:
        covered = project.procurement_coverage.covered_by_ipf_contract_awards
        rows = counts.get(project.project_id, 0)
        if rows > 0:
            status = ProcurementCoverageStatus.RECORDS_PRESENT
        elif covered:
            status = ProcurementCoverageStatus.NO_RECORDS_IN_DATASET
        else:
            status = ProcurementCoverageStatus.NOT_COVERED_BY_THIS_DATASET
        results.append(
            ProcurementCoverage(
                project_id=project.project_id,
                dataset=dataset,
                covered_by_dataset=covered,
                coverage_status=status,
                row_count=rows,
                interpretation=_INTERPRETATION[status],
            )
        )
    return results
