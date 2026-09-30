"""Bronze ingestion orchestration: SOURCE -> BRONZE (Phase 2 stops here).

Resolves source files from configuration, runs each source ingester and
returns in-memory Bronze tables. Persisting them is a separate step through a
``BronzeWriter``, so the same run works locally (JSONL) and, later, on
Databricks (Delta).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from worldbank_copilot.common.config import Settings
from worldbank_copilot.common.exceptions import SourceFileError
from worldbank_copilot.common.logging import get_logger
from worldbank_copilot.common.project_registry import ProjectRegistry
from worldbank_copilot.ingestion.documents import (
    DocumentInventory,
    IsrCompleteness,
    build_document_inventory,
    isr_completeness_by_project,
    load_document_manifest,
)
from worldbank_copilot.ingestion.loans import ingest_loans
from worldbank_copilot.ingestion.procurement import (
    ProcurementCoverage,
    assess_procurement_coverage,
    ingest_procurement,
)
from worldbank_copilot.ingestion.projects import ingest_workbook
from worldbank_copilot.transformations.bronze import (
    LINEAGE_FIELDS,
    BronzeTable,
    BronzeWriter,
    SourceMetadata,
)

logger = get_logger(__name__)

PROCUREMENT_COVERAGE_TABLE = "bronze_procurement_coverage"


@dataclass(frozen=True)
class SourceFiles:
    data_root: Path
    projects_workbook: Path
    loans_snapshot: Path
    procurement_contract_awards: Path
    documents_root: Path
    document_manifest: Path

    def relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.data_root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()


def _resolve_one(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(p for p in directory.glob(pattern) if p.is_file())
    if not matches:
        raise SourceFileError(f"{label}: no file matches {pattern!r} in {directory}")
    if len(matches) > 1:
        names = [m.name for m in matches]
        raise SourceFileError(
            f"{label}: {len(matches)} files match {pattern!r} in {directory}: {names}. "
            "Set an exact file name in configuration."
        )
    return matches[0]


def resolve_source_files(settings: Settings) -> SourceFiles:
    """Locate each source file; fails clearly on missing or ambiguous matches."""
    structured = settings.structured_root
    if not structured.is_dir():
        raise SourceFileError(f"Structured data directory not found: {structured}")
    return SourceFiles(
        data_root=Path(settings.data_root),
        projects_workbook=_resolve_one(
            structured, settings.require("sources.projects_workbook"), "projects_workbook"
        ),
        loans_snapshot=_resolve_one(
            structured, settings.require("sources.loans_snapshot"), "loans_snapshot"
        ),
        procurement_contract_awards=_resolve_one(
            structured,
            settings.require("sources.procurement_contract_awards"),
            "procurement_contract_awards",
        ),
        documents_root=settings.documents_root,
        document_manifest=settings.config_dir / settings.require("sources.document_manifest"),
    )


@dataclass
class BronzeIngestionResult:
    run_id: str
    ingested_at: str
    source_files: SourceFiles
    tables: dict[str, BronzeTable]
    procurement_coverage: list[ProcurementCoverage]
    document_inventory: DocumentInventory
    isr_completeness: dict[str, IsrCompleteness]

    def all_tables(self) -> list[BronzeTable]:
        """Every Bronze table, including inventory and coverage, for writing."""
        return [*self.tables.values(), self.document_inventory.table, self._coverage_table()]

    def _coverage_table(self) -> BronzeTable:
        records = [
            {
                "_source_file": self.tables["bronze_procurement_raw"].metadata.source_file,
                "_source_sheet": None,
                "_source_row": None,
                "_ingested_at": self.ingested_at,
                "_ingestion_run_id": self.run_id,
                **item.model_dump(mode="json"),
            }
            for item in self.procurement_coverage
        ]
        fields = [*LINEAGE_FIELDS, *ProcurementCoverage.model_fields]
        metadata = SourceMetadata(
            source_name="procurement_coverage",
            source_file=self.tables["bronze_procurement_raw"].metadata.source_file,
            ingested_at=self.ingested_at,
            ingestion_run_id=self.run_id,
            rows_matched=len(records),
            notes=["Derived from registry coverage metadata and procurement row counts."],
        )
        return BronzeTable(PROCUREMENT_COVERAGE_TABLE, records, metadata, fields)


def ingest_bronze(
    settings: Settings,
    registry: ProjectRegistry,
    *,
    ingested_at: datetime | None = None,
    run_id: str | None = None,
) -> BronzeIngestionResult:
    """Read all sources for the registered projects into Bronze tables (in memory)."""
    files = resolve_source_files(settings)
    stamp = (ingested_at or datetime.now(UTC)).isoformat()
    run = run_id or uuid.uuid4().hex
    ids = registry.project_ids
    common = {"ingested_at": stamp, "run_id": run}

    logger.info("Reading workbook %s", files.projects_workbook)
    tables = ingest_workbook(
        files.projects_workbook, ids, source_file=files.relative(files.projects_workbook), **common
    )
    logger.info("Reading loans snapshot %s", files.loans_snapshot)
    tables["bronze_loans_raw"] = ingest_loans(
        files.loans_snapshot, ids, source_file=files.relative(files.loans_snapshot), **common
    )
    logger.info("Reading contract awards %s", files.procurement_contract_awards)
    procurement = ingest_procurement(
        files.procurement_contract_awards,
        ids,
        source_file=files.relative(files.procurement_contract_awards),
        **common,
    )
    tables["bronze_procurement_raw"] = procurement

    logger.info("Building document inventory under %s", files.documents_root)
    inventory = build_document_inventory(
        files.documents_root,
        files.data_root,
        registry,
        load_document_manifest(files.document_manifest),
        **common,
    )
    return BronzeIngestionResult(
        run_id=run,
        ingested_at=stamp,
        source_files=files,
        tables=tables,
        procurement_coverage=assess_procurement_coverage(procurement, registry),
        document_inventory=inventory,
        isr_completeness=isr_completeness_by_project(inventory.table, registry),
    )


def write_bronze(result: BronzeIngestionResult, writer: BronzeWriter) -> dict[str, str]:
    """Persist every Bronze table with ``writer``; returns ``{table: locator}``."""
    return {table.name: writer.write(table) for table in result.all_tables()}
