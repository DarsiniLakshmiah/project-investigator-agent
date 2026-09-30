"""Bronze layer representation and storage adapters.

A Bronze table is a list of source-preserving records plus metadata describing
where they came from. Ingestion builds ``BronzeTable`` objects without knowing
how they will be stored. A ``BronzeWriter`` persists them:

* ``LocalJsonlBronzeWriter`` writes JSON Lines for local development.
* A Delta writer will be added once the Databricks runtime decision is made
  (IMPLEMENTATION_PLAN.md §8); it is intentionally not stubbed here.

Record conventions:
* ``_source_file``, ``_source_sheet``, ``_source_row``, ``_ingested_at`` and
  ``_ingestion_run_id`` describe lineage.
* ``project_id`` is the canonical in-scope project ID (the join key).
* All source values are kept as text exactly as read (no trimming or casting).
* Source columns not covered by the contract are kept in ``_extra_fields``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

BronzeRecord = dict[str, Any]

LINEAGE_FIELDS = (
    "_source_file",
    "_source_sheet",
    "_source_row",
    "_ingested_at",
    "_ingestion_run_id",
)


class SourceMetadata(BaseModel):
    """How a Bronze table was read from its source."""

    model_config = ConfigDict(extra="forbid")

    source_name: str
    source_file: str
    source_sheet: str | None = None
    ingested_at: str
    ingestion_run_id: str
    header_row: int | None = None
    secondary_header_row: int | None = None
    source_columns: list[str] = []
    column_mapping: dict[str, str] = {}  # bronze field -> source header (as written)
    api_field_names: dict[str, str] = {}  # bronze field -> API name (secondary header)
    missing_optional_columns: list[str] = []
    unmapped_columns: list[str] = []
    known_gaps: list[str] = []
    rows_scanned: int = 0
    rows_matched: int = 0
    rows_without_project_id: int = 0
    empty_rows: int = 0
    notes: list[str] = []


@dataclass
class BronzeTable:
    name: str
    records: list[BronzeRecord]
    metadata: SourceMetadata
    fields: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def for_project(self, project_id: str) -> list[BronzeRecord]:
        return [r for r in self.records if r.get("project_id") == project_id]

    def count_by_project(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record["project_id"]] = counts.get(record["project_id"], 0) + 1
        return counts


class BronzeWriter(Protocol):
    def write(self, table: BronzeTable) -> str:
        """Persist ``table`` and return a locator (path or table name)."""
        ...


class LocalJsonlBronzeWriter:
    """Writes ``<root>/<table>.jsonl`` plus ``<table>.metadata.json`` (overwrite)."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def write(self, table: BronzeTable) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        data_path = self.root / f"{table.name}.jsonl"
        with data_path.open("w", encoding="utf-8", newline="\n") as fh:
            for record in table.records:
                fh.write(json.dumps(record, ensure_ascii=False, default=str))
                fh.write("\n")
        meta_path = self.root / f"{table.name}.metadata.json"
        meta = table.metadata.model_dump()
        meta["fields"] = table.fields
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(data_path)
