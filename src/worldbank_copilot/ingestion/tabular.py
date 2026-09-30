"""Contract-driven reading of tabular sources into Bronze tables.

Handles the header quirks seen in the real files:

* title/banner rows above the header ("World Bank Projects, data as of ...");
* a second header row of API field names (Projects sheet), which is skipped
  and recorded in metadata;
* alternative header labels ("Project" vs "Project ID") via contract aliases;
* stray or doubled whitespace in header names.

Values are never trimmed, cast or otherwise cleaned: Bronze keeps them as read.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time
from itertools import chain, islice
from pathlib import Path
from typing import Any

from worldbank_copilot.common.exceptions import SourceContractError
from worldbank_copilot.ingestion.contracts import SourceContract
from worldbank_copilot.transformations.bronze import (
    LINEAGE_FIELDS,
    BronzeRecord,
    BronzeTable,
    SourceMetadata,
)

Row = tuple[int, Sequence[Any]]  # (1-based source row number, cell values)

SECONDARY_HEADER_MIN_MATCH = 0.8


def normalize_column_name(value: Any) -> str:
    """Collapse internal whitespace and strip; ``None`` becomes ``""``."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _match_key(value: Any) -> str:
    return normalize_column_name(value).casefold()


def to_raw_text(value: Any) -> str | None:
    """Represent a source cell as text without cleaning it.

    Strings are returned unchanged (including surrounding whitespace). Numbers
    use Python's shortest round-trip representation (``100000000.0`` stays
    ``"100000000.0"``), so the original numeric value is recoverable.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return str(value)


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def iter_csv_rows(path: Path | str) -> Iterator[Row]:
    """Yield CSV records; row number is the 1-based record index (header = 1)."""
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        yield from enumerate(csv.reader(fh), start=1)


def iter_worksheet_rows(workbook: Any, sheet_name: str) -> Iterator[Row]:
    """Yield rows of an openpyxl worksheet with their Excel row numbers."""
    if sheet_name not in workbook.sheetnames:
        raise SourceContractError(
            f"Sheet {sheet_name!r} not found; available sheets: {workbook.sheetnames}"
        )
    worksheet = workbook[sheet_name]
    yield from enumerate(worksheet.iter_rows(min_row=1, values_only=True), start=1)


@dataclass
class ResolvedHeader:
    header_row: int
    data_start: int  # index into the preview list where data rows begin
    source_columns: list[str]
    field_index: dict[str, int]
    column_mapping: dict[str, str]
    secondary_header_row: int | None = None
    api_field_names: dict[str, str] = field(default_factory=dict)
    missing_optional: list[str] = field(default_factory=list)
    unmapped: dict[int, str] = field(default_factory=dict)


def resolve_header(preview: Sequence[Row], contract: SourceContract) -> ResolvedHeader:
    """Find the header row and map its columns to contract fields.

    The header is the first row containing every required column. Raises
    ``SourceContractError`` naming the missing columns if no row qualifies.
    """
    alias_to_field: dict[str, str] = {}
    for spec in contract.columns:
        for name in spec.source_names:
            alias_to_field[_match_key(name)] = spec.field
    required = contract.required_fields

    best: tuple[int, int, list[str]] | None = None  # (matched, row number, missing)
    for position, (row_number, values) in enumerate(preview):
        matched = {alias_to_field[k] for k in map(_match_key, values) if k in alias_to_field}
        missing = [f for f in required if f not in matched]
        if not missing:
            return _map_header(preview, position, contract, alias_to_field)
        if best is None or len(matched) > best[0]:
            best = (len(matched), row_number, missing)

    detail = ""
    if best is not None and best[0] > 0:
        missing_names = [_source_label(contract, f) for f in best[2]]
        detail = f" Closest candidate is row {best[1]}, missing required columns {missing_names}."
    raise SourceContractError(
        f"{contract.name}: no header row with all required columns in the first "
        f"{len(preview)} rows.{detail}"
    )


def _source_label(contract: SourceContract, field_name: str) -> str:
    spec = next(c for c in contract.columns if c.field == field_name)
    return " / ".join(spec.source_names)


def _map_header(
    preview: Sequence[Row],
    position: int,
    contract: SourceContract,
    alias_to_field: dict[str, str],
) -> ResolvedHeader:
    row_number, values = preview[position]
    source_columns = [to_raw_text(v) or "" for v in values]
    field_index: dict[str, int] = {}
    column_mapping: dict[str, str] = {}
    unmapped: dict[int, str] = {}
    for index, header in enumerate(source_columns):
        key = _match_key(header)
        if not key:
            continue
        target = alias_to_field.get(key)
        if target is None:
            unmapped[index] = normalize_column_name(header)
            continue
        if target in field_index:
            raise SourceContractError(
                f"{contract.name}: columns {column_mapping[target]!r} and {header!r} "
                f"both map to field {target!r}"
            )
        field_index[target] = index
        column_mapping[target] = header

    resolved = ResolvedHeader(
        header_row=row_number,
        data_start=position + 1,
        source_columns=source_columns,
        field_index=field_index,
        column_mapping=column_mapping,
        missing_optional=[c.field for c in contract.columns if c.field not in field_index],
        unmapped=unmapped,
    )

    if contract.secondary_header_names and position + 1 < len(preview):
        next_number, next_values = preview[position + 1]
        keys = [_match_key(v) for v in next_values if not _is_blank(v)]
        hits = sum(1 for k in keys if k in contract.secondary_header_names)
        if keys and hits / len(keys) >= SECONDARY_HEADER_MIN_MATCH:
            resolved.secondary_header_row = next_number
            resolved.data_start = position + 2
            resolved.api_field_names = {
                f: to_raw_text(next_values[i]) or ""
                for f, i in field_index.items()
                if i < len(next_values)
            }
    return resolved


def read_tabular_source(
    rows: Iterable[Row],
    contract: SourceContract,
    project_ids: Iterable[str],
    *,
    source_file: str,
    ingested_at: str,
    run_id: str,
    source_sheet: str | None = None,
) -> BronzeTable:
    """Stream ``rows`` into a Bronze table containing only in-scope projects."""
    wanted = set(project_ids)
    iterator = iter(rows)
    preview = list(islice(iterator, contract.max_header_scan_rows + 1))
    header = resolve_header(preview, contract)
    pid_index = header.field_index[contract.project_id_field]
    ordered_fields = [c.field for c in contract.columns]

    records: list[BronzeRecord] = []
    scanned = empty = without_pid = 0
    for row_number, values in chain(preview[header.data_start :], iterator):
        scanned += 1
        if all(_is_blank(v) for v in values):
            empty += 1
            continue
        raw_pid = to_raw_text(values[pid_index]) if pid_index < len(values) else None
        canonical = raw_pid.strip().upper() if raw_pid else ""
        if not canonical:
            without_pid += 1
            continue
        if canonical not in wanted:
            continue
        record: BronzeRecord = {
            "_source_file": source_file,
            "_source_sheet": source_sheet,
            "_source_row": row_number,
            "_ingested_at": ingested_at,
            "_ingestion_run_id": run_id,
            "project_id": canonical,
        }
        for name in ordered_fields:
            index = header.field_index.get(name)
            record[name] = (
                to_raw_text(values[index]) if index is not None and index < len(values) else None
            )
        record["_extra_fields"] = {
            name: to_raw_text(values[i]) if i < len(values) else None
            for i, name in header.unmapped.items()
        }
        records.append(record)

    metadata = SourceMetadata(
        source_name=contract.name,
        source_file=source_file,
        source_sheet=source_sheet,
        ingested_at=ingested_at,
        ingestion_run_id=run_id,
        header_row=header.header_row,
        secondary_header_row=header.secondary_header_row,
        source_columns=header.source_columns,
        column_mapping=header.column_mapping,
        api_field_names=header.api_field_names,
        missing_optional_columns=header.missing_optional,
        unmapped_columns=list(header.unmapped.values()),
        known_gaps=list(contract.known_gaps),
        rows_scanned=scanned,
        rows_matched=len(records),
        rows_without_project_id=without_pid,
        empty_rows=empty,
    )
    fields = [*LINEAGE_FIELDS, "project_id", *ordered_fields, "_extra_fields"]
    return BronzeTable(contract.bronze_table, records, metadata, fields)
