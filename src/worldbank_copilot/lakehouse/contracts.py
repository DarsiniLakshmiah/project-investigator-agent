"""Explicit persisted-data contracts for the Delta tables (Phase 6).

A ``TableContract`` states, for one Delta table: every column with its type,
nullability, role and (where applicable) controlled vocabulary; the natural key;
and the write mode. Rows are validated against the contract *before* anything
is written. Nothing is inferred from the data at write time.

Where a Phase 1-5 Pydantic model already defines the domain schema it stays the
source of truth: ``model_columns`` derives columns from it with a fixed type
mapping, so there is no second, conflicting schema definition. The resulting
contracts are frozen in ``configs/delta_contracts.lock.json``; a schema change is
therefore always an explicit, reviewed diff (``tests/unit/test_lakehouse_contracts``).
"""

from __future__ import annotations

import types
import typing
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum, StrEnum
from typing import Any

from pydantic import BaseModel

from worldbank_copilot.common.exceptions import LakehouseError

# Every money / numeric value is persisted as DECIMAL(38, 6): the largest scale in the
# validated sources is 6 (procurement amounts). Values with a larger scale are rejected,
# never rounded.
DECIMAL_PRECISION = 38
DECIMAL_SCALE = 6


class ContractError(LakehouseError):
    """Rows do not satisfy their table contract (nothing is written)."""


class DType(StrEnum):
    STRING = "STRING"
    BIGINT = "BIGINT"
    BOOLEAN = "BOOLEAN"
    DATE = "DATE"
    TIMESTAMP = "TIMESTAMP"
    DECIMAL = "DECIMAL"
    ARRAY_STRING = "ARRAY<STRING>"
    ARRAY_BIGINT = "ARRAY<BIGINT>"
    MAP_STRING = "MAP<STRING,STRING>"


class Role(StrEnum):
    KEY = "KEY"  # record identity (record_id and natural-key columns)
    DATA = "DATA"  # business content
    PROVENANCE = "PROVENANCE"  # where the value came from (stable across reloads)
    QUALITY = "QUALITY"  # extraction status / quality issues
    INTEGRITY = "INTEGRITY"  # record_hash (content fingerprint)
    OPERATIONAL = "OPERATIONAL"  # load metadata; excluded from the content hash


class WriteMode(StrEnum):
    # The table holds exactly the current source snapshot: insert new records, update
    # records whose content hash changed, delete records absent from the snapshot.
    SNAPSHOT_MERGE = "SNAPSHOT_MERGE"


@dataclass(frozen=True)
class Column:
    name: str
    dtype: DType
    nullable: bool = True
    role: Role = Role.DATA
    vocabulary: tuple[str, ...] | None = None
    description: str | None = None

    @property
    def sql_type(self) -> str:
        if self.dtype is DType.DECIMAL:
            return f"DECIMAL({DECIMAL_PRECISION},{DECIMAL_SCALE})"
        return self.dtype.value

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "type": self.sql_type,
            "nullable": self.nullable,
            "role": self.role.value,
        }
        if self.vocabulary:
            out["vocabulary"] = list(self.vocabulary)
        return out


RECORD_ID = "record_id"
RECORD_HASH = "record_hash"
LOAD_RUN_ID = "_load_run_id"
LOADED_AT = "_loaded_at"
SOURCE_SNAPSHOT_ID = "_source_snapshot_id"
PIPELINE_VERSION = "_pipeline_version"

STANDARD_HEAD = (
    Column(
        RECORD_ID,
        DType.STRING,
        nullable=False,
        role=Role.KEY,
        description="Deterministic id from the natural key (stable across reloads).",
    ),
)
STANDARD_TAIL = (
    Column(
        RECORD_HASH,
        DType.STRING,
        nullable=False,
        role=Role.INTEGRITY,
        description="SHA-256 of the business, provenance and quality columns.",
    ),
    Column(
        SOURCE_SNAPSHOT_ID,
        DType.STRING,
        nullable=False,
        role=Role.OPERATIONAL,
        description="Source snapshot (hash of all source file hashes) of the load.",
    ),
    Column(LOAD_RUN_ID, DType.STRING, nullable=False, role=Role.OPERATIONAL),
    Column(LOADED_AT, DType.TIMESTAMP, nullable=False, role=Role.OPERATIONAL),
    Column(PIPELINE_VERSION, DType.STRING, nullable=False, role=Role.OPERATIONAL),
)


@dataclass(frozen=True)
class TableContract:
    name: str  # Delta table name inside its schema
    layer: str  # bronze | silver
    source_dataset: str  # repository dataset name (Phase 2-5)
    columns: tuple[Column, ...]
    natural_key: tuple[str, ...]
    description: str
    write_mode: WriteMode = WriteMode.SNAPSHOT_MERGE
    profile_groups: tuple[tuple[str, ...], ...] = field(default=())

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def column(self, name: str) -> Column:
        return next(c for c in self.columns if c.name == name)

    @property
    def hashed_columns(self) -> list[str]:
        excluded = {Role.OPERATIONAL, Role.INTEGRITY}
        return [c.name for c in self.columns if c.role not in excluded]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "layer": self.layer,
            "source_dataset": self.source_dataset,
            "natural_key": list(self.natural_key),
            "write_mode": self.write_mode.value,
            "columns": [c.to_dict() for c in self.columns],
        }


def build_contract(
    name: str,
    layer: str,
    source_dataset: str,
    body: Iterable[Column],
    natural_key: Iterable[str],
    description: str,
    profile_groups: Iterable[Iterable[str]] = (),
) -> TableContract:
    columns = (*STANDARD_HEAD, *body, *STANDARD_TAIL)
    names = [c.name for c in columns]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ContractError(f"{name}: duplicate columns {duplicates}")
    key = tuple(natural_key)
    missing = [k for k in key if k not in names]
    if missing:
        raise ContractError(f"{name}: natural key columns {missing} are not in the contract")
    return TableContract(
        name,
        layer,
        source_dataset,
        columns,
        key,
        description,
        profile_groups=tuple(tuple(g) for g in profile_groups),
    )


# ---------------------------------------------------------------------------
# Columns derived from the existing Pydantic domain models
# ---------------------------------------------------------------------------

_SCALARS: dict[type, DType] = {
    str: DType.STRING,
    int: DType.BIGINT,
    bool: DType.BOOLEAN,
    Decimal: DType.DECIMAL,
    date: DType.DATE,
    datetime: DType.TIMESTAMP,
}
JSON_SUFFIX = "_json"


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        nullable = len(args) < len(typing.get_args(annotation))
        if len(args) == 1:
            return args[0], nullable
    return annotation, False


def annotation_column(
    name: str, annotation: Any, *, role: Role = Role.DATA, required: bool = False
) -> Column:
    """Column for one model field. Nested structures become ``<name>_json`` strings."""
    inner, optional = _unwrap_optional(annotation)
    nullable = optional or not required
    if isinstance(inner, type) and issubclass(inner, Enum):
        vocab = tuple(str(m.value) for m in inner)
        return Column(name, DType.STRING, nullable, role, vocabulary=vocab)
    if isinstance(inner, type) and inner in _SCALARS:
        return Column(name, _SCALARS[inner], nullable, role)
    if typing.get_origin(inner) is list:
        (item,) = typing.get_args(inner) or (Any,)
        if item is str:
            return Column(name, DType.ARRAY_STRING, nullable, role)
        if item is int:
            return Column(name, DType.ARRAY_BIGINT, nullable, role)
    # Nested models, lists of models and dicts: kept complete as canonical JSON.
    return Column(name + JSON_SUFFIX, DType.STRING, nullable, role)


def model_columns(
    model: type[BaseModel], *, exclude: Iterable[str] = (), roles: dict[str, Role] | None = None
) -> list[Column]:
    """Columns for every field of ``model`` (in declaration order)."""
    roles = roles or {}
    skip = set(exclude)
    out = []
    for name, info in model.model_fields.items():
        if name in skip:
            continue
        out.append(
            annotation_column(
                name, info.annotation, role=roles.get(name, Role.DATA), required=info.is_required()
            )
        )
    return out


# ---------------------------------------------------------------------------
# Row validation
# ---------------------------------------------------------------------------


def _type_error(column: Column, value: Any) -> str | None:
    dtype = column.dtype
    if dtype is DType.STRING:
        return None if isinstance(value, str) else "expected str"
    if dtype is DType.BIGINT:
        return None if isinstance(value, int) and not isinstance(value, bool) else "expected int"
    if dtype is DType.BOOLEAN:
        return None if isinstance(value, bool) else "expected bool"
    if dtype is DType.DATE:
        ok = isinstance(value, date) and not isinstance(value, datetime)
        return None if ok else "expected date"
    if dtype is DType.TIMESTAMP:
        if not isinstance(value, datetime):
            return "expected datetime"
        return None if value.tzinfo is not None else "timestamp must be timezone-aware"
    if dtype is DType.DECIMAL:
        if not isinstance(value, Decimal) or not value.is_finite():
            return "expected finite Decimal (never float)"
        exponent = value.as_tuple().exponent
        scale = max(0, -exponent) if isinstance(exponent, int) else 0
        integer_digits = len(value.as_tuple().digits) + min(0, exponent)
        if scale > DECIMAL_SCALE:
            return f"scale {scale} exceeds {DECIMAL_SCALE} (would require rounding)"
        if integer_digits > DECIMAL_PRECISION - DECIMAL_SCALE:
            return "too many integer digits"
        return None
    if dtype is DType.ARRAY_STRING:
        ok = isinstance(value, list) and all(isinstance(v, str) for v in value)
        return None if ok else "expected list[str]"
    if dtype is DType.ARRAY_BIGINT:
        ok = isinstance(value, list) and all(
            isinstance(v, int) and not isinstance(v, bool) for v in value
        )
        return None if ok else "expected list[int]"
    if dtype is DType.MAP_STRING:
        ok = isinstance(value, dict) and all(
            isinstance(k, str) and (v is None or isinstance(v, str)) for k, v in value.items()
        )
        return None if ok else "expected dict[str, str | None]"
    return f"unsupported type {dtype}"


def validate_rows(contract: TableContract, rows: list[dict[str, Any]]) -> list[str]:
    """All contract violations (empty list = valid). Includes duplicate protection."""
    problems: list[str] = []
    expected = contract.column_names
    seen_ids: dict[str, int] = {}
    seen_keys: dict[tuple, int] = {}
    for index, row in enumerate(rows):
        if list(row) != expected:
            extra = sorted(set(row) - set(expected))
            missing = sorted(set(expected) - set(row))
            problems.append(f"row {index}: columns differ (missing {missing}, extra {extra})")
            continue
        for column in contract.columns:
            value = row[column.name]
            if value is None:
                if not column.nullable:
                    problems.append(f"row {index}: {column.name} is required")
                continue
            error = _type_error(column, value)
            if error:
                problems.append(f"row {index}: {column.name}: {error} ({value!r:.60})")
            elif column.vocabulary and value not in column.vocabulary:
                problems.append(f"row {index}: {column.name}: {value!r} not in vocabulary")
        record_id = row[RECORD_ID]
        if record_id in seen_ids:
            problems.append(f"row {index}: duplicate record_id (also row {seen_ids[record_id]})")
        seen_ids[record_id] = index
        key = tuple(
            tuple(v) if isinstance(v, list) else v for v in (row[k] for k in contract.natural_key)
        )
        if key in seen_keys:
            problems.append(f"row {index}: duplicate natural key {key!r:.120}")
        seen_keys[key] = index
    return problems


def require_valid(contract: TableContract, rows: list[dict[str, Any]]) -> None:
    problems = validate_rows(contract, rows)
    if problems:
        shown = "; ".join(problems[:10])
        more = f" (+{len(problems) - 10} more)" if len(problems) > 10 else ""
        raise ContractError(f"{contract.layer}.{contract.name}: {shown}{more}")
