"""Governed, read-only, project-scoped table access for tools (Phase 9).

A tool never writes SQL. It describes a ``ReadRequest`` (logical table, project, columns,
typed filters, ordering, bound) and a ``TableReader`` executes it:

* the logical table must be in ``TABLES`` and every column must exist in its contract
  (schema drift fails loudly instead of returning wrong data);
* ``project_id`` is mandatory and is ALWAYS applied as a predicate before anything else;
  every returned row is re-checked (a foreign row raises ``ScopeViolation``);
* results are bounded (``MAX_ROWS``); exceeding the bound is an error, not truncation.

``SparkTableReader`` reads Unity Catalog Delta tables with the DataFrame API (literal
values via ``F.lit``; no string-built predicates). With ``pin_versions`` it records each
table's Delta version at the start of the request and reads ``VERSION AS OF`` that
version, so one request sees one consistent snapshot (recorded in ``data_snapshot``).
``InMemoryReader`` applies identical semantics to rows in memory (unit tests).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal, Protocol

from worldbank_copilot.intelligence.contracts import GOLD_CONTRACTS
from worldbank_copilot.lakehouse.contracts import TableContract
from worldbank_copilot.lakehouse.records import events_contract as silver_events_contract
from worldbank_copilot.lakehouse.records import (
    isr_contract,
    isr_disbursements_contract,
    silver_contract,
)
from worldbank_copilot.lakehouse.sql import qualified
from worldbank_copilot.retrieval.models import ScopeViolation

MAX_ROWS = 2000

# Logical table -> contract. Only these tables are readable by tools (allowlist).
TABLES: dict[str, Callable[[], TableContract]] = {
    "gold.project_360": GOLD_CONTRACTS["project_360"],
    "gold.project_timeline": GOLD_CONTRACTS["project_timeline"],
    "gold.result_progress": GOLD_CONTRACTS["result_progress"],
    "gold.risk_register": GOLD_CONTRACTS["risk_register"],
    "gold.attention_signals": GOLD_CONTRACTS["attention_signals"],
    "silver.projects": lambda: silver_contract("silver_projects"),
    "silver.loans": lambda: silver_contract("silver_loans"),
    "silver.isr_snapshots": isr_contract,
    "silver.isr_loan_disbursements": isr_disbursements_contract,
    "silver.project_events": silver_events_contract,
}

Op = Literal["eq", "in", "ge", "le", "not_null"]


class ReadError(Exception):
    """A read request is malformed or its result violates a bound."""


@dataclass(frozen=True)
class Filter:
    column: str
    op: Op
    value: Any = None


@dataclass(frozen=True)
class ReadRequest:
    table: str
    project_id: str
    columns: tuple[str, ...]
    filters: tuple[Filter, ...] = ()
    order_by: tuple[tuple[str, Literal["asc", "desc"]], ...] = ()  # nulls always last

    def all_columns(self) -> tuple[str, ...]:
        cols = list(self.columns)
        if "project_id" not in cols:
            cols.insert(0, "project_id")
        return tuple(cols)


def validate_request(request: ReadRequest) -> TableContract:
    if request.table not in TABLES:
        raise ReadError(f"table {request.table!r} is not readable by tools")
    if not request.project_id or not request.project_id.strip():
        raise ScopeViolation("every tool read must be scoped to a project_id")
    contract = TABLES[request.table]()
    known = set(contract.column_names)
    named = (
        set(request.all_columns())
        | {f.column for f in request.filters}
        | {c for c, _ in request.order_by}
    )
    unknown = sorted(named - known)
    if unknown:
        raise ReadError(f"{request.table} has no column(s) {unknown} (contract drift?)")
    for f in request.filters:
        if f.op == "in" and not isinstance(f.value, list | tuple):
            raise ReadError(f"filter {f.column} IN needs a list")
    return contract


def check_rows(request: ReadRequest, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) > MAX_ROWS:
        raise ReadError(f"{request.table}: more than {MAX_ROWS} rows for one project")
    foreign = {r.get("project_id") for r in rows} - {request.project_id}
    if foreign:
        raise ScopeViolation(f"{request.table} returned rows of {sorted(map(str, foreign))}")
    return rows


class TableReader(Protocol):
    def pin(self, tables: Iterable[str]) -> None: ...

    def read(self, request: ReadRequest) -> list[dict[str, Any]]: ...

    def snapshot(self) -> dict[str, int | None]: ...


# -- in-memory (tests) -----------------------------------------------------------------


def _comparable(value: Any) -> Any:
    if isinstance(value, datetime | date | Decimal | int | float | str | bool):
        return value
    return str(value)


def _matches(row: dict[str, Any], f: Filter) -> bool:
    value = row.get(f.column)
    if f.op == "not_null":
        return value is not None
    if value is None:
        return False
    if f.op == "eq":
        return value == f.value
    if f.op == "in":
        return value in f.value
    if f.op == "ge":
        return _comparable(value) >= _comparable(f.value)
    return _comparable(value) <= _comparable(f.value)


def sort_rows(rows: list[dict[str, Any]], order_by: Sequence[tuple[str, str]]) -> list[dict]:
    """Stable multi-key sort with nulls last in either direction (same as Spark below)."""
    out = list(rows)
    for column, direction in reversed(order_by):
        present = [r for r in out if r.get(column) is not None]
        missing = [r for r in out if r.get(column) is None]
        present.sort(key=lambda r: _comparable(r[column]), reverse=direction == "desc")  # noqa: B023
        out = present + missing
    return out


@dataclass
class InMemoryReader:
    tables: dict[str, list[dict[str, Any]]]
    versions: dict[str, int | None] = field(default_factory=dict)
    requests: list[ReadRequest] = field(default_factory=list)
    pinned: dict[str, int | None] = field(default_factory=dict)

    def pin(self, tables: Iterable[str]) -> None:
        for table in tables:
            self.pinned.setdefault(table, self.versions.get(table))

    def read(self, request: ReadRequest) -> list[dict[str, Any]]:
        validate_request(request)
        self.pin([request.table])
        self.requests.append(request)
        rows = [
            r
            for r in self.tables.get(request.table, [])
            if r.get("project_id") == request.project_id
        ]
        for f in request.filters:
            rows = [r for r in rows if _matches(r, f)]
        rows = sort_rows(rows, request.order_by)
        cols = request.all_columns()
        return check_rows(request, [{c: r.get(c) for c in cols} for r in rows])

    def snapshot(self) -> dict[str, int | None]:
        return dict(self.pinned)


# -- Spark / Unity Catalog ---------------------------------------------------------------


class SparkTableReader:
    """Reads governed Delta tables; one reader per request (consistent snapshot)."""

    def __init__(
        self,
        spark: Any,
        resolve: Callable[[str], str],
        *,
        pin_versions: bool = True,
    ):
        self.spark = spark
        self.resolve = resolve  # logical name -> catalog.schema.table (or a temp view)
        self.pin_versions = pin_versions
        self.pinned: dict[str, int | None] = {}

    @classmethod
    def for_settings(cls, spark: Any, settings: Any, **kwargs: Any) -> SparkTableReader:
        def resolve(logical: str) -> str:
            layer, table = logical.split(".", 1)
            return settings.table_name(layer, table)

        return cls(spark, resolve, **kwargs)

    def _qualified(self, logical: str) -> str:
        return qualified(*self.resolve(logical).split("."))

    def pin(self, tables: Iterable[str]) -> None:
        for table in tables:
            if table in self.pinned:
                continue
            if table not in TABLES:
                raise ReadError(f"table {table!r} is not readable by tools")
            version = None
            if self.pin_versions:
                history = self.spark.sql(f"DESCRIBE HISTORY {self._qualified(table)} LIMIT 1")
                version = int(history.select("version").collect()[0][0])
            self.pinned[table] = version

    def read(self, request: ReadRequest) -> list[dict[str, Any]]:
        from pyspark.sql import functions as F  # noqa: N812

        validate_request(request)
        self.pin([request.table])
        version = self.pinned[request.table]
        name = self._qualified(request.table)
        frame = (
            self.spark.sql(f"SELECT * FROM {name} VERSION AS OF {int(version)}")
            if version is not None
            else self.spark.table(name)
        )
        frame = frame.where(F.col("project_id") == F.lit(request.project_id))
        for f in request.filters:
            col = F.col(f.column)
            if f.op == "eq":
                frame = frame.where(col == F.lit(f.value))
            elif f.op == "in":
                frame = frame.where(col.isin([F.lit(v) for v in f.value]))
            elif f.op == "ge":
                frame = frame.where(col >= F.lit(f.value))
            elif f.op == "le":
                frame = frame.where(col <= F.lit(f.value))
            else:
                frame = frame.where(col.isNotNull())
        frame = frame.select(*request.all_columns())
        if request.order_by:
            frame = frame.orderBy(
                *[
                    F.col(c).asc_nulls_last() if d == "asc" else F.col(c).desc_nulls_last()
                    for c, d in request.order_by
                ]
            )
        rows = [r.asDict(recursive=True) for r in frame.limit(MAX_ROWS + 1).collect()]
        return check_rows(request, rows)

    def snapshot(self) -> dict[str, int | None]:
        return dict(self.pinned)
