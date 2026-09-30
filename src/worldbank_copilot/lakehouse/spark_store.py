"""Unity Catalog + Delta persistence through a Spark session (Databricks only).

This is the only module that touches Spark. ``pyspark`` is imported lazily, so the
package imports and unit-tests locally without it. The session is passed in
explicitly (the notebook's ``spark``); nothing here creates or discovers clusters.

Behaviour:
* the catalog must already exist (it is validated, never created);
* schemas and Volumes are validated and created only when missing; a permission
  failure is reported with the exact object and operation;
* a table is created from the contract DDL if missing; an existing table must have
  exactly the contract's columns and types, otherwise loading stops (no silent
  schema evolution);
* rows are written through an explicit Spark schema (DECIMAL(38,6), DATE,
  TIMESTAMP, ...) and the snapshot MERGE in ``sql.snapshot_merge_sql``;
* the MERGE metrics of every write are read from the Delta history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from worldbank_copilot.common.exceptions import LakehouseError
from worldbank_copilot.lakehouse.contracts import (
    DECIMAL_PRECISION,
    DECIMAL_SCALE,
    DType,
    TableContract,
)
from worldbank_copilot.lakehouse.sql import (
    create_schema_sql,
    create_table_sql,
    create_volume_sql,
    history_sql,
    qualified,
    snapshot_merge_sql,
)


@dataclass(frozen=True)
class ObjectStatus:
    object_type: str  # CATALOG | SCHEMA | VOLUME | TABLE
    name: str
    status: str  # EXISTS | CREATED


@dataclass(frozen=True)
class WriteResult:
    table: str
    rows_in_snapshot: int
    inserted: int
    updated: int
    deleted: int
    delta_version: int | None


def merge_result(
    table: str, row_count: int, version_before: Any, entry: dict[str, Any]
) -> WriteResult:
    """MERGE metrics from the latest Delta history entry.

    If the MERGE created no new version it changed nothing: the history still shows
    the older operation, whose metrics must not be attributed to this write.
    """
    committed = entry.get("version") != version_before and entry.get("operation") == "MERGE"
    metrics = (entry.get("operationMetrics") or {}) if committed else {}

    def metric(key: str) -> int:
        return int(metrics.get(key, 0))

    return WriteResult(
        table,
        row_count,
        metric("numTargetRowsInserted"),
        metric("numTargetRowsUpdated"),
        metric("numTargetRowsDeleted"),
        entry.get("version"),
    )


def spark_type(dtype: DType) -> Any:
    from pyspark.sql import types as T

    return {
        DType.STRING: T.StringType(),
        DType.BIGINT: T.LongType(),
        DType.BOOLEAN: T.BooleanType(),
        DType.DATE: T.DateType(),
        DType.TIMESTAMP: T.TimestampType(),
        DType.DECIMAL: T.DecimalType(DECIMAL_PRECISION, DECIMAL_SCALE),
        DType.ARRAY_STRING: T.ArrayType(T.StringType()),
        DType.ARRAY_BIGINT: T.ArrayType(T.LongType()),
        DType.MAP_STRING: T.MapType(T.StringType(), T.StringType()),
    }[dtype]


def spark_schema(contract: TableContract) -> Any:
    from pyspark.sql import types as T

    return T.StructType(
        [T.StructField(c.name, spark_type(c.dtype), c.nullable) for c in contract.columns]
    )


def _simple_type(data_type: Any) -> str:
    return data_type.simpleString().upper().replace(" ", "")


class SparkDeltaStore:
    def __init__(self, spark: Any, catalog: str, schemas: dict[str, str]):
        self.spark = spark
        self.catalog = catalog
        self.schemas = schemas  # layer -> schema name

    # --- Unity Catalog objects --------------------------------------------------

    def _sql(self, statement: str, *, object_name: str, operation: str) -> Any:
        try:
            return self.spark.sql(statement)
        except Exception as exc:  # noqa: BLE001 - re-raised with the object and operation
            raise LakehouseError(
                f"{operation} on {object_name} failed: {type(exc).__name__}: {exc}. "
                f"Check that the current principal has the required privilege "
                f"(e.g. USE CATALOG, CREATE SCHEMA, CREATE VOLUME, CREATE TABLE, MODIFY)."
            ) from exc

    def validate_catalog(self) -> ObjectStatus:
        rows = self._sql(
            f"SHOW CATALOGS LIKE '{self.catalog}'",
            object_name=self.catalog,
            operation="SHOW CATALOGS",
        ).collect()
        if not rows:
            raise LakehouseError(
                f"Catalog {self.catalog!r} does not exist or is not visible to the current "
                "principal. Phase 6 uses an existing catalog and never creates or substitutes one."
            )
        self._sql(
            f"USE CATALOG {qualified(self.catalog)}",
            object_name=self.catalog,
            operation="USE CATALOG",
        )
        return ObjectStatus("CATALOG", self.catalog, "EXISTS")

    def ensure_schema(self, layer: str, comment: str) -> ObjectStatus:
        schema = self.schemas[layer]
        name = f"{self.catalog}.{schema}"
        found = self._sql(
            f"SHOW SCHEMAS IN {qualified(self.catalog)} LIKE '{schema}'",
            object_name=name,
            operation="SHOW SCHEMAS",
        ).collect()
        if found:
            return ObjectStatus("SCHEMA", name, "EXISTS")
        self._sql(
            create_schema_sql(self.catalog, schema, comment),
            object_name=name,
            operation="CREATE SCHEMA",
        )
        return ObjectStatus("SCHEMA", name, "CREATED")

    def ensure_volume(self, layer: str, volume: str, comment: str) -> ObjectStatus:
        schema = self.schemas[layer]
        name = f"{self.catalog}.{schema}.{volume}"
        found = self._sql(
            f"SHOW VOLUMES IN {qualified(self.catalog, schema)} LIKE '{volume}'",
            object_name=name,
            operation="SHOW VOLUMES",
        ).collect()
        if found:
            return ObjectStatus("VOLUME", name, "EXISTS")
        self._sql(
            create_volume_sql(self.catalog, schema, volume, comment),
            object_name=name,
            operation="CREATE VOLUME",
        )
        return ObjectStatus("VOLUME", name, "CREATED")

    def list_tables(self, layer: str) -> list[str]:
        schema = self.schemas[layer]
        rows = self._sql(
            f"SHOW TABLES IN {qualified(self.catalog, schema)}",
            object_name=f"{self.catalog}.{schema}",
            operation="SHOW TABLES",
        ).collect()
        return sorted(r["tableName"] for r in rows)

    # --- tables --------------------------------------------------------------------

    def fq(self, contract: TableContract) -> str:
        return f"{self.catalog}.{self.schemas[contract.layer]}.{contract.name}"

    def ensure_table(self, contract: TableContract) -> ObjectStatus:
        schema = self.schemas[contract.layer]
        name = self.fq(contract)
        if not self.spark.catalog.tableExists(name):
            self._sql(
                create_table_sql(self.catalog, schema, contract),
                object_name=name,
                operation="CREATE TABLE",
            )
            status = "CREATED"
        else:
            status = "EXISTS"
        self.require_schema_matches(contract)
        return ObjectStatus("TABLE", name, status)

    def require_schema_matches(self, contract: TableContract) -> None:
        actual = [
            (f.name, _simple_type(f.dataType))
            for f in self.spark.table(self.fq(contract)).schema.fields
        ]
        expected = [(c.name, _simple_type(spark_type(c.dtype))) for c in contract.columns]
        if actual != expected:
            diff = sorted(set(actual) ^ set(expected))
            raise LakehouseError(
                f"{self.fq(contract)} exists with a schema that differs from its contract "
                f"({diff[:6]}). Schemas are never evolved silently: review the contract change "
                "and migrate or drop the table explicitly."
            )

    def _latest_history(self, contract: TableContract) -> dict[str, Any]:
        schema = self.schemas[contract.layer]
        rows = self._sql(
            history_sql(self.catalog, schema, contract.name),
            object_name=self.fq(contract),
            operation="DESCRIBE HISTORY",
        ).collect()
        return rows[0].asDict(recursive=True) if rows else {}

    def write_snapshot(self, contract: TableContract, rows: list[dict[str, Any]]) -> WriteResult:
        schema = self.schemas[contract.layer]
        names = contract.column_names
        before = self._latest_history(contract).get("version")
        frame = self.spark.createDataFrame(
            [tuple(r[n] for n in names) for r in rows],
            schema=spark_schema(contract),
            verifySchema=True,
        )
        view = f"_wbc_stage_{contract.name}"
        frame.createOrReplaceTempView(view)
        try:
            self._sql(
                snapshot_merge_sql(self.catalog, schema, contract, view),
                object_name=self.fq(contract),
                operation="MERGE",
            )
        finally:
            self.spark.catalog.dropTempView(view)
        entry = self._latest_history(contract)
        return merge_result(self.fq(contract), len(rows), before, entry)

    def read_rows(self, contract: TableContract) -> list[dict[str, Any]]:
        """Read the table back (small tables) as dicts with Python values."""
        frame = self.spark.table(self.fq(contract)).select(*contract.column_names)
        return [row.asDict(recursive=True) for row in frame.collect()]
