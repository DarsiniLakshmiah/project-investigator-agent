"""Spark helpers shared by the Gold transformations (native Spark only, no Python UDFs).

* identity: ``record_id`` from the natural key and ``record_hash`` from the hashed
  columns (canonical JSON of a typed struct) are Spark expressions, so the same input
  always yields the same ids and hashes, on any cluster;
* parsing: printed numbers / dates -> DECIMAL(38,6) / DATE with ``try_*`` functions, so
  behaviour is the same with ANSI mode on (serverless, Spark 4) or off (DBR 15.4);
* ``validate_frame`` and ``profile_frame`` check contracts and profile tables inside
  Spark; only aggregates reach the driver.

Everything here uses only JVM-native Spark functions (Photon/serverless friendly).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from worldbank_copilot.lakehouse.contracts import (
    DECIMAL_PRECISION,
    DECIMAL_SCALE,
    LOAD_RUN_ID,
    LOADED_AT,
    PIPELINE_VERSION,
    RECORD_HASH,
    RECORD_ID,
    SOURCE_SNAPSHOT_ID,
    TableContract,
)
from worldbank_copilot.lakehouse.spark_store import spark_type

DECIMAL_SQL = f"DECIMAL({DECIMAL_PRECISION},{DECIMAL_SCALE})"
NUMBER_PATTERN = r"^-?[0-9][0-9,]*(\.[0-9]{1,6})?%?$"
_NULL_TOKEN = "\u0000NULL"
_SEPARATOR = "\u001f"


@dataclass(frozen=True)
class GoldContext:
    """Operational metadata of one Gold build (never part of a record hash)."""

    source_snapshot_id: str  # fingerprint of the Silver inputs
    load_run_id: str
    loaded_at: datetime
    pipeline_version: str


def F():  # noqa: N802 - lazy import keeps the module importable without pyspark
    from pyspark.sql import functions

    return functions


def number(column: str) -> Any:
    """Printed number ('2,799,085', '98', '70.00%') -> DECIMAL(38,6); anything else NULL.

    At most six decimals are accepted, so no value is rounded by the cast.
    """
    f = F()
    parsed = f.expr(
        f"try_cast(regexp_replace(regexp_replace(trim({column}), ',', ''), '%', '') "
        f"AS {DECIMAL_SQL})"
    )
    return f.when(f.trim(f.col(column)).rlike(NUMBER_PATTERN), parsed).cast(DECIMAL_SQL)


def printed_date(column: str) -> Any:
    """'31-Mar-2016' -> 2016-03-31; 'Jun/2026' -> 2026-06-30 (end of month); else NULL."""
    f = F()
    day = f.expr(f"to_date(try_to_timestamp({column}, 'd-MMM-yyyy'))")
    month = f.expr(f"last_day(to_date(try_to_timestamp({column}, 'MMM/yyyy')))")
    return f.coalesce(day, month)


def key_expression(table: str, columns: list[str]) -> Any:
    f = F()
    parts = [f.lit(table)] + [
        f.coalesce(f.col(k).cast("string"), f.lit(_NULL_TOKEN)) for k in columns
    ]
    return f.substring(f.sha2(f.concat_ws(_SEPARATOR, *parts), 256), 1, 32)


def hash_expression(contract: TableContract) -> Any:
    f = F()
    struct = f.struct(*[f.col(c) for c in contract.hashed_columns])
    return f.sha2(f.to_json(struct, {"ignoreNullFields": "false"}), 256)


def finalize_frame(frame: Any, contract: TableContract, ctx: GoldContext) -> Any:
    """Cast to the contract types, add identity, content hash and load metadata."""
    f = F()
    body = [
        c
        for c in contract.columns
        if c.name
        not in {
            RECORD_ID,
            RECORD_HASH,
            SOURCE_SNAPSHOT_ID,
            LOAD_RUN_ID,
            LOADED_AT,
            PIPELINE_VERSION,
        }
    ]
    missing = [c.name for c in body if c.name not in frame.columns]
    if missing:
        raise KeyError(f"{contract.name}: builder did not produce {missing}")
    typed = frame.select(*[f.col(c.name).cast(spark_type(c.dtype)).alias(c.name) for c in body])
    typed = typed.withColumn(RECORD_ID, key_expression(contract.name, list(contract.natural_key)))
    typed = typed.withColumn(RECORD_HASH, hash_expression(contract))
    typed = (
        typed.withColumn(SOURCE_SNAPSHOT_ID, f.lit(ctx.source_snapshot_id))
        .withColumn(LOAD_RUN_ID, f.lit(ctx.load_run_id))
        .withColumn(LOADED_AT, f.lit(ctx.loaded_at).cast("timestamp"))
        .withColumn(PIPELINE_VERSION, f.lit(ctx.pipeline_version))
    )
    return typed.select(*contract.column_names)


def schema_problems(frame: Any, contract: TableContract) -> list[str]:
    actual = [(fld.name, fld.dataType.simpleString()) for fld in frame.schema.fields]
    expected = [(c.name, spark_type(c.dtype).simpleString()) for c in contract.columns]
    if actual == expected:
        return []
    return [f"schema differs: {sorted(set(actual) ^ set(expected))[:6]}"]


def validate_frame(frame: Any, contract: TableContract) -> list[str]:
    """Contract violations computed in Spark (required, vocabulary, duplicates)."""
    f = F()
    problems = schema_problems(frame, contract)
    if problems:
        return problems
    checks = {}
    for column in contract.columns:
        if not column.nullable:
            checks[f"{column.name} is NULL"] = f.sum(f.col(column.name).isNull().cast("int"))
        if column.vocabulary:
            bad = f.col(column.name).isNotNull() & ~f.col(column.name).isin(*column.vocabulary)
            checks[f"{column.name} outside vocabulary"] = f.sum(bad.cast("int"))
    checks["duplicate record_id"] = f.count(f.lit(1)) - f.countDistinct(RECORD_ID)
    key = f.concat_ws(
        _SEPARATOR,
        *[f.coalesce(f.col(k).cast("string"), f.lit(_NULL_TOKEN)) for k in contract.natural_key],
    )
    checks["duplicate natural key"] = f.count(f.lit(1)) - f.countDistinct(key)
    checks["stored record_hash differs from content"] = f.sum(
        (f.col(RECORD_HASH) != hash_expression(contract)).cast("int")
    )
    names = list(checks)
    row = frame.agg(
        *[f.coalesce(checks[n], f.lit(0)).alias(f"c{i}") for i, n in enumerate(names)]
    ).collect()[0]
    return [
        f"{contract.layer}.{contract.name}: {row[f'c{i}']} rows with {name}"
        for i, name in enumerate(names)
        if row[f"c{i}"]
    ]


def profile_frame(frame: Any, contract: TableContract) -> dict[str, Any]:
    """Row count, fingerprint and group counts computed in Spark (aggregates only)."""
    f = F()
    agg = frame.agg(
        f.count(f.lit(1)).alias("rows"),
        f.countDistinct(RECORD_ID).alias("ids"),
        f.sha2(
            f.concat_ws("\n", f.sort_array(f.collect_list(hash_expression(contract)))), 256
        ).alias("fingerprint"),
        f.sum((f.col(RECORD_HASH) != hash_expression(contract)).cast("int")).alias("bad"),
    ).collect()[0]
    groups = {}
    for group in contract.profile_groups:
        counts = frame.groupBy(*group).count().collect()
        groups["+".join(group)] = dict(
            sorted(
                ("|".join("" if r[g] is None else str(r[g]) for g in group), r["count"])
                for r in counts
            )
        )
    return {
        "table": f"{contract.layer}.{contract.name}",
        "row_count": agg["rows"],
        "distinct_record_ids": agg["ids"],
        "fingerprint": agg["fingerprint"] or hashlib.sha256(b"").hexdigest(),
        "stored_hash_mismatches": agg["bad"] or 0,
        "groups": groups,
    }


def input_fingerprint(frames: dict[str, Any]) -> str:
    """Fingerprint of the Silver inputs (sorted record hashes of every table)."""
    f = F()
    parts = []
    for name in sorted(frames):
        value = (
            frames[name]
            .agg(
                f.sha2(f.concat_ws("\n", f.sort_array(f.collect_list(RECORD_HASH))), 256).alias("h")
            )
            .collect()[0]["h"]
        )
        parts.append(f"{name}:{value}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()
