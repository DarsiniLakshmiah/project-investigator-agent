"""Unity Catalog / Delta SQL generation (pure functions, no Spark).

Keeping the statements as plain strings makes the exact DDL and MERGE semantics
reviewable and unit-testable locally; ``spark_store`` only executes them.
"""

from __future__ import annotations

from worldbank_copilot.lakehouse.contracts import RECORD_HASH, RECORD_ID, TableContract


def quote(identifier: str) -> str:
    return "`" + identifier.replace("`", "``") + "`"


def qualified(*parts: str) -> str:
    return ".".join(quote(p) for p in parts)


def _literal(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def create_schema_sql(catalog: str, schema: str, comment: str) -> str:
    return f"CREATE SCHEMA IF NOT EXISTS {qualified(catalog, schema)} COMMENT {_literal(comment)}"


def create_volume_sql(catalog: str, schema: str, volume: str, comment: str) -> str:
    return (
        f"CREATE VOLUME IF NOT EXISTS {qualified(catalog, schema, volume)} "
        f"COMMENT {_literal(comment)}"
    )


def create_table_sql(catalog: str, schema: str, contract: TableContract) -> str:
    """Explicit DDL: every column typed, NOT NULL where the contract requires it."""
    lines = []
    for c in contract.columns:
        null = "" if c.nullable else " NOT NULL"
        comment = f" COMMENT {_literal(c.description)}" if c.description else ""
        lines.append(f"  {quote(c.name)} {c.sql_type}{null}{comment}")
    body = ",\n".join(lines)
    return (
        f"CREATE TABLE IF NOT EXISTS {qualified(catalog, schema, contract.name)} (\n{body}\n)\n"
        f"USING DELTA\nCOMMENT {_literal(contract.description)}\n"
        f"TBLPROPERTIES ('worldbank.source_dataset' = {_literal(contract.source_dataset)}, "
        f"'worldbank.natural_key' = {_literal(','.join(contract.natural_key))}, "
        f"'worldbank.write_mode' = {_literal(contract.write_mode.value)})"
    )


def snapshot_merge_sql(catalog: str, schema: str, contract: TableContract, view: str) -> str:
    """Idempotent snapshot MERGE keyed on ``record_id``.

    * new record -> INSERT
    * same record, different content hash -> UPDATE (all columns)
    * same record, same content hash -> untouched (load metadata keeps its first value)
    * record no longer in the snapshot -> DELETE
    Re-running an unchanged snapshot therefore changes no rows.
    """
    target = qualified(catalog, schema, contract.name)
    columns = ", ".join(quote(c) for c in contract.column_names)
    values = ", ".join(f"s.{quote(c)}" for c in contract.column_names)
    updates = ", ".join(f"t.{quote(c)} = s.{quote(c)}" for c in contract.column_names)
    return (
        f"MERGE INTO {target} AS t\nUSING {quote(view)} AS s\n"
        f"ON t.{quote(RECORD_ID)} = s.{quote(RECORD_ID)}\n"
        f"WHEN MATCHED AND t.{quote(RECORD_HASH)} <> s.{quote(RECORD_HASH)} "
        f"THEN UPDATE SET {updates}\n"
        f"WHEN NOT MATCHED THEN INSERT ({columns}) VALUES ({values})\n"
        f"WHEN NOT MATCHED BY SOURCE THEN DELETE"
    )


def history_sql(catalog: str, schema: str, table: str) -> str:
    return f"DESCRIBE HISTORY {qualified(catalog, schema, table)} LIMIT 1"
