"""Phase 6: Unity Catalog object handling and MERGE metrics of SparkDeltaStore.

These tests use a FAKE Spark session that only records SQL. They prove the control
flow (validate before create, never create the catalog, clear permission errors,
honest idempotency metrics). They do NOT prove anything about a real Databricks
workspace; that is validated by running notebooks/05_platformize_databricks.py.
"""

import pytest

from worldbank_copilot.common.exceptions import LakehouseError
from worldbank_copilot.lakehouse.spark_store import SparkDeltaStore, merge_result


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def collect(self):
        return self.rows


class FakeSpark:
    def __init__(self, existing=(), fail_on=None):
        self.existing = set(existing)
        self.fail_on = fail_on
        self.statements = []

    def sql(self, statement):
        self.statements.append(statement)
        if self.fail_on and statement.startswith(self.fail_on):
            raise PermissionError("PERMISSION_DENIED: User does not have CREATE SCHEMA")
        for name in self.existing:
            if (
                statement.startswith(("SHOW CATALOGS", "SHOW SCHEMAS", "SHOW VOLUMES"))
                and f"LIKE '{name}'" in statement
            ):
                return _Result([{"name": name}])
        return _Result([])


def _store(spark):
    return SparkDeltaStore(spark, "worldbank_ai", {"bronze": "bronze", "silver": "silver"})


def test_missing_catalog_is_never_created():
    spark = FakeSpark()
    with pytest.raises(LakehouseError, match="never creates or substitutes"):
        _store(spark).validate_catalog()
    assert not any(s.startswith("CREATE") for s in spark.statements)


def test_existing_objects_are_reused_and_missing_ones_created():
    spark = FakeSpark(existing={"worldbank_ai", "bronze", "sources"})
    store = _store(spark)
    assert store.validate_catalog().status == "EXISTS"
    assert store.ensure_schema("bronze", "c").status == "EXISTS"
    assert store.ensure_volume("bronze", "sources", "c").status == "EXISTS"
    created = store.ensure_schema("silver", "c")
    assert (created.name, created.status) == ("worldbank_ai.silver", "CREATED")
    creates = [s for s in spark.statements if s.startswith("CREATE")]
    assert creates == ["CREATE SCHEMA IF NOT EXISTS `worldbank_ai`.`silver` COMMENT 'c'"]


def test_permission_failure_names_object_and_operation():
    spark = FakeSpark(existing={"worldbank_ai"}, fail_on="CREATE SCHEMA")
    with pytest.raises(LakehouseError, match=r"CREATE SCHEMA on worldbank_ai\.silver failed"):
        _store(spark).ensure_schema("silver", "c")


def test_merge_metrics_first_load_and_idempotent_rerun():
    first = merge_result(
        "t",
        35,
        None,
        {
            "version": 1,
            "operation": "MERGE",
            "operationMetrics": {
                "numTargetRowsInserted": "35",
                "numTargetRowsUpdated": "0",
                "numTargetRowsDeleted": "0",
            },
        },
    )
    assert (first.inserted, first.updated, first.deleted) == (35, 0, 0)
    # A rerun that commits nothing leaves the old history entry: report zero changes.
    rerun = merge_result(
        "t",
        35,
        1,
        {"version": 1, "operation": "MERGE", "operationMetrics": {"numTargetRowsInserted": "35"}},
    )
    assert (rerun.inserted, rerun.updated, rerun.deleted) == (0, 0, 0)
    empty_commit = merge_result(
        "t",
        35,
        1,
        {"version": 2, "operation": "MERGE", "operationMetrics": {"numTargetRowsInserted": "0"}},
    )
    assert empty_commit.inserted == 0 and empty_commit.delta_version == 2
