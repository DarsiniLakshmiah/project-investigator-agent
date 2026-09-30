"""Local Spark session for Gold transformation tests (marker: spark).

Runs real Spark (JVM) locally; the Gold code uses only native Spark functions, so no
Python workers are needed. Skipped when pyspark or a JDK is unavailable. Run with the
Spark dev environment (README "Testing Gold transformations"):

    .venv-spark\\Scripts\\python -m pytest -m spark
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _java_home() -> str | None:
    if os.environ.get("JAVA_HOME"):
        return os.environ["JAVA_HOME"]
    local = sorted((REPO / ".tools").glob("jdk-*"))
    return str(local[-1]) if local else None


@pytest.fixture(scope="session")
def spark():
    pyspark = pytest.importorskip("pyspark")
    java = _java_home()
    if java is None:
        pytest.skip("no JDK (set JAVA_HOME or unpack one into .tools/)")
    os.environ.setdefault("JAVA_HOME", java)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")
    session = (
        pyspark.sql.SparkSession.builder.master("local[2]")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "localhost")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


_DEC = "__decimal__"


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime | date):
            return o.isoformat()
        if isinstance(o, Decimal):
            return _DEC + format(o, "f")  # replaced by a raw JSON number (exact text)
        return super().default(o)


def to_json_line(row: dict) -> str:
    return re.sub('"' + _DEC + r'(-?[0-9.]+)"', lambda m: m.group(1), json.dumps(row, cls=_Encoder))


@pytest.fixture
def frame(spark, tmp_path):
    """Build a DataFrame with a contract's exact schema from partial row dicts.

    Rows are written as JSON Lines and read back with the contract schema (JVM only).
    Unspecified columns are NULL; Decimals are written as exact JSON numbers.
    """
    from worldbank_copilot.lakehouse.spark_store import spark_schema

    counter = {"n": 0}

    def make(contract, rows):
        counter["n"] += 1
        path = tmp_path / f"{contract.name}_{counter['n']}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(to_json_line({c: row.get(c) for c in contract.column_names}) + "\n")
        return (
            spark.read.schema(spark_schema(contract))
            .option("mode", "FAILFAST")
            .json(str(path))
            .select(*contract.column_names)
        )

    return make
