"""Model-isolation SQL and the probe schema on real (local) Spark; JVM-only SQL."""

import pytest

from worldbank_copilot.retrieval import pipeline as rp
from worldbank_copilot.retrieval.endpoint_probe import ProbeReport, StepResult, result_schema

pytestmark = pytest.mark.spark
QWEN, GTE = "databricks-qwen3-embedding-0-6b", "databricks-gte-large-en"


@pytest.fixture
def views(spark):
    spark.sql("""
        CREATE OR REPLACE TEMP VIEW chunks AS SELECT * FROM VALUES
          ('c1', 'P1', 'structure', 'RETRIEVAL', 'ISR', 'P1-d', CAST(1 AS BIGINT), 'h1', 't1'),
          ('c2', 'P1', 'structure', 'RETRIEVAL', 'ISR', 'P1-d', CAST(1 AS BIGINT), 'h2', 't2'),
          ('c3', 'P1', 'structure', 'PARENT',    'ISR', 'P1-d', CAST(1 AS BIGINT), 'h3', 'parent')
        AS t(chunk_id, project_id, chunk_strategy, chunk_role, document_type, document_id,
             isr_sequence, text_sha256, search_text)""")
    spark.sql(f"""
        CREATE OR REPLACE TEMP VIEW emb AS SELECT * FROM VALUES
          ('h1', '{GTE}',  1024, array_repeat(CAST(0.1 AS FLOAT), 1024)),
          ('h2', '{GTE}',  1024, array_repeat(CAST(0.1 AS FLOAT), 1024)),
          ('h1', '{QWEN}', 1024, array_repeat(CAST(0.2 AS FLOAT), 1024)),
          ('h2', '{QWEN}', 8,    array_repeat(CAST(0.2 AS FLOAT), 8))
        AS t(text_sha256, embedding_model, embedding_dimension, embedding)""")


def test_gte_vectors_never_satisfy_or_feed_a_qwen_build(spark, views):
    missing = {
        r["text_sha256"]
        for r in spark.sql(
            rp.missing_embeddings_sql("chunks", "emb", QWEN, ["structure"])
        ).collect()
    }
    assert missing == set()  # h1, h2 have Qwen rows (h2's is invalid: excluded below)
    gte_only = {
        r["text_sha256"]
        for r in spark.sql(
            rp.missing_embeddings_sql("chunks", "emb", "other-model", ["structure"])
        ).collect()
    }
    assert gte_only == {"h1", "h2"}  # GTE vectors do not count as cached for another model
    rows = spark.sql(rp.index_source_sql("chunks", "emb", QWEN, 1024, ["structure"])).collect()
    assert [(r["chunk_id"], r["embedding_model"], len(r["embedding"])) for r in rows] == [
        ("c1", QWEN, 1024)
    ]  # wrong-dimension Qwen vector and all GTE vectors excluded


def test_foreign_model_rows_are_counted(spark, views):
    spark.sql(
        f"CREATE OR REPLACE TEMP VIEW idx AS SELECT 'c1' AS chunk_id, '{GTE}' AS "
        "embedding_model, array_repeat(CAST(0.1 AS FLOAT), 1024) AS embedding"
    )
    assert spark.sql(rp.foreign_model_rows_sql("idx", QWEN, 1024)).first()["n"] == 1


def test_probe_rows_with_all_null_columns_build_a_typed_dataframe(spark):
    report = ProbeReport(
        QWEN,
        [
            StepResult("single input", 1, 200, None, None, 1826.0, 1, [1024], True),
            StepResult("batch 1", 4, 200, None, None, 212.0, 4, [1024] * 4, True),
        ],
    )
    frame = spark.createDataFrame(report.records(), schema=result_schema())
    types = {f.name: f.dataType.simpleString() for f in frame.schema.fields}
    assert types["error_code"] == "string" and types["retry_after"] == "double"
    assert types["error"] == "string" and types["dimensions"] == "array<bigint>"
