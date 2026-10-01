"""Model identity: GTE and Qwen vectors can never be mixed (cache, index source, index)."""

import pytest
from tests.conftest import REPO_CONFIG_DIR

from worldbank_copilot.common import load_settings
from worldbank_copilot.common.exceptions import LakehouseError, ReconciliationError
from worldbank_copilot.retrieval import pipeline as rp
from worldbank_copilot.retrieval.config import load_retrieval_settings
from worldbank_copilot.retrieval.endpoint_probe import RESULT_COLUMNS, ProbeReport, StepResult
from worldbank_copilot.retrieval.vector_search import VectorSearchIndex

RS = load_retrieval_settings(REPO_CONFIG_DIR)
QWEN, GTE = "databricks-qwen3-embedding-0-6b", "databricks-gte-large-en"


def test_missing_texts_are_computed_per_model():
    qwen = rp.missing_embeddings_sql("c", "e", QWEN, ["structure"])
    gte = rp.missing_embeddings_sql("c", "e", GTE, ["structure"])
    assert f"e.embedding_model = '{QWEN}'" in qwen and GTE not in qwen
    assert f"e.embedding_model = '{GTE}'" in gte
    assert "e.text_sha256 = c.text_sha256" in qwen  # cache key: text hash + model


def test_index_source_takes_exactly_one_model_and_dimension():
    sql = rp.index_source_sql("c", "e", QWEN, 1024, ["fixed", "structure"])
    assert f"e.embedding_model = '{QWEN}'" in sql
    assert "e.embedding_dimension = 1024" in sql and "size(e.embedding) = 1024" in sql
    check = rp.foreign_model_rows_sql("idx", QWEN, 1024)
    assert f"embedding_model <> '{QWEN}'" in check and "size(embedding) <> 1024" in check


def test_identifiers_are_validated():
    with pytest.raises(ValueError):
        rp.missing_embeddings_sql("c", "e", "x' OR '1'='1", ["structure"])
    with pytest.raises(ValueError):
        rp.index_source_sql("c", "e", QWEN, 1024, ["structure'; DROP TABLE x; --"])


def test_pipeline_refuses_a_provider_or_model_other_than_the_configured_one():
    settings = load_settings("databricks", config_dir=REPO_CONFIG_DIR, env={})

    class Provider:
        model, dimension = GTE, 1024

    with pytest.raises(ReconciliationError, match="does not match"):
        rp.update_embedding_cache(object(), settings, RS, Provider(), ["structure"])
    with pytest.raises(ReconciliationError, match="configured"):
        rp.build_index_source(object(), settings, RS, GTE, ["structure"])


class Index:
    def __init__(self, dimension):
        self.dimension = dimension

    def describe(self):
        return {
            "delta_sync_index_spec": {
                "source_table": "c.s.t",
                "embedding_vector_columns": [
                    {"name": "embedding", "embedding_dimension": self.dimension}
                ],
            }
        }


class Client:
    def __init__(self, index):
        self.index = index

    def get_index(self, endpoint_name, index_name):
        return self.index


def test_existing_index_with_another_dimension_is_refused():
    index = VectorSearchIndex(
        RS.retrieval.vector_search, "c.s.i", "c.s.t", 1024, client=Client(Index(768))
    )
    with pytest.raises(LakehouseError, match="dimension"):
        index.ensure_index()
    same = VectorSearchIndex(
        RS.retrieval.vector_search, "c.s.i", "c.s.t", 1024, client=Client(Index(1024))
    )
    assert same.ensure_index() == "EXISTS"


def test_probe_records_follow_the_explicit_schema_order():
    report = ProbeReport(
        QWEN, [StepResult("single input", 1, 200, None, None, 1826.0, 1, [1024], True)]
    )
    record = report.records()[0]
    assert len(record) == len(RESULT_COLUMNS)
    named = dict(zip([n for n, _ in RESULT_COLUMNS], record, strict=True))
    assert named["error_code"] is None and named["retry_after"] is None and named["error"] is None
    assert named["dimensions"] == [1024] and named["latency_ms"] == 1826.0
