"""silver.document_chunks contract through real Spark types (local Spark, JVM only)."""

from datetime import UTC, datetime

import pytest
from tests.conftest import REPO_CONFIG_DIR
from tests.unit.test_retrieval_chunking import doc

from worldbank_copilot.intelligence.frames import profile_frame, validate_frame
from worldbank_copilot.lakehouse.records import LoadContext
from worldbank_copilot.retrieval.config import load_retrieval_settings
from worldbank_copilot.retrieval.corpus import corpus_dataset

pytestmark = pytest.mark.spark


def test_chunk_rows_load_with_the_contract_schema(frame):
    cfg = load_retrieval_settings(REPO_CONFIG_DIR).chunking
    ctx = LoadContext("snap", "run", datetime(2026, 9, 30, tzinfo=UTC), "v", {})
    dataset = corpus_dataset([doc()], cfg, ctx)
    rows = [{**r, "_loaded_at": r["_loaded_at"].isoformat()} for r in dataset.rows]
    df = frame(dataset.contract, rows)
    assert df.count() == len(dataset.rows)
    profile = profile_frame(df, dataset.contract)
    assert profile["distinct_record_ids"] == len(dataset.rows)
    problems = [p for p in validate_frame(df, dataset.contract) if "record_hash" not in p]
    assert problems == []  # Spark-side hash uses a different canonical form (Phase 7 frames)
    pages = df.select("page_numbers").first()[0]
    assert isinstance(pages, list) and all(isinstance(p, int) for p in pages)
