"""Phase 8 orchestration on Databricks (notebook 07 calls these functions).

    Phase 4 parsed documents (artifact Volume)          -- no re-parsing
      -> chunking (all configured strategies)
      -> silver.document_chunks          (contract, snapshot MERGE, read-back reconcile)
      -> silver.chunk_embeddings         (cache keyed by text hash + model; only new text
                                          is embedded)
      -> silver.document_chunk_index     (RETRIEVAL chunks + vectors, Change Data Feed)
      -> Vector Search Delta Sync index  (self-managed vectors, filtered by project)
      -> retrieval experiments + production retriever

Driver-side steps are bounded and documented: chunking reads the 53 parsed documents
(a few MB of JSON); only chunk texts that are NOT yet in the embedding cache are sent to
the embedding endpoint; the governed corpus is loaded once for BM25 and evidence
metadata. Everything else runs as Spark SQL / Delta MERGE.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from worldbank_copilot.common.config import Settings
from worldbank_copilot.common.exceptions import ReconciliationError
from worldbank_copilot.common.project_registry import ProjectRegistry
from worldbank_copilot.lakehouse.contracts import require_valid
from worldbank_copilot.lakehouse.pipeline import load_context
from worldbank_copilot.lakehouse.reconcile import compare, profile
from worldbank_copilot.lakehouse.sources import load_snapshot
from worldbank_copilot.lakehouse.spark_store import SparkDeltaStore, WriteResult, merge_result
from worldbank_copilot.lakehouse.sql import history_sql, qualified
from worldbank_copilot.parsing.pipeline import run_parsing
from worldbank_copilot.retrieval.config import RetrievalSettings
from worldbank_copilot.retrieval.corpus import corpus_dataset
from worldbank_copilot.retrieval.embeddings import EmbeddingProvider
from worldbank_copilot.retrieval.vector_search import VectorSearchIndex

EMBEDDINGS_TABLE = "chunk_embeddings"
Say = Callable[[str], None]


def _say(progress: Say | None) -> Say:
    return progress or (lambda _m: None)


def names(settings: Settings, rs: RetrievalSettings) -> dict[str, str]:
    catalog = settings.require("databricks.catalog")
    silver = settings.require("databricks.silver_schema")
    vs = rs.retrieval.vector_search
    return {
        "catalog": catalog,
        "silver": silver,
        "chunks": qualified(catalog, silver, "document_chunks"),
        "embeddings": qualified(catalog, silver, EMBEDDINGS_TABLE),
        "index_table": qualified(catalog, silver, vs.index_table),
        # Vector Search API uses plain dotted names (no backticks).
        "index_table_plain": f"{catalog}.{silver}.{vs.index_table}",
        "index_name": f"{catalog}.{silver}.{vs.index_name}",
    }


# ---------------------------------------------------------------------------
# Capability probe (runs first; nothing is built when a blocking check fails)
# ---------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    status: str  # OK | FAILED | UNAVAILABLE
    blocking: bool
    detail: str


@dataclass
class CapabilityReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def blocking_failures(self) -> list[Check]:
        return [c for c in self.checks if c.blocking and c.status != "OK"]

    def format(self) -> str:
        return "\n".join(
            f"  [{c.status:<11}] {'BLOCKING' if c.blocking else 'optional'} {c.name}: {c.detail}"
            for c in self.checks
        )


def probe_capabilities(
    rs: RetrievalSettings, provider: EmbeddingProvider, check_reranker: bool = True
) -> CapabilityReport:
    report = CapabilityReport()
    try:
        from databricks.vector_search.client import VectorSearchClient

        client = VectorSearchClient(disable_notice=True)
        endpoints = client.list_endpoints().get("endpoints", []) or []
        found = [e for e in endpoints if e.get("name") == rs.retrieval.vector_search.endpoint]
        state = (found[0].get("endpoint_status") or {}).get("state") if found else "absent"
        report.checks.append(
            Check(
                "vector_search",
                "OK",
                True,
                f"API reachable; {len(endpoints)} endpoint(s) visible; configured endpoint "
                f"{rs.retrieval.vector_search.endpoint!r}: {state}",
            )
        )
    except ImportError:
        report.checks.append(
            Check("vector_search", "UNAVAILABLE", True, "databricks-vectorsearch not installed")
        )
    except Exception as exc:  # permission / feature not enabled
        report.checks.append(
            Check("vector_search", "FAILED", True, f"Vector Search API not usable: {exc}")
        )
    try:
        started = time.perf_counter()
        vector = provider.embed(["Implementation Status and Results Report probe"])[0]
        ms = (time.perf_counter() - started) * 1000
        report.checks.append(
            Check(
                "embedding_endpoint",
                "OK",
                True,
                f"{provider.model}: dimension {len(vector)}, single-call latency {ms:.0f} ms",
            )
        )
    except Exception as exc:
        report.checks.append(
            Check(
                "embedding_endpoint",
                "FAILED",
                True,
                f"{provider.model}: {exc} (a Model Serving embedding endpoint is required; "
                "set configs/retrieval/embeddings.yaml endpoint)",
            )
        )
    if check_reranker:
        try:
            from sentence_transformers import CrossEncoder

            CrossEncoder(rs.retrieval.reranker.model, max_length=rs.retrieval.reranker.max_length)
            report.checks.append(Check("cross_encoder", "OK", False, rs.retrieval.reranker.model))
        except Exception as exc:
            report.checks.append(
                Check(
                    "cross_encoder",
                    "UNAVAILABLE",
                    False,
                    f"{exc}; cross-encoder experiments will be reported as UNAVAILABLE",
                )
            )
    try:
        import mlflow  # noqa: F401

        report.checks.append(Check("mlflow", "OK", False, "experiment logging available"))
    except ImportError:
        report.checks.append(Check("mlflow", "UNAVAILABLE", False, "mlflow not installed"))
    return report


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


@dataclass
class CorpusReport:
    documents: int
    failed_documents: list[str]
    write: WriteResult | None
    profile: dict[str, Any]
    counts: dict[str, dict[str, int]]
    readback_diffs: list[str]


def build_corpus(
    spark: Any,
    settings: Settings,
    rs: RetrievalSettings,
    registry: ProjectRegistry,
    progress: Say | None = None,
) -> CorpusReport:
    say = _say(progress)
    run = run_parsing(settings, registry, None)  # loads the Phase 4 parsed cache only
    missing = [f"{o.project_id}/{o.filename}" for o in run.outcomes if o.action == "missing"]
    if missing:
        raise ReconciliationError(f"parsed documents missing (run Phase 4/6 first): {missing}")
    failed = [d.document_id for d in run.documents if d.parse_status.value != "SUCCESS"]
    ctx = load_context(load_snapshot(settings.config_dir))
    dataset = corpus_dataset(run.documents, rs.chunking, ctx)
    require_valid(dataset.contract, dataset.rows)
    say(f"chunked {len(run.documents)} documents into {len(dataset.rows)} chunks")
    expected = profile(dataset.contract, dataset.rows)
    n = names(settings, rs)
    store = SparkDeltaStore(spark, n["catalog"], {"silver": n["silver"]})
    store.ensure_table(dataset.contract)
    write = store.merge_frame(dataset.contract, _frame(spark, dataset), len(dataset.rows))
    diffs = compare(expected, profile(dataset.contract, store.read_rows(dataset.contract)))
    if diffs:
        raise ReconciliationError(f"silver.document_chunks read-back differs: {diffs}")
    counts: dict[str, Counter] = {
        k: Counter() for k in ("strategy_role", "strategy_type", "project", "document_type")
    }
    for row in dataset.rows:
        s = row["chunk_strategy"]
        counts["strategy_role"][f"{s}|{row['chunk_role']}"] += 1
        if row["chunk_role"] == "RETRIEVAL":
            counts["strategy_type"][f"{s}|{row['chunk_type']}"] += 1
            counts["project"][f"{s}|{row['project_id']}"] += 1
            counts["document_type"][f"{s}|{row['document_type']}"] += 1
    return CorpusReport(
        len(run.documents),
        failed,
        write,
        expected,
        {k: dict(sorted(v.items())) for k, v in counts.items()},
        diffs,
    )


def _frame(spark: Any, dataset: Any) -> Any:
    from worldbank_copilot.lakehouse.spark_store import spark_schema

    names_ = dataset.contract.column_names
    return spark.createDataFrame(
        [tuple(r[n] for n in names_) for r in dataset.rows],
        schema=spark_schema(dataset.contract),
        verifySchema=True,
    )


# ---------------------------------------------------------------------------
# Embedding cache and index source
# ---------------------------------------------------------------------------


def _history(spark: Any, catalog: str, schema: str, table: str) -> dict[str, Any]:
    rows = spark.sql(history_sql(catalog, schema, table)).collect()
    return rows[0].asDict(recursive=True) if rows else {}


@dataclass
class EmbeddingReport:
    model: str
    texts_needed: int
    already_cached: int
    embedded_now: int
    seconds: float


def update_embedding_cache(
    spark: Any,
    settings: Settings,
    rs: RetrievalSettings,
    provider: EmbeddingProvider,
    strategies: Sequence[str],
    progress: Say | None = None,
) -> EmbeddingReport:
    """Embed only chunk texts that are not cached for this model (idempotent)."""
    say = _say(progress)
    n = names(settings, rs)
    spark.sql(
        f"CREATE TABLE IF NOT EXISTS {n['embeddings']} ("
        "text_sha256 STRING NOT NULL, embedding_model STRING NOT NULL, "
        "embedding_dimension INT NOT NULL, embedding ARRAY<FLOAT> NOT NULL, "
        "embedded_at TIMESTAMP NOT NULL) USING DELTA "
        "COMMENT 'Embedding cache: one vector per (search_text sha256, model). Phase 8.'"
    )
    in_list = ", ".join(f"'{s}'" for s in strategies)
    needed = spark.sql(
        f"SELECT DISTINCT text_sha256, search_text FROM {n['chunks']} "
        f"WHERE chunk_role = 'RETRIEVAL' AND chunk_strategy IN ({in_list})"
    )
    cached = spark.table(n["embeddings"]).where(f"embedding_model = '{provider.model}'")
    missing = needed.join(cached.select("text_sha256"), "text_sha256", "left_anti")
    total, rows = needed.count(), missing.collect()  # only texts not yet embedded
    say(f"embeddings: {total} distinct texts, {total - len(rows)} cached, {len(rows)} to embed")
    started = time.perf_counter()
    if rows:
        from pyspark.sql import types as T

        now = datetime.now(UTC)
        batch = max(rs.embeddings.batch_size * 10, 500)
        for i in range(0, len(rows), batch):
            part = rows[i : i + batch]
            vectors = provider.embed([r["search_text"] for r in part])
            schema = T.StructType(
                [
                    T.StructField("text_sha256", T.StringType(), False),
                    T.StructField("embedding_model", T.StringType(), False),
                    T.StructField("embedding_dimension", T.IntegerType(), False),
                    T.StructField("embedding", T.ArrayType(T.FloatType(), False), False),
                    T.StructField("embedded_at", T.TimestampType(), False),
                ]
            )
            frame = spark.createDataFrame(
                [
                    (r["text_sha256"], provider.model, len(v), [float(x) for x in v], now)
                    for r, v in zip(part, vectors, strict=True)
                ],
                schema,
            )
            frame.createOrReplaceTempView("_wbc_new_embeddings")
            spark.sql(
                f"MERGE INTO {n['embeddings']} t USING _wbc_new_embeddings s "
                "ON t.text_sha256 = s.text_sha256 AND t.embedding_model = s.embedding_model "
                "WHEN NOT MATCHED THEN INSERT *"
            )
            say(f"  embedded {min(i + batch, len(rows))}/{len(rows)}")
    return EmbeddingReport(
        provider.model, total, total - len(rows), len(rows), round(time.perf_counter() - started, 1)
    )


def build_index_source(
    spark: Any, settings: Settings, rs: RetrievalSettings, model: str, strategies: Sequence[str]
) -> tuple[WriteResult, int]:
    """MERGE RETRIEVAL chunks + cached vectors into the Vector Search source table."""
    n = names(settings, rs)
    spark.sql(
        f"CREATE TABLE IF NOT EXISTS {n['index_table']} ("
        "chunk_id STRING NOT NULL, project_id STRING NOT NULL, chunk_strategy STRING NOT NULL, "
        "document_type STRING, document_id STRING NOT NULL, isr_sequence BIGINT, "
        "text_sha256 STRING NOT NULL, embedding_model STRING NOT NULL, "
        "embedding ARRAY<FLOAT> NOT NULL) USING DELTA "
        "COMMENT 'Vector Search source (Phase 8): RETRIEVAL chunks of silver.document_chunks "
        "with vectors from silver.chunk_embeddings. Text and provenance stay in the corpus.' "
        "TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )
    in_list = ", ".join(f"'{s}'" for s in strategies)
    source = (
        f"SELECT c.chunk_id, c.project_id, c.chunk_strategy, c.document_type, c.document_id, "
        f"c.isr_sequence, c.text_sha256, e.embedding_model, e.embedding "
        f"FROM {n['chunks']} c JOIN {n['embeddings']} e "
        f"ON c.text_sha256 = e.text_sha256 AND e.embedding_model = '{model}' "
        f"WHERE c.chunk_role = 'RETRIEVAL' AND c.chunk_strategy IN ({in_list})"
    )
    expected = spark.sql(
        f"SELECT count(*) AS n FROM {n['chunks']} WHERE chunk_role = 'RETRIEVAL' "
        f"AND chunk_strategy IN ({in_list})"
    ).collect()[0]["n"]
    available = spark.sql(f"SELECT count(*) AS n FROM ({source})").collect()[0]["n"]
    if available != expected:
        raise ReconciliationError(
            f"{expected - available} retrieval chunks have no cached embedding for {model}"
        )
    before = _history(spark, n["catalog"], n["silver"], rs.retrieval.vector_search.index_table)
    changed = " OR ".join(
        f"NOT (t.{c} <=> s.{c})"
        for c in (
            "project_id",
            "chunk_strategy",
            "document_type",
            "document_id",
            "isr_sequence",
            "text_sha256",
            "embedding_model",
        )
    )
    spark.sql(
        f"MERGE INTO {n['index_table']} t USING ({source}) s ON t.chunk_id = s.chunk_id "
        f"WHEN MATCHED AND ({changed}) THEN UPDATE SET * "
        "WHEN NOT MATCHED THEN INSERT * "
        "WHEN NOT MATCHED BY SOURCE THEN DELETE"
    )
    entry = _history(spark, n["catalog"], n["silver"], rs.retrieval.vector_search.index_table)
    return merge_result(n["index_table"], expected, before.get("version"), entry), expected


def vector_index(
    settings: Settings, rs: RetrievalSettings, client: Any | None = None
) -> VectorSearchIndex:
    n = names(settings, rs)
    return VectorSearchIndex(
        rs.retrieval.vector_search,
        n["index_name"],
        n["index_table_plain"],
        rs.embeddings.expected_dimension,
        client=client,
    )


def ensure_vector_index(index: VectorSearchIndex, expected_rows: int, progress: Say | None = None):
    say = _say(progress)
    state, endpoint_action = index.ensure_endpoint()
    say(f"Vector Search endpoint {index.config.endpoint}: {state} ({endpoint_action})")
    index_action = index.ensure_index()
    say(f"index {index.index_name}: {index_action}; syncing {expected_rows} rows")
    index.sync_and_wait(expected_rows)
    return index.status(state, endpoint_action, index_action)
