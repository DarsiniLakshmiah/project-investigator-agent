"""Databricks Vector Search adapter (Phase 8).

Separation of concerns:

    silver.document_chunks  (governed corpus, text + provenance)
        -> silver.document_chunk_index  (index source: chunk_id, filter columns, vector)
        -> Vector Search Delta Sync index (self-managed embeddings)
        -> DenseSearcher.search(vector, filters, k) -> [(chunk_id, score)]

Downstream code never sees raw Vector Search responses: results are chunk ids and
scores, and all metadata/text comes back from the governed corpus. The client is created
lazily, so importing needs no credentials.

Client library: ``databricks-ai-search`` (``databricks.ai_search.client.AISearchClient``),
the successor of the deprecated ``databricks-vectorsearch``. The old package pins
protobuf<6, which downgraded the runtime's protobuf 6.33.5 (Phase 8 dependency fix).
The endpoint / Delta Sync index / similarity_search API used here is unchanged.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from worldbank_copilot.common.exceptions import LakehouseError
from worldbank_copilot.retrieval.config import VectorSearchConfig

INDEX_COLUMNS = (
    "chunk_id",
    "project_id",
    "chunk_strategy",
    "document_type",
    "document_id",
    "isr_sequence",
)
PRIMARY_KEY = "chunk_id"
VECTOR_COLUMN = "embedding"


class VectorSearchUnavailable(LakehouseError):
    """Vector Search cannot be used in this workspace (capability / permission)."""


def is_not_found(exc: BaseException) -> bool:
    """The AI Search SDK's NotFound (alias ResourceDoesNotExist), or an HTTP 404."""
    names = {type(exc).__name__} | {c.__name__ for c in type(exc).__mro__}
    return bool(names & {"NotFound", "ResourceDoesNotExist"}) or (
        getattr(exc, "status_code", None) == 404
    )


def is_quota_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(word in text for word in ("quota", "exceeded", "limit reached", "maximum number"))


class DenseSearcher(Protocol):
    def search(
        self, vector: Sequence[float], filters: dict[str, Any], k: int
    ) -> list[tuple[str, float]]: ...


@dataclass
class IndexStatus:
    endpoint: str
    endpoint_state: str
    endpoint_action: str  # EXISTS | CREATED
    index: str
    index_action: str  # EXISTS | CREATED
    ready: bool
    detailed_state: str | None
    indexed_rows: int | None
    source_table: str


def parse_results(response: dict[str, Any]) -> list[tuple[str, float]]:
    """similarity_search response -> [(chunk_id, score)] (score is the last column)."""
    columns = [c["name"] for c in response.get("manifest", {}).get("columns", [])]
    rows = response.get("result", {}).get("data_array") or []
    if not rows:
        return []
    if PRIMARY_KEY not in columns:
        raise LakehouseError(f"Vector Search response lacks {PRIMARY_KEY}: {columns}")
    key = columns.index(PRIMARY_KEY)
    return [(str(row[key]), float(row[-1])) for row in rows]


class VectorSearchIndex:
    """Endpoint + Delta Sync index lifecycle and queries."""

    def __init__(
        self,
        config: VectorSearchConfig,
        index_name: str,
        source_table: str,
        dimension: int,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.config = config
        self.index_name = index_name  # catalog.schema.index
        self.source_table = source_table  # catalog.schema.table
        self.dimension = dimension
        self._client = client
        self._sleep = sleep
        self._index: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from databricks.ai_search.client import AISearchClient
            except ImportError as exc:
                raise VectorSearchUnavailable(
                    "databricks-ai-search is not installed (requirements-retrieval.txt)"
                ) from exc
            self._client = AISearchClient(disable_notice=True)
        return self._client

    # -- lifecycle ---------------------------------------------------------------

    def _endpoint_state(self) -> str | None:
        try:
            endpoint = self.client.get_endpoint(self.config.endpoint)
        except Exception as exc:
            if is_not_found(exc):
                return None
            raise VectorSearchUnavailable(
                f"cannot read Vector Search endpoint {self.config.endpoint!r}: {exc}"
            ) from exc
        return (endpoint.get("endpoint_status") or {}).get("state", "UNKNOWN")

    def endpoint_state(self) -> str | None:
        """State of the configured endpoint (None if it does not exist). Read-only."""
        return self._endpoint_state()

    def ensure_endpoint(self) -> tuple[str, str]:
        """Validate (and only if ``create_endpoint`` is true, create) the endpoint."""
        state = self._endpoint_state()
        action = "EXISTS"
        if state is None and not self.config.create_endpoint:
            raise VectorSearchUnavailable(
                f"AI Search endpoint {self.config.endpoint!r} does not exist and "
                "create_endpoint is false (configs/retrieval/retrieval.yaml); no endpoint "
                "was created or modified"
            )
        if state is None:
            try:
                self.client.create_endpoint(name=self.config.endpoint, endpoint_type="STANDARD")
            except Exception as exc:
                raise VectorSearchUnavailable(
                    f"cannot create Vector Search endpoint {self.config.endpoint!r} "
                    f"(requires Vector Search to be enabled and permission to create "
                    f"endpoints): {exc}"
                ) from exc
            action = "CREATED"
        deadline = time.monotonic() + self.config.sync_timeout_seconds
        while True:
            state = self._endpoint_state() or "PROVISIONING"
            if state == "ONLINE":
                return state, action
            if "FAIL" in state or time.monotonic() > deadline:
                raise VectorSearchUnavailable(f"endpoint {self.config.endpoint}: state {state}")
            self._sleep(20)

    def _get_index(self) -> Any | None:
        try:
            return self.client.get_index(
                endpoint_name=self.config.endpoint, index_name=self.index_name
            )
        except Exception as exc:
            if is_not_found(exc):
                return None
            raise

    def list_indexes(self) -> list[dict[str, Any]]:
        """Indexes on the configured endpoint (read-only)."""
        response = self.client.list_indexes(self.config.endpoint) or {}
        return list(response.get("vector_indexes", []) or [])

    def ensure_index(self) -> str:
        index = self._get_index()
        if index is not None:
            spec = index.describe().get("delta_sync_index_spec", {})
            source = spec.get("source_table")
            if source and source != self.source_table:
                raise LakehouseError(
                    f"index {self.index_name} syncs {source}, expected {self.source_table}; "
                    "refusing to reuse it"
                )
            dims = {
                c.get("embedding_dimension")
                for c in spec.get("embedding_vector_columns", [])
                if c.get("name") == VECTOR_COLUMN
            }
            if dims and dims != {self.dimension}:
                raise LakehouseError(
                    f"index {self.index_name} has embedding dimension {sorted(dims)}, the "
                    f"configured embedding model has {self.dimension}; refusing to reuse it"
                )
            self._index = index
            return "EXISTS"
        try:
            self._index = self.client.create_delta_sync_index(
                endpoint_name=self.config.endpoint,
                source_table_name=self.source_table,
                index_name=self.index_name,
                pipeline_type=self.config.pipeline_type,
                primary_key=PRIMARY_KEY,
                embedding_dimension=self.dimension,
                embedding_vector_column=VECTOR_COLUMN,
            )
        except Exception as exc:
            if is_quota_error(exc):
                raise VectorSearchUnavailable(
                    f"index {self.index_name} was not created: quota reached on endpoint "
                    f"{self.config.endpoint!r} ({exc}). No existing index was modified or "
                    "deleted; free capacity or choose another endpoint deliberately."
                ) from exc
            raise
        return "CREATED"

    @property
    def index(self) -> Any:
        if self._index is None:
            self._index = self._get_index()
            if self._index is None:
                raise LakehouseError(f"Vector Search index {self.index_name} does not exist")
        return self._index

    def describe(self) -> dict[str, Any]:
        return self.index.describe()

    def sync_and_wait(self, expected_rows: int) -> dict[str, Any]:
        """Trigger a sync (TRIGGERED pipelines) and wait until the index is ready.

        Ready means: status.ready, detailed state ONLINE / ONLINE_NO_PENDING_UPDATE and
        indexed_row_count == rows of the source table.
        """
        info = self.describe()
        if self.config.pipeline_type == "TRIGGERED" and info.get("status", {}).get("ready"):
            self.index.sync()
            self._sleep(15)
        deadline = time.monotonic() + self.config.sync_timeout_seconds
        while True:
            info = self.describe()
            status = info.get("status", {})
            state = status.get("detailed_state", "") or ""
            if "FAILED" in state:
                raise LakehouseError(f"index {self.index_name}: {state} {status.get('message')}")
            done = (
                status.get("ready")
                and state in ("ONLINE", "ONLINE_NO_PENDING_UPDATE")
                and status.get("indexed_row_count") == expected_rows
            )
            if done:
                return info
            if time.monotonic() > deadline:
                raise LakehouseError(
                    f"index {self.index_name} not ready after "
                    f"{self.config.sync_timeout_seconds}s: {state}, "
                    f"{status.get('indexed_row_count')} of {expected_rows} rows"
                )
            self._sleep(20)

    def status(self, endpoint_state: str, endpoint_action: str, index_action: str) -> IndexStatus:
        status = self.describe().get("status", {})
        return IndexStatus(
            self.config.endpoint,
            endpoint_state,
            endpoint_action,
            self.index_name,
            index_action,
            bool(status.get("ready")),
            status.get("detailed_state"),
            status.get("indexed_row_count"),
            self.source_table,
        )

    # -- queries -------------------------------------------------------------------

    def search(
        self, vector: Sequence[float], filters: dict[str, Any], k: int
    ) -> list[tuple[str, float]]:
        if "project_id" not in filters:
            raise LakehouseError("Vector Search queries must be filtered by project_id")
        response = self.index.similarity_search(
            query_vector=list(vector),
            columns=list(INDEX_COLUMNS),
            filters=filters,
            num_results=k,
        )
        return parse_results(response)
