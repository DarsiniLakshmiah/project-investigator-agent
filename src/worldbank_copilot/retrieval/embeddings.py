"""Embedding providers (Phase 8).

Retrieval depends only on ``EmbeddingProvider``; the implementation is chosen by
configs/retrieval/embeddings.yaml. Production: a Databricks Model Serving embedding
endpoint (Foundation Model API) called through the Databricks SDK. The client is
created lazily, so importing this module needs no Databricks credentials.

Vectors are cached in silver.chunk_embeddings keyed by (text_sha256, model): unchanged
chunk text is never re-embedded (see ``pipeline.update_embedding_cache``).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from worldbank_copilot.common.exceptions import CopilotError
from worldbank_copilot.retrieval.config import EmbeddingConfig


class EmbeddingError(CopilotError):
    """The embedding service failed or returned an unexpected result."""


class EmbeddingProvider(Protocol):
    model: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def batched(items: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


class DatabricksServingEmbeddings:
    """Embeddings from a Databricks Model Serving endpoint (e.g. databricks-gte-large-en)."""

    def __init__(
        self,
        config: EmbeddingConfig,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.config = config
        self.model = config.endpoint
        self.dimension = config.expected_dimension
        self._client = client
        self._sleep = sleep

    @property
    def client(self) -> Any:
        if self._client is None:
            from databricks.sdk import WorkspaceClient

            self._client = WorkspaceClient()
        return self._client

    def _query(self, texts: Sequence[str]) -> list[list[float]]:
        response = self.client.serving_endpoints.query(name=self.model, input=list(texts))
        data = getattr(response, "data", None) or []
        vectors = [list(getattr(item, "embedding", None) or []) for item in data]
        if len(vectors) != len(texts):
            raise EmbeddingError(f"{self.model}: {len(vectors)} vectors for {len(texts)} texts")
        bad = [i for i, v in enumerate(vectors) if len(v) != self.dimension]
        if bad:
            raise EmbeddingError(
                f"{self.model}: expected dimension {self.dimension}, got "
                f"{len(vectors[bad[0]])} (configs/retrieval/embeddings.yaml)"
            )
        return vectors

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for batch in batched(list(texts), self.config.batch_size):
            for attempt in range(self.config.max_retries + 1):
                try:
                    out.extend(self._query(batch))
                    break
                except EmbeddingError:
                    raise
                except Exception as exc:  # transient service / rate-limit errors
                    if attempt == self.config.max_retries:
                        raise EmbeddingError(f"{self.model}: {exc}") from exc
                    self._sleep(min(2**attempt, 30))
        return out


def embedding_provider(config: EmbeddingConfig, client: Any | None = None) -> EmbeddingProvider:
    if config.provider == "databricks_serving":
        return DatabricksServingEmbeddings(config, client=client)
    raise EmbeddingError(f"unknown embedding provider {config.provider!r}")
