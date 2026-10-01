"""Embeddings from the Databricks Model Serving endpoint (Phase 8).

Retrieval depends only on ``EmbeddingProvider.embed(texts)``; the endpoint is
configured in configs/retrieval/embeddings.yaml (databricks-gte-large-en).

Why an explicit transport. The Databricks SDK's ``serving_endpoints.query`` wraps every
call in its own retry loop (``retry_timeout_seconds``, default 300 s) and per-attempt
HTTP timeout (``http_timeout_seconds``, default 60 s): a throttled or slow request is
silently re-sent until "Timed out after 0:05:00", and the caller cannot tell a 429 from
a timeout. Here one HTTP POST is one attempt with our own timeout, and this module owns
the single, bounded retry policy:

* retried: request timeout, connection error, HTTP 429 and 5xx (exponential backoff with
  jitter, ``Retry-After`` honoured, at most ``max_retries`` retries);
* not retried: other 4xx (authentication, permission, bad request, missing endpoint)
  and invalid responses (wrong count or dimension, non-finite values).

Requests are bounded by ``max_inputs_per_request`` and ``max_chars_per_request`` (the
endpoint accepts at most 4 MB per request). Every response is validated before any
vector is returned, so a partial or corrupt response is never cached.

``run_embedding_job`` drives a large build: results are handed to a ``sink`` (the
Delta cache MERGE) every ``checkpoint_every_requests`` successful requests, and the
completed part is flushed before an error is raised, so a rerun only embeds what is
still missing (cache key: text sha256 + model).
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from worldbank_copilot.common.exceptions import CopilotError
from worldbank_copilot.retrieval.config import EmbeddingConfig

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class EmbeddingError(CopilotError):
    """Embedding failed and must not be retried (configuration or invalid response)."""


class RetryableEmbeddingError(CopilotError):
    """A transient failure (timeout, connection, 429, 5xx)."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class EmbeddingJobError(EmbeddingError):
    """A job stopped; ``stats`` says how much was embedded and persisted first."""

    def __init__(self, message: str, stats: JobStats):
        super().__init__(message)
        self.stats = stats


class EmbeddingProvider(Protocol):
    model: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass
class Response:
    status: int
    body: Any
    headers: dict[str, str] = field(default_factory=dict)


class Transport(Protocol):
    def post(self, texts: Sequence[str], timeout: float) -> Response: ...


class DatabricksHttpTransport:
    """POST /serving-endpoints/<endpoint>/invocations with workspace credentials.

    Authentication comes from the Databricks SDK configuration (notebook / job identity);
    no retries happen here. Timeouts and connection errors become RetryableEmbeddingError.
    """

    def __init__(self, endpoint: str, config: Any | None = None, session: Any | None = None):
        self.endpoint = endpoint
        self._config = config
        self._session = session

    def _workspace_config(self) -> Any:
        if self._config is None:
            from databricks.sdk import WorkspaceClient

            self._config = WorkspaceClient().config
        return self._config

    def post(self, texts: Sequence[str], timeout: float) -> Response:
        import requests

        config = self._workspace_config()
        session = self._session or requests
        url = f"{config.host.rstrip('/')}/serving-endpoints/{self.endpoint}/invocations"
        try:
            reply = session.post(
                url, json={"input": list(texts)}, headers=config.authenticate(), timeout=timeout
            )
        except requests.Timeout as exc:
            raise RetryableEmbeddingError(f"request timed out after {timeout}s") from exc
        except requests.ConnectionError as exc:
            raise RetryableEmbeddingError(f"connection error: {exc}") from exc
        try:
            body = reply.json()
        except ValueError:
            body = {"text": reply.text[:500]}
        return Response(reply.status_code, body, dict(reply.headers))


def plan_batches(texts: Sequence[str], max_inputs: int, max_chars: int) -> list[list[int]]:
    """Indexes of consecutive texts per request: <= max_inputs and <= max_chars (a single
    longer text is sent alone)."""
    batches: list[list[int]] = []
    current: list[int] = []
    size = 0
    for index, text in enumerate(texts):
        if current and (len(current) >= max_inputs or size + len(text) > max_chars):
            batches.append(current)
            current, size = [], 0
        current.append(index)
        size += len(text)
    if current:
        batches.append(current)
    return batches


def _retry_after(headers: dict[str, str]) -> float | None:
    value = {k.lower(): v for k, v in headers.items()}.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


class DatabricksServingEmbeddings:
    """Embeddings from a Databricks Model Serving endpoint (databricks-gte-large-en)."""

    def __init__(
        self,
        config: EmbeddingConfig,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ):
        self.config = config
        self.model = config.endpoint
        self.dimension = config.expected_dimension
        self.transport = transport or DatabricksHttpTransport(config.endpoint)
        self._sleep = sleep
        self._rng = rng
        self.retries = 0  # cumulative retries (observability)

    def validate(self, body: Any, count: int) -> list[list[float]]:
        """Vectors of a response, or EmbeddingError (never a partial result)."""
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or len(data) != count:
            got = len(data) if isinstance(data, list) else "no data"
            raise EmbeddingError(f"{self.model}: {got} vectors returned for {count} texts")
        ordered = (
            sorted(data, key=lambda item: item.get("index", 0))
            if all(isinstance(item, dict) and "index" in item for item in data)
            else data
        )
        vectors = []
        for item in ordered:
            vector = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(vector, list) or len(vector) != self.dimension:
                size = len(vector) if isinstance(vector, list) else "none"
                raise EmbeddingError(
                    f"{self.model}: expected dimension {self.dimension}, got {size} "
                    "(configs/retrieval/embeddings.yaml)"
                )
            values = [float(x) for x in vector]
            if not all(math.isfinite(x) for x in values):
                raise EmbeddingError(f"{self.model}: non-finite value in embedding")
            vectors.append(values)
        return vectors

    def _attempt(self, texts: Sequence[str]) -> list[list[float]]:
        reply = self.transport.post(texts, self.config.request_timeout_seconds)
        if reply.status == 200:
            return self.validate(reply.body, len(texts))
        message = f"{self.model}: HTTP {reply.status} {str(reply.body)[:300]}"
        if reply.status in RETRYABLE_STATUS:
            raise RetryableEmbeddingError(message, _retry_after(reply.headers))
        raise EmbeddingError(message)  # other 4xx: deterministic, not retried

    def backoff(self, attempt: int, retry_after: float | None) -> float:
        cfg = self.config
        if retry_after is not None:
            return min(retry_after, cfg.backoff_max_seconds)
        delay = min(cfg.backoff_max_seconds, cfg.backoff_base_seconds * 2**attempt)
        return delay * (0.5 + self._rng() / 2)  # jitter: 50-100 % of the delay

    def embed_request(self, texts: Sequence[str]) -> list[list[float]]:
        """One bounded request with the retry policy."""
        for attempt in range(self.config.max_retries + 1):
            try:
                return self._attempt(texts)
            except RetryableEmbeddingError as exc:
                if attempt == self.config.max_retries:
                    raise EmbeddingError(f"{exc} (gave up after {attempt + 1} attempts)") from exc
                self.retries += 1
                self._sleep(self.backoff(attempt, exc.retry_after))
        raise AssertionError("unreachable")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        out: list[list[float]] = [[] for _ in texts]
        for batch in plan_batches(
            texts, self.config.max_inputs_per_request, self.config.max_chars_per_request
        ):
            for index, vector in zip(
                batch, self.embed_request([texts[i] for i in batch]), strict=True
            ):
                out[index] = vector
        return out


def embedding_provider(
    config: EmbeddingConfig, transport: Transport | None = None
) -> EmbeddingProvider:
    if config.provider == "databricks_serving":
        return DatabricksServingEmbeddings(config, transport=transport)
    raise EmbeddingError(f"unknown embedding provider {config.provider!r}")


# ---------------------------------------------------------------------------
# Checkpointed embedding job (cache build)
# ---------------------------------------------------------------------------


@dataclass
class JobStats:
    cached: int
    remaining: int
    batches: int
    batch: int = 0
    embedded: int = 0  # validated and handed to the sink (persisted)
    retries: int = 0
    failed: int = 0  # texts of the request that failed
    checkpoints: int = 0
    seconds: float = 0.0

    def line(self) -> str:
        return (
            f"cached={self.cached}, remaining={self.remaining}, "
            f"batch={self.batch}/{self.batches}, embedded={self.embedded}, "
            f"retries={self.retries}, failed={self.failed}"
        )


Sink = Callable[[list[tuple[str, list[float]]]], None]


def run_embedding_job(
    items: Sequence[tuple[str, str]],
    provider: DatabricksServingEmbeddings,
    sink: Sink,
    *,
    cached: int,
    checkpoint_every_requests: int,
    log: Callable[[str], None] | None = None,
) -> JobStats:
    """Embed (key, text) items in bounded requests, persisting every N requests.

    ``sink`` receives only validated (key, vector) pairs. On failure the completed
    requests are flushed first, then EmbeddingJobError is raised with the statistics.
    """
    say = log or (lambda _m: None)
    started = time.perf_counter()
    texts = [text for _, text in items]
    batches = plan_batches(
        texts, provider.config.max_inputs_per_request, provider.config.max_chars_per_request
    )
    stats = JobStats(cached=cached, remaining=len(items), batches=len(batches))
    retries_before = provider.retries
    pending: list[tuple[str, list[float]]] = []

    def flush() -> None:
        if pending:
            sink(list(pending))
            stats.embedded += len(pending)
            stats.remaining -= len(pending)
            stats.checkpoints += 1
            pending.clear()

    say(stats.line())
    for number, batch in enumerate(batches, 1):
        stats.batch = number
        try:
            vectors = provider.embed_request([texts[i] for i in batch])
        except EmbeddingError as exc:
            stats.failed = len(batch)
            stats.retries = provider.retries - retries_before
            flush()  # keep everything that succeeded before the failure
            stats.seconds = round(time.perf_counter() - started, 1)
            say(stats.line() + "  STOPPED")
            raise EmbeddingJobError(
                f"{exc} | persisted {stats.embedded} new embeddings; rerun resumes with the "
                f"remaining {stats.remaining}",
                stats,
            ) from exc
        pending.extend((items[i][0], v) for i, v in zip(batch, vectors, strict=True))
        stats.retries = provider.retries - retries_before
        if number % checkpoint_every_requests == 0:
            flush()
            say(stats.line())
    flush()
    stats.seconds = round(time.perf_counter() - started, 1)
    say(stats.line() + "  DONE")
    return stats
