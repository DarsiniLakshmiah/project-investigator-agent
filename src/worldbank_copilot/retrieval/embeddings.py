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


def parse_retry_after(headers: dict[str, str], now: float | None = None) -> float | None:
    """Retry-After in seconds: delta-seconds or an HTTP date (RFC 9110); None if absent."""
    value = {k.lower(): v for k, v in headers.items()}.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        when = parsedate_to_datetime(value).timestamp()
    except (TypeError, ValueError):
        return None
    return max(0.0, when - (now if now is not None else time.time()))


def error_code(body: Any) -> str | None:
    return body.get("error_code") if isinstance(body, dict) else None


class RateLimited(RetryableEmbeddingError):
    """HTTP 429: workspace / endpoint throttling (paced, not hammered)."""

    def __init__(self, message: str, retry_after: float | None, code: str | None):
        super().__init__(message, retry_after)
        self.code = code


class DatabricksServingEmbeddings:
    """Embeddings from a Databricks Model Serving endpoint (databricks-gte-large-en).

    Request pacing: consecutive request starts are at least ``interval`` seconds apart
    (``min_request_interval_seconds``). The pacing state lives on this object, so a
    capability-probe request made with the same provider counts against it. On HTTP 429
    the interval doubles (up to ``max_request_interval_seconds``) and recovers slowly
    (x ``pace_recovery_factor`` per success) down to the minimum.

    Rate limiting (429) is handled separately from transient failures (timeouts,
    connection errors, 5xx):

    * Retry-After present: wait exactly that long (not capped);
    * absent: cooldown ``rate_limit_cooldown_seconds``, doubling per consecutive 429 up to
      ``rate_limit_cooldown_max_seconds``;
    * at most ``max_rate_limit_retries`` per request, and at most
      ``rate_limit_max_total_wait_seconds`` of rate-limit waiting per provider; then the
      request fails cleanly (the job checkpoints and stays resumable).
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] | None = None,
    ):
        self.config = config
        self.model = config.endpoint
        self.dimension = config.expected_dimension
        self.transport = transport or DatabricksHttpTransport(config.endpoint)
        self._sleep = sleep
        self._rng = rng
        self._clock = clock
        self.log = log
        self.interval = config.min_request_interval_seconds
        self._last_start: float | None = None
        self.requests = 0  # HTTP attempts sent
        self.retries = 0  # all retries (rate limit + transient)
        self.rate_limited = 0  # 429 responses
        self.rate_limit_wait = 0.0  # seconds spent waiting because of 429
        self.pacing_wait = 0.0  # seconds spent waiting for the request interval

    def _say(self, message: str) -> None:
        if self.log is not None:
            self.log(message)

    def _wait(self, seconds: float) -> None:
        if seconds > 0:
            self._sleep(seconds)

    def _pace(self) -> None:
        """Keep request starts at least ``interval`` seconds apart."""
        if self._last_start is not None:
            remaining = self.interval - (self._clock() - self._last_start)
            if remaining > 0:
                self.pacing_wait += remaining
                self._wait(remaining)
        self._last_start = self._clock()

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
        self._pace()
        self.requests += 1
        reply = self.transport.post(texts, self.config.request_timeout_seconds)
        if reply.status == 200:
            return self.validate(reply.body, len(texts))
        code = error_code(reply.body)
        message = f"{self.model}: HTTP {reply.status} {code or ''} {str(reply.body)[:300]}"
        if reply.status == 429:
            raise RateLimited(message, parse_retry_after(reply.headers), code)
        if reply.status in RETRYABLE_STATUS:
            raise RetryableEmbeddingError(message, parse_retry_after(reply.headers))
        raise EmbeddingError(message)  # other 4xx: deterministic, not retried

    def backoff(self, attempt: int, retry_after: float | None) -> float:
        """Transient failures (timeout, connection, 5xx): exponential with jitter."""
        cfg = self.config
        if retry_after is not None:
            return retry_after
        delay = min(cfg.backoff_max_seconds, cfg.backoff_base_seconds * 2**attempt)
        return delay * (0.5 + self._rng() / 2)  # jitter: 50-100 % of the delay

    def cooldown(self, consecutive: int, retry_after: float | None) -> float:
        """Rate limiting (429): Retry-After in full, else a conservative doubling cooldown."""
        if retry_after is not None:
            return retry_after
        cfg = self.config
        return min(
            cfg.rate_limit_cooldown_max_seconds,
            cfg.rate_limit_cooldown_seconds * 2 ** (consecutive - 1),
        )

    def embed_request(self, texts: Sequence[str]) -> list[list[float]]:
        """One bounded request with pacing, rate-limit cooldown and transient retries."""
        cfg = self.config
        transient = limited = 0
        while True:
            try:
                vectors = self._attempt(texts)
            except RateLimited as exc:
                limited += 1
                self.rate_limited += 1
                self.interval = min(cfg.max_request_interval_seconds, self.interval * 2)
                wait = self.cooldown(limited, exc.retry_after)
                source = "Retry-After" if exc.retry_after is not None else "cooldown"
                budget_left = cfg.rate_limit_max_total_wait_seconds - self.rate_limit_wait
                self._say(
                    f"rate-limited: HTTP 429 code={exc.code} "
                    f"retry_after={exc.retry_after if exc.retry_after is not None else 'absent'} "
                    f"attempt={limited}/{cfg.max_rate_limit_retries} sleep={wait:.1f}s ({source}) "
                    f"cumulative_rate_limit_wait={self.rate_limit_wait:.1f}s "
                    f"pace={self.interval:.1f}s"
                )
                if limited > cfg.max_rate_limit_retries or wait > budget_left:
                    reason = (
                        "rate-limit retries exhausted"
                        if limited > cfg.max_rate_limit_retries
                        else "rate-limit wait budget exhausted"
                    )
                    raise EmbeddingError(
                        f"{exc} ({reason}: {limited} rate-limited attempts, "
                        f"{self.rate_limit_wait:.0f}s waited)"
                    ) from exc
                self.retries += 1
                self.rate_limit_wait += wait
                self._wait(wait)
                continue
            except RetryableEmbeddingError as exc:
                if transient == cfg.max_retries:
                    raise EmbeddingError(f"{exc} (gave up after {transient + 1} attempts)") from exc
                wait = self.backoff(transient, exc.retry_after)
                self._say(
                    f"transient failure: {exc} attempt={transient + 1}/{cfg.max_retries} "
                    f"sleep={wait:.1f}s"
                )
                transient += 1
                self.retries += 1
                self._wait(wait)
                continue
            self.interval = max(
                cfg.min_request_interval_seconds, self.interval * cfg.pace_recovery_factor
            )
            return vectors

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
    rate_limited: int = 0
    rate_limit_wait: float = 0.0
    pace: float = 0.0

    def line(self) -> str:
        return (
            f"cached={self.cached}, remaining={self.remaining}, "
            f"batch={self.batch}/{self.batches}, embedded={self.embedded}, "
            f"retries={self.retries}, failed={self.failed}, "
            f"rate_limited={self.rate_limited}, rate_limit_wait={self.rate_limit_wait:.0f}s, "
            f"pace={self.pace:.1f}s"
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
    limited_before, waited_before = provider.rate_limited, provider.rate_limit_wait
    if provider.log is None:
        provider.log = say
    pending: list[tuple[str, list[float]]] = []

    def observe() -> None:
        stats.retries = provider.retries - retries_before
        stats.rate_limited = provider.rate_limited - limited_before
        stats.rate_limit_wait = provider.rate_limit_wait - waited_before
        stats.pace = provider.interval

    def flush() -> None:
        if pending:
            sink(list(pending))
            stats.embedded += len(pending)
            stats.remaining -= len(pending)
            stats.checkpoints += 1
            pending.clear()

    stats.pace = provider.interval
    say(stats.line())
    for number, batch in enumerate(batches, 1):
        stats.batch = number
        try:
            vectors = provider.embed_request([texts[i] for i in batch])
        except EmbeddingError as exc:
            stats.failed = len(batch)
            observe()
            flush()  # keep everything that succeeded before the failure
            stats.seconds = round(time.perf_counter() - started, 1)
            say(stats.line() + "  STOPPED")
            raise EmbeddingJobError(
                f"{exc} | persisted {stats.embedded} new embeddings; rerun resumes with the "
                f"remaining {stats.remaining}",
                stats,
            ) from exc
        pending.extend((items[i][0], v) for i, v in zip(batch, vectors, strict=True))
        observe()
        if number % checkpoint_every_requests == 0:
            flush()
            say(stats.line())
    flush()
    stats.seconds = round(time.perf_counter() - started, 1)
    say(stats.line() + "  DONE")
    return stats
