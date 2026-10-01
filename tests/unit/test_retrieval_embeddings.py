"""Embedding provider and checkpointed embedding job (Phase 8 Step 3)."""

import hashlib

import pytest
from tests.conftest import REPO_CONFIG_DIR

from worldbank_copilot.retrieval.config import load_retrieval_settings
from worldbank_copilot.retrieval.embeddings import (
    DatabricksHttpTransport,
    DatabricksServingEmbeddings,
    EmbeddingError,
    EmbeddingJobError,
    Response,
    RetryableEmbeddingError,
    plan_batches,
    run_embedding_job,
)

RS = load_retrieval_settings(REPO_CONFIG_DIR)
DIM = RS.embeddings.expected_dimension


def cfg(**changes):
    base = {"max_inputs_per_request": 2, "max_chars_per_request": 1000, "max_retries": 3,
            "backoff_base_seconds": 1, "backoff_max_seconds": 8,
            "checkpoint_every_requests": 2}  # fmt: skip
    return RS.embeddings.model_copy(update={**base, **changes})


def ok(texts, dim=DIM):
    data = [{"index": i, "embedding": [float(len(t))] * dim, "object": "embedding"}
            for i, t in enumerate(texts)]  # fmt: skip
    return Response(200, {"data": data})


class Script:
    """Fake transport: a list of outcomes per call ('ok', status code, 'timeout', callable)."""

    def __init__(self, outcomes=()):
        self.outcomes = list(outcomes)
        self.calls: list[list[str]] = []

    def post(self, texts, timeout):
        self.calls.append(list(texts))
        outcome = self.outcomes.pop(0) if self.outcomes else "ok"
        if outcome == "ok":
            return ok(texts)
        if outcome == "timeout":
            raise RetryableEmbeddingError(f"request timed out after {timeout}s")
        if callable(outcome):
            return outcome(texts)
        return Response(outcome, {"error_code": f"E{outcome}"},
                        {"Retry-After": "3"} if outcome == 429 else {})  # fmt: skip


def provider(transport, **changes):
    sleeps = []
    p = DatabricksServingEmbeddings(cfg(**changes), transport, sleep=sleeps.append, rng=lambda: 1.0)
    return p, sleeps


def items(n):
    return [(hashlib.sha256(f"t{i}".encode()).hexdigest(), f"text {i}") for i in range(n)]


class Cache:
    """Stands in for silver.chunk_embeddings (insert-only on key)."""

    def __init__(self):
        self.rows: dict[str, list[float]] = {}
        self.writes = 0

    def sink(self, batch):
        self.writes += 1
        for key, vector in batch:
            assert len(vector) == DIM
            self.rows.setdefault(key, vector)

    def missing(self, all_items):
        return [i for i in all_items if i[0] not in self.rows]


# -- request planning and the provider ----------------------------------------------------


def test_batches_respect_input_and_character_limits():
    texts = ["a" * 300, "b" * 300, "c" * 300, "d" * 900, "e" * 2000, "f"]
    assert plan_batches(texts, 2, 1000) == [[0, 1], [2], [3], [4], [5]]
    assert plan_batches([], 2, 1000) == []
    assert plan_batches(["x"] * 5, 2, 1000) == [[0, 1], [2, 3], [4]]


def test_successful_batched_embedding_preserves_order():
    transport = Script()
    p, _ = provider(transport)
    vectors = p.embed(["a", "bb", "ccc"])
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0] and all(len(v) == DIM for v in vectors)
    assert transport.calls == [["a", "bb"], ["ccc"]]


def test_response_order_follows_the_index_field():
    def shuffled(texts):
        reply = ok(texts)
        reply.body["data"].reverse()
        return reply

    p, _ = provider(Script([shuffled]))
    assert [v[0] for v in p.embed(["a", "bb"])] == [1.0, 2.0]


def test_timeout_is_retried_with_exponential_backoff():
    p, sleeps = provider(Script(["timeout", "timeout", "ok"]))
    assert len(p.embed(["a"])) == 1
    assert sleeps == [1.0, 2.0] and p.retries == 2  # base 1 s, x2, full jitter factor


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_throttling_and_server_errors_are_retried(status):
    p, sleeps = provider(Script([status, "ok"]))
    assert len(p.embed(["a"])) == 1
    assert sleeps == ([3.0] if status == 429 else [1.0])  # Retry-After honoured for 429


def test_backoff_is_capped_and_jittered():
    p = DatabricksServingEmbeddings(cfg(), Script(), sleep=lambda _s: None, rng=lambda: 0.0)
    assert p.backoff(0, None) == 0.5 and p.backoff(10, None) == 4.0  # cap 8 s, 50 % jitter
    assert p.backoff(0, 120.0) == 8.0  # Retry-After capped too


def test_retries_are_bounded():
    p, sleeps = provider(Script(["timeout"] * 10), max_retries=2)
    with pytest.raises(EmbeddingError, match="gave up after 3 attempts"):
        p.embed(["a"])
    assert len(sleeps) == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_deterministic_client_errors_are_not_retried(status):
    transport = Script([status, "ok"])
    p, sleeps = provider(transport)
    with pytest.raises(EmbeddingError, match=f"HTTP {status}"):
        p.embed(["a"])
    assert sleeps == [] and len(transport.calls) == 1


def test_wrong_dimension_wrong_count_and_non_finite_values_are_rejected():
    p, sleeps = provider(Script([lambda texts: ok(texts, dim=8)]))
    with pytest.raises(EmbeddingError, match="expected dimension 1024, got 8"):
        p.embed(["a"])
    assert sleeps == []  # an invalid response is not retried
    p, _ = provider(Script([lambda texts: Response(200, {"data": ok(texts).body["data"][:1]})]))
    with pytest.raises(EmbeddingError, match="1 vectors returned for 2 texts"):
        p.embed(["a", "b"])

    def nan(texts):
        reply = ok(texts)
        reply.body["data"][0]["embedding"][0] = float("nan")
        return reply

    p, _ = provider(Script([nan]))
    with pytest.raises(EmbeddingError, match="non-finite"):
        p.embed(["a"])


# -- checkpointed job: persistence, resume, idempotency -----------------------------------


def test_job_checkpoints_every_n_requests_and_logs_progress():
    cache, logs = Cache(), []
    p, _ = provider(Script())
    stats = run_embedding_job(items(9), p, cache.sink, cached=4, checkpoint_every_requests=2,
                              log=logs.append)  # fmt: skip
    assert stats.embedded == 9 and stats.remaining == 0 and stats.batches == 5
    assert cache.writes == 3  # after requests 2 and 4, then the final one
    assert logs[0] == "cached=4, remaining=9, batch=0/5, embedded=0, retries=0, failed=0"
    assert logs[-1].endswith("DONE") and "embedded=9" in logs[-1]


def test_partial_batch_failure_persists_completed_work_then_resume_finishes():
    cache = Cache()
    all_items = items(10)  # 5 requests of 2
    p, _ = provider(Script(["ok", "ok", "ok", 400]))  # 4th request fails deterministically
    with pytest.raises(EmbeddingJobError) as failure:
        run_embedding_job(all_items, p, cache.sink, cached=0, checkpoint_every_requests=2)
    stats = failure.value.stats
    assert stats.embedded == 6 and stats.failed == 2 and stats.remaining == 4
    assert len(cache.rows) == 6 and "rerun resumes with the remaining 4" in str(failure.value)
    # Rerun: only the missing texts are sent.
    transport = Script()
    p2, _ = provider(transport)
    remaining = cache.missing(all_items)
    assert len(remaining) == 4
    stats2 = run_embedding_job(remaining, p2, cache.sink, cached=6, checkpoint_every_requests=2)
    assert stats2.embedded == 4 and len(cache.rows) == 10
    assert sum(len(c) for c in transport.calls) == 4


def test_failure_after_retries_does_not_cache_the_failed_request():
    cache = Cache()
    p, _ = provider(Script(["ok"] + ["timeout"] * 10), max_retries=2)
    with pytest.raises(EmbeddingJobError):
        run_embedding_job(items(4), p, cache.sink, cached=0, checkpoint_every_requests=5)
    assert len(cache.rows) == 2  # first request flushed; the timed-out request is not cached


def test_invalid_vectors_are_never_handed_to_the_cache():
    cache = Cache()
    p, _ = provider(Script(["ok", lambda texts: ok(texts, dim=3)]))
    with pytest.raises(EmbeddingJobError, match="dimension"):
        run_embedding_job(items(4), p, cache.sink, cached=0, checkpoint_every_requests=5)
    assert len(cache.rows) == 2 and all(len(v) == DIM for v in cache.rows.values())


def test_cache_hits_are_skipped_and_rerun_is_idempotent():
    cache = Cache()
    all_items = items(6)
    p, _ = provider(Script())
    run_embedding_job(all_items, p, cache.sink, cached=0, checkpoint_every_requests=2)
    snapshot = dict(cache.rows)
    transport = Script()
    p2, _ = provider(transport)
    stats = run_embedding_job(cache.missing(all_items), p2, cache.sink, cached=6,
                              checkpoint_every_requests=2)  # fmt: skip
    assert stats.embedded == 0 and transport.calls == [] and cache.rows == snapshot
    cache.sink([(all_items[0][0], [0.0] * DIM)])  # re-sending a cached key changes nothing
    assert cache.rows == snapshot


# -- HTTP transport -------------------------------------------------------------------------


class FakeConfig:
    host = "https://example.cloud.databricks.com/"

    def authenticate(self):
        return {"Authorization": "Bearer token"}


class FakeSession:
    def __init__(self, error=None):
        self.error, self.calls = error, []

    def post(self, url, json, headers, timeout):
        self.calls.append((url, json, headers, timeout))
        if self.error:
            raise self.error

        class Reply:
            status_code = 200
            headers = {}
            text = ""

            def json(self_inner):
                return {"data": []}

        return Reply()


def test_transport_posts_one_attempt_with_our_timeout():
    session = FakeSession()
    reply = DatabricksHttpTransport("databricks-gte-large-en", FakeConfig(), session).post(
        ["a"], 60
    )
    url, body, headers, timeout = session.calls[0]
    assert (
        url
        == "https://example.cloud.databricks.com/serving-endpoints/databricks-gte-large-en/invocations"
    )
    assert body == {"input": ["a"]} and headers["Authorization"] == "Bearer token" and timeout == 60
    assert reply.status == 200 and len(session.calls) == 1


def test_transport_maps_timeouts_and_connection_errors_to_retryable():
    import requests

    for error in (requests.Timeout("slow"), requests.ConnectionError("reset")):
        transport = DatabricksHttpTransport("e", FakeConfig(), FakeSession(error))
        with pytest.raises(RetryableEmbeddingError):
            transport.post(["a"], 60)


def test_configuration_values():
    e = RS.embeddings
    assert (e.endpoint, e.expected_dimension) == ("databricks-gte-large-en", 1024)
    assert e.max_inputs_per_request == 16 and e.max_chars_per_request == 24000
    assert e.request_timeout_seconds == 60 and e.max_retries == 5
    assert e.checkpoint_every_requests == 25
