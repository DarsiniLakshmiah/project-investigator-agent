"""Load probe for a candidate embedding endpoint (no retries, controlled pacing)."""

import pytest
import yaml
from tests.conftest import REPO_CONFIG_DIR, REPO_ROOT

from worldbank_copilot.retrieval.config import load_retrieval_settings
from worldbank_copilot.retrieval.embeddings import Response, RetryableEmbeddingError
from worldbank_copilot.retrieval.endpoint_probe import ProbeConfig, load_probe_config, run_probe

CONFIG = ProbeConfig(endpoint="qwen", request_timeout_seconds=60, batch_size=2, pause_seconds=15)
TEXTS = [f"text {i}" for i in range(5)]


class Transport:
    def __init__(self, outcomes, dim=1024):
        self.outcomes, self.dim, self.calls = list(outcomes), dim, []

    def post(self, texts, timeout):
        self.calls.append((list(texts), timeout))
        outcome = self.outcomes.pop(0)
        if outcome == "timeout":
            raise RetryableEmbeddingError("request timed out after 60s")
        if outcome == "ok":
            data = [{"index": i, "embedding": [0.1] * self.dim} for i in range(len(texts))]
            return Response(200, {"data": data})
        headers = {"Retry-After": "30"} if outcome == 429 else {}
        return Response(
            outcome,
            {
                "error_code": "REQUEST_LIMIT_EXCEEDED"
                if outcome == 429
                else "RESOURCE_DOES_NOT_EXIST"
            },
            headers,
        )


def run(outcomes, dim=1024):
    sleeps, logs = [], []
    transport = Transport(outcomes, dim)
    report = run_probe(CONFIG, TEXTS, transport, sleep=sleeps.append, log=logs.append)
    return report, transport, sleeps, logs


def test_usable_endpoint_reports_dimension_and_sends_exactly_three_requests():
    report, transport, sleeps, logs = run(["ok", "ok", "ok"], dim=1024)
    assert report.verdict == "USABLE" and report.dimension == 1024
    assert [len(t) for t, _ in transport.calls] == [1, 2, 2] and sleeps == [15, 15]
    assert all(timeout == 60 for _, timeout in transport.calls)
    assert len(report.rows()) == 3 and "verdict USABLE" in report.format()


def test_dimension_is_reported_not_assumed():
    report, *_ = run(["ok", "ok", "ok"], dim=768)
    assert report.verdict == "USABLE" and report.dimension == 768


def test_rate_limited_endpoint_is_recorded_without_retries():
    report, transport, _, _ = run(["ok", 429, 429])
    assert report.verdict == "RATE_LIMITED" and len(transport.calls) == 3
    limited = report.steps[1]
    assert (limited.status, limited.error_code, limited.retry_after) == (
        429,
        "REQUEST_LIMIT_EXCEEDED",
        30.0,
    )


def test_missing_endpoint_and_timeouts():
    assert run([404, 404, 404])[0].verdict == "UNAVAILABLE"
    report, *_ = run(["ok", "timeout", "ok"])
    assert report.verdict == "PARTIAL" and report.steps[1].status is None
    assert "timed out" in report.steps[1].error


def test_inconsistent_or_wrong_count_responses_are_not_usable():
    class Short(Transport):
        def post(self, texts, timeout):
            reply = super().post(texts, timeout)
            reply.body["data"] = reply.body["data"][:1]
            return reply

    report = run_probe(CONFIG, TEXTS, Short(["ok"] * 3), sleep=lambda _s: None)
    assert report.verdict == "PARTIAL" and not report.steps[1].ok


def test_probe_needs_enough_sample_texts():
    with pytest.raises(ValueError, match="needs 5"):
        run_probe(CONFIG, TEXTS[:3], Transport([]), sleep=lambda _s: None)


def test_qwen_is_configured_and_gte_failure_history_is_kept():
    probe = load_probe_config(REPO_CONFIG_DIR)
    assert probe.endpoint == "databricks-qwen3-embedding-0-6b"
    assert probe.batch_size * 2 + 1 <= 9 and probe.pause_seconds >= 10
    candidates = yaml.safe_load(
        (REPO_CONFIG_DIR / "retrieval" / "embedding_candidates.yaml").read_text(encoding="utf-8")
    )
    gte = candidates["candidates"]["databricks-gte-large-en"]
    assert gte["status"] == "FAILED_RATE_LIMITED" and len(gte["observations"]) >= 4
    qwen = candidates["candidates"]["databricks-qwen3-embedding-0-6b"]
    assert qwen["status"] == "PROBED_USABLE"
    production = load_retrieval_settings(REPO_CONFIG_DIR).embeddings
    assert (production.endpoint, production.expected_dimension) == (
        "databricks-qwen3-embedding-0-6b",
        1024,
    )
    notebook = (REPO_ROOT / "notebooks" / "07a_embedding_endpoint_probe.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "update_embedding_cache",
        "build_index_source",
        "ensure_vector_index",
        "build_corpus",
        "MERGE",
        "INSERT",
    ):
        assert forbidden not in notebook
