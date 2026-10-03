"""Offline safety and observability tests for the separate 10D diagnostic."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from worldbank_copilot.validation import phase10d_models as h

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "phase10d_diagnostics", ROOT / "scripts/phase10d_diagnostics.py"
)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


@pytest.fixture(autouse=True)
def diagnostic_runtime_revision(monkeypatch):
    """Read-only diagnostics must not depend on whether reviewed edits are committed."""
    monkeypatch.setattr(
        h.prior,
        "revision_identity",
        lambda root, declared: h.prior.RevisionIdentity(
            declared_commit_sha=declared,
            runtime_git_status="AMBIGUOUS",
            diagnostic="OFFLINE_TEST_GIT_AMBIGUITY",
        ),
    )


def settings(tmp_path):
    return SimpleNamespace(
        repo_root=ROOT,
        artifact_volume_path=tmp_path,
        environment=SimpleNamespace(value="databricks"),
    )


def test_diagnostic_never_writes_reserves_or_invokes(tmp_path, monkeypatch):
    original_open = Path.open

    def read_only(path, mode="r", *args, **kwargs):
        assert not any(flag in mode for flag in "wax+")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", read_only)
    writer = Mock(side_effect=AssertionError("reservation forbidden"))
    invoke = Mock(side_effect=AssertionError("invocation forbidden"))
    monkeypatch.setattr(h, "Writer", writer)
    monkeypatch.setattr(h.DatabricksModelAdapter, "invoke", invoke)
    report = diagnostic.diagnose(settings(tmp_path), declared_sha=diagnostic.REVIEWED_SHA)
    assert report["failed_gates"] == []
    assert report["acceptance_performed"] is False
    assert report["writes"] == report["model_calls"] == 0
    writer.assert_not_called()
    invoke.assert_not_called()
    assert not list(tmp_path.iterdir())
    assert any(row["check"].startswith("shared_protocol:") for row in report["gates"])


def test_independent_gates_continue_and_exception_text_is_redacted(tmp_path, monkeypatch):
    monkeypatch.setattr(h, "prepare", Mock(side_effect=RuntimeError("SECRET_DO_NOT_PRINT")))
    report = diagnostic.diagnose(settings(tmp_path), declared_sha=diagnostic.REVIEWED_SHA)
    assert "phase10d_production_prepare" in report["failed_gates"]
    assert "SECRET_DO_NOT_PRINT" not in str(report)
    assert any(
        row["check"] == "adapter_constructor:databricks-gpt-oss-20b" for row in report["gates"]
    )
    assert sys.gettrace() is None


def test_hash_mismatch_reports_expected_and_actual(tmp_path, monkeypatch):
    original = h.canonical_sha256

    def hash_probe(path):
        return "0" * 64 if str(path).endswith("requirements-phase10d.txt") else original(path)

    monkeypatch.setattr(h, "canonical_sha256", hash_probe)
    report = diagnostic.diagnose(settings(tmp_path), declared_sha=diagnostic.REVIEWED_SHA)
    row = next(
        row for row in report["gates"] if row["check"].endswith(":file:requirements-phase10d.txt")
    )
    assert row["status"] == "FAIL"
    assert row["expected"] != row["actual"] == "0" * 64
    assert "phase10d_build_lock_equal" in report["failed_gates"]


def test_active_trace_preserved(tmp_path):
    def existing_trace(frame, event, arg):
        return existing_trace

    previous = sys.gettrace()
    try:
        sys.settrace(existing_trace)
        report = diagnostic.diagnose(settings(tmp_path), declared_sha=diagnostic.REVIEWED_SHA)
        assert sys.gettrace() is existing_trace
        assert report["failed_gates"] == []
    finally:
        sys.settrace(previous)
