"""Offline harness tests with real contracts and counted deterministic adapters."""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tests.conftest import REPO_CONFIG_DIR, REPO_ROOT
from tests.support.routing_fixtures import CONFIG, INDEX, chunk_rows
from tests.support.tool_fixtures import context, project_360, tables
from tests.unit.test_phase9_contract import PROFILE, baseline_reranker, retrieval

from worldbank_copilot.common.exceptions import ConfigurationError
from worldbank_copilot.investigation.policy import InvestigationPolicy
from worldbank_copilot.tools.reader import InMemoryReader
from worldbank_copilot.validation import phase9_contract as prior
from worldbank_copilot.validation import phase10c_evidence as h

POLICY = InvestigationPolicy()


def runtime(protocol=None):
    rows = chunk_rows()
    for row in rows:
        if row["project_id"] == "P179039":
            row["document_type"] = "ISR"
            row["isr_sequence"] = 6
            row["chunk_text"] += " Program standing mid-term remains on track"
            row["search_text"] += " Program standing mid-term remains on track"
    _, retriever = retrieval(rows=rows)

    def ctx(request_id):
        governed = tables()
        governed["gold.project_360"].append(project_360("P506272"))
        governed["silver.projects"].append(
            {
                "record_id": "sp-P506272",
                "project_id": "P506272",
                "borrower": "Borrower",
                "implementing_agency": "Agency",
                "project_development_objective": "Objective text",
            }
        )
        result = context(InMemoryReader(governed))
        result.request_id = request_id
        return result

    return h.wire(
        retriever,
        baseline_reranker(),
        CONFIG,
        INDEX,
        ctx,
        protocol or SimpleNamespace(profile=PROFILE),
        {"kind": "OFFLINE_STAND_INS"},
        REPO_CONFIG_DIR,
    )


@pytest.mark.parametrize("case", h.load_cases(REPO_ROOT), ids=lambda c: c.case_id)
def test_each_frozen_case_executes_real_offline_contract(case):
    row = h.execute_case(case, runtime(), POLICY, "offline-" + case.case_id, offline=True)
    assert row["status"] == "PASS", row["failure_reasons"]
    assert all(row["checks"].values())
    assert row["cross_source_atomicity"] == "NOT_ESTABLISHED"
    assert row["counters"]["document_retrieval_calls"] == int(case.document_retrieval is not None)


def test_closure_and_case_lock_consistency():
    protocol, cases, lock = h.prepare(REPO_ROOT)
    assert protocol.profile == PROFILE
    assert len(cases) == 4
    assert lock["case_set_sha256_lf"] == prior.canonical_sha256(REPO_ROOT / h.CASE_FILE)


def test_case_set_is_exact_reused_reviewed_questions():
    cases = h.load_cases(REPO_ROOT)
    assert [c.project_id for c in cases] == ["P130544", "P130544", "P179039", "P506272"]
    assert [c.source_question_id for c in cases] == [None, "q04", "q26", "q46"]
    assert sum(len(c.expected_operation_types) for c in cases) == 7
    assert all(h.requirement(c).expected_id() == h.requirement(c).requirement_id for c in cases)


def test_read_receipt_detects_tampering(tmp_path):
    output = tmp_path / "phase10c_evidence_validation__10c1.json"
    artifact = {
        "cost_accounting": h.CostAccounting().model_dump(mode="json"),
        "case_results": [],
        "preflight": {"status": "FAIL"},
        "summary": h.summary([], "FAIL"),
    }
    writer = h.ArtifactWriter(output)
    writer.write(artifact, final=True)
    assert h.read_completed(output)["summary"]["overall_status"] == "FAIL"
    receipt = json.loads(prior.completion_receipt(output).read_bytes())
    assert receipt["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert receipt["bytes"] == len(output.read_bytes())
    output.write_bytes(output.read_bytes() + b" ")
    with pytest.raises(ValueError, match="RECEIPT"):
        h.read_completed(output)


@pytest.mark.parametrize(
    "existing", [".json", ".json.complete.json", ".attempt.json", ".checkpoint-00.json"]
)
def test_no_overwrite_any_prior_attempt(tmp_path, existing):
    output = tmp_path / "phase10c_evidence_validation__10c1.json"
    existing_path = output.with_name(output.stem + existing)
    existing_path.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        h.ArtifactWriter(output)
    assert existing_path.read_bytes() == b"preserve"


def test_checkpoints_are_not_final(tmp_path):
    output = tmp_path / "run.json"
    artifact = {
        "cost_accounting": h.CostAccounting().model_dump(mode="json"),
        "case_results": [],
        "preflight": {"status": "FAIL"},
        "summary": h.summary([], "FAIL"),
    }
    writer = h.ArtifactWriter(output)
    writer.write(artifact)
    checkpoint = tmp_path / "run.checkpoint-00.json"
    with pytest.raises(ValueError, match="NOT_FINAL"):
        h.read_completed(checkpoint)
    assert h.read_completed(checkpoint, require_final=False)
    assert not output.exists()


def test_full_offline_attempt_finalized(monkeypatch, tmp_path):
    monkeypatch.setattr(
        prior,
        "revision_identity",
        lambda *a: prior.RevisionIdentity(
            declared_commit_sha="a" * 40,
            runtime_git_status="AMBIGUOUS",
            diagnostic="UNRELATED_PARENT_GIT",
        ),
    )
    output = tmp_path / "phase10c_evidence_validation__10c1.json"
    result = h.run_attempt(
        REPO_ROOT, output, commit_sha="a" * 40, policy=POLICY, prepare_runtime=runtime, offline=True
    )
    assert result["summary"] == {
        "cases_executed": 4,
        "cases_passed": 4,
        "overall_status": "PASS",
    }, result
    assert result["persistence"]["finalized"]
    assert prior.completion_receipt(output).exists()
    assert len(list(tmp_path.glob("*.checkpoint-*.json"))) == 12  # six checkpoints and six receipts
    with pytest.raises(FileExistsError):
        h.run_attempt(REPO_ROOT, output, commit_sha="a" * 40, policy=POLICY, prepare_runtime=Mock())


def test_trustworthy_mismatch_fails_before_live_wiring(monkeypatch, tmp_path):
    monkeypatch.setattr(
        prior,
        "revision_identity",
        lambda *a: prior.RevisionIdentity(
            declared_commit_sha="a" * 40,
            runtime_git_head="b" * 40,
            runtime_git_status="MISMATCH",
            diagnostic="PROJECT_ROOT_ASSOCIATED",
        ),
    )
    live = Mock()
    result = h.run_attempt(
        REPO_ROOT, tmp_path / "run.json", commit_sha="a" * 40, policy=POLICY, prepare_runtime=live
    )
    assert result["summary"]["overall_status"] == "FAIL"
    assert result["preflight"]["status"] == "FAIL"
    live.assert_not_called()


def test_failed_case_preserved_no_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(
        prior,
        "revision_identity",
        lambda *a: prior.RevisionIdentity(
            declared_commit_sha="a" * 40, runtime_git_status="UNAVAILABLE", diagnostic="NO_GIT"
        ),
    )
    real = h.execute_case
    calls = []

    def failed(case, *args, **kwargs):
        row = real(case, *args, **kwargs)
        calls.append(case.case_id)
        row["status"] = "FAIL"
        row["failure_reasons"] = ["SIMULATED_LOCAL_ASSERTION_FAILURE"]
        return row

    monkeypatch.setattr(h, "execute_case", failed)
    result = h.run_attempt(
        REPO_ROOT,
        tmp_path / "run.json",
        commit_sha="a" * 40,
        policy=POLICY,
        prepare_runtime=runtime,
        offline=True,
    )
    assert calls == ["D10C-01"]
    assert result["summary"]["overall_status"] == "FAIL"
    assert result["persistence"]["finalized"]


def test_databricks_attempt_accepts_absent_cost_config(monkeypatch, tmp_path):
    monkeypatch.setattr(
        prior,
        "revision_identity",
        lambda *a: prior.RevisionIdentity(
            declared_commit_sha="a" * 40, runtime_git_status="UNAVAILABLE", diagnostic="NO_GIT"
        ),
    )
    # Real adapters are replaced with local stand-ins, while offline remains False
    # to exercise the actual Databricks harness branch without external calls.
    live = Mock(side_effect=runtime)
    result = h.run_attempt(
        REPO_ROOT,
        tmp_path / "run.json",
        commit_sha="a" * 40,
        policy=InvestigationPolicy(),
        prepare_runtime=live,
    )
    live.assert_called_once()
    assert result["summary"]["overall_status"] == "PASS"
    assert result["validation_environment"] == "DATABRICKS"
    assert result["cost_accounting"] == h.CostAccounting().model_dump(mode="json")
    assert result["policy"]["cost_ceiling"] is None
    assert result["policy"]["pricing_version"] is None


def test_invalid_revision_rejected_before_live(tmp_path):
    live = Mock()
    with pytest.raises(ConfigurationError, match="full lowercase"):
        h.run_attempt(
            REPO_ROOT,
            tmp_path / "run.json",
            commit_sha="uncommitted",
            policy=POLICY,
            prepare_runtime=live,
        )
    live.assert_not_called()


def test_notebook_thin_user_only():
    text = (REPO_ROOT / "notebooks/09_phase10c_evidence_validation.py").read_text()
    assert "run_databricks_validation" in text
    assert "commit_sha" in text and "run_id" in text
    assert "cost_ceiling" not in text and "pricing_version" not in text
    assert "EvidenceExecutor(" not in text and "similarity_search(" not in text
    assert "langgraph" not in text.lower()


def test_case_exception_recorded_and_stops(monkeypatch, tmp_path):
    monkeypatch.setattr(
        prior,
        "revision_identity",
        lambda *a: prior.RevisionIdentity(
            declared_commit_sha="a" * 40, runtime_git_status="UNAVAILABLE", diagnostic="NO_GIT"
        ),
    )
    execute = Mock(side_effect=RuntimeError("untrusted-secret-detail"))
    monkeypatch.setattr(h, "execute_case", execute)
    output = tmp_path / "run.json"
    result = h.run_attempt(
        REPO_ROOT, output, commit_sha="a" * 40, policy=POLICY, prepare_runtime=runtime, offline=True
    )
    assert execute.call_count == 1
    assert result["case_results"][0]["case_id"] == "D10C-01"
    assert result["summary"]["overall_status"] == "FAIL"
    assert b"untrusted-secret-detail" not in output.read_bytes()


def test_case_lock_drift_stops_before_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(
        prior,
        "revision_identity",
        lambda *a: prior.RevisionIdentity(
            declared_commit_sha="a" * 40, runtime_git_status="UNAVAILABLE", diagnostic="NO_GIT"
        ),
    )
    original = h.canonical_sha256
    monkeypatch.setattr(
        h,
        "canonical_sha256",
        lambda p: "0" * 64 if p.name == "phase10c_evidence_cases.yaml" else original(p),
    )
    live = Mock()
    result = h.run_attempt(
        REPO_ROOT, tmp_path / "run.json", commit_sha="a" * 40, policy=POLICY, prepare_runtime=live
    )
    assert result["summary"]["overall_status"] == "FAIL"
    live.assert_not_called()


def test_partial_write_never_gets_receipt(monkeypatch, tmp_path):
    output = tmp_path / "run.json"
    artifact = {
        "cost_accounting": h.CostAccounting().model_dump(mode="json"),
        "case_results": [],
        "preflight": {"status": "FAIL"},
        "summary": h.summary([], "FAIL"),
    }
    writer = h.ArtifactWriter(output)

    def interrupted(path, value):
        path.write_bytes(b"{")
        raise OSError("interrupted sequential write")

    monkeypatch.setattr(prior, "_write_new_json", interrupted)
    with pytest.raises(OSError):
        writer.write(artifact, final=True)
    assert output.read_bytes() == b"{"
    assert not prior.completion_receipt(output).exists()
    with pytest.raises(FileExistsError):
        h.ArtifactWriter(output)


def test_verified_head_requires_committed_clean_harness(monkeypatch, tmp_path):
    monkeypatch.setattr(
        prior,
        "revision_identity",
        lambda *a: prior.RevisionIdentity(
            declared_commit_sha="a" * 40,
            runtime_git_head="a" * 40,
            runtime_git_status="VERIFIED",
            commit_sha_source="DECLARED_AND_RUNTIME_VERIFIED",
            diagnostic="PROJECT_ROOT_ASSOCIATED",
        ),
    )
    monkeypatch.setattr(
        h.subprocess, "check_output", lambda *a, **kw: b"?? reviewed_uncommitted.py"
    )
    live = Mock()
    result = h.run_attempt(
        REPO_ROOT, tmp_path / "run.json", commit_sha="a" * 40, policy=POLICY, prepare_runtime=live
    )
    assert result["summary"]["overall_status"] == "FAIL"
    assert result["failure"]["invariant_failures"] == [
        "TRUSTWORTHY_SOURCE_NOT_CLEAN_COMMITTED_SNAPSHOT"
    ]
    live.assert_not_called()


def test_pass_artifact_cannot_hide_failed_assertion():
    row = h.execute_case(
        h.load_cases(REPO_ROOT)[0], runtime(), POLICY, "assertion-test", offline=True
    )
    row["checks"]["SOURCE_PLAN_UNEXECUTED"] = False
    artifact = {
        "cost_accounting": h.CostAccounting().model_dump(mode="json"),
        "case_results": [row],
        "preflight": {"status": "PASS"},
        "summary": h.summary([row], "PASS"),
    }
    with pytest.raises(ValueError, match="PASS_REQUIRES_ALL_ASSERTIONS"):
        h.validate_artifact(artifact)


def test_pass_artifact_package_hash_rechecked():
    row = h.execute_case(h.load_cases(REPO_ROOT)[0], runtime(), POLICY, "hash-test", offline=True)
    row["report"]["package"]["package_fingerprint"] = "package_" + "0" * 64
    artifact = {
        "cost_accounting": h.CostAccounting().model_dump(mode="json"),
        "case_results": [row],
        "preflight": {"status": "PASS"},
        "summary": h.summary([row], "PASS"),
    }
    with pytest.raises(ValueError, match="PASS_REPORT_INTEGRITY"):
        h.validate_artifact(artifact)


@pytest.mark.parametrize(
    "update",
    [
        {"applicable": True},
        {"actual_billed_cost": "0"},
        {"phase10_model_calls": 1},
        {"pricing_version": "invented"},
        {"cost_ceiling": "1"},
    ],
)
def test_cost_accounting_rejects_fabricated_cost_or_model_usage(update):
    from pydantic import ValidationError

    data = h.CostAccounting().model_dump(mode="json")
    data.update(update)
    with pytest.raises(ValidationError):
        h.CostAccounting.model_validate(data)


def test_cost_metadata_required_for_completed_artifact(tmp_path):
    output = tmp_path / "run.json"
    artifact = {
        "cost_accounting": h.CostAccounting().model_dump(mode="json"),
        "case_results": [],
        "preflight": {"status": "FAIL"},
        "summary": h.summary([], "FAIL"),
    }
    writer = h.ArtifactWriter(output)
    writer.write(artifact, final=True)
    assert h.read_completed(output)["cost_accounting"]["actual_billed_cost"] == "UNAVAILABLE"
    artifact.pop("cost_accounting")
    with pytest.raises(KeyError, match="cost_accounting"):
        h.validate_artifact(artifact)


def test_databricks_entry_point_requires_only_revision_and_run_id(monkeypatch):
    import inspect

    assert tuple(inspect.signature(h.run_databricks_validation).parameters) == (
        "spark",
        "settings",
        "commit_sha",
        "run_id",
    )
    captured = {}

    def attempt(repo, output, **kwargs):
        captured.update(kwargs)
        return {"cost_accounting": h.CostAccounting().model_dump(mode="json")}

    monkeypatch.setattr(h, "run_attempt", attempt)
    from worldbank_copilot.common import dependency_health

    monkeypatch.setattr(dependency_health, "check_environment", lambda *a: SimpleNamespace(ok=True))
    settings = SimpleNamespace(
        environment=SimpleNamespace(value="databricks"),
        repo_root=REPO_ROOT,
        config_dir=REPO_CONFIG_DIR,
        artifact_volume_path="/unused-local-test",
    )
    result = h.run_databricks_validation(None, settings, commit_sha="a" * 40, run_id="10c1")
    assert captured["policy"].cost_ceiling is None
    assert captured["policy"].pricing_version is None
    assert result["cost_accounting"]["applicable"] is False


NOTEBOOK_SOURCE = (REPO_ROOT / h.NOTEBOOK_FILE).read_text(encoding="utf-8")


def test_local_notebook_semantic_identity_is_locked():
    lock = json.loads((REPO_ROOT / h.LOCK_FILE).read_text())
    assert h.NOTEBOOK_FILE not in lock["files_sha256_lf"]
    assert h.notebook_identity(NOTEBOOK_SOURCE) == lock["notebook_semantic_identity"]
    assert h.verify_notebook(REPO_ROOT, lock["notebook_semantic_identity"])


def databricks_representation(source):
    cells = source.split("# COMMAND ----------")
    exported = []
    for cell in cells:
        if "# MAGIC %" in cell:
            exported.append(cell)
        else:
            exported.append(
                "\n# MAGIC %python\n"
                + "\n".join("# MAGIC " + line for line in cell.strip().splitlines())
            )
    return "# Databricks notebook source\n" + "\n# COMMAND --------------------\n".join(exported)


@pytest.mark.parametrize(
    "source",
    [
        NOTEBOOK_SOURCE,
        NOTEBOOK_SOURCE.replace("\n", "\r\n"),
        NOTEBOOK_SOURCE.rstrip() + "\n\n",
        NOTEBOOK_SOURCE.replace("# ruff: noqa: E501", "# extra serialization comment"),
        NOTEBOOK_SOURCE.replace(
            "artifact = run_databricks_validation(",
            "# COMMAND ----------\nartifact = run_databricks_validation(",
        ),
        NOTEBOOK_SOURCE.replace('"commit_sha"', "'commit_sha'"),
        databricks_representation(NOTEBOOK_SOURCE),
        databricks_representation(NOTEBOOK_SOURCE).replace("# MAGIC %python\n", "%python\n"),
        NOTEBOOK_SOURCE.replace("# MAGIC %run", "%run").replace("# MAGIC %pip", "%pip"),
    ],
)
def test_harmless_notebook_representations_match(source):
    assert h.notebook_identity(source) == h.notebook_identity(NOTEBOOK_SOURCE)


@pytest.mark.parametrize(
    "old,new",
    [
        ("import run_databricks_validation", "import another_entrypoint"),
        ('dbutils.widgets.get("commit_sha").strip()', '"a" * 40'),
        ('dbutils.widgets.get("run_id").strip()', 'dbutils.widgets.get("other_run_id").strip()'),
        ("    spark,", "    other_spark,"),
        ("    settings,", "    other_settings,"),
        ('!= "PASS"', '== "PASS"'),
        (
            'raise RuntimeError("10C capability validation failed; inspect preserved attempt")',
            "pass",
        ),
        ("%run ./_bootstrap", "%run ./other_bootstrap"),
        ("%pip install -q", "%pip install another-package -q"),
        ("run_id=dbutils.widgets.get", "another_argument=dbutils.widgets.get"),
    ],
)
def test_meaningful_wrapper_changes_rejected(tmp_path, old, new):
    assert old in NOTEBOOK_SOURCE
    path = tmp_path / h.NOTEBOOK_FILE
    path.parent.mkdir(parents=True)
    path.write_text(NOTEBOOK_SOURCE.replace(old, new))
    with pytest.raises(ConfigurationError, match="NOTEBOOK_SEMANTIC_MISMATCH"):
        h.verify_notebook(tmp_path, h.notebook_identity(NOTEBOOK_SOURCE))


@pytest.mark.parametrize(
    "extra",
    [
        'print("run_id widget:", repr(dbutils.widgets.get("run_id")))',
        'print("commit_sha widget:", repr(dbutils.widgets.get("commit_sha")))',
        "import hashlib\nimport json\nfrom pathlib import Path",
        'artifact = {"summary": {"overall_status": "PASS"}}',
        "run_databricks_validation(spark, settings, commit_sha='x', run_id='10c6')",
        "# MAGIC %python\n# MAGIC print('extra executable diagnostic')",
        "%sh echo unapproved",
    ],
)
def test_extra_executable_logic_rejected(tmp_path, extra):
    path = tmp_path / h.NOTEBOOK_FILE
    path.parent.mkdir(parents=True)
    path.write_text(NOTEBOOK_SOURCE + "\n# COMMAND ----------\n" + extra)
    with pytest.raises(ConfigurationError, match="NOTEBOOK_"):
        h.verify_notebook(tmp_path, h.notebook_identity(NOTEBOOK_SOURCE))


def test_semantic_guard_runs_in_production_preflight(monkeypatch):
    original = h.verify_notebook
    seen = []

    def observe(repo, expected):
        seen.append(repo)
        return original(repo, expected)

    monkeypatch.setattr(h, "verify_notebook", observe)
    h.prepare(REPO_ROOT)
    assert seen == [REPO_ROOT]


def test_verified_git_does_not_reintroduce_wrapper_byte_equality(monkeypatch, tmp_path):
    monkeypatch.setattr(
        prior,
        "revision_identity",
        lambda *a: prior.RevisionIdentity(
            declared_commit_sha="a" * 40,
            runtime_git_head="a" * 40,
            runtime_git_status="VERIFIED",
            commit_sha_source="DECLARED_AND_RUNTIME_VERIFIED",
            diagnostic="PROJECT_ROOT_ASSOCIATED",
        ),
    )

    def git_status(args, **kwargs):
        assert h.NOTEBOOK_FILE not in args
        assert h.LOCK_FILE in args
        assert "src/worldbank_copilot/investigation/evidence.py" in args
        return b""

    monkeypatch.setattr(h.subprocess, "check_output", git_status)
    result = h.run_attempt(
        REPO_ROOT,
        tmp_path / "run.json",
        commit_sha="a" * 40,
        policy=POLICY,
        prepare_runtime=runtime,
        offline=True,
    )
    assert result["summary"]["overall_status"] == "PASS"


def test_bootstrap_order_change_rejected(tmp_path):
    source = NOTEBOOK_SOURCE.replace("# MAGIC %run ./_bootstrap", "")
    path = tmp_path / h.NOTEBOOK_FILE
    path.parent.mkdir(parents=True)
    path.write_text(source + "\n# COMMAND ----------\n# MAGIC %run ./_bootstrap\n")
    with pytest.raises(ConfigurationError, match="NOTEBOOK_SEMANTIC_MISMATCH"):
        h.verify_notebook(tmp_path, h.notebook_identity(NOTEBOOK_SOURCE))


def test_bad_wrapper_preflight_stops_before_live_work(monkeypatch, tmp_path):
    monkeypatch.setattr(
        prior,
        "revision_identity",
        lambda *a: prior.RevisionIdentity(
            declared_commit_sha="a" * 40, runtime_git_status="UNAVAILABLE", diagnostic="NO_GIT"
        ),
    )
    monkeypatch.setattr(
        h,
        "notebook_identity",
        lambda source: {"scheme": "databricks_wrapper_ast@1", "sha256": "0" * 64},
    )
    live = Mock()
    result = h.run_attempt(
        REPO_ROOT, tmp_path / "run.json", commit_sha="a" * 40, policy=POLICY, prepare_runtime=live
    )
    live.assert_not_called()
    assert result["summary"]["cases_executed"] == 0
    assert result["preflight"]["status"] == "FAIL"
    assert result["failure"]["invariant_failures"] == [
        "NOTEBOOK_SEMANTIC_MISMATCH:" + h.NOTEBOOK_FILE
    ]
