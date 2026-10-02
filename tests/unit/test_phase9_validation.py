"""9F-C acceptance harness: synthetic rows and deterministic stand-ins only."""

from __future__ import annotations

import ast
import copy
import json
import shutil
from unittest.mock import Mock

import pytest
import yaml
from pydantic import ValidationError
from tests.conftest import REPO_ROOT
from tests.support.retrieval_golden import OverlapCrossEncoder, ScopedDense, TextEmbeddings
from tests.support.routing_fixtures import CONFIG, INDEX, chunk_rows, routing_tables
from tests.support.tool_fixtures import REGISTRY, context, project_360

from worldbank_copilot.retrieval.retriever import ChunkStore, Retriever
from worldbank_copilot.routing import execution as ex
from worldbank_copilot.routing.models import AccessContext, Route
from worldbank_copilot.routing.semantic_eval import canonical_sha256
from worldbank_copilot.tools.models import ProvenanceClass
from worldbank_copilot.tools.reader import InMemoryReader
from worldbank_copilot.validation import phase9_contract as v

PROTOCOL = v.prepare_protocol(REPO_ROOT, dependency_ok=True)


def fake_runtime(protocol=PROTOCOL):
    rows = chunk_rows()
    # Ensure all DOCUMENT cases exercise evidence/citation validation locally.
    for row in rows:
        if row["chunk_id"].endswith("-c0"):
            row["search_text"] = row["chunk_text"] = (
                "The latest ISR reports Program standing at mid-term "
                "and the loan agreement closing date."
            )
    embeddings = TextEmbeddings()
    embeddings.model, embeddings.dimension = (
        protocol.profile.embedding_endpoint,
        protocol.profile.embedding_dimension,
    )
    encoder = OverlapCrossEncoder()
    encoder.config = protocol.settings.retrieval.reranker
    r = Retriever(
        ChunkStore(rows), protocol.settings, REGISTRY.project_ids, embeddings, ScopedDense(rows)
    )
    data = routing_tables()
    data["gold.project_360"].append(project_360("P506272"))
    data["silver.projects"].append(
        dict(data["silver.projects"][0], project_id="P506272", record_id="silver-P506272")
    )

    def ctx(request_id):
        result = context(InMemoryReader(data))
        result.request_id = request_id
        return result

    idx = protocol.manifest["retrieval"]["index"]
    identity = {
        "endpoint": idx["endpoint"],
        "index_name": v.INDEX_NAME,
        "index_rows": idx["index_rows"],
        "corpus_rows": idx["corpus_rows"],
        "ready": True,
    }
    return v.wire_runtime(r, encoder, REGISTRY, CONFIG, INDEX, ctx, protocol, identity)


def observe(case_id):
    runtime = fake_runtime()
    case = next(c for c in PROTOCOL.cases.cases if c.case_id == case_id)
    access = AccessContext(
        user_ref="test",
        authorized_projects=runtime.authorized_projects,
        active_project_id=case.active_project_id,
    )
    result = runtime.service.handle(case.query, access, request_id="fake:" + case_id)
    return case, runtime, result, ex.execution_decision(result), runtime.executor.document_result


def check_observation(observation):
    case, runtime, result, decision, document = observation
    return v.validate_case(
        case,
        result,
        decision,
        document,
        runtime.counters.snapshot(),
        runtime.counters.tool_calls,
        1.0,
    )


def run_fake(tmp_path, *, runtime=None, local=None):
    runtime = runtime or fake_runtime()
    return v.run_first(
        output=tmp_path / v.ARTIFACT_NAME,
        commit_sha="a" * 40,
        environment={"kind": "local", "synthetic": True},
        prepare_local=local or (lambda: PROTOCOL),
        prepare_runtime=lambda protocol: runtime,
    )


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "unknown", "reordered"])
def test_case_set_requires_exact_ten_ids(change):
    data = PROTOCOL.cases.model_dump(mode="json")
    if change == "missing":
        data["cases"].pop()
    elif change == "extra":
        data["cases"].append(copy.deepcopy(data["cases"][0]))
    elif change == "duplicate":
        data["cases"][1]["case_id"] = "C01"
    elif change == "unknown":
        data["cases"][0]["case_id"] = "C11"
    else:
        data["cases"].reverse()
    with pytest.raises(ValidationError):
        v.CaseSet.model_validate(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("expected_execution_mode", "CLARIFY"),
        ("structured_execution_allowed", False),
        ("citations_required", True),
        ("expected_project_id", None),
        ("allowed_provenance", ["AI_INTERPRETATION"]),
        ("expected_tools", ["run_sql"]),
        ("unapproved_extra", 1),
    ],
)
def test_malformed_expectations_are_rejected(field, value):
    data = PROTOCOL.cases.cases[0].model_dump(mode="json") | {field: value}
    with pytest.raises(ValidationError):
        v.Case.model_validate(data)


def test_case_hash_is_frozen_deterministic_and_line_ending_independent(tmp_path):
    lock = json.loads((REPO_ROOT / v.LOCK_FILE).read_text("utf-8"))
    path = REPO_ROOT / v.CASE_FILE
    assert canonical_sha256(path) == canonical_sha256(path) == lock["case_set_sha256_lf"]
    windows = tmp_path / "cases.yaml"
    windows.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert canonical_sha256(windows) == lock["case_set_sha256_lf"]


def test_document_questions_come_from_successful_existing_artifacts():
    assert [
        c.source_question_id for c in PROTOCOL.cases.cases if c.retrieval_execution_allowed
    ] == ["q04", "q26", "q46"]
    assert v.prepare_protocol(REPO_ROOT, dependency_ok=True).integrity == PROTOCOL.integrity


def test_dependency_failure_prevents_even_runtime_construction(tmp_path):
    live = Mock()

    def local():
        return v.prepare_protocol(REPO_ROOT, dependency_ok=False)

    artifact = v.run_first(
        output=tmp_path / v.ARTIFACT_NAME,
        commit_sha="a" * 40,
        environment={"kind": "local"},
        prepare_local=local,
        prepare_runtime=live,
    )
    live.assert_not_called()
    assert artifact["preflight"]["status"] == "FAIL"
    assert "DEPENDENCY_HEALTH" in artifact["preflight"]["failures"]
    assert artifact["case_results"] == [] and artifact["summary"]["not_run"] == 10


@pytest.mark.parametrize(
    "field,bad",
    [
        ("endpoint", "other"),
        ("index_name", "other"),
        ("index_rows", 1),
        ("corpus_rows", 1),
        ("ready", False),
    ],
)
def test_runtime_identity_failure_prevents_c01(tmp_path, field, bad):
    runtime = fake_runtime()
    runtime.identity[field] = bad
    runtime.service.handle = Mock(side_effect=AssertionError("case executed"))
    artifact = run_fake(tmp_path, runtime=runtime)
    assert artifact["preflight"]["status"] == "FAIL"
    runtime.service.handle.assert_not_called()
    assert not sum(runtime.counters.snapshot().values())


@pytest.mark.parametrize(
    "field",
    ["min_rerank_score", "rrf_k", "embedding_endpoint", "embedding_dimension", "production"],
)
def test_runtime_baseline_drift_fails_preflight_before_c01(tmp_path, field):
    runtime = fake_runtime()
    r = runtime.retrieval.search.retriever.target
    rs = r.settings
    if field.startswith("embedding"):
        key, value = (
            ("endpoint", "other") if field == "embedding_endpoint" else ("expected_dimension", 512)
        )
        r.settings = rs.model_copy(
            update={"embeddings": rs.embeddings.model_copy(update={key: value})}
        )
    else:
        cfg = rs.retrieval
        update = (
            {"min_rerank_score": 999}
            if field == "min_rerank_score"
            else {"hybrid": cfg.hybrid.model_copy(update={"rrf_k": 61})}
            if field == "rrf_k"
            else {"production": cfg.production.model_copy(update={"retrieval": "hybrid"})}
        )
        r.settings = rs.model_copy(update={"retrieval": cfg.model_copy(update=update)})
    runtime.service.handle = Mock(side_effect=AssertionError("case executed"))
    artifact = run_fake(tmp_path, runtime=runtime)
    assert artifact["preflight"]["status"] == "FAIL"
    runtime.service.handle.assert_not_called()
    assert not sum(runtime.counters.snapshot().values())


def test_local_production_config_drift_fails_preflight(tmp_path):
    # Copy only the local preregistration/config/code dependencies, never alter the repo.
    for folder in ("configs", "evaluation"):
        shutil.copytree(REPO_ROOT / folder, tmp_path / folder)
    lock = json.loads((REPO_ROOT / v.LOCK_FILE).read_text("utf-8"))
    for name in (*lock["frozen_files_sha256_lf"], "src/worldbank_copilot/agents/__init__.py"):
        path = tmp_path / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO_ROOT / name, path)
    path = tmp_path / "configs/retrieval/retrieval.yaml"
    data = yaml.safe_load(path.read_text("utf-8"))
    data["production"]["retrieval"] = "hybrid"
    path.write_text(yaml.safe_dump(data), "utf-8")
    with pytest.raises(v.PreflightError, match="BASELINE_CONFIGURATION"):
        v.prepare_protocol(tmp_path, dependency_ok=True)


@pytest.mark.parametrize("case_id", v.EXPECTED_IDS)
def test_every_preregistered_case_passes_locally(case_id):
    row = check_observation(observe(case_id))
    assert row["status"] == "PASS", row["failure_reasons"]
    if case_id in ("C06", "C07", "C08", "C09", "C10"):
        assert not sum(row["execution_counters"].values())
    if case_id == "C08":
        assert row["actual_router_route"] == "SEMANTIC_CLASSIFICATION_REQUIRED"
        assert row["actual_execution_mode"] == "CLARIFY" and row["translated"] is True
        assert row["actual_execution_reason"] == "INTENT_NOT_RESOLVED"


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("route", "ROUTER_ROUTE"),
        ("mode", "EXECUTION_MODE"),
        ("router_reason", "ROUTER_REASON"),
        ("execution_reason", "EXECUTION_REASON"),
        ("project", "PROJECT_RESOLUTION"),
        ("options", "CLARIFICATION_OPTIONS_PRESERVED"),
    ],
)
def test_observed_route_mode_reason_project_and_options_mismatches_fail(kind, expected):
    observation = list(observe("C08"))
    result, decision = observation[2:4]
    if kind == "route":
        observation[2] = result.model_copy(
            update={"decision": result.decision.model_copy(update={"route": Route.STRUCTURED})}
        )
    elif kind == "mode":
        observation[3] = decision.model_copy(update={"mode": ex.ExecutionMode.REFUSE})
    elif kind == "router_reason":
        observation[2] = result.model_copy(
            update={"decision": result.decision.model_copy(update={"reason_code": "other"})}
        )
    elif kind == "execution_reason":
        observation[3] = decision.model_copy(update={"reason_code": "other"})
    elif kind == "options":
        observation[3] = decision.model_copy(update={"clarification_options": ("invented",)})
    else:
        p = result.understanding.project.model_copy(update={"project_id": "P179039"})
        observation[2] = result.model_copy(
            update={"understanding": result.understanding.model_copy(update={"project": p})}
        )
    row = check_observation(observation)
    assert row["status"] == "FAIL" and expected in row["failure_reasons"]


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("foreign", "CROSS_PROJECT_EVIDENCE"),
        ("citation", "MISSING_CITATION"),
        ("provenance", "DOCUMENT_PROVENANCE"),
        ("verified", "DOCUMENT_PROVENANCE"),
    ],
)
def test_document_evidence_violations_fail(kind, expected):
    observation = observe("C03")
    doc = observation[4]
    assert doc.status == "OK" and doc.evidence
    item = doc.evidence[0]
    if kind == "foreign":
        doc.evidence[0] = item.model_copy(
            update={"evidence": item.evidence.model_copy(update={"project_id": "P179039"})}
        )
    elif kind == "citation":
        citation = item.evidence.citation.model_copy(update={"pages": []})
        doc.evidence[0] = item.model_copy(
            update={"evidence": item.evidence.model_copy(update={"citation": citation})}
        )
    elif kind == "provenance":
        doc.evidence[0] = item.model_copy(
            update={"provenance_class": ProvenanceClass.AI_INTERPRETATION}
        )
    else:
        doc.evidence[0] = item.model_copy(update={"relevance_verified": True})
    row = check_observation(observation)
    assert row["status"] == "FAIL" and expected in row["failure_reasons"]


@pytest.mark.parametrize("counter", v.COUNT_NAMES)
def test_forbidden_downstream_calls_fail(counter):
    observation = observe("C06")
    observation[1].counters.values[counter] = 1
    row = check_observation(observation)
    assert row["status"] == "FAIL", counter


def test_investigation_executed_true_fails():
    observation = list(observe("C06"))
    result = observation[2]
    observation[2] = result.model_copy(
        update={
            "investigation_plan": result.investigation_plan.model_copy(update={"executed": True})
        }
    )
    row = check_observation(observation)
    assert row["status"] == "FAIL" and "INVESTIGATION_EXECUTED" in row["failure_reasons"]


def test_semantic_configuration_is_refused_before_cases(tmp_path):
    runtime = fake_runtime()
    semantic = Mock()
    runtime.service.semantic = semantic
    artifact = run_fake(tmp_path, runtime=runtime)
    assert artifact["preflight"]["status"] == "FAIL"
    semantic.assert_not_called()
    assert artifact["case_results"] == []


def test_full_fake_run_single_search_artifact_and_no_text(tmp_path):
    artifact = run_fake(tmp_path)
    assert artifact["summary"] == {
        "total": 10,
        "passed": 10,
        "failed": 0,
        "not_run": 0,
        "preflight_status": "PASS",
        "overall_status": "PASS",
    }
    for row in artifact["case_results"][2:5]:
        assert row["execution_counters"]["document_search_calls"] == 1
        assert row["execution_counters"]["dense_search_calls"] == 1
    persisted = json.loads((tmp_path / v.ARTIFACT_NAME).read_text("utf-8"))
    assert persisted == artifact
    v.validate_artifact(artifact)
    assert not any(c.query in json.dumps(artifact) for c in PROTOCOL.cases.cases)
    assert "chunk_text" not in json.dumps(artifact)


@pytest.mark.parametrize("preflight,passed", [("PASS", 9), ("FAIL", 10), ("PASS", 0)])
def test_summary_cannot_claim_pass_without_all_ten_and_preflight(preflight, passed):
    rows = [
        {
            "case_id": cid,
            "status": "PASS" if i < passed else "FAIL",
            "failure_reasons": [] if i < passed else ["failure"],
        }
        for i, cid in enumerate(v.EXPECTED_IDS)
    ]
    summary = v.summary(rows, preflight)
    assert summary["overall_status"] == "FAIL"
    artifact = {
        "case_results": rows,
        "preflight": {"status": preflight},
        "summary": summary | {"overall_status": "PASS"},
    }
    with pytest.raises(ValueError, match="summary"):
        v.validate_artifact(artifact)


def test_failed_first_run_is_preserved_and_cannot_rerun(tmp_path):
    runtime = fake_runtime()
    real = runtime.service.handle
    runtime.service.handle = Mock(
        side_effect=lambda *a, **kw: (
            (_ for _ in ()).throw(RuntimeError("sensitive detail"))
            if kw["request_id"].endswith("C03")
            else real(*a, **kw)
        )
    )
    artifact = run_fake(tmp_path, runtime=runtime)
    assert artifact["summary"]["overall_status"] == "FAIL"
    assert runtime.service.handle.call_count == 10  # exactly once each, no retry
    assert artifact["case_results"][2]["failure_reasons"] == ["CASE_EXCEPTION:RuntimeError"]
    before = (tmp_path / v.ARTIFACT_NAME).read_bytes()
    local, live = Mock(), Mock()
    with pytest.raises(FileExistsError):
        v.run_first(
            output=tmp_path / v.ARTIFACT_NAME,
            commit_sha="a" * 40,
            environment={},
            prepare_local=local,
            prepare_runtime=live,
        )
    local.assert_not_called()
    live.assert_not_called()
    assert (tmp_path / v.ARTIFACT_NAME).read_bytes() == before
    assert b"sensitive detail" not in before


def test_failed_preflight_is_also_preserved_and_consumes_first_run(tmp_path):
    run_fake(tmp_path, local=Mock(side_effect=v.PreflightError(["TEST_FAILURE"])))
    with pytest.raises(FileExistsError):
        run_fake(tmp_path)


def test_thin_notebook_is_parseable_and_has_no_duplicated_decisions():
    source = (REPO_ROOT / "notebooks/07e_phase9_contract_validation.py").read_text("utf-8")
    ast.parse(source)
    assert "run_databricks_validation" in source
    assert "Retriever(" not in source and "RoutingService(" not in source
    assert "similarity_search" not in source
    assert "git push" not in source


def test_live_entry_point_refuses_local_environment():
    from worldbank_copilot.common import load_settings

    settings = load_settings("local", env={})
    with pytest.raises(v.ConfigurationError, match="Databricks-only"):
        v.run_databricks_validation(Mock(), settings, commit_sha="a" * 40)


def test_structured_ai_interpretation_is_rejected():
    observation = observe("C01")
    overview = observation[2].tool_results[0].items[0]
    overview.facts[0] = overview.facts[0].model_copy(
        update={"provenance_class": ProvenanceClass.AI_INTERPRETATION}
    )
    row = check_observation(observation)
    assert row["status"] == "FAIL"
    assert "INVALID_PROVENANCE" in row["failure_reasons"]


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("item_scope", "TOOL_ITEM_SCOPE"),
        ("envelope_scope", "TOOL_RESULT_SCOPE"),
        ("tool_version", "TOOL_VERSION"),
    ],
)
def test_structured_identity_violations_fail(kind, expected):
    observation = observe("C01")
    envelope = observation[2].tool_results[0]
    if kind == "item_scope":
        envelope.items[0].project_id = "P179039"
    elif kind == "envelope_scope":
        envelope.project_id = "P179039"
    else:
        envelope.tool_version = "999"
    row = check_observation(observation)
    assert row["status"] == "FAIL" and expected in row["failure_reasons"]


def test_no_evidence_cannot_hide_evidence():
    observation = observe("C03")
    observation[4].status = v.rc.RetrievalStatus.NO_EVIDENCE
    row = check_observation(observation)
    assert row["status"] == "FAIL" and "DOCUMENT_RESULT_CONTRACT" in row["failure_reasons"]


def test_investigation_planned_foreign_call_fails():
    observation = observe("C06")
    plan = observation[2].investigation_plan
    plan.structured_calls[0].arguments["project_id"] = "P179039"
    row = check_observation(observation)
    assert row["status"] == "FAIL" and "INVESTIGATION_TOOL_VALIDATION" in row["failure_reasons"]


def test_failed_preflight_retains_available_local_integrity(tmp_path):
    artifact = run_fake(tmp_path, local=lambda: v.prepare_protocol(REPO_ROOT, dependency_ok=False))
    assert artifact["integrity"]["case_set_sha256_lf"] == PROTOCOL.integrity["case_set_sha256_lf"]
    assert artifact["integrity"]["index_rows"] is None
    assert artifact["summary"]["overall_status"] == "FAIL"


def test_interrupted_run_checkpoints_completed_cases_and_blocks_rerun(tmp_path):
    runtime = fake_runtime()
    real = runtime.service.handle

    def interrupt(*args, **kwargs):
        if kwargs["request_id"].endswith("C04"):
            raise KeyboardInterrupt()
        return real(*args, **kwargs)

    runtime.service.handle = interrupt
    with pytest.raises(KeyboardInterrupt):
        run_fake(tmp_path, runtime=runtime)
    path = tmp_path / v.ARTIFACT_NAME
    before = path.read_bytes()
    artifact = json.loads(before)
    assert [r["case_id"] for r in artifact["case_results"]] == ["C01", "C02", "C03"]
    assert artifact["summary"]["overall_status"] == "FAIL"
    with pytest.raises(FileExistsError):
        run_fake(tmp_path)
    assert path.read_bytes() == before


def test_local_preflight_hash_failure_never_prepares_runtime(tmp_path):
    # A valid but altered C08 query is still outside the frozen preregistration.
    for folder in ("configs", "evaluation"):
        shutil.copytree(REPO_ROOT / folder, tmp_path / folder)
    lock = json.loads((REPO_ROOT / v.LOCK_FILE).read_text("utf-8"))
    for name in (*lock["frozen_files_sha256_lf"], "src/worldbank_copilot/agents/__init__.py"):
        path = tmp_path / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO_ROOT / name, path)
    path = tmp_path / v.CASE_FILE
    data = yaml.safe_load(path.read_text("utf-8"))
    data["cases"][7]["query"] += " Please."
    path.write_text(yaml.safe_dump(data), "utf-8")
    live = Mock()
    artifact = v.run_first(
        output=tmp_path / v.ARTIFACT_NAME,
        commit_sha="a" * 40,
        environment={"kind": "local"},
        prepare_local=lambda: v.prepare_protocol(tmp_path, dependency_ok=True),
        prepare_runtime=live,
    )
    live.assert_not_called()
    assert "CASE_SET_HASH" in artifact["preflight"]["failures"]
    assert artifact["case_results"] == []
