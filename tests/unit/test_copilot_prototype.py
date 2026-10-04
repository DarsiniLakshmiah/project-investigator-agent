"""Offline checks for the prototype scenarios, invariant checker and Databricks factory."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.unit.test_copilot_runtime import Critic, Synthesizer, copilot

from worldbank_copilot.common.exceptions import ConfigurationError
from worldbank_copilot.copilot.contracts import (
    Citation,
    Claim,
    EvidenceItem,
    InvestigationResult,
    ResultStatus,
)
from worldbank_copilot.copilot.databricks import _require_accepted_index, build_copilot
from worldbank_copilot.validation.copilot_prototype import (
    SCENARIOS,
    check_invariants,
    run_scenario,
)


@pytest.mark.parametrize("scenario_id", sorted(SCENARIOS))
def test_scenarios_hold_invariants_offline(scenario_id):
    synthesizer, critic = Synthesizer(), Critic()
    report = run_scenario(copilot(synthesizer, critic), scenario_id)
    assert report["invariants"] == "PASS", report["failed_invariants"]
    calls = len(report["result"]["model_calls"])
    assert calls == (4 if scenario_id == "S2" else 0)  # S2: Investigator x2, Synthesizer, Critic


def test_scenario_questions_are_the_reviewed_routing_cases():
    assert {s.routing_case for s in SCENARIOS.values()} == {"r037", "r051", "r039", "r062"}
    assert {s.project_id for s in SCENARIOS.values()} == {"P130544"}


def result(**fields):
    base = dict(request_id="r", query="q", project_id="P130544", route="STRUCTURED", message="m")
    return InvestigationResult(**{**base, **fields})


def test_invariants_detect_leakage_and_unsupported_publication():
    scenario = SCENARIOS["S1"]
    leaked = result(
        status=ResultStatus.EVIDENCE_ONLY,
        evidence=(EvidenceItem(provenance=("FACT",), payload={"project_id": "P179039"}),),
    )
    assert not check_invariants(leaked, scenario, critic_enabled=True)["no_cross_project_data"]
    claim = Claim(
        claim_id="C1",
        text="x",
        claim_type="ASSERTION",
        provenance="FACT",
        evidence_ids=("ev_missing",),
        citations=(Citation(evidence_id="ev_missing", source_identity="s"),),
    )
    published = result(status=ResultStatus.EVIDENCE_ONLY, claims=(claim,))
    checks = check_invariants(published, scenario, critic_enabled=True)
    assert not checks["claims_only_in_answer"] and not checks["citations_resolve"]
    misrouted = result(status=ResultStatus.REFUSE, route="INVESTIGATION")
    checks = check_invariants(misrouted, scenario, critic_enabled=True)
    assert not checks["route"] and not checks["status_allowed"]


def test_s2_with_a_failed_critic_holds_invariants_but_never_as_reviewed():
    from worldbank_copilot.investigation.claims import Failure, NodeError

    failing = Critic(error=NodeError(Failure.MODEL_OUTPUT_INVALID))
    report = run_scenario(copilot(Synthesizer(), failing), "S2")
    assert report["invariants"] == "PASS", report["failed_invariants"]
    assert report["result"]["validation"]["critic_status"] == "FAILED"
    assert report["result"]["status"] == "ANSWER"


def test_factory_requires_databricks():
    settings = SimpleNamespace(environment=SimpleNamespace(value="local"))
    with pytest.raises(ConfigurationError):
        build_copilot(spark=None, settings=settings)


def test_index_identity_check_matches_accepted_profile():
    protocol = SimpleNamespace(
        profile=SimpleNamespace(vector_search_endpoint="ep"),
        manifest={"retrieval": {"index": {"index_rows": 10, "corpus_rows": 10}}},
    )
    from worldbank_copilot.validation.phase9_contract import INDEX_NAME

    good = {
        "endpoint_name": "ep",
        "name": INDEX_NAME,
        "status": {"indexed_row_count": 10, "ready": True},
    }
    _require_accepted_index(good, 10, protocol)
    for bad, rows in (
        ({**good, "status": {"indexed_row_count": 10, "ready": False}}, 10),
        ({**good, "name": "other"}, 10),
        (good, 9),
    ):
        with pytest.raises(ConfigurationError):
            _require_accepted_index(bad, rows, protocol)


# -- dependency plan: never replace the runtime's protected mlflow-skinny -------------
REPO = Path(__file__).resolve().parents[2]
MLFLOW_DISTRIBUTIONS = {"mlflow", "mlflow-skinny", "mlflow-tracing"}


def pip_plan(notebook):
    """(requirement files, constraint files) of a notebook's single %pip install line."""
    lines = [
        line
        for line in (REPO / "notebooks" / notebook).read_text(encoding="utf-8").splitlines()
        if "%pip install" in line
    ]
    assert len(lines) == 1, notebook
    tokens = lines[0].split()
    files = {flag: [] for flag in ("-r", "-c")}
    for flag, value in zip(tokens, tokens[1:], strict=False):
        if flag in files:
            files[flag].append(value.removeprefix("../"))
    return set(files["-r"]), set(files["-c"])


def requirement_names(path):
    names = set()
    for raw in (REPO / path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            names.add(re.split(r"[<>=!~;\[ ]", line, maxsplit=1)[0].lower().replace("_", "-"))
    return names


def test_prototype_installs_only_databricks_validated_requirement_sets():
    requirements, constraints = pip_plan("14_copilot_prototype_validation.py")
    accepted_10c, _ = pip_plan("09_phase10c_evidence_validation.py")
    accepted_10d, _ = pip_plan("09_phase10d_model_validation.py")
    assert requirements == accepted_10c | accepted_10d
    assert "requirements-copilot-runtime.txt" not in requirements
    assert constraints == {"constraints-databricks.txt"}


def test_prototype_requirements_never_name_mlflow():
    requirements, _ = pip_plan("14_copilot_prototype_validation.py")
    for path in requirements:
        assert not requirement_names(path) & MLFLOW_DISTRIBUTIONS, path
    # The excluded file is exactly what upgrades the protected runtime package.
    assert "mlflow" in requirement_names("requirements-copilot-runtime.txt")


def test_health_check_covers_exactly_what_the_notebook_installs():
    from worldbank_copilot.copilot.databricks import REQUIREMENT_FILES

    requirements, _ = pip_plan("14_copilot_prototype_validation.py")
    assert set(REQUIREMENT_FILES) == requirements


def test_mlflow_skinny_remains_protected_by_policy():
    import yaml

    policy = yaml.safe_load((REPO / "configs/environments/dependencies.yaml").read_text("utf-8"))
    assert "mlflow-skinny" in policy["protected"]


def fake_mlflow(monkeypatch, *, span_attrs=("trace_id", "set_attributes"), module_attrs=None):
    import sys

    live_span = type("LiveSpan", (), {name: None for name in span_attrs})
    module = SimpleNamespace(
        __version__="3.12.0",
        entities=SimpleNamespace(LiveSpan=live_span),
        **{n: (lambda *a, **k: None) for n in (module_attrs or ("start_span", "set_experiment"))},
    )
    monkeypatch.setitem(sys.modules, "mlflow", module)
    monkeypatch.setitem(sys.modules, "mlflow.entities", module.entities)


def test_tracing_api_check_accepts_runtime_api(monkeypatch):
    from worldbank_copilot.copilot.databricks import require_tracing_api

    fake_mlflow(monkeypatch)
    require_tracing_api()


@pytest.mark.parametrize(
    "kwargs",
    [{"module_attrs": ("set_experiment",)}, {"span_attrs": ("set_attributes",)}],
    ids=["no-start_span", "no-trace_id"],
)
def test_tracing_api_check_fails_fast(monkeypatch, kwargs):
    from worldbank_copilot.copilot.databricks import require_tracing_api

    fake_mlflow(monkeypatch, **kwargs)
    with pytest.raises(ConfigurationError, match="MLFLOW_TRACING_API_UNAVAILABLE"):
        require_tracing_api()
