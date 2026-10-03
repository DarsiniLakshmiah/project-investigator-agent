"""Offline checks for the prototype scenarios, invariant checker and Databricks factory."""

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
    assert calls == (2 if scenario_id == "S2" else 0)


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
