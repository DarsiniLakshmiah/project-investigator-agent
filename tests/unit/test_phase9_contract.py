"""Phase 9F: the frozen Phase 10 execution contract.

Routing execution mapping (9D Candidate A), the profile-owned retrieval contract, the
evidence-status and provenance rules, the investigation handoff, frozen-artifact
protection, and a closure manifest that is checked against the code and artifacts.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from pydantic import BaseModel, ValidationError
from tests.conftest import REPO_CONFIG_DIR, REPO_ROOT
from tests.support.retrieval_golden import OverlapCrossEncoder, ScopedDense, TextEmbeddings
from tests.support.routing_fixtures import ALL, RS, chunk_rows, harness
from tests.support.tool_fixtures import IPF, OTHER, PFORR, context

from worldbank_copilot.common.frozen import (
    FrozenArtifactError,
    artifact_status,
    assert_not_frozen,
)
from worldbank_copilot.retrieval import contract as rc
from worldbank_copilot.retrieval.rerank_policy import NeverRerank
from worldbank_copilot.retrieval.retriever import ChunkStore, Retriever
from worldbank_copilot.routing import execution as ex
from worldbank_copilot.routing.config import load_routing_config
from worldbank_copilot.routing.models import (
    PROVENANCE_REQUIREMENTS,
    Route,
    TemporalKind,
    TemporalScope,
    TemporalStatus,
)
from worldbank_copilot.routing.semantic_eval import canonical_sha256
from worldbank_copilot.routing.service import RoutingService
from worldbank_copilot.tools import evidence as ev
from worldbank_copilot.tools.documents import DocumentEvidence, DocumentSearchConfig
from worldbank_copilot.tools.models import (
    Fact,
    ProvenanceClass,
    ToolResult,
    ToolStatus,
    fact,
)
from worldbank_copilot.tools.registry import TOOL_SPECS, default_executor

MANIFEST = yaml.safe_load((REPO_CONFIG_DIR / "phase9_closure.yaml").read_text("utf-8"))
EVAL = REPO_ROOT / "evaluation"
PROFILE = rc.load_quality_baseline(REPO_CONFIG_DIR, RS)


# -- A. routing execution mapping ------------------------------------------------------------


def test_every_router_route_maps_to_exactly_one_execution_mode():
    assert set(ex.ROUTE_MODES) == set(Route)
    assert set(ex.ROUTE_MODES.values()) == set(ex.ExecutionMode)


@pytest.mark.parametrize(
    ("question", "active", "mode", "reason"),
    [
        (
            "What was the implementation progress rating in ISR 2?",
            IPF,
            "EXECUTE_STRUCTURED",
            None,
        ),
        ("Why was the closing date extended?", PFORR, "EXECUTE_DOCUMENT", None),
        ("What changed and why?", PFORR, "PLAN_INVESTIGATION", None),
        ("What is the current closing date?", None, "CLARIFY", "PROJECT_REQUIRED"),
        ("List the restructurings since the restructuring", IPF, "CLARIFY", "AMBIGUOUS_TIME"),
        ("What is the overall risk rating?", IPF, "CLARIFY", "TIME_REQUIRED"),
        ("What was the PDO rating in 2021?", IPF, "CLARIFY", "TEMPORAL_NOT_SUPPORTED"),
        ("Will the project fail?", IPF, "REFUSE", "PREDICTION_NOT_SUPPORTED"),
        ("What is the closing date of P123456?", IPF, "REFUSE", "UNSUPPORTED_PROJECT"),
    ],
)
def test_router_results_map_deterministically(question, active, mode, reason):
    result = harness().ask(question, active=active)
    decision = ex.execution_decision(result)
    assert decision.mode == mode and decision.translated is False
    assert decision.router_route == result.decision.route
    assert decision.reason_code == (reason or result.decision.reason_code)
    assert ex.execution_decision(result) == decision  # deterministic


def test_semantic_required_becomes_clarify_without_rewriting_the_router_output():
    h = harness()
    result = h.ask("How has the way citizens can register complaints been improved?")
    recorded = result.decision.model_dump()
    decision = ex.execution_decision(result)
    assert (decision.mode, decision.reason_code) == ("CLARIFY", "INTENT_NOT_RESOLVED")
    assert decision.translated is True and decision.policy == "9D_CANDIDATE_A"
    # router output is untouched and still the recorded 9B.2 value
    assert result.decision.route == Route.SEMANTIC_CLASSIFICATION_REQUIRED
    assert decision.router_reason_code == "INTENT_NOT_RESOLVED_BY_RULES"
    assert result.decision.model_dump() == recorded
    # no semantic model exists or ran; nothing executed
    assert h.service.semantic is None and result.semantic is None
    assert h.executor.calls == [] and h.retriever.first_stage_calls == []


def test_semantic_fallback_is_disabled_by_default_and_rejected_by_the_contract():
    assert RoutingService.__dataclass_fields__["semantic"].default is None
    result = harness().ask("How has the way citizens can register complaints been improved?")
    with pytest.raises(ex.ExecutionContractError, match="semantic fallback is disabled"):
        ex.execution_decision(result.model_copy(update={"semantic": _semantic_decision()}))


def _semantic_decision():
    from worldbank_copilot.routing.models import SemanticDecision

    return SemanticDecision(classifier="x", version="1", abstain=True, reason="test")


def test_execution_contract_fails_closed_outside_the_contract():
    h = harness()
    clarify = h.ask("What is the overall risk rating?")
    with pytest.raises(ex.ExecutionContractError, match="executed before"):
        ex.execution_decision(clarify.model_copy(update={"executed_tools": ["get_risk_register"]}))
    abstain = clarify.model_copy(
        update={"decision": clarify.decision.model_copy(update={"reason_code": "SEMANTIC_ABSTAIN"})}
    )
    with pytest.raises(ex.ExecutionContractError, match="not in the contract"):
        ex.execution_decision(abstain)
    plan = h.ask("What changed and why?", active=PFORR)
    with pytest.raises(ex.ExecutionContractError, match="unexecuted plan"):
        ex.execution_decision(plan.model_copy(update={"investigation_plan": None}))


def test_router_and_service_do_not_depend_on_the_execution_mapping():
    for module in ("router.py", "service.py"):
        text = (REPO_ROOT / "src/worldbank_copilot/routing" / module).read_text("utf-8")
        assert "execution" not in text.replace("ExecutionOutcome", "").replace("execute", ""), (
            module
        )
    source = (REPO_ROOT / "src/worldbank_copilot/routing/execution.py").read_text("utf-8")
    for forbidden in (
        "IntentEngine",
        "resolve_project",
        "parse_temporal",
        "classify(",
        "Retriever",
    ):
        assert forbidden not in source, forbidden


def test_router_clarification_codes_are_all_in_the_contract():
    text = (REPO_ROOT / "src/worldbank_copilot/routing/router.py").read_text("utf-8")
    for code in ex.CLARIFICATION_REASON_CODES - {ex.INTENT_NOT_RESOLVED}:
        assert f'"{code}"' in text, code


# -- B. retrieval contract ----------------------------------------------------------------------


class EmptyDense:
    def search(self, vector, filters, k):
        return []


class FailingDense:
    def search(self, vector, filters, k):
        raise RuntimeError("AI Search unavailable")


def baseline_reranker():
    reranker = OverlapCrossEncoder()
    reranker.config = RS.retrieval.reranker
    return reranker


def baseline_embeddings():
    provider = TextEmbeddings()
    provider.model, provider.dimension = RS.embeddings.endpoint, RS.embeddings.expected_dimension
    return provider


def retrieval(dense=None, rows=None):
    rows = rows or chunk_rows()
    retriever = Retriever(
        ChunkStore(rows), RS, list(ALL), baseline_embeddings(), dense or ScopedDense(rows)
    )
    search = rc.build_document_search(PROFILE, retriever, baseline_reranker())
    return rc.DocumentRetrieval(PROFILE, search), retriever


def ask(question="Why was the closing date extended?", project=IPF, authorized=ALL, **kw):
    service, retriever = retrieval(kw.pop("dense", None))
    request = rc.RetrievalRequest(project_id=project, query=question, **kw)
    return service.retrieve(request, context(), authorized_projects=authorized), retriever


def test_profile_is_the_approved_quality_baseline_and_not_production():
    assert PROFILE.model_dump() | {"role": None} == {
        "name": "phase8_quality_baseline",
        "version": "1",
        "role": None,
        "chunk_strategy": "fixed",
        "chunk_strategy_version": RS.chunking.version("fixed"),
        "method": "hybrid",
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "min_rerank_score": None,
        "query_config_sha256": "247a87e4670efd7a8621224294ed256c3717f90d7487751258bbf44f753fbcec",
        "vector_search_endpoint": "worldbank-gep-ai-search",
        "vector_search_index": "document_chunk_index_qwen3_v1",
        "rrf_k": 60,
        "candidate_k": 50,
        "final_k": 5,
        "rerank_policy": "always",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "reranker_max_length": 512,
        "reranker_batch_size": 32,
        "embedding_endpoint": "databricks-qwen3-embedding-0-6b",
        "embedding_dimension": 1024,
        "use_type_filters": False,
        "temporal_filters_applied": False,
        "production": False,
    }
    with pytest.raises(ValidationError):
        PROFILE.candidate_k = 10  # immutable


def test_profile_fails_closed_when_it_disagrees_with_the_configuration():
    drifted = RS.model_copy(
        update={
            "retrieval": RS.retrieval.model_copy(
                update={"hybrid": RS.retrieval.hybrid.model_copy(update={"rrf_k": 61})}
            )
        }
    )
    with pytest.raises(rc.ConfigurationError, match="rrf_k"):
        rc.load_quality_baseline(REPO_CONFIG_DIR, drifted)


def test_request_requires_a_project_and_refuses_every_tunable_parameter():
    with pytest.raises(ValidationError):
        rc.RetrievalRequest(query="closing date")
    with pytest.raises(ValidationError):
        rc.RetrievalRequest(project_id="not-a-project", query="closing date")
    for tunable in (
        "top_k",
        "final_k",
        "candidate_k",
        "rrf_k",
        "reranker",
        "embedding_model",
        "chunk_strategy",
        "method",
        "min_rerank_score",
        "policy",
        "strategy",
        "thresholds",
    ):
        with pytest.raises(ValidationError):
            rc.RetrievalRequest(project_id=IPF, query="closing date", **{tunable: 1})
    with pytest.raises(ValidationError):
        rc.RetrievalRequest(project_id=IPF, query="x", require_citations=False)


def test_ok_result_keeps_provenance_and_equals_the_phase8_retrieval():
    result, retriever = ask()
    assert result.status == "OK" and result.profile == "phase8_quality_baseline@1"
    assert result.top_k == 5 and 0 < len(result.evidence) <= 5
    assert result.rerank_decision.policy == "always" and result.rerank_decision.rerank
    assert result.filters_applied["project_id"] == IPF
    assert result.filters_applied["chunk_strategy"] == "fixed"
    assert result.candidate_count >= len(result.evidence)
    for item in result.evidence:
        assert item.provenance_class == ProvenanceClass.DOCUMENTED_FINDING
        assert item.relevance_verified is False
        assert item.evidence.project_id == IPF and item.evidence.citation.pages
        assert item.evidence.source_hash and item.evidence.rerank_score is not None
    direct = retriever.retrieve(
        "Why was the closing date extended?",
        IPF,
        strategy="fixed",
        method="hybrid",
        reranker=OverlapCrossEncoder(),
        candidate_k=50,
        final_k=5,
    )
    assert [e.evidence.model_dump(mode="json") for e in result.evidence] == [
        e.model_dump(mode="json") for e in direct.evidence
    ]


@pytest.mark.parametrize(
    ("project", "authorized", "question"),
    [
        (PFORR, (IPF,), "Why was the closing date extended?"),  # not authorised
        ("P000000", ALL, "Why was the closing date extended?"),  # not in the corpus
        (IPF, ALL, f"What did {PFORR} report about the closing date?"),  # names a foreign project
    ],
)
def test_project_isolation_survives_the_adapter(project, authorized, question):
    result, retriever = ask(question, project=project, authorized=authorized)
    assert result.status == "SCOPE_REFUSED" and result.evidence == [] and result.error


def test_zero_candidates_is_no_evidence_never_corpus_absence():
    result, _ = ask("zebra quantum unicorn", dense=EmptyDense())
    assert result.status == "NO_EVIDENCE" and result.evidence == []
    assert rc.NO_EVIDENCE_MEANING in result.warnings
    assert "not found in the retrieved candidates" in rc.NO_EVIDENCE_MEANING
    assert MANIFEST["retrieval"]["no_evidence_meaning"].startswith(
        "Relevant evidence was not found in the retrieved candidates"
    )


def test_retrieval_failure_is_retrieval_error_not_evidence():
    result, _ = ask(dense=FailingDense())
    assert result.status == "RETRIEVAL_ERROR" and result.evidence == []
    assert "AI Search unavailable" in result.error


def test_hints_are_recorded_but_never_reported_as_applied():
    scope = TemporalScope(
        kind=TemporalKind.YEAR, status=TemporalStatus.RESOLVED, explicit=True, defaulted=False
    )
    hinted, _ = ask(temporal_scope=scope, document_type_hints=("RESTRUCTURING_PAPER",))
    plain, _ = ask()
    assert hinted.recorded_hints.applied is False
    assert hinted.recorded_hints.document_type_hints == ("RESTRUCTURING_PAPER",)
    assert hinted.recorded_hints.temporal_scope == scope
    assert hinted.warnings[0] == rc.HINTS_NOT_APPLIED
    assert hinted.filters_applied == plain.filters_applied  # nothing was filtered by the hints
    assert hinted.filters_applied["document_types"] == []
    assert hinted.filters_applied["date_from"] is None
    assert [e.evidence.chunk_id for e in hinted.evidence] == [
        e.evidence.chunk_id for e in plain.evidence
    ]
    assert plain.recorded_hints.applied is False and rc.HINTS_NOT_APPLIED not in plain.warnings


def test_callers_cannot_retune_the_profile_through_the_search():
    service, retriever = retrieval()
    with pytest.raises(rc.ConfigurationError, match="rerank policy"):
        rc.DocumentRetrieval(PROFILE, replace(service.search, policy=NeverRerank()))
    with pytest.raises(rc.ConfigurationError, match="candidate_k"):
        rc.DocumentRetrieval(
            PROFILE,
            replace(
                service.search,
                config=DocumentSearchConfig(
                    chunk_strategy="fixed", method="hybrid", candidate_k=10, final_k=5
                ),
            ),
        )
    service.search.config = DocumentSearchConfig(
        chunk_strategy="fixed", method="hybrid", candidate_k=10, final_k=20
    )  # tampered after construction
    result = service.retrieve(
        rc.RetrievalRequest(project_id=IPF, query="closing date"),
        context(),
        authorized_projects=ALL,
    )
    assert result.status == "RETRIEVAL_ERROR" and result.evidence == []


# Every mutation is applied at the actual supported replacement point. No model runs.
MUTATIONS = (
    ("threshold", "min_rerank_score"),
    ("embedding_endpoint", "embedding_endpoint"),
    ("embedding_dimension", "embedding_dimension"),
    ("provider_config_endpoint", "embedding_provider_endpoint"),
    ("provider_config_dimension", "embedding_provider_config_dimension"),
    ("reranker_name", "reranker_name"),
    ("missing_reranker_config", "reranker_model"),
    ("provider_model", "embedding_provider_model"),
    ("provider_dimension", "embedding_provider_dimension"),
    ("transport_endpoint", "embedding_transport_endpoint"),
    ("reranker_model", "reranker_model"),
    ("reranker_max_length", "reranker_max_length"),
    ("reranker_batch_size", "reranker_batch_size"),
    ("loaded_max_length", "loaded_reranker_max_length"),
    ("rrf_k", "rrf_k"),
    ("candidate_k", "candidate_k"),
    ("policy", "rerank policy"),
    ("final_k", "final_k"),
    ("bm25_k1", "bm25_k1"),
    ("bm25_b", "bm25_b"),
    ("cached_bm25", "cached_bm25_k1"),
    ("chunk_version", "chunk_strategy_version"),
    ("strategy", "chunk_strategy"),
    ("method", "method"),
    ("profile_id", "profile_id"),
    ("production", "production_selection"),
    ("production_final_k", "production_final_k"),
    ("query", "query_config_sha256"),
    ("vector_endpoint", "vector_search_endpoint"),
    ("dense_endpoint", "dense_endpoint"),
    ("dense_index", "dense_index"),
    ("dense_dimension", "dense_dimension"),
)


def mutate(search, kind):
    r, rs = search.retriever, search.retriever.settings
    if kind in ("candidate_k", "final_k", "strategy", "method"):
        field, value = {
            "candidate_k": ("candidate_k", 10),
            "final_k": ("final_k", 20),
            "strategy": ("chunk_strategy", "structure"),
            "method": ("method", "dense"),
        }[kind]
        search.config = replace(search.config, **{field: value})
    elif kind == "policy":
        search.policy = NeverRerank()
    elif kind == "profile_id":
        search.profile_id = "other@1"
    elif kind == "reranker_name":
        search.cross_encoder.name = "none"
    elif kind == "missing_reranker_config":
        search.cross_encoder.config = None
    elif kind.startswith("provider_config_"):
        field = "endpoint" if kind.endswith("endpoint") else "expected_dimension"
        r.embeddings.config = rs.embeddings.model_copy(
            update={field: "other-endpoint" if field == "endpoint" else 512}
        )
    elif kind.startswith("reranker_"):
        field = kind.removeprefix("reranker_")
        search.cross_encoder.config = search.cross_encoder.config.model_copy(
            update={field: "other-model" if field == "model" else 256}
        )
    elif kind == "loaded_max_length":
        search.cross_encoder._model = SimpleNamespace(max_length=256)
    elif kind.startswith("provider_"):
        setattr(
            r.embeddings,
            kind.removeprefix("provider_"),
            "other-model" if kind == "provider_model" else 512,
        )
    elif kind == "transport_endpoint":
        r.embeddings.transport = SimpleNamespace(endpoint="other-endpoint")
    elif kind.startswith("embedding_"):
        field = "endpoint" if kind == "embedding_endpoint" else "expected_dimension"
        r.settings = rs.model_copy(
            update={
                "embeddings": rs.embeddings.model_copy(
                    update={field: "other-endpoint" if field == "endpoint" else 512}
                )
            }
        )
    elif kind == "chunk_version":
        r.settings = rs.model_copy(
            update={
                "chunking": rs.chunking.model_copy(update={"min_chars": rs.chunking.min_chars + 1})
            }
        )
    elif kind == "query":
        r.settings = rs.model_copy(update={"query": rs.query.model_copy(update={"acronyms": {}})})
    elif kind == "cached_bm25":
        r.store._lexical["test"] = SimpleNamespace(k1=2, b=PROFILE.bm25_b)
    elif kind.startswith("dense_"):
        field = kind.removeprefix("dense_")
        if field == "endpoint":
            r.dense.config = rs.retrieval.vector_search.model_copy(
                update={"endpoint": "other-endpoint"}
            )
        else:
            setattr(
                r.dense,
                "index_name" if field == "index" else field,
                "other-index" if field == "index" else 512,
            )
    else:
        cfg = rs.retrieval
        if kind == "threshold":
            update = {"min_rerank_score": 999.0}
        elif kind == "rrf_k":
            update = {"hybrid": cfg.hybrid.model_copy(update={"rrf_k": 61})}
        elif kind.startswith("bm25_"):
            update = {"lexical": cfg.lexical.model_copy(update={kind.removeprefix("bm25_"): 2})}
        elif kind == "vector_endpoint":
            update = {
                "vector_search": cfg.vector_search.model_copy(update={"endpoint": "other-endpoint"})
            }
        else:
            update = {
                "production": cfg.production.model_copy(
                    update={"retrieval": "hybrid"} if kind == "production" else {"final_k": 10}
                )
            }
        r.settings = rs.model_copy(update={"retrieval": cfg.model_copy(update=update)})


def execution_spies(service):
    r = service.search.retriever
    # Dense search is the AI Search boundary; the fixture uses its local stand-in.
    spies = [
        (r, "first_stage"),
        (r, "candidates"),
        (r.store, "lexical"),
        (r.dense, "search"),
        (r.embeddings, "embed"),
        (service.search.cross_encoder, "score"),
        (service.executor, "run"),
    ]
    calls = []
    for obj, name in spies:
        spy = Mock(wraps=getattr(obj, name))
        setattr(obj, name, spy)
        calls.append(spy)
    return calls


@pytest.mark.parametrize(("kind", "field"), MUTATIONS)
def test_runtime_integrity_mutations_fail_before_any_execution(kind, field):
    service, _ = retrieval()
    spies = execution_spies(service)
    mutate(service.search, kind)
    result = service.retrieve(
        rc.RetrievalRequest(project_id=IPF, query="closing date"),
        context(),
        authorized_projects=ALL,
    )
    assert result.status == "RETRIEVAL_ERROR" and result.evidence == []
    assert field in result.error
    for spy in spies:
        spy.assert_not_called()


@pytest.mark.parametrize(("kind", "field"), MUTATIONS)
def test_construction_rejects_the_same_runtime_integrity_mutations(kind, field):
    service, _ = retrieval()
    spies = execution_spies(service)
    mutate(service.search, kind)
    with pytest.raises(rc.ConfigurationError, match=field):
        rc.DocumentRetrieval(PROFILE, service.search)
    for spy in spies:
        spy.assert_not_called()


def test_builder_rejects_embedding_configuration_drift():
    service, r = retrieval()
    mutate(service.search, "embedding_endpoint")
    with pytest.raises(rc.ConfigurationError, match="embedding_endpoint"):
        rc.build_document_search(PROFILE, r, baseline_reranker())


def test_valid_baseline_executes_all_retrieval_stages():
    service, _ = retrieval()
    spies = execution_spies(service)
    result = service.retrieve(
        rc.RetrievalRequest(project_id=IPF, query="closing date"),
        context(),
        authorized_projects=ALL,
    )
    assert result.status == "OK"
    assert result.profile == "phase8_quality_baseline@1"
    for spy in spies:
        assert spy.call_count > 0


def test_result_contract_rejects_fabricated_or_foreign_evidence():
    ok, _ = ask()
    item = ok.evidence[0]
    base = ok.model_dump(exclude={"evidence", "status"})
    with pytest.raises(ValidationError):  # OK without evidence
        rc.RetrievalResult(status="OK", evidence=[], **base)
    with pytest.raises(ValidationError):  # NO_EVIDENCE carrying evidence
        rc.RetrievalResult(status="NO_EVIDENCE", evidence=[item], **base)
    with pytest.raises(ValidationError):  # foreign project evidence
        rc.RetrievalResult(status="OK", evidence=[item], **(base | {"project_id": OTHER}))
    for update in (
        {"provenance_class": ProvenanceClass.AI_INTERPRETATION},
        {"provenance_class": ProvenanceClass.FACT},
        {"relevance_verified": True},
    ):
        with pytest.raises(ValidationError):
            rc.RetrievalResult(status="OK", evidence=[item.model_copy(update=update)], **base)


# -- C. provenance and evidence status ----------------------------------------------------------


def test_tool_statuses_map_exhaustively_and_never_to_synthesis_states():
    assert set(ev.TOOL_STATUS_MAP) == set(ToolStatus)
    assert not set(ev.TOOL_STATUS_MAP.values()) & ev.SYNTHESIS_ONLY
    assert {s.value for s in rc.RetrievalStatus} <= {s.value for s in ev.EvidenceStatus}
    contract = MANIFEST["phase10_contract"]["evidence_statuses"]
    assert set(contract["phase10_synthesis_only"]) == {s.value for s in ev.SYNTHESIS_ONLY}
    assert set(contract["produced_by_tools_and_retrieval"]) == {
        s.value for s in ev.EvidenceStatus
    } - {s.value for s in ev.SYNTHESIS_ONLY}
    for status, expected in (
        (ToolStatus.EMPTY, "NO_EVIDENCE"),
        (ToolStatus.NOT_COVERED, "NO_EVIDENCE"),
        (ToolStatus.ERROR, "TOOL_ERROR"),
        (ToolStatus.TIMEOUT, "TOOL_ERROR"),
        (ToolStatus.DATA_INTEGRITY_ERROR, "TOOL_ERROR"),
    ):
        assert ev.tool_evidence_status(_envelope(status)) == expected


def _envelope(status, tool="get_risk_register", items=()):
    return ToolResult(
        tool=tool,
        tool_version="1",
        request_id="r",
        project_id=IPF,
        status=status,
        items=list(items),
        error="boom" if status in (ToolStatus.ERROR, ToolStatus.TIMEOUT) else None,
    )


def test_structured_tools_keep_signals_to_phase7_rule_outputs():
    executor, seen = default_executor(), {}
    for spec in TOOL_SPECS:
        if not spec.tables:
            continue
        res = executor.run(spec.name, {"project_id": IPF}, context(), scope_project_id=IPF)
        assert ev.provenance_violations(res) == [], res.tool
        assert ev.tool_evidence_status(res) != "TOOL_ERROR", (res.tool, res.status, res.error)
        seen[spec.name] = {
            label
            for label, cls in ev._classified(res.items, "items")
            if cls == ProvenanceClass.SYSTEM_DERIVED_SIGNAL
        }
    assert seen["get_attention_signals"]  # non-vacuous: signals do occur
    assert seen["get_project_overview"] and seen["get_project_overview"] <= ev.OVERVIEW_SIGNAL_FACTS
    assert all(not v for k, v in seen.items() if k not in (ev.SIGNAL_TOOL, ev.OVERVIEW_TOOL))


class _Item(BaseModel):
    value: Fact


def test_a_signal_or_interpretation_outside_the_rules_is_a_tool_error():
    signal = fact("days_extended", 3, ProvenanceClass.SYSTEM_DERIVED_SIGNAL)
    res = _envelope(ToolStatus.OK, items=[_Item(value=signal)])
    assert ev.provenance_violations(res) and ev.tool_evidence_status(res) == "TOOL_ERROR"
    with pytest.raises(ValidationError, match="AI_INTERPRETATION"):
        _envelope(
            ToolStatus.OK,
            items=[_Item(value=fact("x", 1, ProvenanceClass.AI_INTERPRETATION))],
        )


def test_unknown_always_keeps_its_reason():
    unknown = fact("disbursed_usd", None, ProvenanceClass.FACT)
    assert unknown.provenance_class == ProvenanceClass.UNKNOWN and unknown.unknown_reason
    with pytest.raises(ValidationError):
        Fact(name="x", provenance_class=ProvenanceClass.UNKNOWN)


def test_document_evidence_defaults_to_documented_finding():
    assert DocumentEvidence.model_fields["provenance_class"].default == (
        ProvenanceClass.DOCUMENTED_FINDING
    )


# -- D. investigation handoff --------------------------------------------------------------------


def test_investigation_plan_records_the_handoff_and_executes_nothing():
    h = harness()
    h.retriever.embeddings = baseline_embeddings()
    h._documents = rc.build_document_search(PROFILE, h.retriever, baseline_reranker())
    result = h.ask("What changed and why?", active=PFORR)
    plan = result.investigation_plan
    assert ex.execution_decision(result).mode == "PLAN_INVESTIGATION"
    assert plan.executed is False and plan.clarification_state == "NONE"
    assert plan.retrieval_profile == "phase8_quality_baseline@1"
    assert plan.provenance_requirements == PROVENANCE_REQUIREMENTS
    assert h.executor.calls == [] and h.retriever.first_stage_calls == []
    assert result.tool_results == [] and not result.retrieval_executed
    request = rc.RetrievalRequest.from_spec(PFORR, plan.document_retrievals[0])
    assert request.query == "What changed and why?" and request.require_citations


def test_plan_without_a_configured_profile_records_none():
    plan = harness().ask("What changed and why?", active=PFORR, documents=False).investigation_plan
    assert plan.retrieval_profile is None and plan.executed is False


# -- E. frozen artifacts ------------------------------------------------------------------------


def test_frozen_guard(tmp_path):
    frozen, provisional, broken = (tmp_path / n for n in ("f.yaml", "p.yaml", "b.yaml"))
    frozen.write_text("status: FROZEN\n", "utf-8")
    provisional.write_text("status: PROVISIONAL\n", "utf-8")
    broken.write_text("status: [unclosed\n", "utf-8")
    with pytest.raises(FrozenArtifactError, match="FROZEN"):
        assert_not_frozen(frozen)
    assert_not_frozen(provisional)
    assert_not_frozen(tmp_path / "missing.yaml")
    with pytest.raises(FrozenArtifactError, match="unreadable"):
        assert_not_frozen(broken)


def test_semantic_dev_script_refuses_to_overwrite_the_frozen_9d_decision(monkeypatch):
    decision = EVAL / "semantic_config_9d.yaml"
    assert artifact_status(decision) == "FROZEN"
    before = {p: p.read_bytes() for p in EVAL.glob("semantic_*9d*")}

    def no_write(*args, **kwargs):
        raise AssertionError("the script tried to write a frozen artifact")

    monkeypatch.setattr(Path, "write_text", no_write)
    spec = importlib.util.spec_from_file_location(
        "semantic_dev_9d", REPO_ROOT / "scripts/semantic_dev_9d.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(FrozenArtifactError, match="semantic_config_9d.yaml is FROZEN"):
        module.main()
    assert {p: p.read_bytes() for p in EVAL.glob("semantic_*9d*")} == before


# -- F. closure manifest ------------------------------------------------------------------------


def test_manifest_routing_matches_code_config_and_frozen_artifacts():
    r = MANIFEST["routing"]
    routing = load_routing_config(REPO_CONFIG_DIR)
    freeze = json.loads((EVAL / "routing_freeze_9c.json").read_text("utf-8"))
    assert r["router_version"] == routing.versions["router"] == freeze["router_version"]
    assert r["execution_policy"] == ex.EXECUTION_POLICY
    assert r["semantic_required_execution"] == {
        "mode": "CLARIFY",
        "reason_code": ex.INTENT_NOT_RESOLVED,
    }
    assert set(r["clarification_reason_codes"]) == ex.CLARIFICATION_REASON_CODES
    assert r["semantic_fallback"] == "disabled"
    c9 = r["evaluation_9c"]
    assert c9["dataset_sha256_lf"] == freeze["dataset_sha256_lf"]
    assert c9["dataset_sha256_lf"] == canonical_sha256(REPO_ROOT / c9["dataset"])
    assert c9["case_count"] == freeze["case_count"] == 80
    d9 = r["decision_9d"]
    frozen = yaml.safe_load((REPO_ROOT / d9["frozen_config"]).read_text("utf-8"))
    assert d9["frozen_config_sha256_lf"] == canonical_sha256(REPO_ROOT / d9["frozen_config"])
    assert frozen["status"] == d9["status"] == "FROZEN" and frozen["test_evaluated"] is False
    assert frozen["selected_strategy"] == d9["selected_strategy"]
    assert frozen["config"] is None and d9["semantic_fallback_promoted"] is False


def test_manifest_retrieval_matches_configuration_and_validated_results():
    m = MANIFEST["retrieval"]
    retrieval_yaml = yaml.safe_load(
        (REPO_CONFIG_DIR / "retrieval" / "retrieval.yaml").read_text("utf-8")
    )
    adaptive = yaml.safe_load(
        (REPO_CONFIG_DIR / "retrieval" / "adaptive_rerank.yaml").read_text("utf-8")
    )
    assert rc.RetrievalProfile.model_validate(m["phase10_quality_baseline"]) == PROFILE
    assert retrieval_yaml["production"] == {
        "chunk_strategy": None,
        "retrieval": None,
        "reranker": None,
        "candidate_k": None,
        "final_k": 5,
    }
    assert m["production_config"]["production"] is None
    idx = m["index"]
    assert idx["endpoint"] == retrieval_yaml["vector_search"]["endpoint"]
    assert idx["endpoint"] == adaptive["expected"]["endpoint"]
    assert idx["index_name"] == retrieval_yaml["vector_search"]["index_name"]
    assert idx["index_name"] == adaptive["expected"]["index_name"]
    assert (idx["index_rows"], idx["corpus_rows"]) == (
        adaptive["expected"]["index_rows"],
        adaptive["expected"]["corpus_rows"],
    )
    p1 = json.loads((EVAL / "adaptive_rerank_9e.json").read_text("utf-8"))["points"]["P1"]
    metrics = m["validated_metrics"]
    for key in (
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "ndcg_at_5",
        "composed_p50_ms",
        "composed_p95_ms",
    ):
        assert metrics[key] == pytest.approx(p1[key], abs=5e-5), key
    references = adaptive["drift"]["references"]["P1"]
    assert metrics["mrr"] == pytest.approx(references["mrr"], abs=5e-5)
    assert set(m["hints_recorded_not_applied"]) == {"temporal_scope", "document_type_hints"}


def test_manifest_adaptive_reranking_matches_the_closed_9e_artifacts():
    from worldbank_copilot.retrieval import adaptive_eval as ae

    a = MANIFEST["adaptive_reranking"]
    assert a["promoted"] is False and a["status"] == "DIAGNOSTIC_ONLY"
    lock = json.loads((REPO_ROOT / a["protocol_lock"]).read_text("utf-8"))
    assert ae.lock_sha256(lock) == a["protocol_lock_sha256"]
    assert lock["policy_definitions_sha256"] == a["policy_definitions_sha256"]
    assert lock["production_null"] is True
    frontier = json.loads((EVAL / "adaptive_rerank_9e.json").read_text("utf-8"))
    selection = json.loads((EVAL / "adaptive_rerank_9e_live_selection.json").read_text("utf-8"))
    assert frontier["collection_artifact_sha256"] == a["collection_artifact_sha256"]
    assert selection["collection_artifact_sha256"] == a["collection_artifact_sha256"]
    assert selection["point"] == a["latency_validation_point"]
    for path, sha in a["committed_results_sha256_lf"].items():
        assert canonical_sha256(REPO_ROOT / path) == sha, path
    plan = (REPO_ROOT / "IMPLEMENTATION_PLAN.md").read_text("utf-8")
    assert a["live_artifact_sha256"] in plan  # recorded at 9E closure (artifact is off-repo)


def test_manifest_tools_and_phase10_contract_match_the_code():
    assert MANIFEST["tools"] == {s.name: s.version for s in TOOL_SPECS}
    contract = MANIFEST["phase10_contract"]
    assert tuple(contract["provenance_requirements"]) == PROVENANCE_REQUIREMENTS
    assert set(contract["provenance_origins"]) == {c.value for c in ProvenanceClass}
    for dotted in contract["allowed_interfaces"].values():
        module, _, attr = dotted.rpartition(".")
        try:
            owner = importlib.import_module(module)
        except ModuleNotFoundError:  # Class.method
            module, _, cls = module.rpartition(".")
            owner = getattr(importlib.import_module(module), cls)
        assert hasattr(owner, attr), dotted
