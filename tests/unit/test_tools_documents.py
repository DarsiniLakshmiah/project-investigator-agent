"""Phase 9 T8 search_project_documents: policy decides, harness executes, evidence is
provenance-only (relevance not verified), and results equal the Phase 8 retriever's."""

import pytest
from tests.support.retrieval_golden import OverlapCrossEncoder, comparable
from tests.support.tool_fixtures import context
from tests.unit.test_retriever_split_golden import PROJECTS, retriever

from worldbank_copilot.common.project_registry import ProjectConfig, ProjectRegistry
from worldbank_copilot.retrieval.rerank import NoReranker
from worldbank_copilot.retrieval.rerank_policy import AlwaysRerank, NeverRerank
from worldbank_copilot.tools import registry
from worldbank_copilot.tools.documents import DocumentSearch, DocumentSearchConfig
from worldbank_copilot.tools.models import MechanicalCode, ProvenanceClass, ToolStatus

EXECUTOR = registry.default_executor()
CONFIG = DocumentSearchConfig(chunk_strategy="fixed", method="hybrid", candidate_k=10, final_k=5)
PID = PROJECTS[0]
SYNTHETIC = ProjectRegistry(
    projects=tuple(
        ProjectConfig(
            project_id=p,
            name=p,
            instrument="Investment Project Financing",
            lifecycle_stage="mid",
            purpose="test",
            analytical_question="test",
            documents_dir=p,
            procurement_coverage={"covered_by_ipf_contract_awards": True, "note": "test"},
        )
        for p in PROJECTS
    )
)


def search(policy, question="Why was the closing date extended?", cross_encoder=None, pid=PID):
    r = retriever()
    ctx = context(documents=DocumentSearch(r, CONFIG, policy, cross_encoder))
    ctx.registry = SYNTHETIC
    res = EXECUTOR.run(
        "search_project_documents",
        {"project_id": pid, "query": question},
        ctx,
        scope_project_id=pid,
    )
    return res, r


def phase8(r, question, reranker):
    return r.retrieve(
        question,
        PID,
        strategy="fixed",
        method="hybrid",
        reranker=reranker,
        candidate_k=10,
        final_k=5,
    )


def test_never_policy_equals_the_no_reranker_retrieval():
    res, r = search(NeverRerank(), cross_encoder=OverlapCrossEncoder())
    (item,) = res.items
    assert res.status == ToolStatus.OK and item.reranker == "none"
    assert item.rerank_decision.policy == "never" and item.rerank_decision.rerank is False
    expected = comparable(phase8(r, "Why was the closing date extended?", NoReranker()))
    assert [e.evidence.model_dump(mode="json") for e in item.evidence] == expected["evidence"]


def test_always_policy_equals_the_reranked_retrieval():
    res, r = search(AlwaysRerank(), cross_encoder=OverlapCrossEncoder())
    (item,) = res.items
    assert item.reranker == "cross_encoder" and item.rerank_decision.rerank is True
    expected = comparable(phase8(r, "Why was the closing date extended?", OverlapCrossEncoder()))
    assert [e.evidence.model_dump(mode="json") for e in item.evidence] == expected["evidence"]
    assert all(e.evidence.rerank_score is not None for e in item.evidence)


def test_evidence_is_documented_provenance_with_relevance_not_verified():
    res, _ = search(NeverRerank())
    (item,) = res.items
    assert all(e.provenance_class == ProvenanceClass.DOCUMENTED_FINDING for e in item.evidence)
    assert all(e.relevance_verified is False for e in item.evidence)
    assert all(e.evidence.project_id == PID and e.evidence.citation.pages for e in item.evidence)
    assert any("not establish" in n and "relevance_verified" in n for n in res.notices)
    assert res.semantic_sufficiency == "NOT_ASSESSED"
    assert item.first_stage_candidates >= len(item.evidence)


def test_zero_chunks_is_mechanically_insufficient():
    res, _ = search(NeverRerank(), question="zebra quantum unicorn")
    assert res.status == ToolStatus.INSUFFICIENT_EVIDENCE and res.items == []
    assert res.mechanical[0].code == MechanicalCode.NO_CHUNKS


def test_a_question_naming_another_project_is_refused():
    res, _ = search(NeverRerank(), question=f"What did {PROJECTS[1]} report?")
    assert res.status == ToolStatus.SCOPE_REFUSED and "cross-project" in res.error


@pytest.mark.parametrize(
    ("documents", "text"),
    [(None, "not configured"), ("always-without-reranker", "no reranker is configured")],
)
def test_misconfiguration_is_an_explicit_error(documents, text):
    r = retriever()
    ctx = context(
        documents=None if documents is None else DocumentSearch(r, CONFIG, AlwaysRerank(), None)
    )
    ctx.registry = SYNTHETIC
    res = EXECUTOR.run(
        "search_project_documents",
        {"project_id": PID, "query": "closing date"},
        ctx,
        scope_project_id=PID,
    )
    assert res.status == ToolStatus.ERROR and text in res.error
