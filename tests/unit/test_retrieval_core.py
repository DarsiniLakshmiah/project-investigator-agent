"""Phase 8 query processing, lexical/hybrid retrieval, retriever guardrails, rerank, evidence."""

from datetime import date

import pytest
from tests.conftest import REPO_CONFIG_DIR

from worldbank_copilot.retrieval.config import load_retrieval_settings
from worldbank_copilot.retrieval.lexical import BM25, reciprocal_rank_fusion, tokenize
from worldbank_copilot.retrieval.models import RetrievalError, RetrievalScope, ScopeViolation
from worldbank_copilot.retrieval.query import process_query
from worldbank_copilot.retrieval.rerank import NoReranker, rerank_order
from worldbank_copilot.retrieval.retriever import REQUIRED_COLUMNS, ChunkStore, Retriever

RS = load_retrieval_settings(REPO_CONFIG_DIR)
PROJECTS = ["P1", "P2"]


def row(
    cid,
    project,
    text,
    *,
    strategy="structure",
    role="RETRIEVAL",
    doc=None,
    seq=None,
    dtype="ISR",
    page=1,
    parent=None,
):
    return {
        "chunk_id": cid,
        "chunk_strategy": strategy,
        "chunk_role": role,
        "chunk_type": "TEXT",
        "parent_chunk_id": parent,
        "project_id": project,
        "document_id": doc or f"{project}-d{seq}",
        "document_type": dtype,
        "document_label": f"ISR Sequence {seq}",
        "document_date": date(2024, 1, seq or 1),
        "isr_sequence": seq,
        "source_file": "f.pdf",
        "source_hash": "h" * 64,
        "page_number": page,
        "page_numbers": [page],
        "section_title": "4. KEY ISSUES",
        "element_ids": ["b1"],
        "chunk_text": text,
        "search_text": f"{project} | ISR | {text}",
    }


ROWS = [
    row("a1", "P1", "Procurement delays affected the water treatment plant contract.", seq=1),
    row("a2", "P1", "The closing date was extended by fifteen months.", seq=2),
    row("a3", "P1", "The closing date was extended by fifteen months.", seq=2, page=2),  # dup
    row("a4", "P1", "Metered household connections increased in the latest report.", seq=3),
    row("b1", "P2", "Procurement delays affected the sewage treatment plant contract.", seq=1),
    row("b2", "P2", "Procurement delays affected the tank rejuvenation.", seq=2),
    row("p1", "P1", "Parent section text with the closing date details.", role="PARENT", seq=2),
]
ROWS[1]["parent_chunk_id"] = "p1"


class FakeEmbeddings:
    model, dimension = "fake", 2

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return [[1.0, 0.0] for _ in texts]


class FakeDense:
    def __init__(self, hits):
        self.hits, self.filters = hits, []

    def search(self, vector, filters, k):
        self.filters.append(filters)
        return self.hits[:k]


class ReverseReranker:
    name = "cross_encoder"

    def score(self, query, texts):
        return [float(i) for i in range(len(texts))]  # last candidate scores highest


def retriever(dense_hits=()):
    return Retriever(ChunkStore(ROWS), RS, PROJECTS, FakeEmbeddings(), FakeDense(list(dense_hits)))


# -- query processing ---------------------------------------------------------------


def test_query_normalisation_expansion_and_isr_references():
    q = process_query("  What did the  latest ISR say about the PDO? ", "P130544", RS.query)
    assert q.text == "What did the latest ISR say about the PDO?"
    assert q.latest_isr and "ISR" in q.document_type_hints
    assert "Project Development Objective" in q.expanded
    assert "Implementation Status and Results Report" in q.expanded
    assert process_query("Ratings in ISR 23?", "P130544", RS.query).isr_sequences == (23,)
    assert process_query("ISR sequence 5 and ISR 6", "P130544", RS.query).isr_sequences == (5, 6)


def test_question_naming_another_project_is_refused():
    with pytest.raises(ScopeViolation, match="P179039"):
        process_query("What did P179039 report?", "P130544", RS.query)
    assert process_query("What did P130544 report?", "P130544", RS.query).projects_mentioned


def test_empty_question_is_rejected():
    with pytest.raises(ValueError):
        process_query("   ", "P130544", RS.query)


# -- lexical and fusion ----------------------------------------------------------------


def test_tokenizer_keeps_identifiers():
    tokens = tokenize("Loan IBRD-86010 for P130544 closed on 2024-11-22")
    assert {"ibrd-86010", "ibrd", "86010", "p130544", "2024-11-22"} <= set(tokens)
    assert "for" not in tokens and "on" not in tokens


def test_bm25_ranks_exact_terms_and_is_deterministic():
    index = BM25(
        ["x", "y", "z"], ["procurement delays in contract", "tank rejuvenation", "procurement plan"]
    )
    hits = index.search("procurement delays", 3)
    assert [h[0] for h in hits] == ["x", "z"]
    assert index.search("nothing matches", 3) == [] and index.search("procurement", 0) == []


def test_reciprocal_rank_fusion_orders_and_breaks_ties_by_id():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert [i for i, _ in fused][:2] == ["a", "b"]  # equal RRF score -> id order
    assert [i for i, _ in reciprocal_rank_fusion([["c"], ["d"]], k=60, limit=1)] == ["c"]


# -- scope and guardrails ----------------------------------------------------------------


def test_lexical_retrieval_is_filtered_by_project_before_scoring():
    result = retriever().retrieve(
        "procurement delays treatment plant",
        "P1",
        strategy="structure",
        method="lexical",
        reranker=NoReranker(),
        candidate_k=10,
        final_k=5,
    )
    assert result.status == "OK"
    assert {e.project_id for e in result.evidence} == {"P1"}
    assert "b1" not in {e.chunk_id for e in result.evidence}


def test_dense_filters_carry_project_and_strategy_and_foreign_hits_raise():
    r = retriever(dense_hits=[("a1", 0.9)])
    r.retrieve(
        "delays",
        "P1",
        strategy="structure",
        method="dense",
        reranker=NoReranker(),
        candidate_k=5,
        final_k=5,
    )
    assert r.dense.filters[-1] == {"project_id": "P1", "chunk_strategy": "structure"}
    leaky = retriever(dense_hits=[("a1", 0.9), ("b1", 0.8)])
    with pytest.raises(ScopeViolation, match="b1"):
        leaky.retrieve(
            "delays",
            "P1",
            strategy="structure",
            method="dense",
            reranker=NoReranker(),
            candidate_k=5,
            final_k=5,
        )


def test_parent_chunks_and_unknown_ids_are_rejected():
    with pytest.raises(ScopeViolation):
        retriever(dense_hits=[("p1", 0.9)]).retrieve(
            "x",
            "P1",
            strategy="structure",
            method="dense",
            reranker=NoReranker(),
            candidate_k=5,
            final_k=5,
        )
    with pytest.raises(RetrievalError, match="not in the governed corpus"):
        retriever(dense_hits=[("zz", 0.9)]).retrieve(
            "x",
            "P1",
            strategy="structure",
            method="dense",
            reranker=NoReranker(),
            candidate_k=5,
            final_k=5,
        )


def test_unsupported_project_and_strategy_are_refused():
    with pytest.raises(ScopeViolation, match="approved corpus"):
        retriever().retrieve(
            "x",
            "P9",
            strategy="structure",
            method="lexical",
            reranker=NoReranker(),
            candidate_k=5,
            final_k=5,
        )
    with pytest.raises(RetrievalError, match="strategy"):
        retriever().retrieve(
            "x",
            "P1",
            strategy="nope",
            method="lexical",
            reranker=NoReranker(),
            candidate_k=5,
            final_k=5,
        )
    with pytest.raises(ValueError):
        RetrievalScope(project_id=" ", chunk_strategy="structure")


def test_latest_isr_resolves_to_a_sequence_filter():
    r = retriever(dense_hits=[("a4", 0.9)])
    result = r.retrieve(
        "What changed in the latest ISR?",
        "P1",
        strategy="structure",
        method="dense",
        reranker=NoReranker(),
        candidate_k=5,
        final_k=5,
    )
    assert result.scope.isr_sequences == (3,)
    assert r.dense.filters[-1]["isr_sequence"] == [3]


def test_duplicates_are_suppressed_and_empty_results_abstain():
    result = retriever().retrieve(
        "closing date extended fifteen months",
        "P1",
        strategy="structure",
        method="lexical",
        reranker=NoReranker(),
        candidate_k=10,
        final_k=5,
    )
    texts = [e.text for e in result.evidence]
    assert len(texts) == len(set(texts))
    empty = retriever().retrieve(
        "zebra giraffe",
        "P1",
        strategy="structure",
        method="lexical",
        reranker=NoReranker(),
        candidate_k=10,
        final_k=5,
    )
    assert empty.status == "INSUFFICIENT_EVIDENCE" and empty.evidence == []


def test_rerank_threshold_abstains():
    rs = RS.model_copy(
        update={"retrieval": RS.retrieval.model_copy(update={"min_rerank_score": 100.0})}
    )
    r = Retriever(ChunkStore(ROWS), rs, PROJECTS)
    result = r.retrieve(
        "procurement delays",
        "P1",
        strategy="structure",
        method="lexical",
        reranker=ReverseReranker(),
        candidate_k=10,
        final_k=5,
    )
    assert result.status == "INSUFFICIENT_EVIDENCE" and "below" in result.notes[0]


def test_reranking_reorders_and_keeps_ties_stable():
    assert [i for i, _, _ in rerank_order(["a", "b", "c"], [0.1, 0.9, 0.1])] == ["b", "a", "c"]
    r = retriever()
    result = r.retrieve(
        "closing date procurement delays metered",
        "P1",
        strategy="structure",
        method="lexical",
        reranker=ReverseReranker(),
        candidate_k=10,
        final_k=5,
    )
    scores = [e.rerank_score for e in result.evidence]
    assert scores == sorted(scores, reverse=True)


def test_evidence_is_citation_ready_with_parent_context():
    result = retriever(dense_hits=[("a2", 0.7)]).retrieve(
        "closing date",
        "P1",
        strategy="structure",
        method="dense",
        reranker=NoReranker(),
        candidate_k=5,
        final_k=5,
    )
    item = result.evidence[0]
    assert item.evidence_id == "E1" and item.source_hash == "h" * 64
    assert item.context_text.startswith("Parent section text")
    assert item.citation.text == "P1 | ISR Sequence 2 | page 1, section: 4. KEY ISSUES"
    assert item.retrieval_method == "dense" and item.retrieval_score == 0.7


def test_malformed_corpus_rows_are_rejected():
    bad = dict(ROWS[0])
    del bad["source_hash"]
    with pytest.raises(RetrievalError, match="lacks"):
        ChunkStore([bad])
    no_page = {**ROWS[0], "page_number": None}
    with pytest.raises(RetrievalError, match="citation metadata"):
        ChunkStore([no_page])
    with pytest.raises(RetrievalError, match="duplicate"):
        ChunkStore([ROWS[0], ROWS[0]])
    assert set(REQUIRED_COLUMNS) <= set(ROWS[0])


def test_hybrid_fuses_lexical_and_dense():
    r = retriever(dense_hits=[("a4", 0.9), ("a1", 0.5)])
    result = r.retrieve(
        "procurement delays",
        "P1",
        strategy="structure",
        method="hybrid",
        reranker=NoReranker(),
        candidate_k=5,
        final_k=5,
    )
    ids = [e.chunk_id for e in result.evidence]
    assert ids[0] == "a1"  # found by both lists
    assert {"a1", "a4"} <= set(ids)
