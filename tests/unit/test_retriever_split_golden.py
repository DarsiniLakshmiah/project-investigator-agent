"""Phase 9 Decision 1: the Retriever split preserves Phase 8 behaviour exactly.

The split ``retrieve`` (first_stage + finish) is compared with a verbatim copy of the
Phase 8 implementation (tests/support/retrieval_golden.py) for every retrieval method,
no-rerank and always-rerank, several depths, type filters, abstention thresholds,
ISR references and every error path. Compared: evidence ids/order, project scope,
metadata/citations, statuses, notes and retrieval/rerank scores (not wall-clock latency).
"""

from datetime import date
from itertools import product

import pytest
from tests.conftest import REPO_CONFIG_DIR
from tests.support.retrieval_golden import (
    OverlapCrossEncoder,
    ScopedDense,
    TextEmbeddings,
    comparable,
    outcome,
    phase8_retrieve,
)

from worldbank_copilot.retrieval.config import load_retrieval_settings
from worldbank_copilot.retrieval.rerank import NoReranker
from worldbank_copilot.retrieval.retriever import ChunkStore, FirstStage, Retriever

RS = load_retrieval_settings(REPO_CONFIG_DIR)
PROJECTS = ["P100001", "P100002", "P100003"]
TEXTS = [
    "Procurement delays affected the water treatment plant contract.",
    "The closing date was extended by fifteen months to allow completion.",
    "Metered household connections increased in the latest report.",
    "Disbursement reached 62 percent of the loan after the restructuring.",
    "The operator fee allocation was increased because of exchange rate losses.",
    "Grievances addressed within 30 days rose to 85 percent.",
    "Bulk works progressed in each city; distribution works lag behind.",
    "The PDO rating was downgraded to Moderately Unsatisfactory.",
]


def make_rows():
    rows = []
    for p, project in enumerate(PROJECTS):
        for s, strategy in enumerate(("fixed", "structure")):
            for i, text in enumerate(TEXTS):
                seq = i % 4 + 1
                dtype = "ISR" if i % 3 else "RESTRUCTURING_PAPER"
                rows.append(
                    {
                        "chunk_id": f"{project}-{strategy}-{i}",
                        "chunk_strategy": strategy,
                        "chunk_role": "RETRIEVAL",
                        "chunk_type": "TABLE" if i % 4 == 0 else "TEXT",
                        "parent_chunk_id": f"{project}-{strategy}-parent" if i == 1 else None,
                        "project_id": project,
                        "document_id": f"{project}-doc{i % 3}",
                        "document_type": dtype,
                        "document_label": f"ISR Sequence {seq}",
                        "document_date": date(2024, 1 + p, seq),
                        "isr_sequence": seq if dtype == "ISR" else None,
                        "source_file": f"{project}/f{i % 3}.pdf",
                        "source_hash": f"{p}{s}" * 32,
                        "page_number": i + 1,
                        "page_numbers": [i + 1, i + 2] if i % 2 else [i + 1],
                        "section_title": None if i == 5 else f"Section {i}",
                        "element_ids": [f"b{i}", f"t{i}"],
                        # a project-specific word so projects never tie
                        "chunk_text": f"{text} ({project.lower()} note)",
                        "search_text": f"{project} | {dtype} | {text}",
                    }
                )
            # exact duplicate text in the same document (de-duplication path)
            dup = dict(rows[-2], chunk_id=f"{project}-{strategy}-dup", page_number=40)
            rows.append(dup)
            rows.append(
                {
                    **rows[-3],
                    "chunk_id": f"{project}-{strategy}-parent",
                    "chunk_role": "PARENT",
                    "parent_chunk_id": None,
                    "chunk_text": "Parent section text with the closing date details.",
                }
            )
    return rows


ROWS = make_rows()
QUESTIONS = [
    "Why was the closing date extended?",
    "What were the procurement delays?",
    "How much had been disbursed after the restructuring paper was approved?",
    "What does the latest ISR say about metered household connections?",
    "What was reported in ISR 2 about grievances?",
    "ISR 3 and ISR 4 bulk works progress",
    "What did the restructuring paper say about the PDO rating?",
    "zebra quantum unicorn",  # nothing matches: INSUFFICIENT_EVIDENCE
    "What about project P100002 and its operator fee?",  # foreign project -> ScopeViolation
    "   ",  # empty -> ValueError
]
CONFIGS = list(
    product(
        ("fixed", "structure"),
        ("lexical", "dense", "hybrid"),
        ("none", "cross_encoder"),
        (3, 10, 50),
        (False, True),
    )
)


def retriever(settings=RS):
    return Retriever(ChunkStore(ROWS), settings, PROJECTS, TextEmbeddings(), ScopedDense(ROWS))


def reranker(name):
    return NoReranker() if name == "none" else OverlapCrossEncoder()


def compare_all(settings, project_ids=("P100001", "P100003", "P999999")):
    r = retriever(settings)
    compared = 0
    for (strategy, method, rerank, k, filters), question, project in product(
        CONFIGS, QUESTIONS, project_ids
    ):
        kwargs = dict(
            strategy=strategy,
            method=method,
            reranker=reranker(rerank),
            candidate_k=k,
            final_k=5,
            use_type_filters=filters,
        )
        before = outcome(lambda: phase8_retrieve(r, question, project, **kwargs))  # noqa: B023
        after = outcome(lambda: r.retrieve(question, project, **kwargs))  # noqa: B023
        assert before == after, (strategy, method, rerank, k, filters, question, project)
        compared += 1
    return compared


def test_split_is_equivalent_to_phase8_for_every_configuration():
    assert compare_all(RS) == len(CONFIGS) * len(QUESTIONS) * 3


def test_split_is_equivalent_with_an_abstention_threshold():
    retrieval = RS.retrieval.model_copy(update={"min_rerank_score": 1.5})
    assert compare_all(RS.model_copy(update={"retrieval": retrieval}), ("P100001",)) > 0


def test_golden_covers_every_status_and_error_path():
    r = retriever()
    seen = set()
    for (strategy, method, rerank, k, filters), question in product(CONFIGS, QUESTIONS):
        kind, value = outcome(
            lambda: r.retrieve(  # noqa: B023
                question,  # noqa: B023
                "P100001",
                strategy=strategy,  # noqa: B023
                method=method,  # noqa: B023
                reranker=reranker(rerank),  # noqa: B023
                candidate_k=k,  # noqa: B023
                final_k=5,
                use_type_filters=filters,  # noqa: B023
            )
        )
        seen.add(value["status"] if kind == "ok" else value[0])
    assert {"OK", "INSUFFICIENT_EVIDENCE", "ScopeViolation", "ValueError"} <= seen
    unknown = outcome(
        lambda: r.retrieve(
            "closing date",
            "P999999",
            strategy="fixed",
            method="lexical",
            reranker=NoReranker(),
            candidate_k=10,
            final_k=5,
        )
    )
    assert unknown[0] == "error" and unknown[1][0] == "ScopeViolation"


def test_rerank_actually_reorders_in_the_golden_set():
    """The comparison is meaningful: always-rerank differs from no-rerank somewhere."""
    r = retriever()
    differs = False
    for question in QUESTIONS[:7]:
        kwargs = dict(strategy="fixed", method="hybrid", candidate_k=10, final_k=5)
        plain = r.retrieve(question, "P100001", reranker=NoReranker(), **kwargs)
        ranked = r.retrieve(question, "P100001", reranker=OverlapCrossEncoder(), **kwargs)
        differs |= [e.chunk_id for e in plain.evidence] != [e.chunk_id for e in ranked.evidence]
        assert all(e.rerank_score is not None for e in ranked.evidence)
    assert differs


def test_first_stage_is_reusable_for_both_rerank_decisions():
    r = retriever()
    stage = r.first_stage(
        "Why was the closing date extended?",
        "P100001",
        strategy="fixed",
        method="hybrid",
        candidate_k=10,
    )
    assert isinstance(stage, FirstStage) and stage.candidates
    assert all(c.rerank_score is None for c in stage.candidates)
    before = [c.model_dump() for c in stage.candidates]
    plain = r.finish(stage, reranker=NoReranker(), final_k=5)
    ranked = r.finish(stage, reranker=OverlapCrossEncoder(), final_k=5)
    again = r.finish(stage, reranker=NoReranker(), final_k=5)
    assert [c.model_dump() for c in stage.candidates] == before  # finish does not mutate
    assert comparable(plain) == comparable(again)
    kwargs = dict(strategy="fixed", method="hybrid", candidate_k=10, final_k=5)
    q = "Why was the closing date extended?"
    assert comparable(plain) == comparable(
        phase8_retrieve(r, q, "P100001", reranker=NoReranker(), **kwargs)
    )
    assert comparable(ranked) == comparable(
        phase8_retrieve(r, q, "P100001", reranker=OverlapCrossEncoder(), **kwargs)
    )


def test_first_stage_refuses_before_any_candidate_is_produced():
    r = retriever()
    with pytest.raises(Exception, match="cross-project retrieval is not permitted"):
        r.first_stage(
            "Tell me about P100002", "P100001", strategy="fixed", method="lexical", candidate_k=10
        )
