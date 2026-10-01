"""Phase 8 on the real parsed corpus (local; lexical only - dense needs Databricks).

Skipped when the Phase 4 parsed cache is absent. Checks ground truth, corpus
determinism and project isolation for every evaluation question.
"""

import glob
import re
from pathlib import Path

import pytest
from tests.conftest import REPO_CONFIG_DIR, REPO_ROOT

from worldbank_copilot.parsing.pipeline import load_parsed
from worldbank_copilot.retrieval.chunking import elements
from worldbank_copilot.retrieval.config import load_retrieval_settings
from worldbank_copilot.retrieval.corpus import corpus_bodies
from worldbank_copilot.retrieval.evaluation import RunConfig, load_questions, normalize, run_config
from worldbank_copilot.retrieval.rerank import NoReranker
from worldbank_copilot.retrieval.retriever import ChunkStore, Retriever

pytestmark = pytest.mark.integration
PARSED = REPO_ROOT / ".local_output" / "parsed"
RS = load_retrieval_settings(REPO_CONFIG_DIR)
PROJECTS = ["P130544", "P179039", "P506272"]


@pytest.fixture(scope="module")
def documents():
    files = sorted(glob.glob(str(PARSED / "P*" / "*.json")))
    if not files:
        pytest.skip("no Phase 4 parsed cache")
    return [load_parsed(Path(f)) for f in files]


@pytest.fixture(scope="module")
def store(documents):
    return ChunkStore(corpus_bodies(documents, RS.chunking))


QUESTIONS = load_questions(REPO_ROOT / RS.evaluation.questions_file)


def test_every_ground_truth_phrase_is_on_its_pages(documents):
    by_id = {d.document_id: d for d in documents}
    for question in QUESTIONS:
        for ev in question.evidence:
            doc = by_id[ev.document_id]
            assert doc.project_id == question.project_id
            pages = {
                p
                for e in elements(doc)
                if normalize(ev.contains) in normalize(e.text)
                for p in e.page_numbers
            }
            assert pages & set(ev.pages), (question.id, ev.document_id, ev.pages, sorted(pages))


def test_corpus_covers_every_document_and_is_reproducible(documents):
    first, second = corpus_bodies(documents, RS.chunking), corpus_bodies(documents, RS.chunking)
    assert first == second
    for strategy in RS.chunking.strategies:
        docs = {r["document_id"] for r in first if r["chunk_strategy"] == strategy}
        assert len(docs) == len(documents) == 53
    assert not [r for r in first if re.match(r"^@#&OPS~", r["chunk_text"])]


def test_most_evidence_is_representable_in_structure_chunks(store):
    """Ground truth reachable by at least one structure chunk (chunking does not lose it)."""
    from worldbank_copilot.retrieval.evaluation import matched_items

    rows = [
        r
        for r in store.rows.values()
        if r["chunk_strategy"] == "structure" and r["chunk_role"] == "RETRIEVAL"
    ]
    for question in QUESTIONS:
        if question.answerable:
            covered = set().union(
                *(
                    matched_items(r, question)
                    for r in rows
                    if r["project_id"] == question.project_id
                )
            )
            assert covered, question.id


@pytest.mark.parametrize("scope", PROJECTS)
def test_no_question_retrieves_another_projects_evidence(store, scope):
    retriever = Retriever(store, RS, PROJECTS)
    for question in QUESTIONS:
        result = retriever.retrieve(
            question.question,
            scope,
            strategy="structure",
            method="lexical",
            reranker=NoReranker(),
            candidate_k=50,
            final_k=50,
        )
        assert {e.project_id for e in result.evidence} <= {scope}


def test_lexical_baseline_runs_on_all_questions(store):
    retriever = Retriever(store, RS, PROJECTS)
    result = run_config(
        retriever,
        QUESTIONS,
        RunConfig("structure", "lexical", "none", 10),
        {"none": NoReranker()},
        [5, 10],
    )
    assert result.status == "OK" and len(result.questions) == len(QUESTIONS)
    assert result.summary["leakage_at_10"] == 0
    assert result.summary["recall_at_10"] > 0.5  # sanity floor, not a tuned target
