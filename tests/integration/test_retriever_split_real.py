"""Phase 9 Decision 1 on the real corpus: split retrieve == Phase 8 retrieve.

All 49 evaluation questions x 3 chunking strategies x {lexical, dense, hybrid} x
{no rerank k10, always rerank k50 (+k20)} on the real parsed corpus. Dense retrieval and
the CrossEncoder are deterministic local stand-ins (the real endpoints need Databricks;
the same comparison against the real index and CrossEncoder is a 9F Databricks step).
Skipped when the Phase 4 parsed cache is absent.
"""

import glob
from itertools import product
from pathlib import Path

import pytest
from tests.conftest import REPO_CONFIG_DIR, REPO_ROOT
from tests.support.retrieval_golden import (
    OverlapCrossEncoder,
    ScopedDense,
    TextEmbeddings,
    outcome,
    phase8_retrieve,
)

from worldbank_copilot.parsing.pipeline import load_parsed
from worldbank_copilot.retrieval.config import load_retrieval_settings
from worldbank_copilot.retrieval.corpus import corpus_bodies
from worldbank_copilot.retrieval.evaluation import load_questions
from worldbank_copilot.retrieval.rerank import NoReranker
from worldbank_copilot.retrieval.retriever import ChunkStore, Retriever

pytestmark = pytest.mark.integration
PARSED = REPO_ROOT / ".local_output" / "parsed"
RS = load_retrieval_settings(REPO_CONFIG_DIR)
PROJECTS = ["P130544", "P179039", "P506272"]
QUESTIONS = load_questions(REPO_ROOT / RS.evaluation.questions_file)
# (method, reranker, candidate_k): the two Phase 8 bounds plus neighbours.
CONFIGS = [
    ("hybrid", "none", 10),  # measured no-reranker baseline
    ("hybrid", "cross_encoder", 50),  # measured always-rerank ceiling
    ("hybrid", "cross_encoder", 20),
    ("lexical", "none", 10),
    ("dense", "cross_encoder", 10),
]


@pytest.fixture(scope="module")
def retriever():
    files = sorted(glob.glob(str(PARSED / "P*" / "*.json")))
    if not files:
        pytest.skip("no Phase 4 parsed cache")
    rows = corpus_bodies([load_parsed(Path(f)) for f in files], RS.chunking)
    return Retriever(ChunkStore(rows), RS, PROJECTS, TextEmbeddings(), ScopedDense(rows))


@pytest.mark.parametrize("strategy", ["fixed", "structure", "parent_child"])
def test_split_matches_phase8_on_every_question(retriever, strategy):
    compared = 0
    for (method, rerank, k), q in product(CONFIGS, QUESTIONS):
        kwargs = dict(
            strategy=strategy,
            method=method,
            reranker=NoReranker() if rerank == "none" else OverlapCrossEncoder(),
            candidate_k=k,
            final_k=5,
        )
        before = outcome(lambda: phase8_retrieve(retriever, q.question, q.project_id, **kwargs))  # noqa: B023
        after = outcome(lambda: retriever.retrieve(q.question, q.project_id, **kwargs))  # noqa: B023
        assert before == after, (strategy, method, rerank, k, q.id)
        assert before[0] == "ok", (q.id, before)
        compared += 1
    assert compared == len(CONFIGS) * len(QUESTIONS)


def test_foreign_project_questions_fail_identically(retriever):
    for project, other in (("P130544", "P179039"), ("P506272", "P130544")):
        kwargs = dict(
            strategy="fixed", method="hybrid", reranker=NoReranker(), candidate_k=10, final_k=5
        )
        q = f"What did {other} report about procurement?"
        before = outcome(lambda: phase8_retrieve(retriever, q, project, **kwargs))  # noqa: B023
        after = outcome(lambda: retriever.retrieve(q, project, **kwargs))  # noqa: B023
        assert before == after and before[1][0] == "ScopeViolation"
