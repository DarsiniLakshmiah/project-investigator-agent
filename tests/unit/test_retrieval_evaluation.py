"""Phase 8 evaluation metrics, experiment selection, adapters and configuration."""

import math

import pytest
from tests.conftest import REPO_CONFIG_DIR, REPO_ROOT
from tests.unit.test_retrieval_core import ROWS, retriever

from worldbank_copilot.common.exceptions import ConfigurationError, LakehouseError
from worldbank_copilot.extraction.results import comment_cell
from worldbank_copilot.retrieval.config import load_retrieval_settings
from worldbank_copilot.retrieval.embeddings import DatabricksServingEmbeddings, EmbeddingError
from worldbank_copilot.retrieval.evaluation import (
    EvidenceRef,
    Question,
    RunConfig,
    RunResult,
    choose,
    load_questions,
    matched_items,
    normalize,
    ranking_metrics,
    run_config,
)
from worldbank_copilot.retrieval.rerank import NoReranker
from worldbank_copilot.retrieval.vector_search import VectorSearchIndex, parse_results

RS = load_retrieval_settings(REPO_CONFIG_DIR)


def q(mode="any", evidence=None, **kw):
    evidence = evidence or [EvidenceRef(document_id="P1-d2", pages=[1], contains="closing date")]
    return Question(
        id="t",
        project_id="P1",
        category="c",
        kind="k",
        question="closing date?",
        evidence_mode=mode,
        evidence=evidence,
        **kw,
    )


def chunk(doc, page, text, project="P1"):
    return {"document_id": doc, "page_numbers": [page], "chunk_text": text, "project_id": project}


# -- relevance and metrics --------------------------------------------------------------


def test_relevance_needs_document_page_and_phrase_ignoring_glyphs():
    question = q(
        evidence=[
            EvidenceRef(
                document_id="P1-d2", pages=[3], contains="Overall Risk Rating | Substantial"
            )
        ]
    )
    assert matched_items(chunk("P1-d2", 3, "Overall Risk Rating | Substantial"), question) == {0}
    assert matched_items(chunk("P1-d2", 4, "Overall Risk Rating Substantial"), question) == set()
    assert matched_items(chunk("P1-d9", 3, "Overall Risk Rating Substantial"), question) == set()
    assert normalize("  A|B, c ") == "a b c"


def test_ranking_metrics_exact_values():
    two = [
        EvidenceRef(document_id="P1-a", pages=[1], contains="alpha"),
        EvidenceRef(document_id="P1-b", pages=[1], contains="beta"),
    ]
    rows = [
        chunk("P1-x", 1, "noise"),
        chunk("P1-a", 1, "alpha here"),
        chunk("P1-x", 1, "noise"),
        chunk("P1-x", 1, "noise"),
        chunk("P1-x", 1, "noise"),
        chunk("P1-b", 1, "beta"),
    ]
    all_mode = ranking_metrics(rows, q("all", two), [5, 10])
    assert all_mode["recall_at_5"] == 0.5 and all_mode["recall_at_10"] == 1.0
    assert all_mode["mrr"] == 0.5 and all_mode["precision_at_5"] == 0.2
    ideal = 1 + 1 / math.log2(3)
    assert all_mode["ndcg_at_5"] == pytest.approx((1 / math.log2(3)) / ideal)
    any_mode = ranking_metrics(rows, q("any", two), [5])
    assert any_mode["recall_at_5"] == 1.0
    empty = ranking_metrics([], q(), [5, 10])
    assert empty["recall_at_5"] == 0 and empty["mrr"] == 0 and empty["ndcg_at_5"] == 0


def test_leakage_is_counted():
    rows = [chunk("P1-d2", 1, "closing date"), chunk("P2-d1", 1, "closing date", project="P2")]
    assert ranking_metrics(rows, q(), [5])["leakage_at_5"] == 1.0


def test_question_validation():
    with pytest.raises(ValueError, match="without evidence"):
        Question(id="x", project_id="P1", category="c", kind="k", question="q", evidence=[])
    with pytest.raises(ValueError, match="another project"):
        q(evidence=[EvidenceRef(document_id="P2-d", pages=[1], contains="abc")])
    no_answer = Question(
        id="x", project_id="P1", category="c", kind="k", question="q", answerable=False
    )
    assert no_answer.evidence == []


def test_repository_question_set_is_well_formed():
    questions = load_questions(REPO_ROOT / RS.evaluation.questions_file)
    assert 40 <= len(questions) <= 50
    assert {x.project_id for x in questions} == {"P130544", "P179039", "P506272"}
    assert sum(not x.answerable for x in questions) >= 4
    kinds = {x.kind for x in questions}
    assert {"exact", "paraphrase", "multi_evidence", "temporal", "no_answer"} <= kinds
    categories = {x.category for x in questions}
    assert {
        "ratings",
        "results",
        "finance",
        "closing_dates",
        "restructuring",
        "cancellation",
        "procurement",
        "risks",
        "assessment_findings",
        "implementation_issues",
    } <= categories


# -- experiment runner and selection ------------------------------------------------------


def test_run_config_measures_every_question_and_never_leaks():
    questions = [
        q(evidence=[EvidenceRef(document_id="P1-d2", pages=[1], contains="fifteen months")]),
        Question(
            id="n",
            project_id="P1",
            category="no_answer",
            kind="no_answer",
            question="zebra",
            answerable=False,
        ),
    ]
    result = run_config(
        retriever(),
        questions,
        RunConfig("structure", "lexical", "none", 10),
        {"none": NoReranker()},
        [5, 10],
    )
    assert result.status == "OK" and len(result.questions) == 2
    assert result.summary["recall_at_5"] == 1.0 and result.summary["leakage_at_5"] == 0
    unavailable = run_config(
        retriever(),
        questions,
        RunConfig("structure", "lexical", "cross_encoder", 10),
        {"none": NoReranker()},
        [5],
    )
    assert unavailable.status == "UNAVAILABLE"


class FixedRun(RunResult):
    def __init__(self, name, recall, status="OK"):
        super().__init__(RunConfig(name, "dense", "none", 10), [], status, None)
        self.recall = recall

    @property
    def summary(self):
        return {"recall_at_10": self.recall}


def test_choose_prefers_simpler_configuration_unless_clearly_better():
    a, b, c = FixedRun("a", 0.80), FixedRun("b", 0.81), FixedRun("c", 0.90)
    best, reason = choose([a, b, c], "recall_at_10", 0.02)
    assert best is c and "not preferred" in reason and "beats" in reason
    best, _ = choose([a, b], "recall_at_10", 0.02)
    assert best is a  # +0.01 is noise: the simpler configuration stays
    best, reason = choose([a, FixedRun("d", 0.99, "UNAVAILABLE")], "recall_at_10", 0.02)
    assert best is a and "unavailable" in reason


# -- adapters -------------------------------------------------------------------------------


class Item:
    def __init__(self, embedding):
        self.embedding = embedding


class Response:
    def __init__(self, n, dim):
        self.data = [Item([0.1] * dim) for _ in range(n)]


class Endpoints:
    def __init__(self, dim=1024, fail_times=0):
        self.dim, self.fail_times, self.calls = dim, fail_times, []

    def query(self, name, input):  # noqa: A002 - SDK keyword
        self.calls.append(len(input))
        if self.fail_times:
            self.fail_times -= 1
            raise TimeoutError("rate limited")
        return Response(len(input), self.dim)


class Client:
    def __init__(self, endpoints):
        self.serving_endpoints = endpoints


def test_embeddings_batch_retry_and_validate_dimension():
    cfg = RS.embeddings.model_copy(update={"batch_size": 2, "max_retries": 2})
    endpoints = Endpoints(fail_times=1)
    provider = DatabricksServingEmbeddings(cfg, Client(endpoints), sleep=lambda _s: None)
    vectors = provider.embed(["a", "b", "c"])
    assert len(vectors) == 3 and len(vectors[0]) == 1024
    assert endpoints.calls == [2, 2, 1]  # first batch retried once
    wrong = DatabricksServingEmbeddings(cfg, Client(Endpoints(dim=8)), sleep=lambda _s: None)
    with pytest.raises(EmbeddingError, match="dimension"):
        wrong.embed(["a"])
    failing = DatabricksServingEmbeddings(
        cfg, Client(Endpoints(fail_times=9)), sleep=lambda _s: None
    )
    with pytest.raises(EmbeddingError, match="rate limited"):
        failing.embed(["a"])


def test_vector_search_results_are_normalised():
    response = {
        "manifest": {"columns": [{"name": "chunk_id"}, {"name": "project_id"}, {"name": "score"}]},
        "result": {"data_array": [["c1", "P1", 0.91], ["c2", "P1", 0.5]]},
    }
    assert parse_results(response) == [("c1", 0.91), ("c2", 0.5)]
    assert parse_results({"manifest": {"columns": []}, "result": {}}) == []
    with pytest.raises(LakehouseError):
        parse_results(
            {"manifest": {"columns": [{"name": "x"}]}, "result": {"data_array": [[1, 2]]}}
        )


class FakeIndex:
    def __init__(self, states, source="cat.silver.document_chunk_index"):
        self.states, self.synced, self.source, self.queries = list(states), 0, source, []

    def describe(self):
        state = self.states[0] if len(self.states) == 1 else self.states.pop(0)
        return {"status": state, "delta_sync_index_spec": {"source_table": self.source}}

    def sync(self):
        self.synced += 1

    def similarity_search(self, **kw):
        self.queries.append(kw)
        return {
            "manifest": {"columns": [{"name": "chunk_id"}, {"name": "score"}]},
            "result": {"data_array": [["c1", 0.9]]},
        }


class FakeVS:
    def __init__(self, index=None, endpoint_state="ONLINE"):
        self.index, self.endpoint_state, self.created = index, endpoint_state, []

    def get_endpoint(self, name):
        if self.endpoint_state is None:
            raise Exception("Endpoint not found")
        return {"endpoint_status": {"state": self.endpoint_state}}

    def create_endpoint(self, name, endpoint_type):
        self.created.append(("endpoint", name, endpoint_type))
        self.endpoint_state = "ONLINE"

    def get_index(self, endpoint_name, index_name):
        if self.index is None:
            raise Exception("Index does not exist")
        return self.index

    def create_delta_sync_index(self, **kw):
        self.created.append(("index", kw))
        self.index = FakeIndex(
            [{"ready": True, "detailed_state": "ONLINE_NO_PENDING_UPDATE", "indexed_row_count": 3}]
        )
        return self.index


def vs(client):
    return VectorSearchIndex(
        RS.retrieval.vector_search,
        "cat.silver.idx",
        "cat.silver.document_chunk_index",
        1024,
        client=client,
        sleep=lambda _s: None,
    )


def test_vector_search_lifecycle_creates_missing_objects_with_self_managed_vectors():
    client = FakeVS(endpoint_state=None)
    index = vs(client)
    assert index.ensure_endpoint() == ("ONLINE", "CREATED")
    assert index.ensure_index() == "CREATED"
    spec = client.created[1][1]
    assert spec["primary_key"] == "chunk_id" and spec["embedding_vector_column"] == "embedding"
    assert spec["embedding_dimension"] == 1024 and spec["pipeline_type"] == "TRIGGERED"


def test_existing_index_on_another_source_is_refused():
    index = vs(FakeVS(FakeIndex([{"ready": True}], source="other.table")))
    with pytest.raises(LakehouseError, match="refusing"):
        index.ensure_index()


def test_sync_waits_for_all_rows_and_queries_need_project_filter():
    fake = FakeIndex(
        [
            {"ready": True, "detailed_state": "ONLINE_NO_PENDING_UPDATE", "indexed_row_count": 2},
            {"ready": True, "detailed_state": "ONLINE_TRIGGERED_UPDATE", "indexed_row_count": 2},
            {"ready": True, "detailed_state": "ONLINE_NO_PENDING_UPDATE", "indexed_row_count": 3},
        ]
    )
    index = vs(FakeVS(fake))
    index.ensure_index()
    index.sync_and_wait(expected_rows=3)
    assert fake.synced == 1
    assert index.search([0.1], {"project_id": "P1"}, 5) == [("c1", 0.9)]
    assert fake.queries[-1]["filters"] == {"project_id": "P1"}
    with pytest.raises(LakehouseError, match="project_id"):
        index.search([0.1], {}, 5)


# -- configuration and the comments fix ------------------------------------------------------


def test_retrieval_configuration_is_validated(tmp_path):
    assert set(RS.chunking.strategies) == {"fixed", "structure", "parent_child"}
    assert "project_id" in RS.retrieval.required_filters
    assert RS.chunking.strategies["structure"].retrieval_max_chars <= 1500
    folder = tmp_path / "retrieval"
    folder.mkdir()
    for name in ("chunking", "embeddings", "retrieval", "query", "evaluation"):
        (folder / f"{name}.yaml").write_text(
            (REPO_CONFIG_DIR / "retrieval" / f"{name}.yaml").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    text = (folder / "retrieval.yaml").read_text(encoding="utf-8")
    (folder / "retrieval.yaml").write_text(
        text.replace("required_filters: [project_id]", "required_filters: []"), encoding="utf-8"
    )
    with pytest.raises(ConfigurationError, match="project_id"):
        load_retrieval_settings(tmp_path)


def test_comment_cell_skips_repeated_label_and_refuses_fragments():
    label = "Comments on achieving targets"
    assert (
        comment_cell([label, "Text of the comment", "Text of the comment"]) == "Text of the comment"
    )
    assert comment_cell(["Text only"]) == "Text only"
    assert comment_cell([label, label]) is None
    assert comment_cell([label, "Revenue 97% in", "collection efficiency as"]) is None
    assert comment_cell([]) is None


def test_rows_fixture_is_consistent():
    assert {r["project_id"] for r in ROWS} == {"P1", "P2"}
