"""Reusing the existing AI Search endpoint with a new, independent project index."""

import pytest
from tests.conftest import REPO_CONFIG_DIR

from worldbank_copilot.common import load_settings
from worldbank_copilot.retrieval import pipeline as rp
from worldbank_copilot.retrieval.config import load_retrieval_settings
from worldbank_copilot.retrieval.vector_search import (
    VectorSearchIndex,
    VectorSearchUnavailable,
    is_quota_error,
)

RS = load_retrieval_settings(REPO_CONFIG_DIR)
SETTINGS = load_settings("databricks", config_dir=REPO_CONFIG_DIR, env={})
QWEN = "databricks-qwen3-embedding-0-6b"
GEP = [
    "worldbank_ai.rag.gep_recursive_v1_qwen3_index",
    "worldbank_ai.rag.gep_fixed_v1_qwen3_index",
    "worldbank_ai.rag.gep_structure_v1_qwen3_index",
]
NAMES = rp.names(SETTINGS, RS)
INDEX = f"{NAMES['catalog']}.{NAMES['silver']}.{RS.retrieval.vector_search.index_name}"
SOURCE = NAMES["index_table_plain"]


class NotFound(Exception):
    pass


class FakeIndex:
    def __init__(self, source):
        self.source = source

    def describe(self):
        return {
            "delta_sync_index_spec": {
                "source_table": self.source,
                "embedding_vector_columns": [{"name": "embedding", "embedding_dimension": 1024}],
            },
            "status": {"ready": True},
        }


class Client:
    """Fake AI Search client; records every mutating call."""

    def __init__(self, state="ONLINE", indexes=None, create_error=None):
        self.state, self.indexes = state, dict(indexes or {})
        self.create_error, self.mutations = create_error, []

    def get_endpoint(self, name):
        if self.state is None:
            raise NotFound("endpoint")
        return {"endpoint_status": {"state": self.state}}

    def create_endpoint(self, **kw):
        self.mutations.append(("create_endpoint", kw))

    def list_indexes(self, name):
        return {"vector_indexes": [{"name": n} for n in self.indexes]}

    def get_index(self, endpoint_name, index_name):
        if index_name not in self.indexes:
            raise NotFound(index_name)
        return self.indexes[index_name]

    def create_delta_sync_index(self, **kw):
        self.mutations.append(("create_index", kw))
        if self.create_error:
            raise self.create_error
        self.indexes[kw["index_name"]] = FakeIndex(kw["source_table_name"])
        return self.indexes[kw["index_name"]]


class Spark:
    """Answers the three count queries of the pre-flight."""

    def __init__(self, foreign=0, rows=12797, expected=12797):
        self.foreign, self.rows, self.expected = foreign, rows, expected

    def sql(self, query):
        if "embedding_model <>" in query:
            value = self.foreign
        elif "chunk_role = 'RETRIEVAL'" in query:
            value = self.expected
        else:
            value = self.rows

        class Result:
            def collect(self_inner):
                return [{"n": value}]

        return Result()


def index_for(client):
    return VectorSearchIndex(
        RS.retrieval.vector_search, INDEX, SOURCE, 1024, client=client, sleep=lambda _s: None
    )


def gep():
    return {name: FakeIndex(name.replace("_index", "_source")) for name in GEP}


def test_repository_config_reuses_the_existing_endpoint_with_a_project_index():
    vs = RS.retrieval.vector_search
    assert vs.endpoint == "worldbank-gep-ai-search" and vs.create_endpoint is False
    assert INDEX == "worldbank_copilot.silver.document_chunk_index_qwen3_v1"
    assert SOURCE == "worldbank_copilot.silver.document_chunk_index"
    assert INDEX not in GEP and not INDEX.startswith("worldbank_ai.")


def test_missing_endpoint_is_never_created_by_default():
    client = Client(state=None)
    with pytest.raises(VectorSearchUnavailable, match="create_endpoint is false"):
        index_for(client).ensure_endpoint()
    assert client.mutations == []


def test_preflight_passes_for_a_new_index_beside_the_gep_indexes():
    client = Client(indexes=gep())
    report = rp.preflight_index(Spark(), SETTINGS, RS, index_for(client), QWEN)
    assert report.ok and report.action == "CREATE"
    assert report.existing_indexes == sorted(GEP)
    assert client.mutations == []  # pre-flight is read-only


def test_preflight_refuses_a_name_collision_with_another_source():
    indexes = gep()
    indexes[INDEX] = FakeIndex("worldbank_ai.rag.gep_structure_v1_source")
    report = rp.preflight_index(Spark(), SETTINGS, RS, index_for(Client(indexes=indexes)), QWEN)
    assert not report.ok and report.action == "COLLISION"


def test_preflight_reuses_this_projects_existing_index_on_rerun():
    indexes = gep()
    indexes[INDEX] = FakeIndex(SOURCE)
    report = rp.preflight_index(Spark(), SETTINGS, RS, index_for(Client(indexes=indexes)), QWEN)
    assert report.ok and report.action == "REUSE"


@pytest.mark.parametrize(
    ("spark", "client_state", "model", "failing"),
    [
        (Spark(foreign=5), "ONLINE", QWEN, "only configured model/dimension in source"),
        (Spark(rows=100), "ONLINE", QWEN, "source covers every retrieval chunk"),
        (Spark(), "PROVISIONING", QWEN, "endpoint ONLINE"),
        (Spark(), "ONLINE", "databricks-gte-large-en", "embedding model configured"),
    ],
)
def test_preflight_failures(spark, client_state, model, failing):
    report = rp.preflight_index(spark, SETTINGS, RS, index_for(Client(client_state, gep())), model)
    assert not report.ok
    assert [c.name for c in report.checks if not c.ok] == [failing]


def test_index_creation_quota_stops_cleanly_without_touching_other_indexes():
    error = Exception("Maximum number of indexes per endpoint exceeded quota of 3")
    client = Client(indexes=gep(), create_error=error)
    with pytest.raises(VectorSearchUnavailable, match="quota reached"):
        index_for(client).ensure_index()
    assert [m[0] for m in client.mutations] == ["create_index"]  # one attempt, no deletes
    assert sorted(client.indexes) == sorted(GEP)


def test_index_is_created_once_and_reused_on_rerun():
    client = Client(indexes=gep())
    assert index_for(client).ensure_index() == "CREATED"
    spec = client.mutations[0][1]
    assert (spec["index_name"], spec["source_table_name"]) == (INDEX, SOURCE)
    assert (
        spec["endpoint_name"] == "worldbank-gep-ai-search" and spec["embedding_dimension"] == 1024
    )
    assert index_for(client).ensure_index() == "EXISTS"
    assert len(client.mutations) == 1


def test_quota_error_detection():
    assert is_quota_error(Exception("Maximum number of AI Search endpoints exceeded quota of 1"))
    assert not is_quota_error(Exception("PERMISSION_DENIED"))
