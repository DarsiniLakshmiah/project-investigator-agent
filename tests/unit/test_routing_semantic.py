"""Phase 9D: bounded semantic fallback - safety envelope, contract and dev-only protocol."""

import json

import pytest
from tests.conftest import REPO_CONFIG_DIR, REPO_ROOT
from tests.support.routing_fixtures import ALL, CONFIG, harness
from tests.support.tool_fixtures import IPF, PFORR

from worldbank_copilot.routing.evaluation import load_dataset
from worldbank_copilot.routing.models import (
    ExecutionOutcome,
    Intent,
    Route,
    SemanticDecision,
)
from worldbank_copilot.routing.semantic import (
    CACHE_NAMESPACE,
    SEMANTIC_INTENTS,
    Example,
    ExampleStore,
    KnnConfig,
    KnnSemanticClassifier,
    LexicalEmbedder,
    ProviderEmbedder,
    SemanticQueryContext,
    cosine,
    route_map,
)
from worldbank_copilot.routing.semantic_eval import (
    FrozenTestEvaluationBlocked,
    assert_split_allowed,
    load_protocol,
    lofo,
    select,
)

DATASET = load_dataset(REPO_ROOT / "evaluation" / "routing_cases.yaml")
ROUTES = route_map(CONFIG.requirements)
UNRESOLVED = "How has the way citizens can register complaints been improved?"


class SpyClassifier:
    """Records every invocation and returns a fixed bounded decision."""

    name = "spy"

    def __init__(self, intent=Intent.DOCUMENT_CONTENT, abstain=False, route=None):
        self.calls = []
        self.intent, self.abstain, self.route = intent, abstain, route

    def classify(self, context):
        self.calls.append(context)
        return SemanticDecision(
            classifier=self.name,
            version="test",
            abstain=self.abstain,
            intent=None if self.abstain else self.intent,
            route=None if self.abstain else (self.route or ROUTES.get(self.intent)),
            confidence=0.9,
            reason="spy",
        )


def with_semantic(classifier, **kw):
    h = harness(**kw)
    h.service.semantic = classifier
    return h


# -- invocation boundary --------------------------------------------------------------------

SECURITY_CASES = [
    c
    for c in DATASET.cases
    if c.expected.reason_code
    in {
        "CROSS_PROJECT",
        "MULTI_PROJECT_NOT_SUPPORTED",
        "UNSUPPORTED_PROJECT",
        "NOT_AUTHORIZED",
        "PROJECT_REQUIRED",
        "INVALID_REQUEST",
        "AMBIGUOUS_PROJECT_REFERENCE",
        "PREDICTION_NOT_SUPPORTED",
        "READ_ONLY",
        "OUT_OF_DOMAIN",
    }
]


@pytest.mark.parametrize("case", SECURITY_CASES, ids=lambda c: c.case_id)
def test_classifier_is_never_invoked_for_pre_execution_decisions(case):
    spy = SpyClassifier(Intent.PROJECT_OVERVIEW)
    h = with_semantic(spy)
    auth = ALL if case.authorized_projects == "ALL" else tuple(case.authorized_projects)
    r = h.ask(case.question, active=case.active_project_id, authorized=auth)
    assert spy.calls == [] and r.semantic is None
    assert r.decision.route in (Route.REFUSE, Route.CLARIFY)
    assert h.executor.calls == [] and h.reads == [] and h.retriever.first_stage_calls == []


@pytest.mark.parametrize(
    "question",
    [
        "What is the current closing date?",
        "Show the PDO rating history",
        "Why was the closing date extended?",
        "Why did the PDO rating drop?",
        "What is the overall risk rating?",
        "List the restructurings since the restructuring",
    ],
)
def test_classifier_is_not_invoked_when_rules_resolve(question):
    spy = SpyClassifier()
    r = with_semantic(spy).ask(question)
    assert spy.calls == [] and r.semantic is None
    assert r.understanding.intent.method == "RULE"


def test_classifier_receives_a_scope_validated_context_only():
    spy = SpyClassifier(Intent.DOCUMENT_CONTENT)
    with_semantic(spy).ask(UNRESOLVED, active=PFORR)
    (context,) = spy.calls
    assert isinstance(context, SemanticQueryContext)
    assert context.project_id == PFORR and context.question == UNRESOLVED
    assert not hasattr(context, "authorized_projects")  # cannot broaden authorisation


# -- bounded decision consumed by the harness -----------------------------------------------


def test_semantic_intent_flows_through_the_deterministic_harness():
    spy = SpyClassifier(Intent.DOCUMENT_CONTENT)
    h = with_semantic(spy)
    r = h.ask(UNRESOLVED, active=PFORR)
    assert (
        r.understanding.intent.method == "SEMANTIC" and r.semantic.intent == Intent.DOCUMENT_CONTENT
    )
    assert r.decision.route == Route.DOCUMENT and r.retrieval_executed
    assert h.retriever.first_stage_calls == [(UNRESOLVED, PFORR)]  # harness, scoped to PFORR
    assert "semantic" in [t.stage for t in r.timings]


def test_semantic_overview_uses_only_the_overview_tool():
    """r046/r061 principle: the harness, not the classifier, chooses the minimal tools."""
    h = with_semantic(SpyClassifier(Intent.PROJECT_OVERVIEW))
    r = h.ask("How is it going?")
    assert r.decision.route == Route.STRUCTURED and r.executed_tools == ["get_project_overview"]


def test_the_classifier_never_changes_project_or_calls_tools():
    class Sneaky(SpyClassifier):
        def classify(self, context):
            self.executor_calls_during = len(executor.calls)
            return super().classify(context)

    sneaky = Sneaky(Intent.PROJECT_OVERVIEW)
    h = with_semantic(sneaky)
    executor = h.executor
    r = h.ask("How is it going?", active=IPF)
    assert sneaky.executor_calls_during == 0  # nothing ran before or during classification
    assert {scope for _, _, scope in h.executor.calls} == {IPF}
    assert r.understanding.project.project_id == IPF
    assert not hasattr(KnnSemanticClassifier, "executor")


def test_abstention_becomes_clarify_and_executes_nothing():
    h = with_semantic(SpyClassifier(abstain=True))
    r = h.ask(UNRESOLVED)
    assert (r.decision.route, r.decision.reason_code) == (Route.CLARIFY, "SEMANTIC_ABSTAIN")
    assert r.outcome == ExecutionOutcome.NOT_EXECUTED and r.semantic.abstain
    assert h.executor.calls == [] and h.retriever.first_stage_calls == [] and h.reads == []


@pytest.mark.parametrize("intent", sorted(set(Intent) - set(SEMANTIC_INTENTS)))
def test_out_of_contract_intents_are_rejected_as_abstention(intent):
    h = with_semantic(SpyClassifier(intent, route="STRUCTURED"))
    r = h.ask(UNRESOLVED)
    assert r.decision.reason_code == "SEMANTIC_ABSTAIN" and "out-of-contract" in r.semantic.reason
    assert h.executor.calls == []


def test_without_a_classifier_9b2_behaviour_is_unchanged():
    r = harness().ask(UNRESOLVED)
    assert r.decision.route == Route.SEMANTIC_CLASSIFICATION_REQUIRED and r.semantic is None


# -- development-only examples and protocol ---------------------------------------------------


def test_example_store_holds_development_cases_only():
    store = ExampleStore.from_dataset(DATASET)
    dev = {c.case_id for c in DATASET.cases if c.split == "dev"}
    assert store.examples and {e.example_id for e in store.examples} <= dev
    test_case = next(c for c in DATASET.cases if c.split == "test")
    leaked = ExampleStore(
        [
            *store.examples,
            Example(test_case.case_id, test_case.family, test_case.question, Intent.RISKS),
        ]
    )
    with pytest.raises(ValueError, match="non-development"):
        leaked.assert_development_only(DATASET)


def test_leave_one_family_out_never_uses_the_own_family():
    store = ExampleStore.from_dataset(DATASET)
    clf = KnnSemanticClassifier(LexicalEmbedder(), store, KnnConfig(k=5), ROUTES)
    family = {e.example_id: e.family for e in store.examples}
    for e in store.examples:
        d = clf.classify(
            SemanticQueryContext(e.example_id, e.question, "DEV", "NONE", ()),
            exclude_families=frozenset({e.family}),
        )
        assert all(family[n.example_id] != e.family for n in d.neighbours)
    assert len(lofo(clf, store.examples)) == len(store.examples)


def test_abstention_thresholds_are_applied():
    store = ExampleStore.from_dataset(DATASET)
    strict = KnnSemanticClassifier(
        LexicalEmbedder(), store, KnnConfig(k=3, min_similarity=0.99), ROUTES
    )
    d = strict.classify(SemanticQueryContext("x", "zebra unicorn quantum", "P", "NONE", ()))
    assert d.abstain and d.intent is None and "below" in d.reason


def test_lexical_embedder_is_deterministic_and_normalised():
    e = LexicalEmbedder()
    a, b = e.embed(["Why was the closing date extended?"] * 2)
    assert a == b and abs(cosine(a, a) - 1.0) < 1e-9


def test_provider_embedder_uses_its_own_namespace_and_cache(tmp_path):
    class FakeProvider:
        model = "databricks-qwen3-embedding-0-6b"

        def __init__(self):
            self.calls = []

        def embed(self, texts):
            self.calls.append(list(texts))
            return [[1.0, 2.0] for _ in texts]

    provider = FakeProvider()
    cache = tmp_path / "routing.json"
    emb = ProviderEmbedder(provider, cache_path=cache)
    emb.embed(["a question", "a question", "another"])
    emb.embed(["a question"])
    assert provider.calls == [["a question", "another"]]  # cached, deduplicated
    keys = json.loads(cache.read_text())
    assert all(k.startswith(f"{CACHE_NAMESPACE}/{provider.model}/") for k in keys)
    assert CACHE_NAMESPACE == "routing-question"


def test_frozen_test_split_cannot_be_evaluated_in_this_checkpoint():
    assert_split_allowed("dev")
    with pytest.raises(FrozenTestEvaluationBlocked):
        assert_split_allowed("test")
    with pytest.raises(FrozenTestEvaluationBlocked):
        assert_split_allowed("test", {"status": "PROVISIONAL"}, authorised=True)
    with pytest.raises(FrozenTestEvaluationBlocked):
        assert_split_allowed("test", {"status": "FROZEN"}, authorised=False)
    assert_split_allowed("test", {"status": "FROZEN"}, authorised=True)  # the only way


def test_selection_rule_is_pre_registered_and_can_reject_everything():
    protocol = load_protocol(REPO_CONFIG_DIR)
    assert protocol.selection == {"min_route_precision": 0.9, "min_coverage": 0.3}

    def r(prec, cov, k=3, s=0.0):
        return {
            "name": f"{prec}-{cov}",
            "config": {"k": k, "weighting": "uniform", "min_similarity": s, "min_vote_share": 0.0},
            "metrics": {"route_precision": prec, "coverage": cov, "intent_precision": prec},
        }

    assert select([r(0.8, 0.9), r(0.95, 0.2)], protocol) is None
    assert select([r(0.9, 0.4), r(0.95, 0.6, k=5), r(1.0, 0.6, k=1, s=0.3)], protocol)["name"] in (
        "0.95-0.6",
        "1.0-0.6",
    )


def test_development_artefacts_record_that_test_was_not_evaluated():
    log = json.loads(
        (REPO_ROOT / "evaluation" / "semantic_dev_experiments_9d.json").read_text(encoding="utf-8")
    )
    dev = {c.case_id for c in DATASET.cases if c.split == "dev"}
    assert log["test_evaluated"] is False and log["split_evaluated"] == "dev"
    assert set(log["dev_case_ids"]) == dev
    for run in log["runs"].values():
        assert run["test_evaluated"] is False and set(run["examples"]) <= dev
        assert all(x["split"] == "dev" for x in run["experiments"])
        assert {p["example_id"] for p in run["base_lofo"]} <= dev
    decision = (REPO_ROOT / "evaluation" / "semantic_config_9d.yaml").read_text(encoding="utf-8")
    assert "test_evaluated: false" in decision and "status: PROVISIONAL" in decision


# -- Databricks development guards ----------------------------------------------------------


def _guard_inputs():
    from worldbank_copilot.routing.config import load_routing_config

    return load_routing_config(REPO_CONFIG_DIR), ExampleStore.from_dataset(DATASET)


def test_development_guards_pass_on_the_frozen_ground_truth():
    from worldbank_copilot.routing.semantic_eval import development_guards, require_guards

    routing, store = _guard_inputs()
    results = development_guards(
        REPO_ROOT, DATASET, routing, store, {"status": "PROVISIONAL", "test_evaluated": False}
    )
    assert len(results) == 8 and all(r["passed"] for r in results), results
    require_guards(results)


def test_development_guards_fail_closed():
    from worldbank_copilot.routing.semantic_eval import (
        DevelopmentGuardFailed,
        development_guards,
        require_guards,
    )

    routing, store = _guard_inputs()
    bad_decision = development_guards(
        REPO_ROOT, DATASET, routing, store, {"status": "FROZEN", "test_evaluated": True}
    )
    with pytest.raises(DevelopmentGuardFailed, match="TEST evaluation"):
        require_guards(bad_decision)
    test_case = next(c for c in DATASET.cases if c.split == "test")
    leaked = ExampleStore(
        [
            *store.examples,
            Example(test_case.case_id, test_case.family, test_case.question, Intent.RISKS),
        ]
    )
    results = development_guards(
        REPO_ROOT, DATASET, routing, leaked, {"status": "PROVISIONAL", "test_evaluated": False}
    )
    failed = {r["guard"] for r in results if not r["passed"]}
    assert failed == {
        "experiment inputs are a subset of DEV",
        "no TEST question is used as an example",
    }


def test_freeze_manifest_matches_the_reviewed_dataset():
    from worldbank_copilot.routing.semantic_eval import canonical_sha256

    manifest = json.loads(
        (REPO_ROOT / "evaluation" / "routing_freeze_9c.json").read_text(encoding="utf-8")
    )
    assert canonical_sha256(REPO_ROOT / manifest["dataset"]) == manifest["dataset_sha256_lf"]
    assert manifest["dev_case_ids"] == sorted(c.case_id for c in DATASET.cases if c.split == "dev")
    assert len(manifest["test_case_ids"]) == 51 and manifest["router_version"] == "9B.2"


def test_canonical_hash_ignores_line_endings(tmp_path):
    from worldbank_copilot.routing.semantic_eval import canonical_sha256

    (tmp_path / "a").write_bytes(b"x: 1\r\ny: 2\r\n")
    (tmp_path / "b").write_bytes(b"x: 1\ny: 2\n")
    assert canonical_sha256(tmp_path / "a") == canonical_sha256(tmp_path / "b")


def test_probe_reports_availability_dimension_and_errors():
    from worldbank_copilot.routing.semantic_eval import probe_embedder

    class Ok:
        model = "m"

        def embed(self, texts):
            return [[0.0] * 1024 for _ in texts]

    class Down:
        model = "m"

        def embed(self, texts):
            raise RuntimeError("429 REQUEST_LIMIT_EXCEEDED")

    ok = probe_embedder(Ok(), 1024)
    assert ok["available"] and ok["dimension"] == 1024 and ok["dimension_matches"]
    assert not probe_embedder(Ok(), 768)["dimension_matches"]
    down = probe_embedder(Down(), 1024)
    assert not down["available"] and "429" in down["error"]


def test_similarity_diagnostic_detects_inert_thresholds():
    from worldbank_copilot.routing.semantic_eval import similarity_diagnostic

    store = ExampleStore.from_dataset(DATASET)
    protocol = load_protocol(REPO_CONFIG_DIR)

    class Constant:
        name = "constant"

        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]  # every cosine = 1.0

    inert = similarity_diagnostic(
        KnnSemanticClassifier(Constant(), store, KnnConfig(), ROUTES), protocol
    )
    assert inert["similarity_thresholds_inert"] is True
    lexical = similarity_diagnostic(
        KnnSemanticClassifier(LexicalEmbedder(), store, KnnConfig(), ROUTES), protocol
    )
    assert lexical["similarity_thresholds_inert"] is False


# -- notebook 08 probe regression (EmbeddingConfig has no `dimension`) ------------------------

NOTEBOOK_08 = REPO_ROOT / "notebooks" / "08_semantic_routing_dev.py"


def test_notebook_08_only_reads_existing_embedding_config_fields():
    """Regression: notebook 08 read `rs.embeddings.dimension`, which does not exist."""
    import ast

    from worldbank_copilot.retrieval.config import EmbeddingConfig

    source = "\n".join(
        line
        for line in NOTEBOOK_08.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("# MAGIC")
    )
    accessed = {
        node.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "embeddings"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "rs"
    }
    assert accessed, "notebook 08 no longer reads the embedding configuration"
    assert accessed <= set(EmbeddingConfig.model_fields), accessed - set(
        EmbeddingConfig.model_fields
    )
    assert "dimension" not in EmbeddingConfig.model_fields  # the field is expected_dimension


def test_notebook_probe_path_measures_the_dimension_from_the_returned_vector(tmp_path):
    """The exact notebook 08 probe path with the real configuration and a fake provider."""
    from worldbank_copilot.retrieval.config import load_retrieval_settings
    from worldbank_copilot.routing.semantic_eval import probe_embedder

    rs = load_retrieval_settings(REPO_CONFIG_DIR)
    assert rs.embeddings.expected_dimension == 1024  # validated Phase 8 value

    class FakeQwen:
        model = rs.embeddings.endpoint

        def __init__(self, dimension, fail=None):
            self.dimension_returned, self.fail = dimension, fail
            self.requests = self.retries = self.rate_limited = 0
            self.rate_limit_wait = 0.0
            self.calls = []

        def embed(self, texts):
            self.calls.append(list(texts))
            self.requests += 1
            if self.fail:
                self.rate_limited += 1
                raise RuntimeError(self.fail)
            return [[0.1] * self.dimension_returned for _ in texts]

    ok = FakeQwen(1024)
    embedder = ProviderEmbedder(ok, cache_path=tmp_path / "routing.json")
    embedder.embed(["routing probe"])  # even if the probe text were cached ...
    probe = probe_embedder(embedder.provider, rs.embeddings.expected_dimension)
    assert ok.calls == [["routing probe"], ["routing probe"]]  # ... the probe calls for real
    assert (probe["model"], probe["dimension"], probe["expected_dimension"]) == (
        "databricks-qwen3-embedding-0-6b",
        1024,
        1024,
    )
    assert probe["available"] and probe["dimension_matches"] and probe["error"] is None
    assert probe["provider_counters"]["requests"] == 1
    assert probe["provider_counters"]["rate_limited"] == 0

    wrong = probe_embedder(FakeQwen(768), rs.embeddings.expected_dimension)
    assert wrong["dimension"] == 768 and not wrong["dimension_matches"]
    limited = probe_embedder(
        FakeQwen(1024, fail="HTTP 429 REQUEST_LIMIT_EXCEEDED"), rs.embeddings.expected_dimension
    )
    assert not limited["available"] and "429" in limited["error"]
    assert limited["provider_counters"]["rate_limited"] == 1
