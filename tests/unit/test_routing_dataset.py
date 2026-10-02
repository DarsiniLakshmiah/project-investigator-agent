"""Phase 9C routing dataset: structure, review state and frozen split.

These tests never require the deterministic router to agree with the reviewed labels
(disagreements are measured in evaluation/routing_baseline_<version>.yaml). The only
router behaviour asserted here is the security guarantee: cases that must be refused or
clarified before execution execute nothing.
"""

import re

import pytest
import yaml
from tests.conftest import REPO_CONFIG_DIR, REPO_ROOT
from tests.support.routing_fixtures import harness

from worldbank_copilot.common import load_project_registry
from worldbank_copilot.retrieval.evaluation import load_questions
from worldbank_copilot.routing.config import load_routing_config
from worldbank_copilot.routing.evaluation import (
    HUMAN_ROUTES,
    RoutingDataset,
    load_dataset,
    similarity,
)
from worldbank_copilot.routing.models import ExecutionOutcome, ProjectStatus, TemporalKind
from worldbank_copilot.routing.requirements import SPECS

DATASET = load_dataset(REPO_ROOT / "evaluation" / "routing_cases.yaml")
CASES = DATASET.cases
BY_ID = {c.case_id: c for c in CASES}
REGISTRY = load_project_registry(REPO_CONFIG_DIR)
CONFIG = load_routing_config(REPO_CONFIG_DIR)
VERSION = CONFIG.settings.router_version
PHASE8 = {q.id: q for q in load_questions(REPO_ROOT / "evaluation" / "retrieval_questions.yaml")}
P179039_AGREED = {"q28", "q29", "q30", "q35", "q36"}
# Frozen 2026-10-01 (human-approved 29 dev / 51 test). Changing it needs a new review.
FROZEN_DEV = (
    "r002 r006 r012 r014 r015 r016 r017 r020 r025 r029 r030 r034 r037 r042 r044 r045 r048 "
    "r050 r052 r055 r056 r059 r060 r063 r065 r070 r076 r078 r080"
).split()
APPROVED_ALIASES = {
    "P130544": (
        "IN Karnataka Urban Water Supply Modernization Project",
        "Karnataka Urban Water Supply Modernization Project",
        "Karnataka Urban Water Supply Modernization",
        "Urban Water Supply Modernization Project",
        "Urban Water Supply Modernization",
    ),
    "P179039": (
        "Karnataka Sustainable Rural Water Supply Program",
        "Sustainable Rural Water Supply Program",
        "Sustainable Rural Water Supply",
    ),
    "P506272": (
        "Karnataka Water Security and Resilience Program",
        "Water Security and Resilience Program",
        "Water Security and Resilience",
    ),
}
PRE_EXECUTION_CODES = {
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


def norm(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


# -- review state --------------------------------------------------------------------------


def test_eighty_reviewed_labels_with_recorded_rules():
    assert len(CASES) == 80 and len(BY_ID) == 80
    assert DATASET.review_status == "REVIEWED" and {c.label_status for c in CASES} == {"REVIEWED"}
    assert DATASET.reviewed_on is not None and len(DATASET.semantic_rules) == 7


def test_a_reviewed_dataset_cannot_mix_draft_labels_or_omit_its_rules():
    raw = yaml.safe_load(
        (REPO_ROOT / "evaluation" / "routing_cases.yaml").read_text(encoding="utf-8")
    )
    mixed = {**raw, "cases": [{**raw["cases"][0], "label_status": "DRAFT"}, *raw["cases"][1:]]}
    with pytest.raises(ValueError, match="review_status"):
        RoutingDataset.model_validate(mixed)
    with pytest.raises(ValueError, match="semantic rules"):
        RoutingDataset.model_validate({**raw, "semantic_rules": []})


def test_split_is_frozen_exactly_as_approved():
    assert DATASET.split_status == "FROZEN"
    assert sorted(c.case_id for c in CASES if c.split == "dev") == sorted(FROZEN_DEV)
    assert sum(c.split == "test" for c in CASES) == 51


def test_aliases_are_exactly_the_approved_eleven_and_reviewed():
    assert CONFIG.aliases.reviewed_by
    assert {p: tuple(a) for p, a in CONFIG.aliases.aliases.items()} == APPROVED_ALIASES


# -- structure ------------------------------------------------------------------------------


def test_project_ids_are_valid_or_the_case_expects_a_non_execution():
    for c in CASES:
        e = c.expected
        if e.project_id is not None:
            assert e.project_id in REGISTRY, c.case_id
        if c.authorized_projects != "ALL":
            assert all(p in REGISTRY for p in c.authorized_projects), c.case_id
        if c.active_project_id is not None and c.active_project_id not in REGISTRY:
            assert e.project_status == ProjectStatus.UNSUPPORTED, c.case_id
        if e.project_status == ProjectStatus.RESOLVED:
            assert e.project_id and e.project_basis, c.case_id
        assert e.route in HUMAN_ROUTES


def test_phase8_references_exist_and_are_reused_verbatim():
    for c in CASES:
        if c.phase8_ref:
            q = PHASE8[c.phase8_ref]
            assert q.project_id == c.expected.project_id == c.active_project_id, c.case_id
            assert norm(q.question) == norm(c.question), c.case_id


def test_expected_tools_exist_and_their_key_arguments_validate():
    for c in CASES:
        for t in c.expected.tools:
            assert t.tool in SPECS, (c.case_id, t.tool)
            SPECS[t.tool].args_model.model_validate(
                {
                    "project_id": c.expected.project_id,
                    **({"query": c.question} if t.tool == "search_project_documents" else {}),
                    **t.args,
                }
            )


def test_families_never_cross_the_split():
    splits = {}
    for c in CASES:
        assert splits.setdefault(c.family, c.split) == c.split, c.family
    bad = {
        "version": 1,
        "cases": [
            {
                **CASES[0].model_dump(mode="json"),
                "case_id": "x1",
                "split": "dev",
                "label_status": "DRAFT",
            },
            {
                **CASES[0].model_dump(mode="json"),
                "case_id": "x2",
                "split": "test",
                "label_status": "DRAFT",
            },
        ],
    }
    with pytest.raises(ValueError, match="split across"):
        RoutingDataset.model_validate(bad)


def test_no_identical_or_near_identical_questions_across_splits():
    dev = [c for c in CASES if c.split == "dev"]
    test = [c for c in CASES if c.split == "test"]
    assert not {norm(c.question) for c in dev} & {norm(c.question) for c in test}
    assert max(similarity(a.question, b.question) for a in dev for b in test) < 0.7


def test_both_splits_cover_projects_and_the_test_split_covers_every_route():
    test = [c for c in CASES if c.split == "test"]
    dev = [c for c in CASES if c.split == "dev"]
    assert {c.expected.route for c in test} == set(HUMAN_ROUTES)
    assert {c.expected.project_id for c in test if c.expected.project_id} == set(
        REGISTRY.project_ids
    )
    assert sum((c.expected.project_id or c.active_project_id) == "P506272" for c in dev) >= 4


# -- human decisions ------------------------------------------------------------------------

DECIDED = {
    "r001": ("STRUCTURED", "INTENT_CURRENT_RATINGS"),
    "r003": ("STRUCTURED", "INTENT_FINANCIAL_STATUS"),
    "r004": ("STRUCTURED", "INTENT_RISKS"),
    "r005": ("STRUCTURED", "INTENT_RISKS"),
    "r007": ("CLARIFY", "TIME_REQUIRED"),
    "r008": ("DOCUMENT", "INTENT_DOCUMENT_CONTENT"),
    "r009": ("DOCUMENT", "INTENT_DOCUMENT_CONTENT"),
    "r010": ("DOCUMENT", "INTENT_DOCUMENT_CONTENT"),
    "r011": ("STRUCTURED", "INTENT_RISKS"),
    "r015": ("DOCUMENT", "INTENT_EXPLANATION"),
    "r017": ("DOCUMENT", "INTENT_EXPLANATION"),
    "r019": ("DOCUMENT", "INTENT_EXPLANATION"),
    "r027": ("INVESTIGATION", "INTENT_CHANGE_INVESTIGATION"),
    "r073": ("CLARIFY", "AMBIGUOUS_PROJECT_REFERENCE"),
    "r018": ("STRUCTURED", "INTENT_TIMELINE_EVENTS"),
    "r020": ("DOCUMENT", "INTENT_DOCUMENT_CONTENT"),
    "r044": ("STRUCTURED", "INTENT_RISKS"),
    "r046": ("STRUCTURED", "INTENT_MULTI"),
    "r050": ("DOCUMENT", "INTENT_EXPLANATION"),
    "r051": ("INVESTIGATION", "INTENT_CHANGE_INVESTIGATION"),
    "r052": ("INVESTIGATION", "INTENT_CHANGE_INVESTIGATION"),
    "r058": ("INVESTIGATION", "INTENT_CHANGE_INVESTIGATION"),
    "r056": ("INVESTIGATION", "INTENT_CHANGE_INVESTIGATION"),
    "r057": ("CLARIFY", "AMBIGUOUS_TIME"),
    "r060": ("STRUCTURED", "INTENT_FINANCIAL_STATUS"),
    "r061": ("STRUCTURED", "INTENT_PROJECT_OVERVIEW"),
}


def test_every_human_decision_is_applied_and_recorded():
    for case_id, (route, reason) in DECIDED.items():
        c = BY_ID[case_id]
        assert (c.expected.route, c.expected.reason_code) == (route, reason), case_id
        assert c.human_decision, case_id
    assert BY_ID["r073"].expected.project_status == ProjectStatus.AMBIGUOUS_REFERENCE
    assert "verification" in BY_ID["r005"].review_note.lower()


def test_r046_expects_only_the_overview_which_holds_every_requested_field():
    from worldbank_copilot.tools.project import DIRECT

    assert [t.tool for t in BY_ID["r046"].expected.tools] == ["get_project_overview"]
    assert {"project_status", "disbursed_usd", "financial_snapshot_date"} <= set(DIRECT)


def test_r060_relative_scope_has_an_explicit_governed_contract():
    t = BY_ID["r060"].expected.temporal
    assert (t.kind, t.explicit) == (TemporalKind.RELATIVE, True)
    assert (str(t.date_from), str(t.date_to), str(t.as_of)) == (
        "2025-09-01",
        "2026-08-31",
        "2026-08-31",
    )
    assert "snapshot" in t.as_of_basis and BY_ID["r060"].expected.route == "STRUCTURED"


def test_p179039_routes_and_q28_q30_notes_are_preserved():
    by_ref = {c.phase8_ref: c for c in CASES if c.phase8_ref}
    for ref in P179039_AGREED:
        assert by_ref[ref].expected.route == "DOCUMENT", ref
        assert "P179039_LIMITATION" in by_ref[ref].review_flags
    assert "LABEL_REVIEW" in by_ref["q28"].review_flags
    assert "label-review" in by_ref["q28"].review_note.lower()
    assert "total program size" in by_ref["q30"].review_note.lower()


# -- generated artefacts ------------------------------------------------------------------------


def test_generated_baselines_and_review_cover_every_case():
    current = yaml.safe_load(
        (REPO_ROOT / "evaluation" / f"routing_baseline_{VERSION}.yaml").read_text(encoding="utf-8")
    )
    assert current["router_versions"]["router"] == VERSION
    assert [r["case_id"] for r in current["results"]] == [c.case_id for c in CASES]
    original = yaml.safe_load(
        (REPO_ROOT / "evaluation" / "routing_baseline_9B.1.yaml").read_text(encoding="utf-8")
    )
    assert original["router_versions"]["router"] == "9B.1"
    r073 = next(r for r in original["results"] if r["case_id"] == "r073")
    assert r073["executed_tools"] == ["get_financial_status"]  # the pre-fix defect, as recorded
    review = (REPO_ROOT / "evaluation" / "routing_review_9c.md").read_text(encoding="utf-8")
    assert all(re.search(rf"\b{c.case_id}\b", review) for c in CASES)
    assert "REVIEWED" in review and all(
        f"| {a} |" in review for aliases in APPROVED_ALIASES.values() for a in aliases
    )


# -- security guarantee (the only router behaviour asserted here) ---------------------------------


@pytest.mark.parametrize(
    "case_id",
    [c.case_id for c in CASES if c.expected.reason_code in PRE_EXECUTION_CODES],
)
def test_pre_execution_refusals_and_clarifications_execute_nothing(case_id):
    c = BY_ID[case_id]
    h = harness()
    authorized = REGISTRY.project_ids if c.authorized_projects == "ALL" else c.authorized_projects
    r = h.ask(c.question, active=c.active_project_id, authorized=tuple(authorized))
    assert r.outcome == ExecutionOutcome.NOT_EXECUTED
    assert h.executor.calls == [] and h.reads == [] and h.retriever.first_stage_calls == []
    assert r.executed_tools == [] and r.anchor_results == [] and not r.retrieval_executed
