"""Unit tests: Investigator action governance, governed evidence, packing, enrichment,
per-claim integrity and claim-level finalization. Offline; no models or network."""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from tests.unit.test_phase10d_unknown_provenance import entry

from worldbank_copilot.copilot.finalizer import (
    CONTRADICTION_NOTE,
    UNKNOWN_NOTE,
    check_integrity,
    finalize,
)
from worldbank_copilot.copilot.governed import (
    OBJECTIVE,
    Gathered,
    GovernanceViolation,
    GovernedExecutor,
    anchor,
    approved_context,
    pack,
)
from worldbank_copilot.copilot.investigator import (
    Action,
    ActionTool,
    InvestigatorDecision,
    govern,
    tool_catalog,
)
from worldbank_copilot.copilot.semantic import (
    SemanticReview,
    SemanticSynthesis,
    enrich,
    project,
)
from worldbank_copilot.investigation.evidence_models import EvidenceReference
from worldbank_copilot.investigation.model_protocol import StructuredTool
from worldbank_copilot.tools.registry import TOOL_SPECS
from worldbank_copilot.validation.phase10d_fixtures import fixture

PROJECT = "P000001"


# -- Investigator action governance --------------------------------------------------
def test_action_tools_are_exactly_the_governed_tools():
    structured = {s.name for s in TOOL_SPECS if s.tables}
    assert {t.value for t in ActionTool} == structured | {"search_documents"}
    assert {t.value for t in StructuredTool} == structured
    catalog = tool_catalog()
    assert all("project_id" not in c["arguments"] for c in catalog)


def act(tool, query=None, **arguments):
    return Action(
        tool=tool,
        arguments=tuple({"name": k, "value": json.dumps(v)} for k, v in arguments.items()),
        query=query,
        purpose="p",
    )


def test_project_is_injected_by_code_and_arguments_are_schema_checked():
    call = govern(act("get_rating_history", rating_types=["PDO"]), PROJECT)
    assert call.arguments["project_id"] == PROJECT and call.arguments["rating_types"] == ["PDO"]
    assert govern(act("get_rating_history", rating_types=["NOT_A_RATING"]), PROJECT) is None
    assert govern(act("get_rating_history", project_id="P999999"), PROJECT) is None
    assert govern(act("get_rating_history", query_text="x"), PROJECT) is None


@pytest.mark.parametrize(
    "query", ["", "drop table x", "http://x.org", "/Volumes/a/b", "C:\\secret"]
)
def test_unsafe_or_empty_search_queries_are_rejected(query):
    assert govern(act("search_documents", query=query), PROJECT) is None


def test_search_query_is_model_text_executed_only_as_search():
    call = govern(act("search_documents", query="procurement delays"), PROJECT)
    assert call.is_document and call.query == "procurement delays" and call.arguments is None
    assert govern(act("search_documents", query="x", limit=5), PROJECT) is None


def test_decision_contract_is_bounded():
    with pytest.raises(ValidationError):
        InvestigatorDecision.model_validate(
            {
                "disposition": "INVESTIGATE",
                "objective": "o",
                "actions": [act("search_documents", query="q").model_dump()] * 7,
                "review_evidence": False,
            }
        )
    with pytest.raises(ValidationError):
        Action.model_validate({"tool": "run_sql", "arguments": [], "purpose": "p"})


# -- governed evidence --------------------------------------------------------------
def ref(kind="DOCUMENTED_FINDING", record="r1", **payload):
    data = entry(kind, record)
    data["project_id"] = PROJECT
    data["identity"]["project_id"] = PROJECT
    data.pop("source_identity")
    data["payload"] = {**data["payload"], **payload}
    from worldbank_copilot.investigation.models import fingerprint

    data["evidence_id"] = fingerprint("ev", data["identity"])
    return EvidenceReference.model_validate(data)


def gathered(*calls):
    g = Gathered()
    for refs in calls:
        g.add(SimpleNamespace(tool="t", status="OK", evidence=0, query=None), refs)
    return g


def test_packing_is_round_robin_in_ranked_order_and_trims_text():
    a = [ref(record=f"a{i}", text="x" * 5000) for i in range(3)]
    b = [ref(record=f"b{i}") for i in range(3)]
    g = gathered(a, b)
    entries = pack(g, set(g.refs), max_bytes=100_000, max_text_chars=300)
    order = [e["evidence_id"] for e in entries]
    assert order[:2] == [a[0].evidence_id, b[0].evidence_id]  # each call's best first
    assert all(len(e["payload"].get("text") or "") <= 300 for e in entries)
    small = pack(g, set(g.refs), max_bytes=3000, max_text_chars=300)
    assert 0 < len(small) < 6 and small[0]["evidence_id"] == a[0].evidence_id


def test_packing_keeps_identity_and_citation_fields_intact():
    r = ref(text="y" * 3000)
    (e,) = pack(gathered([r]), {r.evidence_id}, max_bytes=50_000, max_text_chars=200)
    assert e["evidence_id"] == r.evidence_id and e["source_identity"]
    assert e["citation"] == r.model_dump(mode="json")["citation"]


def test_anchor_uses_the_events_source_stated_date():
    event = ref("FACT", "ev", event_type="RESTRUCTURING", event_date="2020-06-01")
    before = ref(record="d1", document_date="2019-01-01")
    after = ref(record="d2", document_date="2021-01-01")
    undated = ref(record="d3")
    g = gathered([event, before, after, undated])
    kept, periods, dropped, usable = anchor(g, event.evidence_id, "BEFORE")
    assert usable and dropped == 1
    assert kept == {event.evidence_id, before.evidence_id, undated.evidence_id}
    assert periods[after.evidence_id] == "AFTER" and periods[undated.evidence_id] == "UNDATED"
    kept, _, dropped, _ = anchor(g, event.evidence_id, "COMPARE")
    assert dropped == 0 and len(kept) == 4


def test_anchor_without_a_source_stated_date_is_not_used():
    estimated = ref("FACT", "ev", event_type="RESTRUCTURING", event_date=None)
    g = gathered([estimated, ref(record="d1", document_date="2019-01-01")])
    kept, periods, dropped, usable = anchor(g, estimated.evidence_id, "BEFORE")
    assert not usable and dropped == 0 and len(kept) == 2


class FakeTools:
    def __init__(self, result):
        self.result = result

    def run(self, *args, **kwargs):
        return self.result


def test_foreign_project_data_from_a_tool_is_a_governance_violation():
    from worldbank_copilot.tools.models import ToolResult

    spec = next(s for s in TOOL_SPECS if s.name == "get_attention_signals")
    foreign = ToolResult(
        tool=spec.name,
        tool_version=spec.version,
        request_id="r",
        project_id="P179039",
        status="EMPTY",
    )
    executor = GovernedExecutor(FakeTools(foreign), None, lambda rid: None, PROJECT, "r")
    call = govern(act("get_attention_signals"), PROJECT)
    with pytest.raises(GovernanceViolation):
        executor.run(call, Gathered())


# -- context, projection and enrichment ----------------------------------------------
def context_for(*refs):
    g = gathered(list(refs))
    entries = pack(g, set(g.refs), max_bytes=100_000, max_text_chars=1000)
    return approved_context(
        request_id="r",
        project_id=PROJECT,
        question="q",
        objective="o",
        temporal_scope=fixture().temporal_scope,
        entries=entries,
        gathered=g,
    )


def semantic(*claims, insufficient=False):
    return SemanticSynthesis.model_validate(
        {
            "claims": [{"text": t, "evidence": list(h), "interpretation": i} for t, h, i in claims],
            "insufficient_evidence": insufficient,
            "limitations": [],
        }
    )


def test_projection_has_handles_objective_and_no_identities():
    context = context_for(ref("FACT", "a"), ref(record="b"))
    projection = project(context, objective="obj", periods=None)
    payload = json.dumps(projection.payload)
    assert projection.payload["objective"] == "obj"
    assert [e["handle"] for e in projection.payload["evidence"]] == ["E1", "E2"]
    for evidence_id in projection.evidence.values():
        assert evidence_id not in payload


def test_enrichment_owns_identity_and_drops_only_invalid_claims():
    context = context_for(ref("FACT", "a"), ref(record="b"))
    projection = project(context, objective="o")
    output, dropped, stats = enrich(
        semantic(
            ("Fact.", ["E1"], False),
            ("Bad handle.", ["E9"], False),
            ("Duplicate.", ["E1", "E1"], False),
            ("Mixed assertion.", ["E1", "E2"], False),
            ("Mixed interpretation.", ["E1", "E2"], True),
        ),
        projection,
        context,
    )
    assert [c.claim_id for c in output.candidate_claims] == ["C1", "C5"]
    assert dropped == {"EVIDENCE_REFERENCE_INVALID": 2, "PROVENANCE_VIOLATION": 1}
    first = output.candidate_claims[0]
    assert first.project_id == PROJECT and first.requirement_ids == (OBJECTIVE,)
    assert first.temporal_scope.model_dump(mode="json") == context.temporal_scope
    assert output.candidate_claims[1].provenance_label == "AI_INTERPRETATION"
    assert stats.handles_failed == 2 and stats.claims_dropped == 3


# -- integrity and claim-level finalization ------------------------------------------
def review(*supports):
    return SemanticReview.model_validate(
        {
            "findings": [
                {"claim": f"C{i}", "support": s, "rationale": f"note {i}"}
                for i, s in enumerate(supports, 1)
            ]
        }
    )


def draft(*claims, insufficient=False):
    context = context_for(ref("FACT", "a"), ref(record="b"), ref("UNKNOWN", "u"))
    output, _, _ = enrich(
        semantic(*claims, insufficient=insufficient), project(context, objective="o"), context
    )
    return output, context


def test_enriched_valid_claims_pass_the_unchanged_validator():
    output, context = draft(("Fact.", ["E1"], False), ("Finding.", ["E2"], False))
    integrity = check_integrity(output, context)
    assert len(integrity.claims) == 2 and not integrity.removed and not integrity.security_failures


def test_claim_level_publication():
    output, context = draft(
        ("A.", ["E1"], False), ("B.", ["E2"], False), ("C.", ["E1"], False), ("D.", ["E2"], False)
    )
    final = finalize(
        check_integrity(output, context),
        review("SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED"),
        critic_enabled=True,
        model_insufficient=False,
        limitations=(),
    )
    assert final.disposition == "PUBLISH_WITH_LIMITATIONS"
    assert [(p.claim.claim_id, p.support, p.qualifier) for p in final.published] == [
        ("C1", "SUPPORTED", None),
        ("C2", "PARTIALLY_SUPPORTED", "note 2"),
    ]
    assert final.removed == {"UNSUPPORTED": 1, "CONTRADICTED": 1}
    assert CONTRADICTION_NOTE in final.limitations


def test_unknown_claim_is_withheld_without_blocking_others():
    output, context = draft(("Known.", ["E1"], False), ("Missing.", ["E3"], False))
    assert output.candidate_claims[1].provenance_label == "UNKNOWN"
    final = finalize(
        check_integrity(output, context),
        review("SUPPORTED", "SUPPORTED"),
        critic_enabled=True,
        model_insufficient=False,
        limitations=(),
    )
    assert [p.claim.claim_id for p in final.published] == ["C1"]
    assert final.removed == {"UNKNOWN_VALUE": 1} and UNKNOWN_NOTE in final.limitations


def test_interpretation_over_unknown_is_removed_by_the_unchanged_validator():
    output, context = draft(("Known.", ["E1"], False), ("Guess.", ["E3"], True))
    integrity = check_integrity(output, context)
    assert integrity.removed == {"PROVENANCE_VIOLATION": 1} and len(integrity.claims) == 1


def test_security_failure_fails_the_whole_response():
    output, context = draft(("Unlike P999999, a delay.", ["E1"], False), ("B.", ["E2"], False))
    integrity = check_integrity(output, context)
    final = finalize(
        integrity, review("SUPPORTED", "SUPPORTED"), critic_enabled=True,
        model_insufficient=False, limitations=(),
    )  # fmt: skip
    assert final.disposition == "FAIL_CLOSED" and not final.published
    assert final.failures == ("PROJECT_ISOLATION_VIOLATION",)


def test_nothing_publishable_is_insufficient_and_model_flag_is_only_a_limitation():
    output, context = draft(("A.", ["E1"], False))
    integrity = check_integrity(output, context)
    none = finalize(integrity, review("UNSUPPORTED"), critic_enabled=True,
                    model_insufficient=False, limitations=())  # fmt: skip
    assert none.disposition == "INSUFFICIENT_EVIDENCE"
    partial = finalize(integrity, review("SUPPORTED"), critic_enabled=True,
                       model_insufficient=True, limitations=())  # fmt: skip
    assert partial.disposition == "PUBLISH_WITH_LIMITATIONS" and partial.limitations
