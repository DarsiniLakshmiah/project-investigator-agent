"""Unit tests: Investigator action governance, governed evidence, packing, enrichment,
per-claim integrity and claim-level finalization. Offline; no models or network."""

import json
from datetime import date
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from tests.unit.test_phase10d_unknown_provenance import entry

from worldbank_copilot.copilot.finalizer import (
    CONTRADICTION_NOTE,
    CRITIC_UNAVAILABLE_NOTE,
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
    matching_events,
    pack,
    resolve_anchor,
)
from worldbank_copilot.copilot.investigator import (
    Action,
    ActionTool,
    InvestigatorDecision,
    Rejection,
    TemporalAnchor,
    govern,
    tool_catalog,
)
from worldbank_copilot.copilot.semantic import (
    TEMPORAL_RELATION_UNVERIFIED,
    SemanticReview,
    SemanticSynthesis,
    enrich,
    project,
)
from worldbank_copilot.investigation.evidence_models import EvidenceReference
from worldbank_copilot.investigation.model_protocol import StructuredTool
from worldbank_copilot.routing.models import AnchorCandidate, TemporalStatus
from worldbank_copilot.routing.temporal import parse_temporal
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
    call, reason = govern(act("get_rating_history", rating_types=["PDO"]), PROJECT)
    assert reason is None
    assert call.arguments["project_id"] == PROJECT and call.arguments["rating_types"] == ["PDO"]
    for bad, code in (
        (act("get_rating_history", rating_types=["NOT_A_RATING"]), Rejection.SCHEMA_INVALID),
        (act("get_rating_history", project_id="P999999"), Rejection.PROJECT_ARGUMENT),
        (act("get_rating_history", query_text="x"), Rejection.SCHEMA_INVALID),
    ):
        assert govern(bad, PROJECT) == (None, code)


def test_structured_action_formatting_is_normalised_not_rejected():
    # D3: an empty query, a bare enum string and a scalar for a list field were rejected.
    bare = Action(
        tool="get_project_timeline",
        arguments=({"name": "event_types", "value": "RESTRUCTURING"},),
        query="",
        purpose="p",
    )
    call, reason = govern(bare, PROJECT)
    assert reason is None and call.arguments["event_types"] == ["RESTRUCTURING"]
    call, reason = govern(act("get_rating_history", query="ignored", rating_types="PDO"), PROJECT)
    assert reason is None and call.arguments["rating_types"] == ["PDO"]


@pytest.mark.parametrize(
    "query", ["", "drop table x", "http://x.org", "/Volumes/a/b", "C:\\secret"]
)
def test_unsafe_or_empty_search_queries_are_rejected(query):
    call, reason = govern(act("search_documents", query=query), PROJECT)
    assert call is None
    assert reason == (Rejection.EMPTY_QUERY if not query else Rejection.UNSAFE_QUERY)


def test_search_query_is_model_text_executed_only_as_search():
    call, _ = govern(act("search_documents", query="procurement delays"), PROJECT)
    assert call.is_document and call.query == "procurement delays" and call.arguments is None
    call, _ = govern(act("search_documents", query="x", limit=5), PROJECT)
    assert call.arguments is None  # model arguments never reach document search


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


WHEN = date(2020, 6, 1)


def test_anchor_filters_by_the_authoritative_date_and_keeps_undated_as_undated():
    event = ref("FACT", "ev", event_type="RESTRUCTURING", event_date="2020-06-01")
    before = ref(record="d1", document_date="2019-01-01")
    after = ref(record="d2", document_date="2021-01-01")
    undated = ref(record="d3")
    g = gathered([event, before, after, undated])
    kept, periods, dropped = anchor(g, WHEN, "BEFORE", event.evidence_id)
    assert dropped == 1
    assert kept == {event.evidence_id, before.evidence_id, undated.evidence_id}
    assert periods[event.evidence_id] == "EVENT" and periods[before.evidence_id] == "BEFORE"
    assert periods[after.evidence_id] == "AFTER" and periods[undated.evidence_id] == "UNDATED"
    kept, _, dropped = anchor(g, WHEN, "COMPARE")
    assert dropped == 0 and len(kept) == 4


# -- event-relative anchor resolution (D3) --------------------------------------------
BEFORE_RESTRUCTURING = parse_temporal("What happened before restructuring?")


def scope(*dates):
    candidates = tuple(
        AnchorCandidate(
            timeline_event_id=f"t{n}", event_type="RESTRUCTURING", event_date=d, title=f"R{n}"
        )
        for n, d in enumerate(dates, 1)
    )
    status = TemporalStatus.RESOLVED if len(candidates) == 1 else TemporalStatus.UNRESOLVED
    return BEFORE_RESTRUCTURING.model_copy(
        update={"status": status, "anchor_candidates": candidates}
    )


def NONE(choice):  # a governed timeline with no matching event
    return ()


def resolver_of(result):
    calls = []

    def resolver(parsed):
        calls.append(parsed)
        return result

    resolver.calls = calls
    return resolver


def test_one_source_dated_event_resolves_the_anchor():
    outcome = resolve_anchor(BEFORE_RESTRUCTURING, None, None, resolver_of(scope(WHEN)), NONE)
    assert outcome.resolved and outcome.when == WHEN and outcome.relation == "BEFORE"


def test_several_source_dated_events_are_ambiguous_not_guessed():
    outcome = resolve_anchor(
        BEFORE_RESTRUCTURING, None, None, resolver_of(scope(WHEN, date(2022, 3, 1))), NONE
    )
    assert outcome.status == "AMBIGUOUS" and outcome.when is None
    assert outcome.candidates == ("R1 (2020-06-01)", "R2 (2022-03-01)")


def test_known_event_without_a_source_stated_date_is_unresolved():
    outcome = resolve_anchor(BEFORE_RESTRUCTURING, None, None, resolver_of(scope()), NONE)
    assert outcome.status == "NO_SOURCE_DATED_EVENT" and not outcome.resolved
    estimated = ref("FACT", "ev", event_type="RESTRUCTURING", event_date=None)
    choice = TemporalAnchor(event="E1", relation="BEFORE")
    outcome = resolve_anchor(
        parse_temporal("What changed?"), choice, estimated, resolver_of(None), NONE
    )
    assert outcome.status == "NO_SOURCE_DATED_EVENT"


def test_investigator_chosen_dated_event_wins_without_a_timeline_lookup():
    event = ref("FACT", "ev", event_type="RESTRUCTURING", event_date="2020-06-01")
    resolver = resolver_of(scope())
    choice = TemporalAnchor(event="E1", relation="AFTER")
    outcome = resolve_anchor(BEFORE_RESTRUCTURING, choice, event, resolver, NONE)
    assert outcome.resolved and outcome.when == WHEN and outcome.relation == "AFTER"
    assert outcome.event_id == event.evidence_id and not resolver.calls


def test_question_that_is_not_event_relative_needs_no_anchor():
    resolver = resolver_of(scope())
    outcome = resolve_anchor(parse_temporal("What changed?"), None, None, resolver, NONE)
    assert outcome.status == "NOT_EVENT_RELATIVE" and not resolver.calls


# -- an event identified by the Investigator, confirmed by the governed timeline ----------
def event(event_type, on, candidate=None, title="Restructuring"):
    return SimpleNamespace(
        event_type=event_type,
        event_date=on,
        candidate_event_date=candidate,
        event_title=title,
        event_sequence=1,
        source=SimpleNamespace(record_id=f"{event_type}-{on or candidate}"),
    )


TIMELINE = (
    event("RESTRUCTURING", date(2021, 5, 20)),
    event("RESTRUCTURING", date(2024, 7, 23)),
    event("RESTRUCTURING", date(2024, 12, 10)),
    event("RESTRUCTURING", date(2026, 6, 29)),
    event("ISR_REPORT", date(2024, 7, 23), title="ISR 9"),
    event("RESTRUCTURING", None, candidate=date(2018, 6, 1)),  # derived date only
)


def identified(relation="BEFORE", event_type="RESTRUCTURING", **parts):
    return TemporalAnchor(relation=relation, event_type=event_type, **parts)


def confirmed(choice, question="What happened before the July 23, 2024 restructuring?"):
    calls = []

    def confirm(c):
        calls.append(c)
        return matching_events(TIMELINE, c)

    outcome = resolve_anchor(parse_temporal(question), choice, None, resolver_of(None), confirm)
    return outcome, calls


@pytest.mark.parametrize("relation", ["BEFORE", "AFTER"])
def test_explicit_date_and_named_event_resolve_from_the_governed_timeline(relation):
    # Cases 1-3: date + named event + BEFORE/AFTER, exactly one governed match.
    outcome, calls = confirmed(identified(relation, year=2024, month=7, day=23))
    assert outcome.resolved and outcome.relation == relation
    assert outcome.when == date(2024, 7, 23) and len(calls) == 1


def test_partial_date_matches_only_where_the_governed_timeline_agrees():
    # "Before the July 23 restructuring": no year, still exactly one governed match.
    outcome, _ = confirmed(identified(month=7, day=23))
    assert outcome.resolved and outcome.when == date(2024, 7, 23)
    outcome, _ = confirmed(identified(year=2024))  # two 2024 restructurings
    assert outcome.status == "AMBIGUOUS"
    assert outcome.candidates == ("Restructuring (2024-07-23)", "Restructuring (2024-12-10)")


def test_explicit_date_matching_no_governed_event_is_not_resolved():
    # Case 4.
    outcome, _ = confirmed(identified(year=2024, month=7, day=22))
    assert outcome.status == "NO_MATCHING_EVENT" and outcome.when is None


def test_same_date_matching_several_governed_events_is_ambiguous():
    # Case 5: the Investigator did not say which kind of event.
    outcome, _ = confirmed(identified(event_type=None, year=2024, month=7, day=23))
    assert outcome.status == "AMBIGUOUS"
    assert outcome.candidates == ("Restructuring (2024-07-23)", "ISR 9 (2024-07-23)")


def test_user_date_is_never_authoritative_without_governed_confirmation():
    # Case 7: a derived candidate date never confirms an event, and the parsed DATE scope
    # of the question is never turned into an anchor by itself.
    outcome, _ = confirmed(identified(year=2018, month=6, day=1))
    assert outcome.status == "NO_MATCHING_EVENT" and outcome.when is None
    question = parse_temporal("What happened before the July 23, 2024 restructuring?")
    assert question.kind == "DATE"  # the parser reads only the date
    outcome = resolve_anchor(question, None, None, resolver_of(None), NONE)
    assert outcome.status == "NOT_EVENT_RELATIVE" and not outcome.resolved


def test_type_only_anchor_lists_the_governed_choices():
    outcome, _ = confirmed(identified())
    assert outcome.status == "AMBIGUOUS" and len(outcome.candidates) == 4


def temporal_case():
    """E1 dated before the anchor, E2 dated after, E3 undated (document_date=None)."""
    refs = (
        ref(record="d1", document_date="2019-01-01"),
        ref(record="d2", document_date="2021-01-01"),
        ref(record="d3", document_date=None),
    )
    context = context_for(*refs)
    _, periods, _ = anchor(gathered(list(refs)), WHEN, "COMPARE")
    return context, periods


def related(*claims):
    return SemanticSynthesis.model_validate(
        {
            "claims": [
                {"text": t, "evidence": list(h), "interpretation": False, "temporal_relation": r}
                for t, h, r in claims
            ],
            "insufficient_evidence": False,
            "limitations": [],
        }
    )


def test_before_after_claims_need_a_resolved_anchor():
    # Case 6: the Synthesizer adds "before restructuring" while no anchor was resolved.
    context, _ = temporal_case()
    projection = project(context, objective="o")
    output, dropped, _ = enrich(
        related(("Before the restructuring, X.", ["E1"], "BEFORE"), ("X.", ["E1"], "NONE")),
        projection,
        context,
    )
    assert [c.claim_id for c in output.candidate_claims] == ["C2"]
    assert dropped == {TEMPORAL_RELATION_UNVERIFIED: 1}
    assert projection.payload["anchor"] is None


def test_undated_evidence_cannot_establish_before_or_after():
    # Cases 4 and 5: undated alone, and undated mixed with dated evidence.
    context, periods = temporal_case()
    projection = project(context, objective="o", periods=periods, anchor={"relation": "BEFORE"})
    output, dropped, _ = enrich(
        related(
            ("Dated before.", ["E1"], "BEFORE"),
            ("Undated before.", ["E3"], "BEFORE"),
            ("Mixed before.", ["E1", "E3"], "BEFORE"),
            ("Wrong side.", ["E2"], "BEFORE"),
            ("After.", ["E2"], "AFTER"),
        ),
        projection,
        context,
    )
    assert [c.claim_id for c in output.candidate_claims] == ["C1", "C5"]
    assert dropped == {TEMPORAL_RELATION_UNVERIFIED: 3}
    periods_shown = [e["period"] for e in projection.payload["evidence"]]
    assert periods_shown == ["BEFORE", "AFTER", "UNDATED"]


def test_undated_evidence_still_supports_ordinary_facts():
    # Case 7: non-temporal claims from undated evidence remain usable, anchor or not.
    context, periods = temporal_case()
    for projection in (
        project(context, objective="o"),
        project(context, objective="o", periods=periods, anchor={"relation": "BEFORE"}),
    ):
        output, dropped, _ = enrich(related(("Undated fact.", ["E3"], "NONE")), projection, context)
        assert len(output.candidate_claims) == 1 and not dropped


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
    call, _ = govern(act("get_attention_signals"), PROJECT)
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
            "claims": [
                {"text": t, "evidence": list(h), "interpretation": i, "temporal_relation": "NONE"}
                for t, h, i in claims
            ],
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
            ],
            "limitations": [],
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


def test_failed_critic_publishes_valid_claims_unassessed_but_never_unknowns():
    output, context = draft(("Known.", ["E1"], False), ("Missing.", ["E3"], False))
    final = finalize(
        check_integrity(output, context), None, critic_enabled=True, critic_failed=True,
        model_insufficient=False, limitations=(),
    )  # fmt: skip
    assert [(p.claim.claim_id, p.support) for p in final.published] == [("C1", "NOT_ASSESSED")]
    assert CRITIC_UNAVAILABLE_NOTE in final.limitations
    output, context = draft(("Missing.", ["E3"], False))
    none = finalize(
        check_integrity(output, context), None, critic_enabled=True, critic_failed=True,
        model_insufficient=False, limitations=(),
    )  # fmt: skip
    assert none.disposition == "INSUFFICIENT_EVIDENCE" and not none.published
    assert CRITIC_UNAVAILABLE_NOTE not in none.limitations


def test_limitation_publishes_only_when_grounded_and_once():
    output, context = draft(("A.", ["E1"], False))
    integrity = check_integrity(output, context)
    verdicts = SemanticReview.model_validate(
        {
            "findings": [{"claim": "C1", "support": "SUPPORTED", "rationale": None}],
            "limitations": [
                {"limitation": "L1", "grounded": True},
                {"limitation": "L2", "grounded": False},
                {"limitation": "L3", "grounded": True},
                {"limitation": "L3", "grounded": True},  # duplicate verdict: not trusted
            ],
        }
    )
    final = finalize(
        integrity, verdicts, critic_enabled=True, model_insufficient=False, limitations=("app",),
        model_limitations=("kept", "ungrounded", "ambiguous", "unreviewed"),
    )  # fmt: skip
    assert final.limitations == ("app", "kept") and final.limitations_withheld == 3
