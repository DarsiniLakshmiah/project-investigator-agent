"""Offline tests for the canonical runtime (Copilot.investigate) over real tools/retrieval.

Model replies are fakes built from what the runtime sends; they test the harness, never
model quality. No network, endpoint or Databricks access.
"""

import json
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from tests.unit.test_phase10c_validation import runtime

from worldbank_copilot.common.exceptions import ConfigurationError
from worldbank_copilot.copilot import Copilot, CriticStatus, ResultStatus, load_copilot_config
from worldbank_copilot.copilot.finalizer import (
    CONTRADICTION_NOTE,
    CRITIC_DISABLED_NOTE,
    CRITIC_UNAVAILABLE_NOTE,
)
from worldbank_copilot.copilot.semantic import SYNTHESIS_INSTRUCTIONS
from worldbank_copilot.investigation.claims import Failure, ModelReply, NodeError

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
PROJECT = "P130544"
INVESTIGATION = "What implementation challenges were documented?"  # rules alone cannot route it
# The offline corpus documents the restructuring and cancellation rationale.
SEARCH = {
    "tool": "search_documents",
    "arguments": [],
    "query": "restructuring cancellation rationale",
}
SIGNALS = {"tool": "get_attention_signals", "arguments": [], "query": None}


def action(**fields):
    return {"purpose": "evidence", "query": None, "arguments": [], **fields}


def decision(**fields):
    base = {
        "disposition": "INVESTIGATE",
        "objective": "Understand the documented implementation challenges.",
        "clarification": None,
        "actions": [action(**SEARCH), action(**SIGNALS)],
        "temporal_anchor": None,
        "review_evidence": True,
    }
    return {**base, **fields}


def reply(value, **usage):
    text = value if isinstance(value, str) else json.dumps(value)
    return ModelReply(text=text, model_identity="offline-fake", **usage)


class Investigator:
    """Round 1 proposes actions; round 2 (if reached) answers now unless told otherwise."""

    def __init__(self, error=None, raw=None, round2=None, **round1):
        self.error, self.raw, self.requests = error, raw, []
        self.round1 = decision(**round1)
        self.round2 = round2

    def invoke(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        if self.raw is not None:
            return reply(self.raw)
        payload = json.loads(request.context_json)
        if payload["round"] == 1:
            return reply(self.round1, input_tokens=10, output_tokens=5)
        second = self.round2(payload) if callable(self.round2) else self.round2
        return reply(
            second or decision(disposition="ANSWER_NOW", actions=[], review_evidence=False),
            input_tokens=10,
            output_tokens=5,
        )


class Synthesizer:
    """One single-handle claim per supplied evidence (first ``count``); ``changes`` edit C1."""

    def __init__(self, abstain=False, raw=None, error=None, count=2, limitations=(), **changes):
        self.abstain, self.raw, self.error, self.count = abstain, raw, error, count
        self.limitations = list(limitations)
        self.changes, self.requests = changes, []

    def invoke(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        payload = json.loads(request.context_json)
        claims = [
            {"text": "The record reports an implementation fact.", "evidence": [e["handle"]],
             "interpretation": False, "temporal_relation": "NONE"}
            for e in payload["evidence"][: self.count]
        ]  # fmt: skip
        if claims and not self.abstain:
            claims[0].update({k: v(payload) if callable(v) else v for k, v in self.changes.items()})
        output = {
            "claims": [] if self.abstain else claims,
            "insufficient_evidence": self.abstain,
            "limitations": self.limitations,
        }
        return reply(self.raw or output, input_tokens=10, output_tokens=5)


class Critic:
    def __init__(self, supports=("SUPPORTED",), raw=None, error=None, grounded=(True,)):
        self.supports, self.raw, self.error, self.requests = supports, raw, error, []
        self.grounded = grounded

    def invoke(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        payload = json.loads(request.context_json)
        claims = payload["candidate_claims"]
        verdicts = [
            {"limitation": item["limitation"], "grounded": self.grounded[i % len(self.grounded)]}
            for i, item in enumerate(payload["candidate_limitations"])
        ]
        findings = [
            {
                "claim": c["claim"],
                "support": self.supports[i % len(self.supports)],
                "rationale": "Offline fake finding.",
            }
            for i, c in enumerate(claims)
        ]
        return reply(self.raw or {"findings": findings, "limitations": verdicts})


def config(**models):
    base = load_copilot_config(CONFIG_DIR, env={})
    return base.model_copy(update={"models": base.models.model_copy(update=models)})


def copilot(synthesizer=None, critic=None, investigator=None, **models):
    wired = runtime()
    return Copilot(
        wired.service,
        wired.documents,
        config(**models),
        wired.config_dir,
        investigator=investigator or Investigator(),
        synthesizer=synthesizer,
        critic=critic,
    )


def run(question=INVESTIGATION, **parts):
    app = copilot(**parts)
    return app, app.investigate(question, PROJECT)


def roles(result):
    return [c.role for c in result.model_calls]


# -- the semantic path -------------------------------------------------------------
def test_natural_question_is_investigated_and_answered_with_citations():
    app, result = run(synthesizer=Synthesizer(), critic=Critic())
    assert result.status == ResultStatus.ANSWER, result.message
    assert result.route == "INVESTIGATOR"
    assert roles(result) == ["INVESTIGATOR", "INVESTIGATOR", "SYNTHESIZER", "CRITIC"]
    assert result.objective and result.activity.decision_rounds == 2
    assert result.activity.tool_calls == 2 and result.activity.evidence_retrieved >= 2
    evidence_ids = {e.evidence_id for e in result.evidence}
    for claim in result.claims:
        assert claim.support == "SUPPORTED" and claim.citations
        assert set(claim.evidence_ids) <= evidence_ids
    assert result.validation.critic_status == CriticStatus.REVIEWED
    assert result.model_capability_note and "12/19" in result.model_capability_note


@pytest.mark.parametrize(
    "question",
    [
        "What implementation challenges were documented?",
        "What problems did this project face?",
        "What were the main implementation issues?",
        "Were there procurement problems?",
        "What did the ISR documents say about institutional capacity?",
        "What caused implementation delays?",
        "What happened before financing was cancelled?",
    ],
)
def test_paraphrases_reach_the_investigator_not_clarify(question):
    _, result = run(question, synthesizer=Synthesizer(), critic=Critic())
    assert result.route == "INVESTIGATOR" and result.status == ResultStatus.ANSWER
    assert roles(result)[0] == "INVESTIGATOR"


def test_one_round_when_investigator_does_not_ask_to_review():
    _, result = run(
        investigator=Investigator(review_evidence=False),
        synthesizer=Synthesizer(),
        critic=Critic(),
    )
    assert roles(result) == ["INVESTIGATOR", "SYNTHESIZER", "CRITIC"]
    assert result.activity.decision_rounds == 1


def test_second_round_sees_handles_and_call_summaries_but_no_identities():
    investigator = Investigator()
    _, result = run(investigator=investigator, synthesizer=Synthesizer(), critic=Critic())
    second = json.loads(investigator.requests[1].context_json)
    assert second["round"] == 2 and second["evidence"] and second["calls_made"]
    assert all(e["handle"].startswith("E") for e in second["evidence"])
    text = investigator.requests[1].context_json
    for item in result.evidence:
        assert item.evidence_id not in text
    assert '"project_id"' not in investigator.requests[0].context_json


def test_tool_calls_are_bounded_across_rounds():
    many = [action(**SEARCH, purpose=f"p{i}") for i in range(5)]
    extra = decision(actions=[action(**SIGNALS)] * 3, review_evidence=False)
    _, result = run(
        investigator=Investigator(actions=many, round2=extra),
        synthesizer=Synthesizer(),
        critic=Critic(),
    )
    assert result.activity.tool_calls == 6 and result.activity.rejected_actions == 2


@pytest.mark.parametrize(
    "bad",
    [
        action(tool="search_documents", query="SELECT * FROM gold.project_360"),
        action(tool="search_documents", query="see https://example.org"),
        action(
            tool="get_attention_signals",
            arguments=[{"name": "project_id", "value": "P179039"}],
        ),
        action(tool="get_attention_signals", arguments=[{"name": "sql", "value": "x"}]),
        action(tool="search_documents", query=None),
    ],
    ids=["sql", "url", "project-argument", "unknown-argument", "empty-query"],
)  # fmt: skip
def test_invalid_actions_are_rejected_not_executed(bad):
    _, result = run(
        investigator=Investigator(actions=[bad, action(**SIGNALS)], review_evidence=False),
        synthesizer=Synthesizer(),
        critic=Critic(),
    )
    assert result.activity.rejected_actions == 1 and result.activity.tool_calls == 1


@pytest.mark.parametrize(
    ("disposition", "status"),
    [("PREDICTION", "REFUSE"), ("OUT_OF_SCOPE", "REFUSE"), ("CLARIFY", "CLARIFY")],
)
def test_investigator_dispositions_stop_before_any_tool(disposition, status):
    synthesizer = Synthesizer()
    _, result = run(
        investigator=Investigator(disposition=disposition, clarification="Which period?"),
        synthesizer=synthesizer,
        critic=Critic(),
    )
    assert result.status == status and not synthesizer.requests
    assert not result.activity or result.activity.tool_calls == 0
    assert roles(result) == ["INVESTIGATOR"]
    if disposition == "PREDICTION":
        assert "does not have a validated model" in result.message


def test_no_evidence_abstains_without_synthesis():
    synthesizer = Synthesizer()
    _, result = run(
        investigator=Investigator(disposition="ANSWER_NOW", actions=[], review_evidence=False),
        synthesizer=synthesizer,
        critic=Critic(),
    )
    assert result.status == ResultStatus.INSUFFICIENT_EVIDENCE and not synthesizer.requests


def test_investigator_chosen_event_without_source_stated_date_is_insufficient():
    def anchored(payload):
        timeline = [e for e in payload["evidence"] if "event_type" in e["content"]]
        return decision(
            disposition="ANSWER_NOW",
            actions=[],
            review_evidence=False,
            temporal_anchor={"event": timeline[0]["handle"], "relation": "BEFORE"},
        )

    timeline = action(
        tool="get_project_timeline",
        arguments=[{"name": "event_types", "value": '["RESTRUCTURING"]'}],
    )
    synthesizer = Synthesizer()
    _, result = run(
        "What happened before restructuring?",
        investigator=Investigator(actions=[action(**SEARCH), timeline], round2=anchored),
        synthesizer=synthesizer,
        critic=Critic(),
    )
    assert result.status == ResultStatus.INSUFFICIENT_EVIDENCE and not synthesizer.requests
    assert result.activity.temporal_anchor is None
    assert result.activity.anchor_resolution == "NO_SOURCE_DATED_EVENT"


# -- event-relative questions (D3) ---------------------------------------------------
BEFORE_RESTRUCTURING = "What happened before restructuring?"


def timeline_has(monkeypatch, *dates):
    """The router's existing resolver, answering from fixed source-dated timeline events."""
    from worldbank_copilot.routing.models import AnchorCandidate, TemporalStatus
    from worldbank_copilot.routing.service import RoutingService

    calls = []

    def resolve(self, temporal, pid, access, ctx, anchors):
        calls.append(pid)
        candidates = tuple(
            AnchorCandidate(
                timeline_event_id=f"t{n}", event_type="RESTRUCTURING", event_date=d, title=f"R{n}"
            )
            for n, d in enumerate(dates, 1)
        )
        status = TemporalStatus.RESOLVED if len(dates) == 1 else TemporalStatus.UNRESOLVED
        return temporal.model_copy(update={"status": status, "anchor_candidates": candidates})

    monkeypatch.setattr(RoutingService, "_resolve_anchor", resolve)
    return calls


def test_one_authoritative_event_date_filters_evidence_and_answers(monkeypatch):
    calls = timeline_has(monkeypatch, date(2100, 1, 1))  # every dated record is BEFORE it
    synthesizer = Synthesizer(temporal_relation="BEFORE", count=1)
    _, result = run(BEFORE_RESTRUCTURING, synthesizer=synthesizer, critic=Critic())
    assert calls == [PROJECT]
    assert result.status == ResultStatus.ANSWER, result.message
    assert result.activity.anchor_resolution == "RESOLVED"
    assert result.activity.temporal_anchor == "BEFORE"
    payload = json.loads(synthesizer.requests[0].context_json)
    assert payload["anchor"] == {"relation": "BEFORE", "date": "2100-01-01"}
    periods = {e["handle"]: e["period"] for e in payload["evidence"]}
    first = payload["evidence"][0]["handle"]
    # the published BEFORE claim cites evidence source-dated before the event
    assert periods[first] == "BEFORE" and len(result.claims) == 1
    assert set(periods.values()) <= {"BEFORE", "EVENT", "UNDATED"}  # AFTER was filtered out


def test_several_authoritative_events_ask_which_one(monkeypatch):
    timeline_has(monkeypatch, date(2019, 5, 1), date(2022, 3, 1))
    synthesizer = Synthesizer()
    _, result = run(BEFORE_RESTRUCTURING, synthesizer=synthesizer, critic=Critic())
    assert result.status == ResultStatus.CLARIFY and not synthesizer.requests
    assert "R1 (2019-05-01)" in result.message and "R2 (2022-03-01)" in result.message
    assert result.activity.anchor_resolution == "AMBIGUOUS" and not result.claims


def test_event_without_authoritative_date_publishes_no_temporal_answer():
    # The offline timeline holds no source-stated restructuring date for this project.
    synthesizer = Synthesizer()
    for question in (
        BEFORE_RESTRUCTURING,
        "What were the biggest implementation problems and did they improve after restructuring?",
    ):
        _, result = run(question, synthesizer=synthesizer, critic=Critic())
        assert result.status == ResultStatus.INSUFFICIENT_EVIDENCE and not result.claims
        assert result.activity.anchor_resolution == "NO_SOURCE_DATED_EVENT"
    assert not synthesizer.requests


# -- a named, dated event (follow-up to an ambiguous anchor) ------------------------------
JULY_23 = "What happened before the July 23, 2024 restructuring?"


def anchor_on(relation="BEFORE", event_type="RESTRUCTURING", year=None, month=None, day=None):
    """What the Investigator understood: the strict contract requires every key."""
    return {
        "relation": relation,
        "event": None,
        "event_type": event_type,
        "year": year,
        "month": month,
        "day": day,
    }


def governed_timeline(monkeypatch, *events):
    """The real governed timeline call, with source-stated (event_type, date) events added."""
    from worldbank_copilot.tools.executor import ToolExecutor

    original = ToolExecutor.run

    def run(self, tool, arguments, *args, **kwargs):
        result = original(self, tool, arguments, *args, **kwargs)
        if tool != "get_project_timeline" or not result.items:
            return result
        template = result.items[0]
        added = [
            template.model_copy(
                update={"event_type": kind, "event_date": on, "event_title": kind.title()}
            )
            for kind, on in events
            if kind in (arguments.get("event_types") or (kind,))
        ]
        return result.model_copy(update={"items": [*result.items, *added]})

    monkeypatch.setattr(ToolExecutor, "run", run)


FOUR = tuple(
    ("RESTRUCTURING", d)
    for d in (date(2021, 5, 20), date(2024, 7, 23), date(2024, 12, 10), date(2026, 6, 29))
)
FULL_DATE = {"year": 2024, "month": 7, "day": 23}
MONTH_DAY = {"month": 7, "day": 23}


@pytest.mark.parametrize(
    ("question", "parts"),
    [
        (JULY_23, FULL_DATE),
        ("What happened before the 2024-07-23 restructuring?", FULL_DATE),
        ("Before the July 23 restructuring, what problems were documented?", MONTH_DAY),
    ],
)  # fmt: skip
def test_named_dated_event_is_confirmed_and_bounds_the_evidence(monkeypatch, question, parts):
    governed_timeline(monkeypatch, *FOUR)
    synthesizer = Synthesizer()
    _, result = run(
        question,
        investigator=Investigator(temporal_anchor=anchor_on(**parts)),
        synthesizer=synthesizer,
        critic=Critic(),
    )
    assert result.status == ResultStatus.ANSWER, result.message
    assert result.activity.anchor_resolution == "RESOLVED"
    assert result.activity.temporal_anchor == "BEFORE"
    payload = json.loads(synthesizer.requests[0].context_json)
    assert payload["anchor"] == {"relation": "BEFORE", "date": "2024-07-23"}


def test_named_dated_event_after_relation(monkeypatch):
    governed_timeline(monkeypatch, *FOUR)
    synthesizer = Synthesizer()
    _, result = run(
        "What changed after the July 23, 2024 restructuring?",
        investigator=Investigator(temporal_anchor=anchor_on("AFTER", year=2024, month=7, day=23)),
        synthesizer=synthesizer,
        critic=Critic(),
    )
    assert result.activity.anchor_resolution == "RESOLVED"
    assert result.activity.temporal_anchor == "AFTER"
    payload = json.loads(synthesizer.requests[0].context_json)
    assert payload["anchor"] == {"relation": "AFTER", "date": "2024-07-23"}
    assert all(e["period"] in ("AFTER", "EVENT", "UNDATED") for e in payload["evidence"])


def test_named_date_with_no_governed_event_is_insufficient(monkeypatch):
    governed_timeline(monkeypatch, *FOUR)
    synthesizer = Synthesizer()
    _, result = run(
        "What happened before the July 22, 2024 restructuring?",
        investigator=Investigator(temporal_anchor=anchor_on(year=2024, month=7, day=22)),
        synthesizer=synthesizer,
        critic=Critic(),
    )
    assert result.status == ResultStatus.INSUFFICIENT_EVIDENCE and not synthesizer.requests
    assert result.activity.anchor_resolution == "NO_MATCHING_EVENT"


def test_same_date_on_several_governed_events_asks_which(monkeypatch):
    governed_timeline(monkeypatch, *FOUR, ("ISR_REPORT", date(2024, 7, 23)))
    synthesizer = Synthesizer()
    _, result = run(
        "What happened before July 23, 2024?",
        investigator=Investigator(
            temporal_anchor=anchor_on(event_type=None, year=2024, month=7, day=23)
        ),
        synthesizer=synthesizer,
        critic=Critic(),
    )
    assert result.status == ResultStatus.CLARIFY and not synthesizer.requests
    assert result.activity.anchor_resolution == "AMBIGUOUS"


def test_candidate_date_in_the_question_is_not_authoritative():
    # Offline fixture: the restructuring has only a derived candidate date (2018-06-01).
    synthesizer = Synthesizer()
    _, result = run(
        "What happened before the June 1, 2018 restructuring?",
        investigator=Investigator(temporal_anchor=anchor_on(year=2018, month=6, day=1)),
        synthesizer=synthesizer,
        critic=Critic(),
    )
    assert result.status == ResultStatus.INSUFFICIENT_EVIDENCE and not synthesizer.requests
    assert result.activity.anchor_resolution == "NO_MATCHING_EVENT"
    assert result.activity.temporal_anchor is None


# -- planning failures are observable ------------------------------------------------
BAD_ANCHOR = decision(temporal_anchor={**anchor_on(), "event": "2024-07-23"})  # not a handle


@pytest.mark.parametrize(
    ("investigator", "code"),
    [
        (lambda: Investigator(raw="not json"), "OUTPUT_PARSE_FAILED"),
        (lambda: Investigator(raw=BAD_ANCHOR), "SCHEMA_INVALID"),
        (lambda: Investigator(error=NodeError(Failure.MODEL_UNAVAILABLE)), "MODEL_CALL_FAILED"),
    ],
)
def test_first_round_planning_failure_is_reported_with_a_reason(investigator, code):
    synthesizer = Synthesizer()
    _, result = run(JULY_23, investigator=investigator(), synthesizer=synthesizer, critic=Critic())
    assert result.status == ResultStatus.FAIL_CLOSED and not synthesizer.requests
    assert result.message == f"The investigation could not be planned ({code})."
    assert result.activity.planning_failure == code


def test_failed_review_round_answers_from_first_round_evidence():
    # The live D3-follow-up shape: round 1 succeeded and ran tools; round 2 was invalid.
    _, result = run(
        JULY_23,
        investigator=Investigator(round2=BAD_ANCHOR),
        synthesizer=Synthesizer(),
        critic=Critic(),
    )
    assert result.activity.planning_failure == "SCHEMA_INVALID"
    assert result.activity.decision_rounds == 1 and result.activity.tool_calls == 2
    assert result.activity.evidence_retrieved > 0
    assert result.status == ResultStatus.ANSWER and result.objective
    assert any("review round failed (SCHEMA_INVALID)" in n for n in result.limitations)


@pytest.mark.parametrize(
    ("first_round", "code"),
    [
        ({"actions": [], "review_evidence": False}, "NO_ACTIONS"),
        ({"disposition": "ANSWER_NOW", "actions": []}, "UNSUPPORTED_DISPOSITION"),
        ({"actions": [action(tool="search_documents", query="")]}, "GOVERNANCE_REJECTED"),
    ],
)
def test_plans_that_gather_nothing_say_why(first_round, code):
    _, result = run(investigator=Investigator(**first_round), synthesizer=Synthesizer())
    assert result.status == ResultStatus.INSUFFICIENT_EVIDENCE
    assert result.activity.planning_failure == code


def test_synthesizer_before_claim_without_anchor_is_removed():
    critic = Critic()
    _, result = run(synthesizer=Synthesizer(temporal_relation="BEFORE"), critic=critic)
    assert result.activity.anchor_resolution == "NOT_EVENT_RELATIVE"
    assert result.status == ResultStatus.ANSWER and len(result.claims) == 1
    assert result.validation.claims_removed == {"TEMPORAL_RELATION_UNVERIFIED": 1}
    reviewed = json.loads(critic.requests[0].context_json)["candidate_claims"]
    assert [c["temporal_relation"] for c in reviewed] == ["NONE"]


# -- claim-level grounding -----------------------------------------------------------
def test_unsupported_secondary_claim_is_removed_and_the_rest_published():
    _, result = run(synthesizer=Synthesizer(), critic=Critic(("SUPPORTED", "UNSUPPORTED")))
    assert result.status == ResultStatus.ANSWER and len(result.claims) == 1
    assert result.validation.claims_removed == {"UNSUPPORTED": 1}


def test_partially_supported_claim_is_kept_with_its_qualifier():
    _, result = run(synthesizer=Synthesizer(count=1), critic=Critic(("PARTIALLY_SUPPORTED",)))
    (claim,) = result.claims
    assert claim.support == "PARTIALLY_SUPPORTED" and claim.qualifier == "Offline fake finding."


def test_contradicted_claim_is_removed_and_flagged():
    _, result = run(synthesizer=Synthesizer(), critic=Critic(("SUPPORTED", "CONTRADICTED")))
    assert len(result.claims) == 1 and CONTRADICTION_NOTE in result.limitations


def test_nothing_supported_is_insufficient_evidence():
    _, result = run(synthesizer=Synthesizer(), critic=Critic(("UNSUPPORTED",)))
    assert result.status == ResultStatus.INSUFFICIENT_EVIDENCE and not result.claims
    assert result.validation.claims_removed == {"UNSUPPORTED": 2}


def test_model_abstention_publishes_nothing_and_needs_no_critic():
    critic = Critic()
    _, result = run(synthesizer=Synthesizer(abstain=True), critic=critic)
    assert result.status == ResultStatus.INSUFFICIENT_EVIDENCE and not critic.requests


def test_invalid_handle_drops_only_that_claim():
    _, result = run(synthesizer=Synthesizer(evidence=["E99"]), critic=Critic())
    assert result.status == ResultStatus.ANSWER and len(result.claims) == 1
    assert result.validation.claims_removed == {"EVIDENCE_REFERENCE_INVALID": 1}


def mixed(payload):
    seen = {}
    for e in payload["evidence"]:
        seen.setdefault(e["provenance"], e["handle"])
    assert len(seen) >= 2
    return list(seen.values())[:2]


def test_mixed_provenance_assertion_drops_only_that_claim():
    _, result = run(synthesizer=Synthesizer(evidence=mixed), critic=Critic())
    assert result.validation.claims_removed == {"PROVENANCE_VIOLATION": 1}
    assert result.status == ResultStatus.ANSWER


def test_interpretation_may_combine_provenance_types():
    _, result = run(synthesizer=Synthesizer(evidence=mixed, interpretation=True), critic=Critic())
    assert result.claims[0].provenance == "AI_INTERPRETATION"


def test_foreign_project_in_a_claim_fails_the_whole_response():
    critic = Critic()
    _, result = run(synthesizer=Synthesizer(text="Unlike P179039, delays occurred."), critic=critic)
    assert result.status == ResultStatus.FAIL_CLOSED and not result.claims
    assert result.validation.failures == ("PROJECT_ISOLATION_VIOLATION",)
    assert not critic.requests


def test_prediction_wording_in_a_claim_is_removed_deterministically():
    _, result = run(synthesizer=Synthesizer(text="The project will fail."), critic=Critic())
    assert result.validation.claims_removed == {"OVERCLAIMED": 1} and len(result.claims) == 1


def test_disabled_critic_publishes_integrity_valid_claims_as_not_assessed():
    critic = Critic()
    _, result = run(synthesizer=Synthesizer(), critic=critic, critic_enabled=False)
    assert result.status == ResultStatus.ANSWER and not critic.requests
    assert {c.support for c in result.claims} == {"NOT_ASSESSED"}
    assert result.validation.critic_status == CriticStatus.DISABLED
    assert CRITIC_DISABLED_NOTE in result.limitations


# -- failures ------------------------------------------------------------------------
@pytest.mark.parametrize(
    "part",
    [
        {"investigator": Investigator(error=NodeError(Failure.MODEL_UNAVAILABLE))},
        {"investigator": Investigator(raw="not JSON")},
        {"synthesizer": Synthesizer(error=TimeoutError())},
        {"synthesizer": Synthesizer(raw="not JSON")},
    ],
    ids=["investigator-unavailable", "investigator-malformed", "synth-timeout", "synth-malformed"],
)
def test_model_failure_fails_closed(part):
    parts = {"synthesizer": Synthesizer(), "critic": Critic(), **part}
    _, result = run(**parts)
    assert result.status == ResultStatus.FAIL_CLOSED and not result.claims
    assert result.model_calls[-1].outcome != "COMPLETED"


# -- a failed Critic is a missing secondary review, not a gate ------------------------
INVALID_FINDING = {
    "findings": [{"claim": "C1", "support": "MAYBE", "rationale": None}],
    "limitations": [],
}


@pytest.mark.parametrize(
    ("critic", "outcome"),
    [
        (Critic(error=NodeError(Failure.MODEL_UNAVAILABLE)), "MODEL_UNAVAILABLE"),
        (Critic(error=TimeoutError()), "MODEL_TIMEOUT"),
        (Critic(raw="not JSON"), "MODEL_OUTPUT_INVALID"),
        (Critic(raw=json.dumps(INVALID_FINDING)), "SCHEMA_VALIDATION_FAILED"),
    ],
    ids=["call-failed", "timeout", "parse-failed", "schema-invalid"],
)
def test_critic_failure_after_valid_integrity_degrades_to_answer(critic, outcome):
    _, result = run(synthesizer=Synthesizer(), critic=critic)
    assert result.status == ResultStatus.ANSWER and len(result.claims) == 2
    assert {c.support for c in result.claims} == {"NOT_ASSESSED"}  # never "approved"
    assert result.validation.critic_status == CriticStatus.FAILED
    assert result.validation.semantic_support == "NOT_ASSESSED"
    assert result.validation.mechanical_validity == "VALID"
    assert CRITIC_UNAVAILABLE_NOTE in result.limitations
    assert "semantic review unavailable" in result.message
    call = result.model_calls[-1]
    assert call.role == "CRITIC" and call.outcome == outcome


def test_critic_failure_never_bypasses_deterministic_integrity():
    # Security failure: fails closed before the (failing) Critic is ever called.
    critic = Critic(error=NodeError(Failure.MODEL_UNAVAILABLE))
    _, result = run(synthesizer=Synthesizer(text="Unlike P179039, delays occurred."), critic=critic)
    assert result.status == ResultStatus.FAIL_CLOSED and not result.claims
    assert result.validation.failures == ("PROJECT_ISOLATION_VIOLATION",)
    assert not critic.requests
    # Claim-level integrity removals still apply when the Critic then fails.
    for synthesizer, removed in (
        (Synthesizer(text="The project will fail."), "OVERCLAIMED"),
        (Synthesizer(temporal_relation="BEFORE"), "TEMPORAL_RELATION_UNVERIFIED"),
        (Synthesizer(evidence=mixed), "PROVENANCE_VIOLATION"),
    ):
        _, result = run(synthesizer=synthesizer, critic=Critic(raw="not JSON"))
        assert result.validation.claims_removed == {removed: 1} and len(result.claims) == 1
        assert result.validation.critic_status == CriticStatus.FAILED


def test_no_valid_claims_left_is_not_an_answer_even_if_the_critic_fails():
    critic = Critic(raw="not JSON")
    _, result = run(synthesizer=Synthesizer(text="The project will fail.", count=1), critic=critic)
    assert result.status == ResultStatus.INSUFFICIENT_EVIDENCE and not result.claims
    assert not critic.requests  # nothing left to review


def test_successful_critic_judgments_still_control_publication():
    # UNSUPPORTED and CONTRADICTED are judgments, not failures: never degraded to publish.
    _, result = run(synthesizer=Synthesizer(), critic=Critic(("UNSUPPORTED", "CONTRADICTED")))
    assert result.status == ResultStatus.INSUFFICIENT_EVIDENCE and not result.claims
    assert result.validation.critic_status == CriticStatus.REVIEWED
    assert result.validation.claims_removed == {"UNSUPPORTED": 1, "CONTRADICTED": 1}
    assert CONTRADICTION_NOTE in result.limitations and CRITIC_UNAVAILABLE_NOTE not in str(
        result.limitations
    )


def test_critic_sees_only_cited_evidence_and_may_omit_rationale():
    critic = Critic(raw=json.dumps({"findings": [
        {"claim": "C1", "support": "SUPPORTED", "rationale": None},
        {"claim": "C2", "support": "SUPPORTED", "rationale": None},
    ], "limitations": []}))  # fmt: skip
    synthesizer = Synthesizer()
    _, result = run(synthesizer=synthesizer, critic=critic)
    assert result.status == ResultStatus.ANSWER and len(result.claims) == 2
    shown = json.loads(synthesizer.requests[0].context_json)["evidence"]
    reviewed = json.loads(critic.requests[0].context_json)
    cited = {h for c in reviewed["candidate_claims"] for h in c["evidence"]}
    assert {e["handle"] for e in reviewed["evidence"]} == cited
    assert len(reviewed["evidence"]) < len(shown)


# -- D3C: temporal claims, relevance guidance and grounded limitations ----------------
def test_resolved_before_question_publishes_verified_before_claims(monkeypatch):
    governed_timeline(monkeypatch, *FOUR)
    synthesizer = Synthesizer(temporal_relation="BEFORE", count=1)
    _, result = run(
        JULY_23,
        investigator=Investigator(temporal_anchor=anchor_on(**FULL_DATE)),
        synthesizer=synthesizer,
        critic=Critic(),
    )
    assert result.status == ResultStatus.ANSWER, result.message
    assert [c.temporal_relation for c in result.claims] == ["BEFORE"]
    payload = json.loads(synthesizer.requests[0].context_json)
    assert payload["anchor"] == {"relation": "BEFORE", "date": "2024-07-23"}
    assert payload["time_scope"] is None  # no competing DATE scope read off the event date
    cited = result.claims[0].evidence_ids
    periods = {e["handle"]: e["period"] for e in payload["evidence"]}
    assert periods[payload["evidence"][0]["handle"]] == "BEFORE" and len(cited) == 1


def test_background_facts_may_stay_none_beside_before_claims(monkeypatch):
    governed_timeline(monkeypatch, *FOUR)
    _, result = run(
        JULY_23,
        investigator=Investigator(temporal_anchor=anchor_on(**FULL_DATE)),
        synthesizer=Synthesizer(temporal_relation="BEFORE"),  # C1 BEFORE, C2 background
        critic=Critic(),
    )
    assert result.status == ResultStatus.ANSWER
    assert sorted(c.temporal_relation for c in result.claims) == ["BEFORE", "NONE"]


def test_temporal_question_asks_for_relevant_developments_not_a_chronology():
    # Contract test only: selectivity is a semantic choice, measured live, never a rule.
    text = " ".join(SYNTHESIS_INSTRUCTIONS.split())
    assert "do not list every record or retell the project chronology" in text
    assert "what characterises that period" in text
    assert "sets temporal_relation to that relation" in text


def test_ungrounded_model_limitation_is_not_published():
    contradicting = "No evidence covers project outcomes after the 2018 ISR reports."
    grounded = "The evidence does not cover disbursement amounts."
    critic = Critic(grounded=(False, True))
    _, result = run(synthesizer=Synthesizer(limitations=[contradicting, grounded]), critic=critic)
    assert result.status == ResultStatus.ANSWER
    assert contradicting not in result.limitations and grounded in result.limitations
    assert result.validation.limitations_withheld == 1
    reviewed = json.loads(critic.requests[0].context_json)
    assert [x["limitation"] for x in reviewed["candidate_limitations"]] == ["L1", "L2"]
    assert all("content" not in e for e in reviewed["evidence_index"])


@pytest.mark.parametrize(
    "parts",
    [{"critic": Critic(raw="not JSON")}, {"critic": Critic(), "critic_enabled": False}],
    ids=["critic-failed", "critic-disabled"],
)
def test_unreviewed_model_limitations_are_withheld(parts):
    _, result = run(synthesizer=Synthesizer(limitations=["Unchecked gap statement."]), **parts)
    assert result.status == ResultStatus.ANSWER
    assert "Unchecked gap statement." not in result.limitations
    assert result.validation.limitations_withheld == 1


def test_critic_missing_a_finding_removes_that_claim_only():
    one = {
        "findings": [{"claim": "C1", "support": "SUPPORTED", "rationale": "ok"}],
        "limitations": [],
    }
    _, result = run(synthesizer=Synthesizer(), critic=Critic(raw=json.dumps(one)))
    assert len(result.claims) == 1 and result.validation.claims_removed == {"NOT_REVIEWED": 1}


def test_retrieval_failure_is_graceful():
    app = copilot(Synthesizer(), Critic())
    app.documents.retrieve = Mock(side_effect=RuntimeError("index unavailable"))
    result = app.investigate(INVESTIGATION, PROJECT)
    assert app.documents.retrieve.called
    assert result.status != ResultStatus.FAIL_CLOSED or result.validation.failures
    assert result.activity.tool_calls == 2  # the failed call is recorded, the other ran


def test_router_failure_fails_closed_without_leaking_detail():
    app = copilot(Synthesizer(), Critic())
    app.router.context_factory = Mock(side_effect=RuntimeError("secret internal detail"))
    result = app.investigate(INVESTIGATION, PROJECT)
    assert result.status == ResultStatus.FAIL_CLOSED and "secret" not in result.message


# -- deterministic fast path and refusals --------------------------------------------
@pytest.mark.parametrize(
    "question", ["What deserves my attention?", "Show the timeline of restructurings."]
)
def test_confident_structured_questions_never_call_models(question):
    investigator = Investigator()
    _, result = run(question, investigator=investigator, synthesizer=Synthesizer())
    assert result.status == ResultStatus.EVIDENCE_ONLY and result.route == "STRUCTURED"
    assert not result.model_calls and not investigator.requests


@pytest.mark.parametrize(
    "question", ["Will this project fail?", "What is the probability this project will fail?"]
)
def test_prediction_requests_are_refused_without_model_calls(question):
    investigator = Investigator()
    _, result = run(question, investigator=investigator, synthesizer=Synthesizer())
    assert result.status == ResultStatus.REFUSE and not investigator.requests


def test_unauthorized_project_is_refused_before_routing():
    app = copilot(Synthesizer(), Critic())
    app.router.context_factory = Mock(side_effect=AssertionError("must not read"))
    result = app.investigate(INVESTIGATION, "P999999")
    assert result.status == ResultStatus.REFUSE and result.route is None


def test_cross_project_question_is_refused():
    _, result = run("What is the current closing date of P179039?", synthesizer=Synthesizer())
    assert result.status == ResultStatus.REFUSE and not result.model_calls


def test_attention_signals_are_project_scoped_and_typed():
    _, result = run("What deserves my attention?")
    assert result.attention_signals
    assert all(s.provenance == "SYSTEM_DERIVED_SIGNAL" for s in result.attention_signals)


# -- configuration ------------------------------------------------------------------
def test_config_loads_defaults_bounds_and_env_overrides():
    cfg = load_copilot_config(CONFIG_DIR, env={})
    assert cfg.models.investigator_endpoint == "databricks-qwen35-122b-a10b"
    assert cfg.investigation.max_decision_rounds == 2 and cfg.investigation.max_tool_calls == 6
    assert cfg.models.max_output_tokens == 5000 and PROJECT in cfg.allowed_projects
    cfg = load_copilot_config(
        CONFIG_DIR,
        env={"WBC_COPILOT_CRITIC_ENABLED": "false", "WBC_COPILOT_INVESTIGATOR_ENDPOINT": "x-ep"},
    )
    assert cfg.models.critic_enabled is False and cfg.models.investigator_endpoint == "x-ep"
    with pytest.raises(ValueError):
        load_copilot_config(CONFIG_DIR, env={"WBC_COPILOT_CRITIC_ENDPOINT": "bad endpoint/x"})


def test_missing_config_file_is_explicit(tmp_path):
    with pytest.raises(ConfigurationError):
        load_copilot_config(tmp_path, env={})


# -- tracing ------------------------------------------------------------------------
class FakeSpan:
    def __init__(self, name, log):
        self.name, self.attributes, self.trace_id = name, {}, "trace-1"
        log.append(self)

    def set_attributes(self, attributes):
        self.attributes.update(attributes)


def traced(monkeypatch):
    spans = []

    @contextmanager
    def start_span(name):
        yield FakeSpan(name, spans)

    monkeypatch.setitem(__import__("sys").modules, "mlflow", SimpleNamespace(start_span=start_span))
    return spans


SECRET = "SECRET-CLAIM-TEXT-51ab"


def test_trace_has_named_stages_and_only_structural_metadata(monkeypatch):
    spans = traced(monkeypatch)
    app = copilot(Synthesizer(text=SECRET), Critic())
    app.mlflow_enabled = True
    question = INVESTIGATION + " PLANTEDQUERYMARKER"
    result = app.investigate(question, PROJECT)
    assert result.trace_id == "trace-1" and result.status == ResultStatus.ANSWER
    assert [s.name for s in spans] == [
        "copilot.investigate",
        "routing",
        "attention",
        "investigator_1",
        "tools_1",
        "investigator_2",
        "anchor",
        "synthesis",
        "enrichment",
        "integrity",
        "critic",
        "finalization",
    ]
    root = spans[0].attributes
    assert root["route"] == "INVESTIGATOR" and root["decision_rounds"] == 2
    assert root["tool_calls"] == 2 and root["model_call_count"] == 4
    serialized = json.dumps([s.attributes for s in spans])
    for leaked in (SECRET, "PLANTEDQUERYMARKER", question, result.objective):
        assert leaked not in serialized
    assert all(isinstance(v, (str, int, float, bool)) for s in spans for v in s.attributes.values())
