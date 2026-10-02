"""Phase 9B stage 4: deterministic intent rules, precedence and the DOCUMENT /
INVESTIGATION boundary. These are unit tests of the rules, not the Phase 9 routing
evaluation (9C)."""

import pytest
import yaml
from tests.conftest import REPO_CONFIG_DIR
from tests.support.routing_fixtures import CONFIG

from worldbank_copilot.routing.intents import IntentEngine
from worldbank_copilot.routing.models import Intent, IntentStatus
from worldbank_copilot.routing.temporal import parse_temporal

ENGINE = IntentEngine(CONFIG.rules)


def decide(text):
    return ENGINE.decide(text, parse_temporal(text))


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        # structured
        ("Give me an overview of the project", Intent.PROJECT_OVERVIEW),
        ("What is the current closing date?", Intent.PROJECT_OVERVIEW),
        ("What is the project status?", Intent.PROJECT_OVERVIEW),
        ("What were the overall ratings in the latest ISR?", Intent.CURRENT_RATINGS),
        ("What is the PDO rating?", Intent.CURRENT_RATINGS),
        ("Show the PDO rating history", Intent.RATING_HISTORY),
        ("What was the implementation progress rating in ISR 4?", Intent.RATING_HISTORY),
        ("How did the ratings change from ISR 5 to 9?", Intent.RATING_HISTORY),
        ("Show me the project timeline", Intent.TIMELINE_EVENTS),
        ("How many restructurings has the project had?", Intent.TIMELINE_EVENTS),
        ("When was the project restructured?", Intent.TIMELINE_EVENTS),
        ("How much has been disbursed?", Intent.FINANCIAL_STATUS),
        ("What is the undisbursed balance of the loan?", Intent.FINANCIAL_STATUS),
        ("What are the baseline and target values?", Intent.RESULTS_PROGRESS),
        ("Show progress against the indicators", Intent.RESULTS_PROGRESS),
        ("What are the main environmental risks identified?", Intent.RISKS),
        ("What rating was given to institutional capacity at appraisal?", Intent.RISKS),
        ("What was the overall risk rating at appraisal?", Intent.RISKS),
        ("What deserves my attention?", Intent.ATTENTION),
        ("Are there any red flags?", Intent.ATTENTION),
        # document
        ("What does the restructuring paper say about the cancellation?", Intent.DOCUMENT_CONTENT),
        (
            "Which procurement guidelines apply according to the Project Appraisal Document?",
            Intent.DOCUMENT_CONTENT,
        ),
        ("How much had been disbursed according to ISR 24?", Intent.DOCUMENT_CONTENT),
        ("What does the ESSA say about groundwater?", Intent.DOCUMENT_CONTENT),
        ("Why, according to the restructuring paper, was the AF cancelled?", Intent.EXPLANATION),
        # refusal
        ("Update the closing date to 2030", Intent.WRITE_REQUEST),
        ("Please delete the risk register", Intent.WRITE_REQUEST),
        ("Will the project fail?", Intent.PREDICTION_REQUEST),
        ("What is the probability that the program will be delayed?", Intent.PREDICTION_REQUEST),
        ("Is the project failing?", Intent.PREDICTION_REQUEST),
        ("What is the weather in Bengaluru?", Intent.OUT_OF_DOMAIN),
        ("Write me a poem about water", Intent.OUT_OF_DOMAIN),
    ],
)
def test_each_deterministic_intent(text, intent):
    d = decide(text)
    assert (d.status, d.intent, d.method) == (IntentStatus.RESOLVED, intent, "RULE"), d.reason
    assert d.decided_by and all(h.rule_id and h.span[1] > h.span[0] for h in d.hits)


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        # documented rationale of a decision -> DOCUMENT
        ("Why was the closing date extended?", Intent.EXPLANATION),
        ("Why was the currency of the Additional Financing loan changed?", Intent.EXPLANATION),
        ("What justified the most recent restructuring?", Intent.EXPLANATION),
        ("Why was part of the loan cancelled?", Intent.EXPLANATION),
        # a structured change must first be established -> INVESTIGATION
        ("Why did the PDO rating drop?", Intent.CHANGE_INVESTIGATION),
        ("Why is disbursement lagging?", Intent.CHANGE_INVESTIGATION),
        ("Why has the indicator target not been met?", Intent.CHANGE_INVESTIGATION),
        ("What changed in the project and why?", Intent.CHANGE_INVESTIGATION),
        ("Which risks anticipated during appraisal later appeared?", Intent.CHANGE_INVESTIGATION),
        ("What early implementation signals are emerging?", Intent.CHANGE_INVESTIGATION),
        ("What deserves my attention and why?", Intent.CHANGE_INVESTIGATION),
    ],
)
def test_document_vs_investigation_boundary(text, intent):
    assert decide(text).intent == intent


def test_why_a_risk_is_rated_mixes_subjects_and_is_left_unresolved():
    """'risk' (documented) + 'rated' (rating change wording) with 'why': the boundary is
    not settled by rules, so it is left to 9D rather than guessed."""
    d = decide("Why is the fiduciary risk rated Substantial?")
    assert d.intent is None and set(d.subjects) == {"RATINGS", "RISKS"}


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (
            "How has the way citizens can register complaints been improved?",
            "no deterministic rule",
        ),
        (
            "Why did the operator fee allocation have to be increased?",
            "without a deterministic subject",
        ),
        ("Which cities was the project being restructured to include?", "decision is mentioned"),
        ("On what basis will Program funds be disbursed?", "decision is mentioned"),
        ("What is the project status and the disbursement?", "several subject families"),
        ("Why did the rating drop after the restructuring?", "mixing"),
        ("How is it going?", "no deterministic rule"),
    ],
)
def test_unresolved_questions_require_semantic_classification(text, reason):
    d = decide(text)
    assert (d.status, d.intent, d.method) == (
        IntentStatus.SEMANTIC_CLASSIFICATION_REQUIRED,
        None,
        "NONE",
    )
    assert reason in d.reason


def test_refusals_take_precedence_over_every_other_rule():
    d = decide("Update the PDO rating according to the restructuring paper")
    assert d.intent == Intent.WRITE_REQUEST and d.decided_by == ("WRITE_IMPERATIVE",)
    assert any(h.group == "document_scope" for h in d.hits)  # recorded, not used
    assert (
        decide("Will the project fail given the disbursement lag?").intent
        == Intent.PREDICTION_REQUEST
    )


def test_document_scope_beats_structured_subjects_and_suppresses_contained_words():
    d = decide("What does the restructuring paper say about the PDO rating?")
    assert d.intent == Intent.DOCUMENT_CONTENT and d.document_types == ("RESTRUCTURING_PAPER",)
    assert "SUBJECT_DECISION" in {h.rule_id for h in d.suppressed_hits}  # inside the doc name


def test_longer_matches_suppress_contained_generic_hits():
    d = decide("What is the overall risk rating?")
    assert d.rating_types == ("OVERALL_RISK",)
    assert {h.rule_id for h in d.suppressed_hits} >= {
        "SUBJECT_RATING_GENERIC",
        "SUBJECT_RISK_REGISTER",
    }


def test_in_the_latest_isr_alone_is_not_document_scope():
    assert decide("What were the ratings in the latest ISR?").intent == Intent.CURRENT_RATINGS
    assert decide("What progress was made by the latest ISR?").intent is None
    reported = decide("What progress was reported in the latest ISR?")  # explicit reporting
    assert reported.intent == Intent.DOCUMENT_CONTENT


def test_hits_record_rule_group_precedence_and_pattern():
    d = decide("Why was the closing date extended?")
    groups = {h.group: h for h in d.hits}
    assert groups["explanatory"].precedence < groups["subject"].precedence
    assert groups["explanatory"].text.lower().startswith("why")
    assert d.rules_version == CONFIG.rules.version


def test_configuration_has_no_generic_why_rule_or_default_intent():
    raw = yaml.safe_load((REPO_CONFIG_DIR / "routing" / "intents.yaml").read_text(encoding="utf-8"))
    for rule in raw["rules"]:
        if rule["group"] == "explanatory":
            assert "intent" not in rule  # 'why' is a modifier, never an intent by itself
    assert all(
        r.intent != Intent.CHANGE_INVESTIGATION or r.group == "investigation"
        for r in CONFIG.rules.rules
    )
