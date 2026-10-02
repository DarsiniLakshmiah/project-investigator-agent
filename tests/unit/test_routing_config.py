"""Phase 9B routing configuration integrity (versions, coverage, tool references)."""

import pytest
from tests.conftest import REPO_CONFIG_DIR
from tests.support.routing_fixtures import CONFIG

from worldbank_copilot.common.exceptions import ConfigurationError
from worldbank_copilot.routing.config import IntentRules, load_routing_config
from worldbank_copilot.routing.models import REFUSAL_INTENTS, Intent
from worldbank_copilot.routing.requirements import SPECS, validate_call


def test_configuration_loads_with_versions():
    assert load_routing_config(REPO_CONFIG_DIR).versions == {
        "router": "9B.2",
        "intent_rules": "1",
        "requirements": "1",
        "aliases": "1",
    }


def test_every_non_refusal_intent_has_a_requirement_and_known_tools():
    assert set(CONFIG.requirements.intents) == set(Intent) - REFUSAL_INTENTS
    for intent, req in CONFIG.requirements.intents.items():
        assert set(req.tools) <= set(SPECS), intent
        assert req.default_time in req.supported_time, intent  # a default must be executable


def test_every_planned_investigation_call_is_valid_for_its_tool():
    for kind, calls in CONFIG.requirements.investigation_plans.items():
        for call in calls:
            assert validate_call(call.tool, {"project_id": "P130544", **call.args}) is None, kind


def test_every_intent_rule_intent_and_subject_is_known():
    groups = {r.group for r in CONFIG.rules.rules}
    assert groups == {"refusal", "investigation", "document_scope", "explanatory", "subject"}
    assert {r.intent for r in CONFIG.rules.rules if r.group == "refusal"} == set(REFUSAL_INTENTS)


def test_invalid_rule_configuration_fails_loudly():
    base = {"version": 1, "document_types": {"ISR": ["isr"]}}
    with pytest.raises(ValueError, match="invalid pattern"):
        IntentRules.model_validate(
            {
                **base,
                "rules": [
                    {"id": "X", "group": "subject", "subject": "RATINGS", "patterns": ["(unclosed"]}
                ],
            }
        )
    with pytest.raises(ValueError, match="need an intent"):
        IntentRules.model_validate(
            {**base, "rules": [{"id": "X", "group": "refusal", "patterns": ["x"]}]}
        )
    with pytest.raises(ValueError, match="duplicate"):
        IntentRules.model_validate(
            {
                **base,
                "rules": [
                    {"id": "X", "group": "explanatory", "patterns": ["x"]},
                    {"id": "X", "group": "explanatory", "patterns": ["y"]},
                ],
            }
        )
    with pytest.raises(ConfigurationError, match="not found"):
        load_routing_config(REPO_CONFIG_DIR / "missing")
