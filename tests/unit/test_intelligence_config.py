"""Phase 7 pure checks: rule configuration, rating scales, Gold contracts (no Spark)."""

import json
import subprocess
import sys

import pytest
import yaml

from worldbank_copilot.common.exceptions import ConfigurationError
from worldbank_copilot.intelligence.contracts import (
    GOLD_LOCK_FILE,
    PROVENANCE_CLASSES,
    attention_signals_contract,
    contracts_lock,
    result_progress_contract,
    timeline_contract,
)
from worldbank_copilot.intelligence.rules import load_rules, load_scales
from worldbank_copilot.intelligence.signals import BUILDERS, check_rule_coverage


def test_reviewed_rules_load_with_versions_thresholds_and_rationale(repo_config_dir):
    rules = load_rules(repo_config_dir)
    assert set(rules.enabled) == set(BUILDERS)
    for rule in rules.rules:
        assert rule.version >= 1 and rule.rationale.strip() and rule.description.strip()
        assert rule.rule_version == f"{rule.rule_id}@v{rule.version}"
    assert rules.get("SCHEDULE_CLOSING_DATE_EXTENDED").threshold("high_extension_months") == 24
    assert rules.get("FINANCE_DISBURSEMENT_LAG").applies_to_instruments == [
        "Investment Project Financing"
    ]
    assert len(rules.deferred) >= 5 and all(d.reason for d in rules.deferred)
    check_rule_coverage(rules)


def _write(tmp_path, rules):
    folder = tmp_path / "intelligence"
    folder.mkdir(parents=True)
    (folder / "attention_rules.yaml").write_text(yaml.safe_dump({"rules": rules}), encoding="utf-8")
    return tmp_path


BASE = {
    "rule_id": "X",
    "version": 1,
    "enabled": True,
    "category": "SCHEDULE",
    "title": "t",
    "description": "d",
    "required_fields": [],
    "severity": {"INFO": "any"},
    "thresholds": {},
    "rationale": "r",
}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"category": "PREDICTION"}, "unknown category"),
        ({"severity": {"CRITICAL": "x"}}, "severity levels"),
        ({"version": 0}, "greater than or equal"),
    ],
)
def test_invalid_rule_configuration_fails(tmp_path, change, message):
    with pytest.raises(ConfigurationError, match=message):
        load_rules(_write(tmp_path, [{**BASE, **change}]))


def test_duplicate_rules_missing_thresholds_and_unimplemented_rules_fail(tmp_path):
    with pytest.raises(ConfigurationError, match="duplicate rule ids"):
        load_rules(_write(tmp_path / "a", [BASE, BASE]))
    rules = load_rules(_write(tmp_path / "b", [BASE]))
    with pytest.raises(ConfigurationError, match="no threshold"):
        rules.get("X").threshold("missing")
    with pytest.raises(ConfigurationError, match="rules without builders"):
        check_rule_coverage(rules)


def test_rating_scales_are_ordinal(repo_config_dir):
    scales = load_scales(repo_config_dir)
    order = sorted(scales.performance, key=scales.performance.get, reverse=True)
    assert order == [
        "Highly Satisfactory",
        "Satisfactory",
        "Moderately Satisfactory",
        "Moderately Unsatisfactory",
        "Unsatisfactory",
        "Highly Unsatisfactory",
    ]
    assert scales.performance["Moderately Unsatisfactory"] == (
        scales.performance_below_satisfactory_max_rank
    )
    assert scales.risk["High"] > scales.risk["Substantial"] > scales.risk["Moderate"]


def test_gold_contracts_match_the_committed_lock(repo_config_dir):
    lock = json.loads((repo_config_dir / GOLD_LOCK_FILE).read_text(encoding="utf-8"))
    assert contracts_lock() == lock


def test_gold_contracts_encode_the_semantics():
    signals = attention_signals_contract()
    assert signals.column("provenance_class").vocabulary == ("SYSTEM_DERIVED_SIGNAL",)
    assert signals.column("severity").vocabulary == ("INFO", "WATCH", "HIGH")
    assert not signals.column("supporting_record_ids").nullable
    timeline = timeline_contract()
    assert timeline.column("event_date_status").vocabulary == (
        "SOURCE_STATED",
        "SCHEDULED",
        "DERIVED_CANDIDATE",
        "UNDATED",
    )
    assert "AI_INTERPRETATION" in PROVENANCE_CLASSES
    assert "AI_INTERPRETATION" not in timeline.column("provenance_class").vocabulary
    progress = result_progress_contract()
    assert progress.column("progress_percentage").sql_type == "DECIMAL(38,6)"
    assert progress.natural_key == ("source_record_id",)


def test_intelligence_imports_without_spark():
    code = (
        "import sys, worldbank_copilot.intelligence.pipeline, "
        "worldbank_copilot.intelligence.signals; print('pyspark' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
