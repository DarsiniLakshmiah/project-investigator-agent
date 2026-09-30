"""Gold on real Silver data with local Spark (development rehearsal, nothing persisted).

Input: the Silver rows exported by the Phase 6 dry run (scripts/platformize.py), which
reconciled exactly with the Silver Delta tables in Databricks. Skipped when absent.
Assertions are invariants against Silver, not remembered Gold counts.
"""

from collections import Counter
from datetime import UTC, date, datetime

import pytest

from worldbank_copilot.common import load_settings
from worldbank_copilot.intelligence.frames import GoldContext, profile_frame
from worldbank_copilot.intelligence.pipeline import build_gold, silver_from_json, validate_build
from worldbank_copilot.intelligence.rules import load_rules, load_scales

pytestmark = [pytest.mark.spark, pytest.mark.integration]

SETTINGS = load_settings("local", env={})
EXPORT = SETTINGS.local_output_root / "lakehouse"


@pytest.fixture(scope="module")
def real(spark):
    if not (EXPORT / "silver.project_results.jsonl").exists():
        pytest.skip("no Phase 6 Silver export; run scripts/platformize.py")
    silver = silver_from_json(spark, EXPORT)
    rules, scales = load_rules(SETTINGS.config_dir), load_scales(SETTINGS.config_dir)
    ctx = GoldContext("real", "run-a", datetime(2026, 9, 30, tzinfo=UTC), "test")
    build = build_gold(spark, silver, rules, scales, ctx)
    validate_build(build)
    return spark, silver, build, rules, scales


def collect(build, table):
    return [r.asDict(recursive=True) for r in build.frames[table].collect()]


def test_all_invariants_pass_and_counts_reconcile_with_silver(real):
    _, silver, build, _, _ = real
    assert all(r.status in ("PASS", "OBSERVED") for r in build.checks)
    assert build.frames["project_360"].count() == silver["projects"].count() == 3
    assert build.frames["result_progress"].count() == silver["project_results"].count() == 837
    events = silver["project_events"].count()
    isrs = silver["isr_snapshots"].count()
    assert isrs == 35
    # 3 milestones per project + original closing + every ISR + every formal event.
    assert build.frames["project_timeline"].count() == 3 * 3 + 3 + isrs + events
    appraisal = build.frames["risk_register"].where("record_type != 'ISR_SORT_RATING'").count()
    assert appraisal == silver["appraisal_risks"].count() == 89


def test_known_facts(real):
    _, _, build, _, _ = real
    p360 = {r["project_id"]: r for r in collect(build, "project_360")}
    assert (
        p360["P130544"]["original_closing_date"],
        p360["P130544"]["current_closing_date"],
        p360["P130544"]["days_extended"],
    ) == (date(2022, 11, 30), date(2027, 9, 30), 1765)
    assert p360["P130544"]["number_of_restructurings"] == 4
    assert p360["P130544"]["number_of_restructuring_papers"] == 4
    assert {p: r["latest_isr_sequence"] for p, r in p360.items()} == {
        "P130544": 24,
        "P179039": 8,
        "P506272": 3,
    }
    timeline = [
        r
        for r in collect(build, "project_timeline")
        if r["project_id"] == "P179039" and r["event_type"] == "ISR_REPORT"
    ]
    ordered = [r["isr_sequence"] for r in sorted(timeline, key=lambda r: r["event_sequence"])]
    assert ordered == list(range(1, 9))
    (anomaly,) = [r for r in timeline if r["date_sequence_anomaly"]]
    assert (anomaly["isr_sequence"], anomaly["event_date"]) == (5, date(2025, 5, 31))


def test_no_pending_alias_merged_and_no_dli_progress(real):
    _, silver, build, _, _ = real
    progress = collect(build, "result_progress")
    keys = {r["canonical_indicator_id"] for r in progress}
    assert len(keys) == 237
    for pair in silver["indicator_match_candidates"].collect():
        assert pair["indicator_key_a"] in keys and pair["indicator_key_b"] in keys
    assert not [r for r in progress if r["layout"] == "DLI" and r["progress_percentage"]]
    assert Counter(r["identity_review_status"] for r in progress)["REVIEWED_ALIAS"] == 0


def test_signals_are_traceable_and_neutral(real):
    _, _, build, rules, _ = real
    signals = collect(build, "attention_signals")
    assert {s["rule_id"] for s in signals} <= set(rules.enabled)
    for s in signals:
        assert s["source_record_id"] in s["supporting_record_ids"]
        assert s["provenance_class"] == "SYSTEM_DERIVED_SIGNAL"
    (ext,) = [s for s in signals if s["rule_id"] == "SCHEDULE_CLOSING_DATE_EXTENDED"]
    assert (ext["project_id"], ext["severity"]) == ("P130544", "HIGH")
    fin = [s for s in signals if s["rule_id"] == "FINANCE_DISBURSEMENT_LAG"]
    assert {s["project_id"] for s in fin} <= {"P130544"}  # IPF only


def test_second_build_is_identical(real):
    spark, silver, build, rules, scales = real
    ctx = GoldContext("real", "run-b", datetime(2027, 1, 1, tzinfo=UTC), "test")
    again = build_gold(spark, silver, rules, scales, ctx)
    for name, frame in build.frames.items():
        first = profile_frame(frame, build.contracts[name])
        second = profile_frame(again.frames[name], again.contracts[name])
        assert first == {**second, "table": first["table"]}, name


def test_phase7_validation_sql_runs_against_the_gold_tables(real):
    """Every query in sql/phase7_validation.sql executes (UC names -> temp views, test only)."""
    import re

    from worldbank_copilot.lakehouse.validation_sql import load_queries

    spark, silver, build, _, _ = real
    for name, frame in silver.items():
        frame.createOrReplaceGlobalTempView(f"silver__{name}")
    for name, frame in build.frames.items():
        frame.createOrReplaceGlobalTempView(f"gold__{name}")
    spark.read.json(str(EXPORT / "bronze.document_inventory.jsonl")).createOrReplaceGlobalTempView(
        "bronze__document_inventory"
    )
    queries = load_queries(SETTINGS, SETTINGS.repo_root / "sql" / "phase7_validation.sql")
    assert len(queries) >= 14
    catalog = SETTINGS.databricks.catalog
    for name, query in queries.items():
        local = re.sub(rf"\b{catalog}\.(\w+)\.(\w+)", r"global_temp.\1__\2", query)
        rows = spark.sql(local).collect()
        if name == "no_ai_interpretation":
            assert all(r["ai_rows"] == 0 for r in rows)
        if name == "project_360":
            assert len(rows) == 3
