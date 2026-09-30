"""Gold transformations on real (local) Spark with a small synthetic Silver scenario.

The scenario is structural only (made-up projects P1/P2), built with the Phase 6 Silver
contract schemas. It exercises the production code path (build_gold / run_gold) with
the reviewed rule configuration.
"""

from datetime import UTC, date, datetime
from decimal import Decimal as D

import pytest

from worldbank_copilot.common import load_settings
from worldbank_copilot.intelligence.frames import GoldContext, profile_frame, validate_frame
from worldbank_copilot.intelligence.pipeline import (
    SILVER_INPUTS,
    build_gold,
    run_gold,
    validate_build,
)
from worldbank_copilot.intelligence.rules import load_rules, load_scales

pytestmark = pytest.mark.spark

SETTINGS = load_settings("local", env={})
RULES, SCALES = load_rules(SETTINGS.config_dir), load_scales(SETTINGS.config_dir)
CTX = GoldContext("fixture-snapshot", "run-1", datetime(2026, 9, 30, tzinfo=UTC), "test")


def isr(pid, seq, day, pdo, ip, risk="Substantial"):
    return {
        "record_id": f"isr-{pid}-{seq}",
        "project_id": pid,
        "document_id": f"doc-{pid}-{seq}",
        "isr_sequence": seq,
        "isr_number": f"ISR{seq}",
        "canonical_report_date": day,
        "header_date": day,
        "archive_date": day,
        "canonical_date_basis": "header_date",
        "pdo_rating": pdo,
        "pdo_rating_status": "EXACT",
        "pdo_rating_page_number": 1,
        "pdo_rating_extraction_method": "DOCLING_TABLE",
        "implementation_progress_rating": ip,
        "implementation_progress_rating_status": "EXACT",
        "implementation_progress_rating_page_number": 1,
        "implementation_progress_rating_extraction_method": "DOCLING_TABLE",
        "overall_risk_rating": risk,
        "overall_risk_rating_status": "EXACT",
        "overall_risk_rating_page_number": 2,
        "overall_risk_rating_extraction_method": "DOCLING_TABLE",
    }


def result(
    rid,
    key,
    seq,
    day,
    base,
    cur,
    tgt,
    *,
    unit="Number",
    layout="WIDE",
    status="EXACT",
    target_date="Jun/2020",
    pid="P1",
):
    return {
        "record_id": rid,
        "project_id": pid,
        "indicator_key": key,
        "indicator_name_raw": f"{key} ({unit})",
        "indicator_name_normalized": key.lower(),
        "indicator_type": "PDO",
        "unit": unit,
        "layout": layout,
        "identity_basis": "EXACT_NAME",
        "isr_sequence": seq,
        "observation_date": day,
        "baseline_value": base,
        "current_value": cur,
        "target_value": tgt,
        "target_date": target_date,
        "status": status,
        "extraction_method": "DOCLING_TABLE",
        "document_id": f"doc-{pid}-{seq}",
        "evidence_page_number": 5,
        "evidence_table_id": "t0005",
        "evidence_section": "Results",
    }


def event(rid, etype, day=None, **kw):
    row = {
        "record_id": rid,
        "project_id": "P1",
        "event_id": rid,
        "event_type": etype,
        "event_date": day,
        "status": "EXACT",
        "evidence_document_id": "doc-res",
        "evidence_page_number": 4,
        "evidence_extraction_method": "DOCLING_TABLE",
    }
    row.update(kw)
    return row


SCENARIO = {
    "projects": [
        {
            "record_id": "prj-1",
            "project_id": "P1",
            "project_name": "Project One",
            "financing_instrument": "Investment Project Financing",
            "project_status": "Active",
            "approval_date": date(2015, 6, 1),
            "effective_date": date(2016, 1, 1),
            "current_closing_date": date(2026, 12, 31),
        },
        {
            "record_id": "prj-2",
            "project_id": "P2",
            "project_name": "Project Two",
            "financing_instrument": "Program-for-Results Financing",
            "project_status": "Active",
            "approval_date": date(2022, 6, 1),
            "effective_date": date(2023, 1, 1),
            "current_closing_date": date(2028, 1, 1),
        },
    ],
    "project_financial_summary": [
        {
            "record_id": "fin-1",
            "project_id": "P1",
            "snapshot_date": date(2025, 12, 31),
            "loan_count": 1,
            "original_principal_total_usd": D("100"),
            "cancelled_total_usd": D("0"),
            "net_principal_after_cancellation_usd": D("100"),
            "disbursed_total_usd": D("40"),
            "undisbursed_total_usd": D("60"),
            "disbursement_vs_net_principal_pct": D("40.00"),
            "loans_with_valuation_caveats": [],
        },
        {
            "record_id": "fin-2",
            "project_id": "P2",
            "snapshot_date": date(2025, 12, 31),
            "loan_count": 1,
            "disbursement_vs_net_principal_pct": D("10.00"),
            "cancelled_total_usd": D("0"),
            "loans_with_valuation_caveats": [],
        },
    ],
    "project_enrichment": [
        {
            "record_id": "enr-1",
            "project_id": "P1",
            "target_table": "silver_projects",
            "target_field": "original_closing_date",
            "loan_number": "IBRDX",
            "value": date(2023, 12, 31),
            "status": "DERIVED_FROM_EXPLICIT_SOURCE",
            "rule": "first loan",
            "evidence_document_id": "doc-isr",
            "evidence_page_number": 6,
            "evidence_extraction_method": "DOCLING_TABLE",
        },
        {
            "record_id": "enr-2",
            "project_id": "P2",
            "target_table": "silver_projects",
            "target_field": "original_closing_date",
            "loan_number": "IBRDY",
            "value": date(2028, 1, 1),
            "status": "DERIVED_FROM_EXPLICIT_SOURCE",
            "rule": "first loan",
        },
    ],
    "isr_snapshots": [
        isr("P1", 1, date(2016, 6, 1), "Satisfactory", "Satisfactory"),
        isr("P1", 2, date(2017, 6, 1), "Moderately Satisfactory", "Moderately Satisfactory"),
        isr("P1", 3, date(2018, 6, 1), "Moderately Unsatisfactory", "Unsatisfactory"),
        isr("P1", 4, date(2019, 6, 1), "Unsatisfactory", "Unsatisfactory"),
        isr("P1", 5, date(2020, 6, 1), "Unsatisfactory", "Moderately Unsatisfactory"),
        isr("P1", 6, date(2021, 6, 1), "Moderately Satisfactory", "Moderately Satisfactory"),
        # P2: ISR 1 printed with a date later than ISR 2 (the P179039-like anomaly).
        isr("P2", 1, date(2024, 6, 30), "Satisfactory", "Satisfactory", "Moderate"),
        isr("P2", 2, date(2024, 1, 15), "Satisfactory", "Satisfactory", "Moderate"),
        isr("P2", 3, date(2025, 1, 1), "Satisfactory", "Satisfactory", "Moderate"),
    ],
    "isr_sort_ratings": [
        {
            "record_id": "sort-1",
            "isr_record_id": "isr-P1-6",
            "project_id": "P1",
            "isr_sequence": 6,
            "canonical_report_date": date(2021, 6, 1),
            "risk_category": "Fiduciary",
            "rating_at_approval": "Moderate",
            "previous_rating": "Moderate",
            "current_rating": "Substantial",
            "current_rating_raw": "Substantial",
            "current_rating_status": "EXACT",
            "evidence_document_id": "doc-P1-6",
            "evidence_page_number": 2,
            "evidence_extraction_method": "DOCLING_TABLE",
        },
    ],
    "project_events": [
        event("ev-r1", "RESTRUCTURING", date(2020, 1, 1)),
        event(
            "ev-r1-paper",
            "RESTRUCTURING",
            None,
            candidate_event_date=date(2020, 1, 1),
            candidate_date_basis="EARLIEST_ISR_RESTRUCTURING",
            candidate_date_status="DERIVED_FROM_EXPLICIT_SOURCE",
            status="AMBIGUOUS",
        ),
        event(
            "ev-c1",
            "CLOSING_DATE_CHANGE",
            date(2020, 2, 1),
            loan_number="IBRDX",
            old_closing_date=date(2023, 12, 31),
            new_closing_date=date(2025, 6, 30),
        ),
        event(
            "ev-c1-again",
            "CLOSING_DATE_CHANGE",
            None,
            loan_number="IBRDX",
            old_closing_date=date(2023, 12, 31),
            new_closing_date=date(2025, 6, 30),
        ),
        event(
            "ev-c2",
            "CLOSING_DATE_CHANGE",
            date(2022, 1, 1),
            loan_number="IBRDX",
            old_closing_date=date(2025, 6, 30),
            new_closing_date=date(2026, 12, 31),
        ),
        event(
            "ev-af",
            "ADDITIONAL_FINANCING",
            date(2021, 1, 1),
            additional_financing_amount=D("50.00"),
            additional_financing_currency="USD_MILLIONS",
        ),
    ],
    "project_results": [
        result("r-k1-4", "K1", 4, date(2019, 6, 1), "0", "10", "100"),
        result("r-k1-5", "K1", 5, date(2020, 6, 1), "0", "5", "100"),  # away
        result("r-k1-6", "K1", 6, date(2021, 6, 1), "0", "50", "100"),  # toward
        result("r-k2-6", "K2", 6, date(2021, 6, 1), "5", "5", "5"),  # zero distance
        result("r-k3-6", "K3", 6, date(2021, 6, 1), "No", "No", "Yes", unit="Yes/No"),
        result("r-k4-6", "K4", 6, date(2021, 6, 1), None, "3", "10"),  # no baseline
        result("r-k5-6", "K5", 6, date(2021, 6, 1), "0", "7", "10", layout="DLI"),
        result("r-k6-6", "K6", 6, date(2021, 6, 1), "0", "4", "8"),
        result("r-k7-6", "K7", 6, date(2021, 6, 1), "0", "2", "8"),
        result("r-k8-5", "K8", 5, date(2020, 6, 1), "0", "1", "10"),
        result("r-k8-6", "K8", 6, date(2021, 6, 1), "0", "2", None),  # target missing
        result("r-k9-6", "K9", 6, date(2021, 6, 1), "1,000", "1,000", "1,000.1234567"),
    ],
    "indicator_match_candidates": [
        {
            "record_id": "cand-1",
            "project_id": "P1",
            "indicator_key_a": "K6",
            "indicator_key_b": "K7",
            "review_status": "PENDING_REVIEW",
        },
    ],
    "appraisal_risks": [
        {
            "record_id": "risk-1",
            "project_id": "P1",
            "risk_id": "R1",
            "framing": "FORMAL_RISK_RATING",
            "risk_category": "Fiduciary",
            "risk_rating": "Substantial",
            "risk_rating_raw": "Substantial",
            "status": "EXACT",
            "source_document_type": "APPRAISAL_DOCUMENT",
            "identified_date": date(2015, 5, 1),
            "evidence_document_id": "doc-pad",
            "evidence_page_number": 10,
            "evidence_extraction_method": "DOCLING_TABLE",
        },
        {
            "record_id": "risk-2",
            "project_id": "P1",
            "risk_id": "R2",
            "framing": "ASSESSMENT_FINDING",
            "risk_category": "Technical",
            "risk_description": "Synthetic finding text.",
            "status": "EXACT",
            "source_document_type": "TECHNICAL_ASSESSMENT",
            "evidence_document_id": "doc-ta",
            "evidence_page_number": 3,
            "evidence_extraction_method": "DOCLING_TABLE",
        },
    ],
    "data_quality_observations": [
        {"record_id": "dq-1", "severity": "WARNING", "check_code": "SEQUENCE_DATE_ANOMALY"},
    ],
}


@pytest.fixture(scope="module")
def silver(spark, tmp_path_factory):
    from tests.spark.conftest import to_json_line  # noqa: F401 (shared encoder)

    from worldbank_copilot.lakehouse.spark_store import spark_schema

    root = tmp_path_factory.mktemp("silver")
    frames = {}
    for name, contract_fn in SILVER_INPUTS.items():
        contract = contract_fn()
        path = root / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in SCENARIO[name]:
                fh.write(to_json_line({c: row.get(c) for c in contract.column_names}) + "\n")
        frames[name] = (
            spark.read.schema(spark_schema(contract)).option("mode", "FAILFAST").json(str(path))
        )
    return frames


@pytest.fixture(scope="module")
def gold(spark, silver):
    build = build_gold(spark, silver, RULES, SCALES, CTX)
    validate_build(build)  # contracts + ERROR invariants
    return build


def rows(build, table, **where):
    out = [r.asDict(recursive=True) for r in build.frames[table].collect()]
    return [r for r in out if all(r.get(k) == v for k, v in where.items())]


def signals(build, rule_id, **where):
    return rows(build, "attention_signals", rule_id=rule_id, **where)


def test_project_360_one_row_per_project_with_derived_state(gold):
    (p1,) = rows(gold, "project_360", project_id="P1")
    assert len(rows(gold, "project_360")) == 2
    assert (p1["original_closing_date"], p1["current_closing_date"], p1["days_extended"]) == (
        date(2023, 12, 31),
        date(2026, 12, 31),
        1096,
    )
    assert (p1["latest_isr_sequence"], p1["latest_do_rating"], p1["previous_do_rating"]) == (
        6,
        "Moderately Satisfactory",
        "Unsatisfactory",
    )
    assert p1["do_rating_change"] == "UPGRADE" and p1["ip_rating_change"] == "UPGRADE"
    assert (p1["number_of_restructurings"], p1["number_of_restructuring_papers"]) == (1, 1)
    assert p1["disbursement_pct_of_net_principal"] == D("40.000000")
    assert {"prj-1", "fin-1", "enr-1", "isr-P1-6", "isr-P1-5"} <= set(p1["source_record_ids"])


def test_timeline_keeps_isr_sequence_order_and_anomaly(gold):
    p2 = sorted(
        rows(gold, "project_timeline", project_id="P2", event_type="ISR_REPORT"),
        key=lambda r: r["event_sequence"],
    )
    assert [r["isr_sequence"] for r in p2] == [1, 2, 3]
    first = p2[0]
    assert first["event_date"] == date(2024, 6, 30)  # printed date kept
    assert first["ordering_date"] == date(2024, 1, 15)
    assert first["ordering_basis"] == "ISR_SEQUENCE_ADJUSTED" and first["date_sequence_anomaly"]
    assert not p2[1]["date_sequence_anomaly"]


def test_candidate_dates_are_never_promoted(gold):
    (paper,) = rows(gold, "project_timeline", source_record_id="ev-r1-paper")
    assert paper["event_date"] is None
    assert paper["event_date_status"] == "DERIVED_CANDIDATE"
    assert (paper["candidate_event_date"], paper["ordering_basis"]) == (
        date(2020, 1, 1),
        "CANDIDATE_DATE",
    )
    (dated,) = rows(gold, "project_timeline", source_record_id="ev-r1")
    assert dated["event_date_status"] == "SOURCE_STATED"


def test_result_progress_is_safe_and_explained(gold):
    by = {r["source_record_id"]: r for r in rows(gold, "result_progress")}
    k1 = by["r-k1-6"]
    assert (k1["progress_percentage"], k1["target_gap"], k1["calculation_status"]) == (
        D("50.000000"),
        D("50.000000"),
        "OK",
    )
    assert (k1["previous_number"], k1["absolute_change"], k1["trend_direction"]) == (
        D("5.000000"),
        D("45.000000"),
        "TOWARD_TARGET",
    )
    assert by["r-k1-5"]["trend_direction"] == "AWAY_FROM_TARGET"
    assert by["r-k1-4"]["trend_direction"] == "NOT_COMPARABLE"  # first observation
    expected = {
        "r-k2-6": "ZERO_TARGET_DISTANCE",
        "r-k3-6": "QUALITATIVE_UNIT",
        "r-k4-6": "MISSING_BASELINE",
        "r-k5-6": "DLI_LAYOUT_NOT_EVALUATED",
        "r-k8-6": "MISSING_TARGET",
        "r-k9-6": "NON_NUMERIC_VALUE",
    }
    for rid, status in expected.items():
        assert by[rid]["calculation_status"] == status, rid
        assert (
            by[rid]["progress_percentage"] is None and by[rid]["target_status"] == "NOT_EVALUABLE"
        )
    assert by["r-k9-6"]["target_number"] is None  # >6 decimals is never rounded
    assert by["r-k8-6"]["target_changed"] is False  # missing target is not a change


def test_pending_alias_pair_stays_two_series(gold):
    by = {r["source_record_id"]: r for r in rows(gold, "result_progress")}
    assert by["r-k6-6"]["canonical_indicator_id"] == "K6"
    assert by["r-k7-6"]["canonical_indicator_id"] == "K7"
    assert {by["r-k6-6"]["identity_review_status"], by["r-k7-6"]["identity_review_status"]} == {
        "PENDING_ALIAS_REVIEW"
    }
    assert by["r-k1-6"]["identity_review_status"] == "EXACT_IDENTITY"


def test_rating_signals_use_sequence_and_ordinal_ranks(gold):
    downgrades = {
        (s["subject"], s["effective_sequence"]): s["severity"]
        for s in signals(gold, "RATING_DOWNGRADE")
    }
    assert downgrades == {
        ("PDO rating", 2): "WATCH",
        ("PDO rating", 3): "HIGH",
        ("PDO rating", 4): "HIGH",
        ("Implementation progress rating", 2): "WATCH",
        ("Implementation progress rating", 3): "HIGH",
    }
    recoveries = {(s["subject"], s["effective_sequence"]) for s in signals(gold, "RATING_RECOVERY")}
    assert recoveries == {("PDO rating", 6), ("Implementation progress rating", 6)}
    persistent = {
        (s["subject"], s["current_value"], s["severity"], s["signal_status"])
        for s in signals(gold, "RATING_BELOW_SATISFACTORY_PERSISTENT")
    }
    assert persistent == {
        ("PDO rating", "3", "WATCH", "HISTORICAL"),
        ("Implementation progress rating", "3", "WATCH", "HISTORICAL"),
    }
    assert not signals(gold, "RATING_DOWNGRADE", project_id="P2")


def test_schedule_signals_and_duplicate_reports_counted_once(gold):
    (ext,) = signals(gold, "SCHEDULE_CLOSING_DATE_EXTENDED")
    assert (ext["severity"], ext["current_value"], ext["comparison_value"]) == (
        "HIGH",
        "2026-12-31",
        "2023-12-31",
    )  # 36 months >= 24
    assert ext["source_table"] == "silver.project_enrichment" and ext["page_number"] == 6
    assert {"prj-1", "enr-1", "ev-c1", "ev-c2"} <= set(ext["supporting_record_ids"])
    (repeated,) = signals(gold, "SCHEDULE_REPEATED_CLOSING_DATE_CHANGES")
    assert repeated["current_value"] == "2"  # ev-c1 and its duplicate report count once
    assert not signals(gold, "SCHEDULE_CLOSING_DATE_EXTENDED", project_id="P2")


def test_financial_rule_applies_to_ipf_only(gold):
    (lag,) = signals(gold, "FINANCE_DISBURSEMENT_LAG")
    assert lag["project_id"] == "P1"  # P2 is PforR: rule does not apply
    assert lag["severity"] == "HIGH"  # elapsed ~91% vs 40% disbursed
    assert lag["source_table"] == "silver.project_financial_summary"


def test_restructuring_and_scope_signals_are_neutral(gold):
    (restructuring,) = signals(gold, "CHANGE_RESTRUCTURING")
    assert restructuring["current_value"] == "1" and restructuring["severity"] == "INFO"
    assert "undated restructuring papers" in restructuring["signal_description"]
    assert set(restructuring["supporting_record_ids"]) == {"ev-r1", "ev-r1-paper"}
    (af,) = signals(gold, "CHANGE_ADDITIONAL_FINANCING")
    assert af["current_value"] == "50 USD_MILLIONS"
    for s in rows(gold, "attention_signals"):
        assert "fail" not in s["signal_description"].lower()
        assert s["provenance_class"] == "SYSTEM_DERIVED_SIGNAL"
        assert s["source_record_id"] in s["supporting_record_ids"]


def test_result_signals(gold):
    passed = {s["subject"]: s for s in signals(gold, "RESULT_TARGET_DATE_PASSED_NOT_MET")}
    assert {k: s["severity"] for k, s in passed.items()} == {
        "K1 (Number)": "WATCH",
        "K6 (Number)": "WATCH",
        "K7 (Number)": "HIGH",
    }  # 50/50/25 %
    assert passed["K7 (Number)"]["caveats"] and not passed["K1 (Number)"]["caveats"]
    (away,) = signals(gold, "RESULT_MOVED_AWAY_FROM_TARGET")
    assert away["effective_sequence"] == 5
    assert set(away["supporting_record_ids"]) == {"r-k1-4", "r-k1-5"}
    assert not signals(gold, "RESULT_TARGET_CHANGED")  # K8 target missing, not changed


def test_risk_register_preserves_framing(gold):
    register = {r["source_record_id"]: r for r in rows(gold, "risk_register")}
    assert register["risk-1"]["record_type"] == "FORMAL_RISK_RATING"
    assert register["risk-1"]["rating_rank"] == 3
    assert register["risk-2"]["record_type"] == "ASSESSMENT_FINDING"
    assert register["sort-1"]["record_type"] == "ISR_SORT_RATING"
    assert {r["resolution_status"] for r in register.values()} == {"NOT_STATED"}
    (overall,) = signals(gold, "RISK_OVERALL_RATING_ELEVATED")
    assert overall["project_id"] == "P1" and "6 consecutive ISRs" in overall["signal_description"]


def test_ids_and_hashes_are_deterministic_and_operational_metadata_excluded(spark, silver, gold):
    other = GoldContext("fixture-snapshot", "run-2", datetime(2027, 1, 1, tzinfo=UTC), "test")
    again = build_gold(spark, silver, RULES, SCALES, other)
    for name, frame in gold.frames.items():
        a = profile_frame(frame, gold.contracts[name])
        b = profile_frame(again.frames[name], again.contracts[name])
        assert a["fingerprint"] == b["fingerprint"] and a["row_count"] == b["row_count"], name
        ids_a = sorted(r["record_id"] for r in frame.select("record_id").collect())
        ids_b = sorted(r["record_id"] for r in again.frames[name].select("record_id").collect())
        assert ids_a == ids_b


def test_contract_validation_catches_duplicates_and_bad_values(gold):
    from pyspark.sql import functions as F

    contract = gold.contracts["attention_signals"]
    frame = gold.frames["attention_signals"]
    doubled = frame.unionByName(frame.limit(1))
    assert any("duplicate record_id" in p for p in validate_frame(doubled, contract))
    bad = frame.withColumn("severity", F.lit("CRITICAL"))
    assert any("severity outside vocabulary" in p for p in validate_frame(bad, contract))


def test_invariants_block_a_promoted_candidate_date(spark, silver, gold):
    from pyspark.sql import functions as F

    from worldbank_copilot.intelligence.checks import blocking, run_checks

    frames = dict(gold.frames)
    frames["project_timeline"] = frames["project_timeline"].withColumn(
        "event_date", F.coalesce("event_date", "candidate_event_date")
    )  # simulated defect
    failed = {r.check_id for r in blocking(run_checks(silver, frames, RULES))}
    assert "TIMELINE_NO_DERIVED_DATE_PROMOTED" in failed


def test_local_run_is_a_dry_run(spark, silver):
    report = run_gold(spark, SETTINGS, silver=silver, write=False)
    assert report.mode.startswith("LOCAL_SPARK_DRY_RUN") and report.writes == []
    assert all(r.status in ("PASS", "OBSERVED") for r in report.checks)
