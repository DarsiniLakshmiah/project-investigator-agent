"""Phase 9 tools on real data with local Spark (development rehearsal, nothing persisted).

Input: the Phase 6 Silver export (reconciled with Databricks) and Gold built from it by
the Phase 7 code, registered as temporary views. Tools read them through
``SparkTableReader`` (DataFrame API; Delta version pinning is exercised in Databricks).
Assertions are reconciliation invariants against the Gold/Silver frames and facts
already validated in Phase 7, plus exact Spark == in-memory reader equivalence.
Skipped when the export is absent.
"""

from datetime import UTC, datetime

import pytest

from worldbank_copilot.common import load_project_registry, load_settings
from worldbank_copilot.intelligence.frames import GoldContext
from worldbank_copilot.intelligence.pipeline import build_gold, silver_from_json, validate_build
from worldbank_copilot.intelligence.rules import load_rules, load_scales
from worldbank_copilot.lakehouse.spark_store import spark_schema
from worldbank_copilot.tools import registry
from worldbank_copilot.tools.base import ToolContext
from worldbank_copilot.tools.models import ProvenanceClass, ToolStatus
from worldbank_copilot.tools.reader import TABLES, InMemoryReader, SparkTableReader

pytestmark = [pytest.mark.spark, pytest.mark.integration]

SETTINGS = load_settings("local", env={})
EXPORT = SETTINGS.local_output_root / "lakehouse"
REGISTRY = load_project_registry(SETTINGS.config_dir)
RULES, SCALES = load_rules(SETTINGS.config_dir), load_scales(SETTINGS.config_dir)
PROJECTS = REGISTRY.project_ids
EXECUTOR = registry.default_executor()
CALLS = [
    ("get_project_overview", {}),
    ("get_project_timeline", {}),
    ("get_rating_history", {}),
    ("get_financial_status", {"as_of_isr": "latest", "include_events": True}),
    ("get_results_progress", {"history": True}),
    ("get_results_progress", {}),
    ("get_risk_register", {}),
    ("get_attention_signals", {"status": "ALL"}),
]


def view(logical: str) -> str:
    return "tools_" + logical.replace(".", "_")


@pytest.fixture(scope="module")
def real(spark):
    if not (EXPORT / "silver.project_results.jsonl").exists():
        pytest.skip("no Phase 6 Silver export; run scripts/platformize.py")
    silver = silver_from_json(spark, EXPORT)
    ctx = GoldContext("real", "run-a", datetime(2026, 9, 30, tzinfo=UTC), "test")
    build = build_gold(spark, silver, RULES, SCALES, ctx)
    validate_build(build)
    rows = {}
    for logical, contract_fn in TABLES.items():
        layer, name = logical.split(".", 1)
        if layer == "gold":
            frame = build.frames[name]
        else:
            contract = contract_fn()
            frame = (
                spark.read.schema(spark_schema(contract))
                .option("mode", "FAILFAST")
                .json(str(EXPORT / f"{logical}.jsonl"))
                .select(*contract.column_names)
            )
        frame = frame.localCheckpoint()  # materialise once; tools read it many times
        frame.createOrReplaceTempView(view(logical))
        rows[logical] = [r.asDict(recursive=True) for r in frame.collect()]
    return spark, rows


def run(reader, name, args, project_id):
    ctx = ToolContext(reader, REGISTRY, SCALES, RULES, request_id="spark-test")
    return EXECUTOR.run(name, {"project_id": project_id, **args}, ctx, scope_project_id=project_id)


def spark_reader(spark):
    return SparkTableReader(spark, view, pin_versions=False)


def stable(result):
    return result.model_dump(mode="json", exclude={"latency_ms", "data_snapshot"})


@pytest.mark.parametrize(("name", "args"), CALLS)
def test_spark_reader_equals_in_memory_reader_on_real_data(real, name, args):
    spark, rows = real
    for pid in PROJECTS:
        via_spark = run(spark_reader(spark), name, args, pid)
        via_memory = run(InMemoryReader(rows), name, args, pid)
        assert via_spark.status != ToolStatus.ERROR, (name, pid, via_spark.error)
        assert stable(via_spark) == stable(via_memory), (name, pid)


def test_tool_outputs_reconcile_with_gold_counts(real):
    spark, rows = real
    totals = {"timeline": 0, "results": 0, "risks": 0, "signals": 0}
    for pid in PROJECTS:
        r = spark_reader(spark)
        totals["timeline"] += len(run(r, "get_project_timeline", {"limit": 500}, pid).items)
        totals["results"] += len(run(r, "get_results_progress", {"history": True}, pid).items)
        totals["risks"] += len(run(r, "get_risk_register", {}, pid).items)
        totals["signals"] += len(run(r, "get_attention_signals", {"status": "ALL"}, pid).items)
    assert totals == {
        "timeline": len(rows["gold.project_timeline"]),
        "results": len(rows["gold.result_progress"]),
        "risks": len(rows["gold.risk_register"]),
        "signals": len(rows["gold.attention_signals"]),
    }
    assert totals == {"timeline": 74, "results": 837, "risks": 116, "signals": 69}  # Phase 7


def test_validated_phase7_facts_through_the_tools(real):
    spark, _ = real
    latest = {}
    for pid in PROJECTS:
        res = run(spark_reader(spark), "get_project_overview", {}, pid)
        f = {x.name: x for x in res.items[0].facts}
        latest[pid] = f["latest_isr_sequence"].value
        if pid == "P130544":
            days = f["days_extended"]
            assert (days.value, days.provenance_class) == (1765, ProvenanceClass.DOCUMENTED_FINDING)
        else:  # Program-for-Results: the IPF-only disbursement rule is stated as not evaluated
            assert any("FINANCE_DISBURSEMENT_LAG" in c for c in res.caveats)
    assert latest == {"P130544": 24, "P179039": 8, "P506272": 3}
    ratings = run(spark_reader(spark), "get_rating_history", {}, "P179039")
    assert [i.isr_sequence for i in ratings.items] == list(range(1, 9))  # by sequence


def test_real_loans_and_scope(real):
    spark, _ = real
    res = run(spark_reader(spark), "get_financial_status", {"as_of_isr": "latest"}, "P130544")
    (item,) = res.items
    assert [loan.loan_number for loan in item.loans] == ["IBRD86010", "IBRD93240"]
    assert {line.isr_sequence for line in item.isr_reported} == {24}
    foreign = run(
        spark_reader(spark), "get_financial_status", {"loan_number": "IBRD-8601-0"}, "P179039"
    )
    assert foreign.status == ToolStatus.SCOPE_REFUSED


def test_every_signal_is_system_derived_and_dli_layout_progress_is_never_computed(real):
    spark, _ = real
    for pid in PROJECTS:
        signals = run(spark_reader(spark), "get_attention_signals", {"status": "ALL"}, pid)
        assert all(
            s.provenance_class == ProvenanceClass.SYSTEM_DERIVED_SIGNAL for s in signals.items
        )
        # Phase 7 invariant: DLI-LAYOUT tables are never read as progress.
        history = run(spark_reader(spark), "get_results_progress", {"history": True}, pid)
        for obs in history.items:
            if obs.layout == "DLI":
                assert obs.progress_percentage.provenance_class == ProvenanceClass.UNKNOWN


def test_r005_overall_appraisal_risk_row_exists_in_the_governed_register(real):
    """9C human-review condition for r005 / q32: the overall appraisal risk rating of
    P179039 must be in gold.risk_register (FORMAL_RISK_RATING) with provenance."""
    spark, _ = real
    res = run(
        spark_reader(spark),
        "get_risk_register",
        {"record_type": "FORMAL_RISK_RATING", "category": "Overall"},
        "P179039",
    )
    (row,) = res.items
    assert (row.rating, row.source_document_type, row.source.page_number) == (
        "Moderate",
        "APPRAISAL_DOCUMENT",
        8,
    )
    assert row.provenance_class == ProvenanceClass.DOCUMENTED_FINDING
