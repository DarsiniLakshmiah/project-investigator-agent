"""Phase 7 orchestration: governed Silver Delta tables -> Gold Delta tables (Databricks).

Production path (Databricks, notebook 06_build_gold_intelligence):

    worldbank_copilot.silver.*  (validated against the Phase 6 contracts)
      -> Spark transformations (timeline, result_progress, risk_register, signals, 360)
      -> Gold contracts (types, required, vocabularies, duplicate keys, content hashes)
      -> Gold invariants (ERROR blocks the write) + quality observations
      -> snapshot MERGE into worldbank_copilot.gold.*
      -> read back and reconcile profiles with the validated build

``silver_from_json`` exists only for local development rehearsal and tests: it loads
Silver rows exported by the Phase 6 dry run with the same contract schemas. It is not a
production input.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from worldbank_copilot.common.config import Settings
from worldbank_copilot.common.exceptions import ReconciliationError
from worldbank_copilot.intelligence.checks import (
    CheckResult,
    blocking,
    quality_frame,
    run_checks,
)
from worldbank_copilot.intelligence.contracts import GOLD_CONTRACTS
from worldbank_copilot.intelligence.frames import (
    GoldContext,
    finalize_frame,
    input_fingerprint,
    profile_frame,
    validate_frame,
)
from worldbank_copilot.intelligence.project_360 import build_project_360
from worldbank_copilot.intelligence.results import build_result_progress
from worldbank_copilot.intelligence.risks import build_risk_register
from worldbank_copilot.intelligence.rules import RatingScales, RuleSet, load_rules, load_scales
from worldbank_copilot.intelligence.signals import build_signals
from worldbank_copilot.intelligence.timeline import build_timeline
from worldbank_copilot.lakehouse.contracts import ContractError, TableContract
from worldbank_copilot.lakehouse.pipeline import pipeline_version
from worldbank_copilot.lakehouse.reconcile import compare
from worldbank_copilot.lakehouse.records import (
    enrichment_contract,
    events_contract,
    isr_contract,
    isr_sort_contract,
    quality_contract,
    results_contract,
    risks_contract,
    silver_contract,
)
from worldbank_copilot.lakehouse.review import candidates_contract

# Silver tables Gold reads, with the Phase 6 contracts they must satisfy.
SILVER_INPUTS: dict[str, Callable[[], TableContract]] = {
    "projects": lambda: silver_contract("silver_projects"),
    "project_financial_summary": lambda: silver_contract("silver_project_financial_summary"),
    "project_enrichment": enrichment_contract,
    "isr_snapshots": isr_contract,
    "isr_sort_ratings": isr_sort_contract,
    "project_events": events_contract,
    "project_results": results_contract,
    "appraisal_risks": risks_contract,
    "indicator_match_candidates": candidates_contract,
    "data_quality_observations": quality_contract,
}

GOLD_ORDER = (
    "project_360",
    "project_timeline",
    "result_progress",
    "risk_register",
    "attention_signals",
    "quality_observations",
)


def read_silver(spark: Any, settings: Settings) -> dict[str, Any]:
    """Governed Silver Delta tables, each checked against its Phase 6 contract."""
    from worldbank_copilot.intelligence.frames import schema_problems

    frames = {}
    for name, contract_fn in SILVER_INPUTS.items():
        table = settings.table_name("silver", name)
        if not spark.catalog.tableExists(table):
            raise ContractError(f"required Silver table {table} does not exist (run Phase 6)")
        contract = contract_fn()
        frame = spark.table(table).select(*contract.column_names)
        problems = schema_problems(frame, contract)
        if problems:
            raise ContractError(f"{table} does not match its Phase 6 contract: {problems}")
        frames[name] = frame
    return frames


def silver_from_json(spark: Any, directory: Path) -> dict[str, Any]:
    """DEVELOPMENT ONLY: Silver rows exported by scripts/platformize.py (Phase 6 dry run)."""
    from worldbank_copilot.lakehouse.spark_store import spark_schema

    frames = {}
    for name, contract_fn in SILVER_INPUTS.items():
        contract = contract_fn()
        path = Path(directory) / f"silver.{name}.jsonl"
        frames[name] = (
            spark.read.schema(spark_schema(contract))
            .option("mode", "FAILFAST")
            .json(str(path))
            .select(*contract.column_names)
        )
    return frames


@dataclass
class GoldBuild:
    context: GoldContext
    frames: dict[str, Any]  # table -> finalized DataFrame
    contracts: dict[str, TableContract]
    checks: list[CheckResult]


def gold_context(silver: dict[str, Any], loaded_at: datetime | None = None) -> GoldContext:
    return GoldContext(
        source_snapshot_id=input_fingerprint(silver),
        load_run_id=uuid.uuid4().hex,
        loaded_at=loaded_at or datetime.now(UTC),
        pipeline_version=pipeline_version(),
    )


def build_gold(
    spark: Any, silver: dict[str, Any], rules: RuleSet, scales: RatingScales, ctx: GoldContext
) -> GoldBuild:
    contracts = {name: fn() for name, fn in GOLD_CONTRACTS.items()}
    progress = build_result_progress(silver)
    signals = build_signals(silver, progress, rules, scales)
    raw = {
        "project_timeline": build_timeline(silver),
        "result_progress": progress,
        "risk_register": build_risk_register(silver, scales),
        "attention_signals": signals,
        "project_360": build_project_360(silver, signals, scales),
    }
    frames = {name: finalize_frame(raw[name], contracts[name], ctx) for name in raw}
    checks = run_checks(silver, frames, rules)
    frames["quality_observations"] = finalize_frame(
        quality_frame(spark, checks), contracts["quality_observations"], ctx
    )
    return GoldBuild(ctx, {n: frames[n] for n in GOLD_ORDER}, contracts, checks)


@dataclass
class GoldReport:
    mode: str
    source_snapshot_id: str
    load_run_id: str
    profiles: dict[str, dict[str, Any]]
    checks: list[CheckResult]
    objects: list[Any] = field(default_factory=list)
    writes: list[Any] = field(default_factory=list)
    readback_diffs: dict[str, list[str]] = field(default_factory=dict)

    @property
    def idempotent_write(self) -> bool | None:
        if not self.writes:
            return None
        return all(w.inserted == w.updated == w.deleted == 0 for w in self.writes)


def validate_build(build: GoldBuild) -> None:
    problems = [
        p
        for name, frame in build.frames.items()
        for p in validate_frame(frame, build.contracts[name])
    ]
    if problems:
        raise ContractError("Gold contract violations: " + "; ".join(problems[:10]))
    errors = blocking(build.checks)
    if errors:
        raise ReconciliationError(
            "Gold invariants failed (nothing written): "
            + "; ".join(f"{r.check_id}={r.count}" for r in errors)
        )


def run_gold(
    spark: Any,
    settings: Settings,
    *,
    silver: dict[str, Any] | None = None,
    write: bool = True,
    progress: Callable[[str], None] | None = None,
) -> GoldReport:
    """Build, validate and (with ``write``) persist Gold; always returns the report."""
    say = progress or (lambda _msg: None)
    rules, scales = load_rules(settings.config_dir), load_scales(settings.config_dir)
    silver = silver if silver is not None else read_silver(spark, settings)
    say("Silver inputs validated against Phase 6 contracts")
    build = build_gold(spark, silver, rules, scales, gold_context(silver))
    validate_build(build)
    say("Gold contracts and invariants passed")
    profiles = {n: profile_frame(frame, build.contracts[n]) for n, frame in build.frames.items()}
    report = GoldReport(
        "DATABRICKS" if write else "LOCAL_SPARK_DRY_RUN (nothing persisted)",
        build.context.source_snapshot_id,
        build.context.load_run_id,
        profiles,
        build.checks,
    )
    if not write:
        return report

    from worldbank_copilot.lakehouse.spark_store import SparkDeltaStore

    store = SparkDeltaStore(
        spark,
        settings.require("databricks.catalog"),
        {"gold": settings.require("databricks.gold_schema")},
    )
    report.objects.append(store.validate_catalog())
    report.objects.append(
        store.ensure_schema(
            "gold", "Deterministic Gold intelligence (Phase 7): facts and rule-based signals."
        )
    )
    for name, frame in build.frames.items():
        contract = build.contracts[name]
        say(f"MERGE gold.{name}")
        report.objects.append(store.ensure_table(contract))
        report.writes.append(store.merge_frame(contract, frame))
    for name in build.frames:
        contract = build.contracts[name]
        diffs = compare(profiles[name], profile_frame(store.read_frame(contract), contract))
        if diffs:
            report.readback_diffs[name] = diffs
    if report.readback_diffs:
        raise ReconciliationError(
            f"Gold read-back differs from the validated build: {report.readback_diffs}"
        )
    return report


def format_report(report: GoldReport) -> str:
    lines = [
        f"Phase 7 Gold: {report.mode}",
        f"  silver input fingerprint {report.source_snapshot_id}",
        f"  load run                 {report.load_run_id}",
    ]
    for obj in report.objects:
        lines.append(f"  {obj.object_type:<7} {obj.name:<55} {obj.status}")
    writes = {w.table.rsplit(".", 1)[-1]: w for w in report.writes}
    for name, p in report.profiles.items():
        w = writes.get(name)
        merge = f"  inserted {w.inserted} updated {w.updated} deleted {w.deleted}" if w else ""
        lines.append(
            f"    gold.{name:<24} {p['row_count']:>5} rows  {p['fingerprint'][:12]}{merge}"
        )
    lines.append("  checks:")
    for r in report.checks:
        lines.append(f"    [{r.status:<8}] {r.severity:<7} {r.check_id:<36} {r.count}")
    if report.writes:
        lines.append(
            "  read-back reconciliation: "
            + ("OK" if not report.readback_diffs else str(report.readback_diffs))
        )
        lines.append(f"  idempotent (no rows changed by this run): {report.idempotent_write}")
    return "\n".join(lines)
