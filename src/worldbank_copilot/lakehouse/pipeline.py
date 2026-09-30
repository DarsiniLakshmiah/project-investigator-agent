"""Phase 6 orchestration: validated Bronze/Silver/document-derived data -> Delta.

Flow (identical locally and in Databricks up to the persistence step):

1. load the committed source snapshot manifest and verify every source file hash
   under the environment's data root (hard stop on any mismatch or missing file);
2. rebuild Bronze, structured Silver and document-derived Silver with the existing
   Phase 2-5 code (Docling is not re-run; the Phase 4 parsed cache is used) and check
   that every parsed document was built from the verified source file;
3. shape rows by explicit contracts, validate them (types, nulls, vocabularies,
   Decimal scale, duplicate keys) and profile them;
4. reconcile the profiles with the committed expected profiles for this snapshot
   before anything is written;
5. Databricks only (a Spark session is supplied): validate/create Unity Catalog
   objects, MERGE every table, read every table back and require the read-back
   profile to equal the validated one.

Without a Spark session the run is a LOCAL DRY RUN: nothing is persisted and the
report says so.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from worldbank_copilot.common.config import Settings
from worldbank_copilot.common.exceptions import ReconciliationError, SourceIntegrityError
from worldbank_copilot.common.project_registry import ProjectRegistry
from worldbank_copilot.common.quality import DataQualityReport
from worldbank_copilot.extraction.identity import load_aliases
from worldbank_copilot.extraction.pipeline import (
    ALIASES_FILE,
    EXTRACTOR_VERSION,
    ExtractionRun,
    build_quality_report,
    run_extraction,
)
from worldbank_copilot.ingestion.data_quality import build_data_quality_report
from worldbank_copilot.ingestion.pipeline import ingest_bronze
from worldbank_copilot.lakehouse.contracts import require_valid
from worldbank_copilot.lakehouse.reconcile import compare, load_expected, profile
from worldbank_copilot.lakehouse.records import (
    Dataset,
    LoadContext,
    bronze_dataset,
    enrichment_dataset,
    events_dataset,
    isr_child_datasets,
    isr_dataset,
    quality_dataset,
    record_quality_inputs,
    report_inputs,
    results_dataset,
    risks_dataset,
    silver_dataset,
)
from worldbank_copilot.lakehouse.review import candidates_dataset
from worldbank_copilot.lakehouse.sources import (
    IntegrityReport,
    SourceSnapshot,
    load_snapshot,
    require_integrity,
    verify_snapshot,
)
from worldbank_copilot.parsing.report import build_parsing_report
from worldbank_copilot.transformations.silver import build_silver
from worldbank_copilot.transformations.silver_quality import build_silver_quality_report

LOCAL_DRY_RUN = "LOCAL_DRY_RUN (nothing persisted)"
DATABRICKS = "DATABRICKS"


def code_hash() -> str:
    """Hash of the lakehouse, extraction and intelligence code (operational version tag)."""
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for package in ("lakehouse", "extraction", "intelligence"):
        for path in sorted((root / package).glob("*.py")):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def pipeline_version() -> str:
    try:
        version = metadata.version("worldbank-implementation-intelligence")
    except metadata.PackageNotFoundError:
        version = "source"
    return f"{version}+extractor.{EXTRACTOR_VERSION}+code.{code_hash()}"


def ingestion_run_id(snapshot: SourceSnapshot) -> str:
    """Deterministic: Silver lineage (source_refs) is identical across reloads."""
    return f"snapshot-{snapshot.snapshot_id[:16]}"


def load_context(snapshot: SourceSnapshot, loaded_at: datetime | None = None) -> LoadContext:
    return LoadContext(
        source_snapshot_id=snapshot.snapshot_id,
        load_run_id=uuid.uuid4().hex,  # operational only; records use deterministic ids
        loaded_at=loaded_at or datetime.now(UTC),
        pipeline_version=pipeline_version(),
        source_hashes=snapshot.hashes,
    )


@dataclass
class PlatformBuild:
    context: LoadContext
    snapshot: SourceSnapshot
    integrity: IntegrityReport
    datasets: dict[str, Dataset]  # "<layer>.<table>" -> dataset
    reports: dict[str, DataQualityReport]
    extraction: ExtractionRun


def _check_parsed_lineage(extraction: ExtractionRun, snapshot: SourceSnapshot) -> None:
    hashes = snapshot.hashes
    bad = []
    for doc in extraction.documents.values():
        expected = hashes.get(doc.source_ref.relative_path)
        if expected != doc.source_hash:
            bad.append(
                f"{doc.source_ref.relative_path} (parsed {doc.source_hash[:12]}, "
                f"snapshot {(expected or 'absent')[:12]})"
            )
    if bad:
        raise SourceIntegrityError(
            "parsed documents were not built from the verified source files: " + "; ".join(bad[:10])
        )
    if extraction.hashes_unchanged is False:
        raise SourceIntegrityError(
            f"source files changed during the run: {extraction.hash_mismatches}"
        )


def build_platform_datasets(
    settings: Settings, registry: ProjectRegistry, *, progress: Callable[[str], None] | None = None
) -> PlatformBuild:
    say = progress or (lambda _msg: None)
    snapshot = load_snapshot(settings.config_dir)
    say(f"Verifying {len(snapshot.files)} source files under {settings.data_root}")
    integrity = verify_snapshot(snapshot, settings.data_root)
    require_integrity(integrity)
    ctx = load_context(snapshot)

    say("Bronze ingestion (Phase 2 code)")
    bronze = ingest_bronze(
        settings, registry, ingested_at=ctx.loaded_at, run_id=ingestion_run_id(snapshot)
    )
    say("Structured Silver (Phase 3 code)")
    silver = build_silver(bronze, registry)
    say("Document-derived Silver (Phase 5 code, Phase 4 parsed cache)")
    extraction = run_extraction(
        settings,
        registry,
        silver.tables["silver_loans"].rows,
        silver.tables["silver_projects"].rows,
    )
    _check_parsed_lineage(extraction, snapshot)
    reports = {
        "bronze": build_data_quality_report(bronze, registry),
        "silver": build_silver_quality_report(silver, registry),
        "parsing": build_parsing_report(extraction.parse_run, registry),
        "silver_documents": build_quality_report(extraction),
    }

    datasets: dict[str, Dataset] = {}

    def add(dataset: Dataset) -> None:
        datasets[f"{dataset.contract.layer}.{dataset.contract.name}"] = dataset

    for table in bronze.all_tables():
        add(bronze_dataset(table, ctx))
    for name, table in silver.tables.items():
        add(silver_dataset(name, table.rows, ctx))
    add(isr_dataset(extraction.snapshots, extraction.documents, ctx))
    for child in isr_child_datasets(extraction.snapshots, ctx):
        add(child)
    add(results_dataset(extraction.results, ctx))
    add(risks_dataset(extraction.risks, ctx))
    add(events_dataset(extraction.events, ctx))
    add(enrichment_dataset(extraction.enrichment, ctx))
    aliases = load_aliases(settings.config_dir / ALIASES_FILE)
    add(candidates_dataset(extraction.issues, extraction.results, aliases, ctx))

    inputs = [item for layer, report in reports.items() for item in report_inputs(layer, report)]
    inputs = (
        record_quality_inputs(
            datasets,
            {
                "silver.isr_snapshots": extraction.snapshots,
                "silver.project_results": extraction.results,
                "silver.appraisal_risks": extraction.risks,
                "silver.project_events": extraction.events,
            },
        )
        + inputs
    )
    add(quality_dataset(inputs, ctx))

    for dataset in datasets.values():
        require_valid(dataset.contract, dataset.rows)
    return PlatformBuild(ctx, snapshot, integrity, datasets, reports, extraction)


@dataclass
class PlatformReport:
    mode: str
    snapshot_id: str
    load_run_id: str
    integrity: dict[str, int]
    unexpected_files: list[str]
    profiles: dict[str, dict[str, Any]]
    expected_diffs: dict[str, list[str]]
    objects: list[Any] = field(default_factory=list)
    writes: list[Any] = field(default_factory=list)
    readback_diffs: dict[str, list[str]] = field(default_factory=dict)
    quality_counts: dict[str, int] = field(default_factory=dict)

    @property
    def idempotent_write(self) -> bool | None:
        if not self.writes:
            return None
        return all(w.inserted == w.updated == w.deleted == 0 for w in self.writes)


def reconcile_with_expected(
    build: PlatformBuild, expected: dict[str, Any] | None, profiles: dict[str, Any]
) -> dict[str, list[str]]:
    if expected is None:
        return {
            "*": [
                "no committed expected profiles (configs/reconciliation/expected_"
                "profiles.json); run scripts/platformize.py --write-expected locally"
            ]
        }
    if expected.get("source_snapshot_id") != build.snapshot.snapshot_id:
        return {
            "*": [
                f"expected profiles are for snapshot "
                f"{str(expected.get('source_snapshot_id'))[:16]}, sources are "
                f"{build.snapshot.snapshot_id[:16]}; regenerate them locally after "
                "reviewing the new snapshot"
            ]
        }
    diffs = {}
    exp_profiles = expected.get("profiles", {})
    for name in sorted(set(exp_profiles) | set(profiles)):
        if name not in exp_profiles or name not in profiles:
            diffs[name] = [
                "table missing from " + ("expected" if name not in exp_profiles else "this build")
            ]
            continue
        found = compare(exp_profiles[name], profiles[name])
        if found:
            diffs[name] = found
    return diffs


def run_platform(
    settings: Settings,
    registry: ProjectRegistry,
    *,
    spark: Any = None,
    progress: Callable[[str], None] | None = None,
    build: PlatformBuild | None = None,
) -> PlatformReport:
    """Build, validate and reconcile; with ``spark`` also persist and verify Delta."""
    say = progress or (lambda _msg: None)
    store = None
    objects = []
    if spark is not None:
        from worldbank_copilot.lakehouse.spark_store import SparkDeltaStore

        store = SparkDeltaStore(
            spark,
            settings.require("databricks.catalog"),
            {
                "bronze": settings.require("databricks.bronze_schema"),
                "silver": settings.require("databricks.silver_schema"),
            },
        )
        objects += setup_unity_catalog(store, settings)

    build = build or build_platform_datasets(settings, registry, progress=say)
    profiles = {name: profile(ds.contract, ds.rows) for name, ds in build.datasets.items()}
    expected_diffs = reconcile_with_expected(build, load_expected(settings.config_dir), profiles)
    quality = build.datasets["silver.data_quality_observations"].rows
    report = PlatformReport(
        mode=DATABRICKS if store else LOCAL_DRY_RUN,
        snapshot_id=build.snapshot.snapshot_id,
        load_run_id=build.context.load_run_id,
        integrity=build.integrity.counts(),
        unexpected_files=list(build.integrity.unexpected_files),
        profiles=profiles,
        expected_diffs=expected_diffs,
        objects=objects,
        quality_counts={
            s: sum(r["severity"] == s for r in quality) for s in ("ERROR", "WARNING", "INFO")
        },
    )
    if store is None:
        return report
    if expected_diffs:
        raise ReconciliationError(
            "validated data does not reconcile with the committed expected profiles; nothing "
            f"was written: {dict(list(expected_diffs.items())[:3])}"
        )
    for name, dataset in build.datasets.items():
        say(f"MERGE {name} ({len(dataset.rows)} rows)")
        report.objects.append(store.ensure_table(dataset.contract))
        report.writes.append(store.write_snapshot(dataset.contract, dataset.rows))
    for name, dataset in build.datasets.items():
        persisted = profile(dataset.contract, store.read_rows(dataset.contract))
        diffs = compare(profiles[name], persisted)
        if diffs:
            report.readback_diffs[name] = diffs
    if report.readback_diffs:
        raise ReconciliationError(
            f"Delta read-back differs from the validated data: "
            f"{dict(list(report.readback_diffs.items())[:3])}"
        )
    return report


def setup_unity_catalog(store: Any, settings: Settings) -> list[Any]:
    """Validate the catalog; validate/create the bronze and silver schemas and Volumes.

    The gold schema is reserved for Phase 7 and deliberately not created.
    """
    return [
        store.validate_catalog(),
        store.ensure_schema("bronze", "Source-aligned Bronze data (World Bank copilot)."),
        store.ensure_schema("silver", "Normalized structured and document-derived Silver."),
        store.ensure_volume(
            "bronze",
            settings.require("databricks.source_volume"),
            "Original source files (validated snapshot; never modified).",
        ),
        store.ensure_volume(
            "silver",
            settings.require("databricks.artifact_volume"),
            "Pipeline artefacts derived from sources (parsed documents).",
        ),
    ]


def format_report(report: PlatformReport) -> str:
    lines = [
        f"Phase 6 platformization: {report.mode}",
        f"  source snapshot   {report.snapshot_id}",
        f"  load run          {report.load_run_id}",
        f"  source hashes     {report.integrity}",
    ]
    if report.unexpected_files:
        lines.append(f"  unexpected source-like files (ignored): {report.unexpected_files}")
    for obj in report.objects:
        lines.append(f"  {obj.object_type:<7} {obj.name:<60} {obj.status}")
    lines.append("  tables:")
    writes = {w.table.split(".", 1)[1]: w for w in report.writes}
    for name, p in report.profiles.items():
        w = writes.get(name)
        merge = (
            (
                f"  inserted {w.inserted} updated {w.updated} deleted {w.deleted} "
                f"(v{w.delta_version})"
            )
            if w
            else ""
        )
        lines.append(f"    {name:<42} {p['row_count']:>5} rows  {p['fingerprint'][:12]}{merge}")
    lines.append(f"  quality observations: {report.quality_counts}")
    lines.append(
        "  expected-profile reconciliation: "
        + ("OK" if not report.expected_diffs else f"DIFFERENCES {report.expected_diffs}")
    )
    if report.writes:
        lines.append(
            "  read-back reconciliation: "
            + ("OK" if not report.readback_diffs else f"DIFFERENCES {report.readback_diffs}")
        )
        lines.append(f"  idempotent (no rows changed by this run): {report.idempotent_write}")
    return "\n".join(lines)
