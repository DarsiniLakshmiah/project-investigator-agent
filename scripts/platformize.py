"""Phase 6 local entry point: build, validate and reconcile the Delta-ready datasets.

Locally this is a DRY RUN: nothing is written to Databricks. Persistence happens only
in Databricks through notebooks/05_platformize_databricks.py.

Usage:
    python scripts/platformize.py                    # dry run + reconciliation vs expected
    python scripts/platformize.py --write-snapshot   # (re)record configs/source_snapshot.json
    python scripts/platformize.py --write-expected   # (re)record expected profiles + contract lock
    python scripts/platformize.py --print-ddl        # CREATE TABLE statements
    python scripts/platformize.py --upload-commands  # Databricks CLI commands for the Volumes

Also writes the persisted-shape rows to <local_output_root>/lakehouse/ (inspection) and
the indicator alias review sheet to review/indicator_alias_candidates.csv.
Exit code 1 on any integrity, contract or reconciliation failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import worldbank_copilot  # noqa: F401
except ModuleNotFoundError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldbank_copilot.common import load_project_registry, load_settings
from worldbank_copilot.common.exceptions import CopilotError
from worldbank_copilot.common.logging import configure_logging
from worldbank_copilot.ingestion.pipeline import resolve_source_files
from worldbank_copilot.lakehouse.identity import canonical
from worldbank_copilot.lakehouse.pipeline import (
    build_platform_datasets,
    format_report,
    run_platform,
)
from worldbank_copilot.lakehouse.reconcile import write_expected
from worldbank_copilot.lakehouse.review import REVIEW_FILE, write_review_csv
from worldbank_copilot.lakehouse.sources import build_snapshot, write_snapshot
from worldbank_copilot.lakehouse.sql import create_table_sql

CONTRACT_LOCK = "delta_contracts.lock.json"


def record_snapshot(settings, registry) -> Path:
    files = resolve_source_files(settings)
    structured = [files.projects_workbook, files.loans_snapshot, files.procurement_contract_awards]
    documents = []
    for pid in registry.project_ids:
        folder = Path(settings.documents_root) / registry.get(pid).documents_dir
        documents += [(pid, p) for p in sorted(folder.glob("*.pdf"))]
    snapshot = build_snapshot(Path(settings.data_root), structured, documents)
    return write_snapshot(snapshot, settings.config_dir)


def upload_commands(settings) -> str:
    local_data = Path(settings.data_root)
    parsed = settings.local_output_root / "parsed"
    source = (
        f"dbfs:/Volumes/{settings.databricks.catalog}/{settings.databricks.bronze_schema}/"
        f"{settings.databricks.source_volume}/data"
    )
    artefacts = (
        f"dbfs:/Volumes/{settings.databricks.catalog}/"
        f"{settings.databricks.silver_schema}/{settings.databricks.artifact_volume}/parsed"
    )
    return "\n".join(
        [
            "# Databricks CLI (authenticated to your workspace). Files are copied byte-for-byte;",
            "# the notebook verifies every hash against configs/source_snapshot.json.",
            f'databricks fs cp --recursive "{local_data}" "{source}"',
            f'databricks fs cp --recursive "{parsed}" "{artefacts}"',
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-snapshot", action="store_true")
    parser.add_argument("--write-expected", action="store_true")
    parser.add_argument("--print-ddl", action="store_true")
    parser.add_argument("--upload-commands", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    settings = load_settings("local")
    configure_logging(settings.log_level)
    registry = load_project_registry(settings.config_dir)
    if args.upload_commands:
        print(upload_commands(settings))
        return 0
    if args.write_snapshot:
        print(f"wrote {record_snapshot(settings, registry)}")

    say = None if args.quiet else print
    try:
        build = build_platform_datasets(settings, registry, progress=say)
        report = run_platform(settings, registry, build=build)
    except CopilotError as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1

    if args.write_expected:
        path = write_expected(report.profiles, build.snapshot.snapshot_id, settings.config_dir)
        lock = {name: ds.contract.to_dict() for name, ds in build.datasets.items()}
        (settings.config_dir / CONTRACT_LOCK).write_text(
            json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {path} and {settings.config_dir / CONTRACT_LOCK}")
        report = run_platform(settings, registry, build=build)

    out = settings.local_output_root / "lakehouse"
    out.mkdir(parents=True, exist_ok=True)
    for name, dataset in build.datasets.items():
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in dataset.rows:
                fh.write(json.dumps(canonical(row), ensure_ascii=False) + "\n")
    review = write_review_csv(
        build.datasets["silver.indicator_match_candidates"], settings.repo_root / REVIEW_FILE
    )
    if args.print_ddl:
        for dataset in build.datasets.values():
            schema = getattr(settings.databricks, f"{dataset.contract.layer}_schema")
            print(create_table_sql(settings.databricks.catalog, schema, dataset.contract) + ";\n")
    print(format_report(report))
    print(f"persisted-shape rows: {out}")
    print(f"indicator alias review sheet: {review}")
    return 1 if report.expected_diffs else 0


if __name__ == "__main__":
    raise SystemExit(main())
