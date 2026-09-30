"""Validate local source files: Bronze ingestion -> Silver transformation -> Silver validation.

Usage:
    python scripts/validate_data.py [--layer bronze|silver] [--json PATH]
                                    [--write-bronze] [--write-silver] [--env local]

Reads source files only; never writes to the data directory. With --write-*,
tables are written as JSONL under data.local_output_root/{bronze,silver}.
Exit code 1 if any ERROR observation is found in a layer that was run.
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
from worldbank_copilot.common.logging import configure_logging
from worldbank_copilot.ingestion.data_quality import build_data_quality_report
from worldbank_copilot.ingestion.pipeline import ingest_bronze, write_bronze
from worldbank_copilot.ingestion.validation_report import format_text_report, to_json_dict
from worldbank_copilot.transformations.bronze import LocalJsonlBronzeWriter
from worldbank_copilot.transformations.silver import (
    LocalJsonlSilverWriter,
    build_silver,
    write_silver,
)
from worldbank_copilot.transformations.silver_quality import build_silver_quality_report
from worldbank_copilot.transformations.silver_report import format_silver_report, silver_json_dict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--layer", choices=("bronze", "silver"), default="silver",
        help="last layer to build and validate (default: silver, which includes bronze)",
    )  # fmt: skip
    parser.add_argument("--json", type=Path, help="also write the full report as JSON")
    parser.add_argument("--write-bronze", action="store_true", help="write Bronze JSONL")
    parser.add_argument("--write-silver", action="store_true", help="write Silver JSONL")
    parser.add_argument("--env", help="force environment (local | databricks)")
    args = parser.parse_args(argv)

    settings = load_settings(args.env)
    configure_logging(settings.log_level)
    registry = load_project_registry(settings.config_dir)

    result = ingest_bronze(settings, registry)
    bronze_report = build_data_quality_report(result, registry)
    print(format_text_report(result, bronze_report, registry))
    payload = to_json_dict(result, bronze_report, registry)
    has_errors = bronze_report.has_errors

    if args.layer == "silver":
        silver = build_silver(result, registry)
        silver_report = build_silver_quality_report(silver, registry)
        print(format_silver_report(silver, silver_report, registry))
        payload["silver"] = silver_json_dict(silver, silver_report)
        has_errors = has_errors or silver_report.has_errors
        if args.write_silver:
            writer = LocalJsonlSilverWriter(settings.local_output_root / "silver")
            written = write_silver(silver, writer)
            writer.write_lineage()
            print(f"\nSilver tables written to {writer.root} ({len(written)} tables + lineage)")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON report written to {args.json}")
    if args.write_bronze:
        target = settings.local_output_root / "bronze"
        written = write_bronze(result, LocalJsonlBronzeWriter(target))
        print(f"\nBronze tables written to {target} ({len(written)} tables)")
    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
