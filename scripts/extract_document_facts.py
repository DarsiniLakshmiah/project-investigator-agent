"""Extract document-derived Silver datasets from the Phase 4 parsed cache.

Usage:
    python scripts/extract_document_facts.py                  # all projects (cache reused)
    python scripts/extract_document_facts.py --project P130544
    python scripts/extract_document_facts.py --no-cache       # re-run extractors
    python scripts/extract_document_facts.py --quiet

Reads <local_output_root>/parsed (run scripts/parse_documents.py first) and the
structured sources (for cross-source checks). Writes
<local_output_root>/silver_documents/. Source files are only read. Exit code 1
if any ERROR observation is found.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

try:
    import worldbank_copilot  # noqa: F401
except ModuleNotFoundError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from worldbank_copilot.common import load_project_registry, load_settings
from worldbank_copilot.common.logging import configure_logging
from worldbank_copilot.extraction.pipeline import (
    OUTPUT_DIR,
    build_quality_report,
    run_extraction,
    summarize,
    write_outputs,
)
from worldbank_copilot.ingestion.pipeline import ingest_bronze
from worldbank_copilot.transformations.silver import build_silver


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", action="append", help="project ID (repeatable)")
    parser.add_argument("--no-cache", action="store_true", help="ignore the extraction cache")
    parser.add_argument("--quiet", action="store_true", help="no per-document progress")
    parser.add_argument("--env", help="force environment (local | databricks)")
    args = parser.parse_args(argv)

    settings = load_settings(args.env)
    configure_logging(settings.log_level)
    registry = load_project_registry(settings.config_dir)
    silver = build_silver(ingest_bronze(settings, registry), registry)
    run = run_extraction(
        settings,
        registry,
        silver.tables["silver_loans"].rows,
        silver.tables["silver_projects"].rows,
        projects=args.project,
        use_cache=not args.no_cache,
        progress=None if args.quiet else print,
    )
    report = build_quality_report(run)
    written = write_outputs(run, report, settings.local_output_root / OUTPUT_DIR)

    summary = summarize(run)
    print("\nRow counts:", summary["row_counts"])
    print(
        "Cache hits:",
        summary["cache_hits"],
        "| source hashes unchanged:",
        summary["source_hashes_unchanged"],
    )
    print("PDF text fallback:", summary["pdf_text_fallback"])
    print(
        "Observations:", report.counts(), dict(Counter(o.check.value for o in report.observations))
    )
    for observation in report.observations:
        if observation.severity.value in ("ERROR", "WARNING"):
            print(
                f"  [{observation.severity.value}] {observation.check.value}: {observation.message}"
            )
    for name, path in written.items():
        print(f"wrote {name}: {path}")
    return 1 if report.has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
