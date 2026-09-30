"""Parse the in-scope project PDFs with Docling and report parsing quality.

Usage:
    python scripts/parse_documents.py                        # all projects (cached docs reused)
    python scripts/parse_documents.py --project P130544
    python scripts/parse_documents.py --document <document_id or filename>
    python scripts/parse_documents.py --force                # reparse even if cached
    python scripts/parse_documents.py --failed-only          # retry failed/missing only
    python scripts/parse_documents.py --report-only          # validate existing outputs, no parsing
    python scripts/parse_documents.py --json .local_output/parsing_report.json

Outputs go to <local_output_root>/parsed/<PROJECT_ID>/<document_id>.json. Source
PDFs are only read. Exit code 1 if any ERROR observation is found.
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
from worldbank_copilot.parsing.pipeline import run_parsing
from worldbank_copilot.parsing.report import (
    aggregates,
    build_parsing_report,
    format_parsing_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project", action="append", help="project ID (repeatable)")
    parser.add_argument("--document", help="single document_id or filename")
    parser.add_argument("--force", action="store_true", help="reparse even if cached")
    parser.add_argument("--failed-only", action="store_true", help="only failed/missing docs")
    parser.add_argument("--report-only", action="store_true", help="no parsing; validate outputs")
    parser.add_argument("--json", type=Path, help="write the full report as JSON")
    parser.add_argument("--env", help="force environment (local | databricks)")
    args = parser.parse_args(argv)

    settings = load_settings(args.env)
    configure_logging(settings.log_level)
    registry = load_project_registry(settings.config_dir)

    document_parser = None
    if not args.report_only:
        from worldbank_copilot.parsing.docling_parser import DoclingDocumentParser

        document_parser = DoclingDocumentParser()
    run = run_parsing(
        settings,
        registry,
        document_parser,
        projects=args.project,
        document=args.document,
        force=args.force,
        failed_only=args.failed_only,
        progress=lambda message: print(message, flush=True),
    )
    report = build_parsing_report(run, registry)
    print()
    print(format_parsing_report(run, report, registry))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "aggregates": aggregates(run),
            "observations": [o.model_dump(mode="json") for o in report.observations],
            "counts": report.counts(),
            "total_seconds": run.total_seconds,
        }
        args.json.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        print(f"\nJSON report written to {args.json}")
    return 1 if report.has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
