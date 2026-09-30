"""Parsing-quality report: per-document metrics, portfolio validation, aggregates."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

from worldbank_copilot.common.project_registry import ProjectRegistry
from worldbank_copilot.common.quality import (
    CheckCode,
    DataQualityReport,
    Observation,
    Severity,
    sort_observations,
)
from worldbank_copilot.ingestion.documents import check_isr_completeness
from worldbank_copilot.parsing.models import ParsedDocument, ParseStatus, ValidationStatus
from worldbank_copilot.parsing.pipeline import ParseRun


def document_metrics(doc: ParsedDocument) -> dict[str, Any]:
    ok = doc.parse_status is ParseStatus.SUCCESS
    statuses = {v.field: v.status.value for v in doc.metadata_validation}
    return {
        "project_id": doc.project_id,
        "document_id": doc.document_id,
        "filename": doc.filename,
        "document_type": doc.document_type,
        "display_name": doc.display_name(),
        "parse_status": doc.parse_status.value,
        "parse_seconds": doc.parse_seconds,
        "page_count": doc.page_count,
        "text_blocks": sum(1 for b in doc.blocks if not b.is_furniture) if ok else 0,
        "furniture_blocks": sum(1 for b in doc.blocks if b.is_furniture) if ok else 0,
        "tables": len(doc.tables),
        "sections": len(doc.sections),
        "empty_pages": sum(1 for p in doc.pages if p.is_empty),
        "clean_chars": sum(p.char_count_clean for p in doc.pages),
        "isr_sequence": doc.isr_sequence,
        "document_date": doc.document_date.isoformat() if doc.document_date else None,
        "report_date": doc.report_date.isoformat() if doc.report_date else None,
        "archive_date": doc.archive_date.isoformat() if doc.archive_date else None,
        "report_number": doc.report_number,
        "loan_number": doc.loan_number,
        "metadata_validation": statuses,
        "issues": [f"{i.severity}:{i.code}" for i in doc.quality_issues],
        "parser_warnings": len(doc.parser_warnings),
    }


def build_parsing_report(run: ParseRun, registry: ProjectRegistry) -> DataQualityReport:
    obs: list[Observation] = []
    by_name = {(d.project_id, d.filename): d for d in run.documents}
    for pid, filename in run.expected:
        if (pid, filename) not in by_name:
            obs.append(
                Observation(
                    check=CheckCode.PARSED_DOCUMENT_MISSING,
                    severity=Severity.ERROR,
                    project_id=pid,
                    source=filename,
                    message=f"expected document {filename} has no parsed output",
                )
            )
    expected = set(run.expected)
    for doc in run.documents:
        if (doc.project_id, doc.filename) not in expected:
            obs.append(
                Observation(
                    check=CheckCode.UNEXPECTED_DOCUMENT,
                    severity=Severity.WARNING,
                    project_id=doc.project_id,
                    source=doc.filename,
                    message="parsed document is not in the curated manifest",
                )
            )

    for path, before in run.hashes_before.items():
        after = run.hashes_after.get(path)
        if after != before:
            obs.append(
                Observation(
                    check=CheckCode.SOURCE_HASH_CHANGED,
                    severity=Severity.ERROR,
                    source=path,
                    message="source file changed during the run",
                    details={"before": before, "after": after},
                )
            )
    for doc in run.documents:
        current = run.hashes_after.get(doc.source_ref.relative_path)
        if current is not None and current != doc.source_hash:
            obs.append(
                Observation(
                    check=CheckCode.SOURCE_HASH_CHANGED,
                    severity=Severity.ERROR,
                    project_id=doc.project_id,
                    source=doc.filename,
                    message="parsed output was produced from different file content",
                )
            )
        for issue in doc.quality_issues:
            obs.append(
                Observation(
                    check=CheckCode(issue.code),
                    severity=Severity(issue.severity),
                    project_id=doc.project_id,
                    source=doc.filename,
                    message=f"{doc.display_name()}: {issue.message}",
                    details={"page_number": issue.page_number, "document_id": doc.document_id},
                )
            )

    if run.full_projects:
        for pid in run.selected_projects:
            project = registry.get(pid)
            isr_docs = [
                d for d in run.documents if d.project_id == pid and d.document_type == "ISR"
            ]
            result = check_isr_completeness(
                pid, [d.isr_sequence for d in isr_docs], project.expected_isr_count
            )
            failed = [d.filename for d in isr_docs if d.parse_status is not ParseStatus.SUCCESS]
            if not result.is_complete or failed:
                obs.append(Observation(
                    check=CheckCode.ISR_INVENTORY_INCOMPLETE, severity=Severity.ERROR,
                    project_id=pid, source="parsed ISRs",
                    message=f"{pid}: ISR sequences {result.found_count}/{result.expected_count}; "
                            f"missing {result.missing}; duplicates {result.duplicates}; "
                            f"failed parses {failed}",
                    details=result.model_dump()))  # fmt: skip
    return DataQualityReport(
        generated_at=datetime.now(UTC).isoformat(),
        ingestion_run_id=run.started_at,
        observations=sort_observations(obs),
        layer="parsed",
    )


def aggregates(run: ParseRun) -> dict[str, Any]:
    metrics = [document_metrics(d) for d in run.documents]
    by_project: dict[str, Counter] = defaultdict(Counter)
    by_type: dict[str, Counter] = defaultdict(Counter)
    for m in metrics:
        for key, bucket in ((m["project_id"], by_project), (m["document_type"] or "UNRESOLVED",
                                                             by_type)):  # fmt: skip
            bucket[key]["documents"] += 1
            bucket[key][m["parse_status"].lower()] += 1
            bucket[key]["pages"] += m["page_count"]
            bucket[key]["tables"] += m["tables"]
            bucket[key]["sections"] += m["sections"]
    validation = defaultdict(Counter)
    for doc in run.documents:
        for v in doc.metadata_validation:
            validation[v.field][v.status.value] += 1
    isr = {}
    for pid in run.selected_projects:
        docs = [d for d in run.documents if d.project_id == pid and d.document_type == "ISR"]
        isr[pid] = {
            "parsed": sum(1 for d in docs if d.parse_status is ParseStatus.SUCCESS),
            "sequences": sorted(d.isr_sequence for d in docs if d.isr_sequence is not None),
        }
    return {
        "metrics": metrics,
        "by_project": {k: dict(v) for k, v in sorted(by_project.items())},
        "by_type": {k: dict(v) for k, v in sorted(by_type.items())},
        "by_status": dict(Counter(m["parse_status"] for m in metrics)),
        "actions": dict(Counter(o.action for o in run.outcomes)),
        "metadata_validation": {k: dict(v) for k, v in validation.items()},
        "isr": isr,
    }


def format_parsing_report(
    run: ParseRun, report: DataQualityReport, registry: ProjectRegistry
) -> str:
    agg = aggregates(run)
    metrics = agg["metrics"]
    parsed_ok = sum(1 for m in metrics if m["parse_status"] == "SUCCESS")
    failed = sum(1 for m in metrics if m["parse_status"] == "FAILED")
    parse_total = sum(m["parse_seconds"] or 0 for m in metrics)
    lines = [
        f"Parsing run started {run.started_at}; wall time {run.total_seconds}s; actions "
        f"{agg['actions']}",
        f"Total Docling parse time (sum of per-document parse times, incl. cached): "
        f"{parse_total:.0f}s ({parse_total / 60:.1f} min)",
        "",
        f"Documents expected: {len(run.expected)}",
        f"Documents parsed:   {parsed_ok}",
        f"Documents failed:   {failed}",
        "",
        "ISRs:",
    ]
    for pid, info in agg["isr"].items():
        expected = registry.get(pid).expected_isr_count
        lines.append(f"  {pid}: {info['parsed']}/{expected} parsed; sequences {info['sequences']}")
    total_tables = sum(m["tables"] for m in metrics)
    total_sections = sum(m["sections"] for m in metrics)
    total_pages = sum(m["page_count"] for m in metrics)
    lines += ["", f"Pages: {total_pages}   Tables extracted: {total_tables}   "
              f"Sections detected: {total_sections}", "", "By project:"]  # fmt: skip
    for key, row in agg["by_project"].items():
        lines.append(f"  {key}: {row}")
    lines.append("By document type:")
    for key, row in agg["by_type"].items():
        lines.append(f"  {key}: {row}")
    lines += ["", "Metadata validation (field: status counts):"]
    for key, row in agg["metadata_validation"].items():
        lines.append(f"  {key}: {row}")
    totals = Counter()
    for row in agg["metadata_validation"].values():
        totals.update(row)
    lines.append(
        f"  overall: confirmed {totals[ValidationStatus.CONFIRMED.value]}, corrected "
        f"{totals[ValidationStatus.CORRECTED_FROM_DOCUMENT.value]}, conflicts "
        f"{totals[ValidationStatus.CONFLICT.value]}, manifest-only "
        f"{totals[ValidationStatus.MANIFEST_ONLY.value]}, document-only "
        f"{totals[ValidationStatus.DOCUMENT_ONLY.value]}, unknown "
        f"{totals[ValidationStatus.UNKNOWN.value]}"
    )
    hashes_ok = all(run.hashes_before[p] == run.hashes_after.get(p) for p in run.hashes_before)
    lines.append(f"\nSource hashes unchanged: {hashes_ok} ({len(run.hashes_before)} files)")
    counts = report.counts()
    lines += ["", f"Parsing observations ({counts['ERROR']} errors, {counts['WARNING']} warnings, "
              f"{counts['INFO']} info):"]  # fmt: skip
    for o in report.observations:
        lines.append(f"  [{o.severity.value:<7}] {o.check.value:<32} {o.project_id or '-':<8} "
                     f"{o.message}")  # fmt: skip
    return "\n".join(lines)
