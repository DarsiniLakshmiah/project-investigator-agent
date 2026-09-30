"""Phase 5 pipeline: parsed documents -> document-derived Silver datasets.

Reads the Phase 4 parsed cache only (Docling is never re-run). The PDF text layer
is opened only when an extractor triggers a logged fallback. Per-document
extraction results are cached under ``<output>/_cache`` and keyed by the source
hash, parser config hash and extractor version/config.

Outputs (local; never committed):
    isr_snapshots.jsonl, project_results.jsonl, appraisal_risks.jsonl,
    project_events.jsonl, project_enrichment.jsonl, silver_projects_enriched.jsonl,
    pdf_text_fallback_log.jsonl, document_extraction_quality.json
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from worldbank_copilot.common.config import Settings
from worldbank_copilot.common.project_registry import ProjectRegistry
from worldbank_copilot.common.quality import (
    CheckCode,
    DataQualityReport,
    Observation,
    Severity,
    sort_observations,
)
from worldbank_copilot.extraction import isr as isr_module
from worldbank_copilot.extraction import results as results_module
from worldbank_copilot.extraction.closing_dates import (
    apply_project_enrichment,
    reconcile_original_closing,
)
from worldbank_copilot.extraction.crosscheck import crosscheck_project
from worldbank_copilot.extraction.events import (
    FORMAL_PAPER_TYPES,
    extract_formal_events,
    isr_events,
    restructuring_date_candidates,
)
from worldbank_copilot.extraction.identity import load_aliases, resolve_identities
from worldbank_copilot.extraction.models import (
    AppraisalRisk,
    IsrSnapshot,
    ProjectEnrichment,
    ProjectEvent,
    ResultObservation,
)
from worldbank_copilot.extraction.provenance import ExtractionIssue, issue
from worldbank_copilot.extraction.risks import APPRAISAL_TYPES, extract_appraisal_risks
from worldbank_copilot.extraction.text_source import FallbackLog, LoggedTextSource, PdfTextLayer
from worldbank_copilot.parsing.models import ParsedDocument, ParseStatus
from worldbank_copilot.parsing.pipeline import run_parsing

logger = logging.getLogger(__name__)

EXTRACTOR_VERSION = "phase5.1"
EXTRACTOR_CONFIG = {
    "material_date_difference_days": isr_module.MATERIAL_DATE_DIFFERENCE_DAYS,
    "results_low_coverage": results_module.LOW_COVERAGE,
}
OUTPUT_DIR = "silver_documents"
ALIASES_FILE = "indicator_aliases.yaml"


class DocumentExtraction(BaseModel):
    """Cache unit: everything extracted from one parsed document."""

    model_config = ConfigDict(extra="forbid")

    cache_key: str
    document_id: str
    project_id: str
    document_type: str | None
    snapshot: IsrSnapshot | None = None
    results: list[ResultObservation] = Field(default_factory=list)
    risks: list[AppraisalRisk] = Field(default_factory=list)
    events: list[ProjectEvent] = Field(default_factory=list)
    issues: list[ExtractionIssue] = Field(default_factory=list)
    fallback_pages: list[int] = Field(default_factory=list)
    fallback_invocations: list[dict] = Field(default_factory=list)


@dataclass
class ExtractionRun:
    generated_at: str
    documents: dict[str, ParsedDocument]
    extractions: list[DocumentExtraction]
    snapshots: list[IsrSnapshot] = field(default_factory=list)
    results: list[ResultObservation] = field(default_factory=list)
    risks: list[AppraisalRisk] = field(default_factory=list)
    events: list[ProjectEvent] = field(default_factory=list)
    enrichment: list[ProjectEnrichment] = field(default_factory=list)
    enriched_projects: list[Any] = field(default_factory=list)
    enrichment_lineage: list[dict] = field(default_factory=list)
    issues: list[ExtractionIssue] = field(default_factory=list)
    cache_hits: int = 0
    hashes_unchanged: bool | None = None
    hash_mismatches: list[str] = field(default_factory=list)
    parse_run: Any = None  # Phase 4 ParseRun the documents were loaded from


def extractor_code_hash() -> str:
    """Hash of the extraction package source: any code change invalidates the cache."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def cache_key(doc: ParsedDocument) -> str:
    payload = json.dumps(
        {
            "source_hash": doc.source_hash,
            "parser": doc.parser.config_hash,
            "extractor": EXTRACTOR_VERSION,
            "code": extractor_code_hash(),
            "config": EXTRACTOR_CONFIG,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def extract_document(doc: ParsedDocument, text_path: Path | None) -> DocumentExtraction:
    """Run the extractors that apply to this document type."""
    log = FallbackLog()
    text = LoggedTextSource(
        PdfTextLayer(text_path) if text_path else None, log, doc.document_id, doc.filename
    )
    out = DocumentExtraction(
        cache_key=cache_key(doc),
        document_id=doc.document_id,
        project_id=doc.project_id,
        document_type=doc.document_type,
    )
    if doc.document_type == "ISR":
        snapshot = isr_module.extract_isr_snapshot(doc, text)
        observations, issues, pages = results_module.extract_results(
            doc, text, snapshot.canonical_report_date
        )
        out.snapshot, out.results, out.fallback_pages = snapshot, observations, pages
        out.issues += issues
    if doc.document_type in APPRAISAL_TYPES:
        risks, issues = extract_appraisal_risks(doc, text)
        out.risks, out.issues = risks, out.issues + issues
    if doc.document_type in FORMAL_PAPER_TYPES:
        events, issues = extract_formal_events(doc)
        out.events, out.issues = events, out.issues + issues
    out.fallback_invocations = [asdict(i) for i in log.invocations]
    return out


def _load_cached(path: Path, key: str) -> DocumentExtraction | None:
    if not path.exists():
        return None
    try:
        cached = DocumentExtraction.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return cached if cached.cache_key == key else None


def run_extraction(
    settings: Settings,
    registry: ProjectRegistry,
    silver_loans: list[Any],
    silver_projects: list[Any],
    *,
    projects: list[str] | None = None,
    output_root: Path | None = None,
    use_cache: bool = True,
    progress: Callable[[str], None] | None = None,
) -> ExtractionRun:
    root = output_root or settings.local_output_root / OUTPUT_DIR
    cache_dir = root / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    parse_run = run_parsing(settings, registry, None, projects=projects)
    documents = {d.document_id: d for d in parse_run.documents}
    run = ExtractionRun(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        documents=documents,
        extractions=[],
        parse_run=parse_run,
    )
    for outcome in parse_run.outcomes:
        if outcome.action == "missing":
            run.issues.append(
                issue(
                    CheckCode.ISR_SNAPSHOT_MISSING,
                    "ERROR",
                    f"{outcome.project_id} {outcome.filename}: no parsed output; "
                    "run Phase 4 parsing first",
                    project_id=outcome.project_id,
                )
            )

    data_root = Path(settings.data_root)
    for index, doc in enumerate(sorted(documents.values(), key=lambda d: d.document_id), 1):
        if doc.parse_status is not ParseStatus.SUCCESS:
            continue
        key = cache_key(doc)
        path = cache_dir / f"{doc.document_id}.json"
        cached = _load_cached(path, key) if use_cache else None
        if cached is not None:
            run.cache_hits += 1
            extraction = cached
        else:
            extraction = extract_document(doc, data_root / doc.source_ref.relative_path)
            path.write_text(extraction.model_dump_json(), encoding="utf-8")
        run.extractions.append(extraction)
        if progress:
            progress(
                f"[{index}/{len(documents)}] {doc.project_id} {doc.display_name()}: "
                f"{'cached' if cached else 'extracted'}"
            )

    aliases = load_aliases(settings.config_dir / ALIASES_FILE)
    for project_id in parse_run.selected_projects:
        _assemble_project(run, project_id, registry, silver_loans, aliases)
    run.enriched_projects, run.enrichment_lineage = apply_project_enrichment(
        [p for p in silver_projects if p.project_id in parse_run.selected_projects],
        run.enrichment,
    )
    run.issues += _global_limitations()
    run.hash_mismatches = sorted(
        path
        for path, digest in parse_run.hashes_before.items()
        if parse_run.hashes_after.get(path) != digest
    )
    run.hashes_unchanged = not run.hash_mismatches
    return run


def _assemble_project(
    run: ExtractionRun,
    project_id: str,
    registry: ProjectRegistry,
    silver_loans: list[Any],
    aliases: dict,
) -> None:
    extractions = [e for e in run.extractions if e.project_id == project_id]
    documents = {k: d for k, d in run.documents.items() if d.project_id == project_id}
    snapshots = sorted(
        (e.snapshot for e in extractions if e.snapshot),
        key=lambda s: (s.isr_sequence is None, s.isr_sequence),
    )
    results = [r for e in extractions for r in e.results]
    risks = [r for e in extractions for r in e.risks]
    paper_events = [ev for e in extractions for ev in e.events]
    for extraction in extractions:
        run.issues += extraction.issues
        if extraction.snapshot:
            run.issues += extraction.snapshot.quality_issues

    run.issues += resolve_identities(results, aliases)
    dated_events, event_issues = isr_events(snapshots, documents)
    run.issues += event_issues
    run.issues += restructuring_date_candidates(
        paper_events, [e for e in dated_events if e.event_type == "RESTRUCTURING"], documents
    )
    events = sorted(
        dated_events + paper_events,
        key=lambda e: (e.event_date is None, e.event_date, e.event_type),
    )
    papers = [d for d in documents.values() if d.document_type in FORMAL_PAPER_TYPES]
    enrichment, closing_issues = reconcile_original_closing(
        project_id, snapshots, papers, silver_loans
    )
    run.issues += closing_issues
    run.issues += crosscheck_project(project_id, snapshots, events, silver_loans)
    run.issues += _project_checks(project_id, registry, snapshots, risks, documents)

    run.snapshots += snapshots
    run.results += results
    run.risks += risks
    run.events += events
    run.enrichment += enrichment


def _project_checks(
    project_id: str,
    registry: ProjectRegistry,
    snapshots: list[IsrSnapshot],
    risks: list[AppraisalRisk],
    documents: dict[str, ParsedDocument],
) -> list[ExtractionIssue]:
    out: list[ExtractionIssue] = []
    expected = registry.get(project_id).expected_isr_count
    sequences = sorted(s.isr_sequence for s in snapshots if s.isr_sequence is not None)
    if expected is not None:
        missing = sorted(set(range(1, expected + 1)) - set(sequences))
        if missing or len(snapshots) != expected:
            out.append(
                issue(
                    CheckCode.ISR_SNAPSHOT_MISSING,
                    "ERROR",
                    f"{project_id}: {len(snapshots)} ISR snapshots, expected {expected}; "
                    f"missing sequences {missing}",
                    project_id=project_id,
                )
            )

    # Canonical (header-first) dates should follow ISR sequence order.
    dated = [s for s in snapshots if s.canonical_report_date and s.isr_sequence is not None]
    for prev, cur in zip(dated, dated[1:], strict=False):
        if cur.canonical_report_date < prev.canonical_report_date:
            out.append(
                issue(
                    CheckCode.SEQUENCE_DATE_ANOMALY,
                    "WARNING",
                    f"{project_id}: canonical date of ISR {prev.isr_sequence} "
                    f"({prev.canonical_report_date}, {prev.canonical_date_basis}) is later "
                    f"than ISR {cur.isr_sequence} ({cur.canonical_report_date}); dates "
                    "are preserved as printed, sequence order is authoritative for ordering",
                    prev.source_refs[0] if prev.source_refs else None,
                    project_id=project_id,
                    header_date=str(prev.header_date),
                    archive_date=str(prev.archive_date),
                    isr_sequence=prev.isr_sequence,
                    next_isr_sequence=cur.isr_sequence,
                )
            )

    # SORT tables with fewer categories than other ISRs of the same project.
    sizes = {s.isr_sequence: len(s.sort_ratings) for s in snapshots if s.sort_ratings}
    if sizes:
        typical = Counter(sizes.values()).most_common(1)[0][0]
        for sequence, size in sorted(sizes.items()):
            if size < typical:
                out.append(
                    issue(
                        CheckCode.EXTRACTION_LIMITATION,
                        "INFO",
                        f"{project_id} ISR {sequence}: SORT table has {size} categories "
                        f"(typical {typical} in this project); printed table extracted as is",
                        project_id=project_id,
                    )
                )

    for doc in documents.values():
        image_pages = [p.page_number for p in doc.pages if p.is_empty and p.picture_count]
        if image_pages:
            out.append(
                issue(
                    CheckCode.OCR_NOT_REQUIRED_FOR_CURRENT_SCOPE,
                    "INFO",
                    f"{doc.display_name()}: pages {image_pages} have pictures but no text layer; "
                    "not OCRed (OCR disabled by decision)",
                    project_id=project_id,
                    document=doc.filename,
                    pages=image_pages,
                )
            )
        if doc.document_type in APPRAISAL_TYPES and not any(
            r.source_document == doc.filename for r in risks
        ):
            out.append(
                issue(
                    CheckCode.EXTRACTION_LIMITATION,
                    "INFO",
                    f"{doc.display_name()}: no risk table or SORT found; narrative risk "
                    "discussion is not converted into risk records",
                    project_id=project_id,
                    document=doc.filename,
                )
            )
    return out


def _global_limitations() -> list[ExtractionIssue]:
    return [
        issue(
            CheckCode.EXTRACTION_LIMITATION,
            "INFO",
            "Appraisal risks are extracted only from SORT tables, explicit risk tables and "
            "Technical Assessment 'Risk - n' blocks; risks discussed only in narrative "
            "paragraphs are not extracted (no LLM extraction in Phase 5)",
        )
    ]


# ---------------------------------------------------------------------------
# Quality report and outputs
# ---------------------------------------------------------------------------


def to_observation(item: ExtractionIssue) -> Observation:
    details = dict(item.details)
    project_id = details.pop("project_id", None) or (
        item.evidence.project_id if item.evidence else None
    )
    if item.evidence:
        details["evidence"] = item.evidence.model_dump(mode="json", exclude_none=True)
    return Observation(
        check=CheckCode(item.code),
        severity=Severity(item.severity),
        project_id=project_id,
        source=item.evidence.label if item.evidence else "silver_documents",
        message=item.message,
        details=json.loads(json.dumps(details, default=str)),
    )


def build_quality_report(run: ExtractionRun) -> DataQualityReport:
    seen, observations = set(), []
    for item in run.issues:
        observation = to_observation(item)
        # Details are part of the identity: distinct findings can share message and source
        # (e.g. two POSSIBLE_INDICATOR_MATCH pairs anchored on the same page).
        key = (observation.check, observation.message, observation.source,
               json.dumps(observation.details, sort_keys=True, default=str))  # fmt: skip
        if key not in seen:
            seen.add(key)
            observations.append(observation)
    if run.hashes_unchanged is False:
        observations.append(
            Observation(
                check=CheckCode.EXTRACTION_CONFLICT,
                severity=Severity.ERROR,
                source="data/documents",
                message=f"source hashes changed: {run.hash_mismatches}",
            )
        )
    return DataQualityReport(
        generated_at=run.generated_at,
        ingestion_run_id="extract-" + run.generated_at,
        observations=sort_observations(observations),
        layer="silver_documents",
    )


def summarize(run: ExtractionRun) -> dict[str, Any]:
    def rating(r):
        return r.normalized_rating if r else None

    fallback = [inv for e in run.extractions for inv in e.fallback_invocations]
    by_project: dict[str, Any] = defaultdict(dict)
    for s in run.snapshots:
        by_project[s.project_id].setdefault("ratings_by_isr", []).append(
            {
                "isr_sequence": s.isr_sequence,
                "canonical_report_date": str(s.canonical_report_date),
                "pdo": rating(s.pdo_rating),
                "ip": rating(s.implementation_progress_rating),
                "overall_risk": rating(s.overall_risk_rating),
                "sort_categories": len(s.sort_ratings),
                "loans": len(s.loan_disbursements),
            }
        )
    for project_id in {s.project_id for s in run.snapshots} | {r.project_id for r in run.results}:
        obs = [r for r in run.results if r.project_id == project_id]
        entry = by_project[project_id]
        entry["isr_snapshots"] = sum(s.project_id == project_id for s in run.snapshots)
        entry["result_observations"] = len(obs)
        entry["indicators"] = len({r.indicator_key for r in obs})
        entry["observations_by_method"] = dict(Counter(r.extraction_method.value for r in obs))
        entry["observations_by_status"] = dict(Counter(r.status.value for r in obs))
        risks = [r for r in run.risks if r.project_id == project_id]
        entry["risks_by_framing"] = dict(Counter(r.framing for r in risks))
        entry["risks_by_document_type"] = dict(Counter(r.source_document_type for r in risks))
        entry["events_by_type"] = dict(
            Counter(e.event_type for e in run.events if e.project_id == project_id)
        )
    return {
        "generated_at": run.generated_at,
        "extractor_version": EXTRACTOR_VERSION,
        "extractor_config": EXTRACTOR_CONFIG,
        "extractor_code_hash": extractor_code_hash(),
        "documents": len(run.extractions),
        "cache_hits": run.cache_hits,
        "source_hashes_unchanged": run.hashes_unchanged,
        "row_counts": {
            "isr_snapshots": len(run.snapshots),
            "project_results": len(run.results),
            "appraisal_risks": len(run.risks),
            "project_events": len(run.events),
            "project_enrichment": len(run.enrichment),
        },
        "projects": dict(sorted(by_project.items())),
        "pdf_text_fallback": {
            "invocations": len(fallback),
            "documents": len({f["document_id"] for f in fallback}),
            "by_element": dict(Counter(f["element"].split(" p")[0] for f in fallback)),
        },
        "enrichment_lineage": run.enrichment_lineage,
    }


def _write_jsonl(path: Path, rows: list[BaseModel]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row.model_dump_json() + "\n")


def write_outputs(run: ExtractionRun, report: DataQualityReport, root: Path) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "isr_snapshots": (root / "isr_snapshots.jsonl", run.snapshots),
        "project_results": (root / "project_results.jsonl", run.results),
        "appraisal_risks": (root / "appraisal_risks.jsonl", run.risks),
        "project_events": (root / "project_events.jsonl", run.events),
        "project_enrichment": (root / "project_enrichment.jsonl", run.enrichment),
        "silver_projects_enriched": (
            root / "silver_projects_enriched.jsonl",
            run.enriched_projects,
        ),
    }
    for path, rows in files.values():
        _write_jsonl(path, rows)
    fallback_path = root / "pdf_text_fallback_log.jsonl"
    with fallback_path.open("w", encoding="utf-8") as fh:
        for extraction in run.extractions:
            for invocation in extraction.fallback_invocations:
                fh.write(json.dumps(invocation) + "\n")
    quality_path = root / "document_extraction_quality.json"
    quality_path.write_text(
        json.dumps(
            {"summary": summarize(run), "report": report.model_dump(mode="json")},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    written = {name: str(path) for name, (path, _) in files.items()}
    written["pdf_text_fallback_log"] = str(fallback_path)
    written["document_extraction_quality"] = str(quality_path)
    return written
