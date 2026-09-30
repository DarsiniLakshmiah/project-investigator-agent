"""Portfolio parsing run: inventory -> (cached | parsed | failed) ParsedDocuments on disk.

Cache identity = (document_id, source SHA-256, parser name/version/config hash).
``document_id`` already contains the SHA prefix, so a changed file gets a new id.
Metadata extraction and reconciliation are cheap and re-run on every load, so a
manifest change is reflected without reparsing.

One failing PDF never stops the run: it is recorded as a FAILED document and can
be retried alone (``failed_only`` or ``document``).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from worldbank_copilot.common.config import Settings
from worldbank_copilot.common.logging import get_logger
from worldbank_copilot.common.project_registry import ProjectRegistry
from worldbank_copilot.ingestion.documents import build_document_inventory, load_document_manifest
from worldbank_copilot.parsing.document_builder import (
    build_document,
    content_of,
    failed_document,
)
from worldbank_copilot.parsing.models import ParsedDocument, ParseStatus
from worldbank_copilot.parsing.parser import DocumentParser, ParserError

logger = get_logger(__name__)

INDEX_FILE = "_parse_index.json"
SCHEMA_FILE = "_parsed_document.schema.json"


@dataclass
class ParseOutcome:
    document_id: str
    project_id: str
    filename: str
    action: str  # parsed | cached | failed | loaded | missing
    seconds: float | None = None
    path: str | None = None
    error: str | None = None


@dataclass
class ParseRun:
    started_at: str
    total_seconds: float
    outcomes: list[ParseOutcome]
    documents: list[ParsedDocument]
    expected: list[tuple[str, str]]  # (project_id, filename) from the curated manifest
    selected_projects: list[str]
    full_projects: bool  # True when whole projects were selected (ISR completeness valid)
    hashes_before: dict[str, str] = field(default_factory=dict)  # relative_path -> sha256
    hashes_after: dict[str, str] = field(default_factory=dict)
    inventory_records: list[dict[str, Any]] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pdf_page_texts(path: Path) -> list[str] | None:
    """The PDF's own text layer per page, read independently of the parser.

    Used for the page-count cross-check and text-coverage measurement. pypdfium2
    ships with Docling; returns None if unavailable or unreadable (checks skipped).
    """
    try:
        import pypdfium2

        pdf = pypdfium2.PdfDocument(str(path))
    except Exception:  # noqa: BLE001 - an unreadable text layer only disables cross-checks
        return None
    texts = []
    try:
        for page in pdf:
            text_page = page.get_textpage()
            texts.append(text_page.get_text_range())
            text_page.close()
            page.close()
    except Exception:  # noqa: BLE001
        return None
    finally:
        pdf.close()
    return texts


def output_path(root: Path, project_id: str, document_id: str) -> Path:
    return root / project_id / f"{document_id}.json"


def load_parsed(path: Path) -> ParsedDocument | None:
    if not path.is_file():
        return None
    return ParsedDocument.model_validate_json(path.read_text(encoding="utf-8"))


def save_parsed(document: ParsedDocument, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document.model_dump_json(indent=1), encoding="utf-8")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def run_parsing(
    settings: Settings,
    registry: ProjectRegistry,
    parser: DocumentParser | None,
    *,
    projects: list[str] | None = None,
    document: str | None = None,
    force: bool = False,
    failed_only: bool = False,
    output_root: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> ParseRun:
    """Parse (or, with ``parser=None``, only load) the selected documents."""
    started = time.perf_counter()
    started_at = _now()
    root = output_root or settings.local_output_root / "parsed"
    data_root = Path(settings.data_root)
    manifest = load_document_manifest(
        settings.config_dir / settings.require("sources.document_manifest")
    )
    inventory = build_document_inventory(
        settings.documents_root, data_root, registry, manifest, ingested_at=started_at,
        run_id="parse-" + started_at,
    )  # fmt: skip
    selected_projects = [registry.validate_project_id(p) for p in projects] if projects else (
        registry.project_ids)  # fmt: skip
    records = [r for r in inventory.table.records if r["project_id"] in selected_projects]
    if document:
        records = [r for r in records if document in (r["document_id"], r["filename"])]
        if not records:
            raise ValueError(f"No inventoried document matches {document!r}")
    expected = [(pid, e.filename) for pid, entries in manifest.documents.items()
                if pid in selected_projects for e in entries]  # fmt: skip
    if document:
        expected = [(r["project_id"], r["filename"]) for r in records]

    info = parser.info if parser else None
    outcomes: list[ParseOutcome] = []
    documents: list[ParsedDocument] = []
    hashes_before = {r["relative_path"]: r["sha256"] for r in records}

    for index, record in enumerate(records, start=1):
        path = output_path(root, record["project_id"], record["document_id"])
        existing = load_parsed(path)
        source = data_root / record["relative_path"]
        label = f"[{index}/{len(records)}] {record['project_id']} {record['filename']}"

        if parser is None:
            if existing is None:
                outcomes.append(ParseOutcome(record["document_id"], record["project_id"],
                                             record["filename"], "missing"))  # fmt: skip
                continue
            documents.append(_refresh(existing, record, source))
            outcomes.append(ParseOutcome(record["document_id"], record["project_id"],
                                         record["filename"], "loaded", path=str(path)))  # fmt: skip
            continue

        cache_valid = (
            existing is not None
            and existing.parse_status is ParseStatus.SUCCESS
            and existing.source_hash == record["sha256"]
            and existing.parser.config_hash == info.config_hash
        )
        if failed_only:
            needs_parse = existing is None or existing.parse_status is ParseStatus.FAILED
        else:
            needs_parse = force or not cache_valid
        if not needs_parse and existing is not None:
            refreshed = _refresh(existing, record, source)
            save_parsed(refreshed, path)
            documents.append(refreshed)
            outcomes.append(ParseOutcome(record["document_id"], record["project_id"],
                                         record["filename"], "cached", path=str(path)))  # fmt: skip
            if progress:
                progress(f"{label}: cached")
            continue

        t0 = time.perf_counter()
        parsed_at = _now()
        try:
            content = parser.parse(source)
            seconds = time.perf_counter() - t0
            doc = build_document(record, content, info, parsed_at, seconds,
                                 pdf_page_texts=pdf_page_texts(source))  # fmt: skip
            action, error = "parsed", None
        except (ParserError, Exception) as exc:  # noqa: BLE001 - isolate per-document failures
            error = f"{type(exc).__name__}: {exc}"
            doc = failed_document(record, info, parsed_at, time.perf_counter() - t0, error)
            action = "failed"
            logger.error("Parsing failed for %s: %s", record["filename"], error)
        save_parsed(doc, path)
        documents.append(doc)
        outcomes.append(ParseOutcome(record["document_id"], record["project_id"],
                                     record["filename"], action, doc.parse_seconds, str(path),
                                     error))  # fmt: skip
        if progress:
            progress(f"{label}: {action} in {doc.parse_seconds}s")

    hashes_after = {
        r["relative_path"]: sha256_file(data_root / r["relative_path"]) for r in records
    }
    total = time.perf_counter() - started
    run = ParseRun(
        started_at=started_at,
        total_seconds=round(total, 1),
        outcomes=outcomes,
        documents=documents,
        expected=expected,
        selected_projects=selected_projects,
        full_projects=document is None,
        hashes_before=hashes_before,
        hashes_after=hashes_after,
        inventory_records=records,
    )
    if parser is not None:
        _write_index(root, run)
    return run


def _refresh(existing: ParsedDocument, record: dict[str, Any], source: Path) -> ParsedDocument:
    """Re-derive metadata and checks from cached content (cheap; reflects manifest edits)."""
    if existing.parse_status is not ParseStatus.SUCCESS:
        return existing
    return build_document(
        record,
        content_of(existing),
        existing.parser,
        existing.parsed_at,
        existing.parse_seconds,
        pdf_page_texts=pdf_page_texts(source),
    )


def _write_index(root: Path, run: ParseRun) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "started_at": run.started_at,
        "total_seconds": run.total_seconds,
        "outcomes": [o.__dict__ for o in run.outcomes],
    }
    (root / INDEX_FILE).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    (root / SCHEMA_FILE).write_text(
        json.dumps(ParsedDocument.model_json_schema(), indent=1), encoding="utf-8"
    )
