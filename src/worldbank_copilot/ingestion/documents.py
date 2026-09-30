"""Bronze document inventory (no PDF content is parsed in Phase 2).

Each file under ``<documents_root>/<project documents_dir>/`` becomes one
inventory record. Classification is deterministic and uses two sources:

1. **Filename patterns**, e.g. ``...-Sequence-No-05.pdf`` → ISR sequence 5,
   ``ISR-Disclosable-P130544-04-10-2017-...`` → ISR dated 2017-04-10.
2. **The document manifest** (``configs/document_manifest.yaml``): curated
   labels for known files, many of which have opaque hash-like names. An entry
   applies only if the file size still matches.

If both sources give a value and they disagree, the value is left empty and
the conflict is recorded; neither source silently wins. No LLM is involved.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from worldbank_copilot.common.dates import FILENAME_MM_DD_YYYY, parse_date
from worldbank_copilot.common.exceptions import ConfigurationError
from worldbank_copilot.common.identifiers import normalize_loan_number
from worldbank_copilot.common.project_registry import ProjectRegistry, validate_project_id_format
from worldbank_copilot.transformations.bronze import (
    LINEAGE_FIELDS,
    BronzeRecord,
    BronzeTable,
    SourceMetadata,
)

INVENTORY_TABLE = "bronze_document_inventory"


class DocumentType(StrEnum):
    ISR = "ISR"
    APPRAISAL_DOCUMENT = "APPRAISAL_DOCUMENT"
    RESTRUCTURING_PAPER = "RESTRUCTURING_PAPER"
    ADDITIONAL_FINANCING = "ADDITIONAL_FINANCING"
    LOAN_AGREEMENT = "LOAN_AGREEMENT"
    PERFORMANCE_INDICATORS = "PERFORMANCE_INDICATORS"
    TECHNICAL_ASSESSMENT = "TECHNICAL_ASSESSMENT"
    FIDUCIARY_ASSESSMENT = "FIDUCIARY_ASSESSMENT"
    ESSA = "ESSA"
    CANCELLATION = "CANCELLATION"
    OTHER = "OTHER"


class ClassificationMethod(StrEnum):
    FILENAME_PATTERN = "FILENAME_PATTERN"
    MANIFEST = "MANIFEST"
    FILENAME_PATTERN_AND_MANIFEST = "FILENAME_PATTERN_AND_MANIFEST"
    CONFLICT = "CONFLICT"
    UNCLASSIFIED = "UNCLASSIFIED"


class ManifestStatus(StrEnum):
    MATCHED = "MATCHED"
    NOT_IN_MANIFEST = "NOT_IN_MANIFEST"
    SIZE_MISMATCH = "SIZE_MISMATCH"


# ---------------------------------------------------------------------------
# Filename classification
# ---------------------------------------------------------------------------

# "...-Sequence-No-05" and the mangled "Disclosable0Ve04000Sequence0No00015".
_ISR_SEQUENCE = re.compile(r"Sequence[-_ 0]?No[-_ 0]*?(\d{1,3})(?!\d)", re.IGNORECASE)
_ISR_MARKER = re.compile(r"ISR|Disclosable", re.IGNORECASE)
# "ISR-Disclosable-P130544-04-10-2017-1491820303083"
_ISR_DATED = re.compile(r"ISR[-_]Disclosable[-_]P\d{6}[-_](\d{2}-\d{2}-\d{4})", re.IGNORECASE)

# Ordered: first match wins.
_TYPE_RULES: tuple[tuple[str, re.Pattern[str], DocumentType], ...] = (
    ("restructuring_paper", re.compile(r"Restructuring[-_ ]Paper", re.I),
     DocumentType.RESTRUCTURING_PAPER),
    ("additional_financing", re.compile(r"Additional[-_ ]Financing", re.I),
     DocumentType.ADDITIONAL_FINANCING),
    ("cancellation", re.compile(r"Cancell?ation", re.I), DocumentType.CANCELLATION),
    ("loan_agreement", re.compile(r"Loan[-_ ]Agreement", re.I), DocumentType.LOAN_AGREEMENT),
    ("performance_indicators",
     re.compile(r"Performance[-_ ]Monitoring|Supplemental[-_ ]Letter", re.I),
     DocumentType.PERFORMANCE_INDICATORS),
    ("technical_assessment", re.compile(r"Technical[-_ ]Assessment", re.I),
     DocumentType.TECHNICAL_ASSESSMENT),
    ("fiduciary_assessment", re.compile(r"Fiduciary", re.I), DocumentType.FIDUCIARY_ASSESSMENT),
    ("essa", re.compile(r"(?<![A-Za-z])ESSA(?![A-Za-z])|Environmental[-_ ]and[-_ ]Social[-_ ]"
                        r"Systems", re.I), DocumentType.ESSA),
    ("appraisal_document", re.compile(r"(?:^|[-_])PAD(?:\d+)?(?:[-_]|$)"),
     DocumentType.APPRAISAL_DOCUMENT),
)  # fmt: skip


@dataclass(frozen=True)
class FilenameClassification:
    document_type: DocumentType | None = None
    isr_sequence: int | None = None
    document_date: date | None = None
    date_basis: str | None = None
    rule: str | None = None


def classify_filename(filename: str) -> FilenameClassification:
    """Classify a document from its filename alone; empty result if nothing matches."""
    stem = Path(filename).stem
    match = _ISR_SEQUENCE.search(stem)
    if match and _ISR_MARKER.search(stem):
        return FilenameClassification(
            DocumentType.ISR, isr_sequence=int(match.group(1)), rule="isr_sequence"
        )
    match = _ISR_DATED.search(stem)
    if match:
        return FilenameClassification(
            DocumentType.ISR,
            document_date=parse_date(match.group(1), [FILENAME_MM_DD_YYYY]),
            date_basis="filename_date",
            rule="isr_dated",
        )
    for rule, pattern, document_type in _TYPE_RULES:
        if pattern.search(stem):
            return FilenameClassification(document_type, rule=rule)
    return FilenameClassification()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str
    size_bytes: int
    document_type: DocumentType
    isr_sequence: int | None = None
    document_date: date | None = None
    date_basis: str | None = None
    report_number: str | None = None
    loan_number_raw: str | None = None
    evidence: str


class DocumentManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    documents: dict[str, list[ManifestEntry]]

    @model_validator(mode="after")
    def _validate(self) -> DocumentManifest:
        for project_id, entries in self.documents.items():
            validate_project_id_format(project_id)
            names = [e.filename for e in entries]
            dupes = sorted({n for n in names if names.count(n) > 1})
            if dupes:
                raise ValueError(f"{project_id}: duplicate manifest filenames {dupes}")
        return self

    def get(self, project_id: str, filename: str) -> ManifestEntry | None:
        for entry in self.documents.get(project_id, []):
            if entry.filename == filename:
                return entry
        return None


def load_document_manifest(path: Path | str) -> DocumentManifest:
    path = Path(path)
    if not path.is_file():
        raise ConfigurationError(f"Document manifest not found: {path}")
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    try:
        return DocumentManifest.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid document manifest {path}:\n{exc}") from exc


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


@dataclass
class DocumentInventory:
    table: BronzeTable
    missing_manifest_files: list[dict[str, str]] = field(default_factory=list)
    missing_project_dirs: list[str] = field(default_factory=list)


_INVENTORY_FIELDS = [
    "project_id",
    "document_id",
    "filename",
    "relative_path",
    "extension",
    "file_size_bytes",
    "sha256",
    "document_type",
    "classification_method",
    "isr_sequence",
    "document_date",
    "date_basis",
    "report_number",
    "raw_loan_number",
    "normalized_loan_number",
    "manifest_status",
    "filename_rule",
    "filename_document_type",
    "filename_isr_sequence",
    "filename_document_date",
    "classification_conflicts",
    "filename_project_ids",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _merge(name: str, from_filename: Any, from_manifest: Any, conflicts: list[str]) -> Any:
    if from_filename is not None and from_manifest is not None and from_filename != from_manifest:
        conflicts.append(f"{name}: filename={from_filename} manifest={from_manifest}")
        return None
    return from_manifest if from_manifest is not None else from_filename


def build_document_inventory(
    documents_root: Path,
    data_root: Path,
    registry: ProjectRegistry,
    manifest: DocumentManifest,
    *,
    ingested_at: str,
    run_id: str,
) -> DocumentInventory:
    """Inventory every file in each registered project's document directory."""
    records: list[BronzeRecord] = []
    missing_dirs: list[str] = []
    seen: set[tuple[str, str]] = set()

    for project in registry.projects:
        project_dir = Path(documents_root) / project.documents_dir
        if not project_dir.is_dir():
            missing_dirs.append(project.project_id)
            continue
        for path in sorted(p for p in project_dir.iterdir() if p.is_file()):
            seen.add((project.project_id, path.name))
            records.append(
                _inventory_record(path, project.project_id, data_root, manifest, ingested_at,
                                  run_id)
            )  # fmt: skip

    missing_manifest = [
        {"project_id": pid, "filename": entry.filename}
        for pid, entries in manifest.documents.items()
        for entry in entries
        if (pid, entry.filename) not in seen
    ]
    metadata = SourceMetadata(
        source_name="document_inventory",
        source_file=_relative(Path(documents_root), data_root) or ".",
        ingested_at=ingested_at,
        ingestion_run_id=run_id,
        rows_scanned=len(records),
        rows_matched=len(records),
        notes=[
            "No PDF content is parsed in Phase 2; classification uses filename patterns "
            "and configs/document_manifest.yaml.",
        ],
    )
    table = BronzeTable(INVENTORY_TABLE, records, metadata, [*LINEAGE_FIELDS, *_INVENTORY_FIELDS])
    return DocumentInventory(table, missing_manifest, missing_dirs)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _inventory_record(
    path: Path,
    project_id: str,
    data_root: Path,
    manifest: DocumentManifest,
    ingested_at: str,
    run_id: str,
) -> BronzeRecord:
    size = path.stat().st_size
    sha = _sha256(path)
    relative_path = _relative(path, data_root)
    by_name = classify_filename(path.name)

    entry = manifest.get(project_id, path.name)
    if entry is None:
        status = ManifestStatus.NOT_IN_MANIFEST
    elif entry.size_bytes != size:
        status = ManifestStatus.SIZE_MISMATCH
    else:
        status = ManifestStatus.MATCHED
    trusted = entry if status is ManifestStatus.MATCHED else None

    conflicts: list[str] = []
    document_type = _merge(
        "document_type", by_name.document_type, trusted and trusted.document_type, conflicts
    )
    isr_sequence = _merge(
        "isr_sequence", by_name.isr_sequence, trusted and trusted.isr_sequence, conflicts
    )
    document_date = _merge(
        "document_date", by_name.document_date, trusted and trusted.document_date, conflicts
    )
    if document_date is None:
        date_basis = None
    elif trusted is not None and trusted.document_date is not None:
        date_basis = trusted.date_basis
    else:
        date_basis = by_name.date_basis

    if any(c.startswith("document_type") for c in conflicts):
        method = ClassificationMethod.CONFLICT
    elif by_name.document_type and trusted:
        method = ClassificationMethod.FILENAME_PATTERN_AND_MANIFEST
    elif trusted:
        method = ClassificationMethod.MANIFEST
    elif by_name.document_type:
        method = ClassificationMethod.FILENAME_PATTERN
    else:
        method = ClassificationMethod.UNCLASSIFIED
        document_type = DocumentType.OTHER

    raw_loan = trusted.loan_number_raw if trusted else None
    return {
        "_source_file": relative_path,
        "_source_sheet": None,
        "_source_row": None,
        "_ingested_at": ingested_at,
        "_ingestion_run_id": run_id,
        "project_id": project_id,
        "document_id": f"{project_id}-{sha[:12]}",
        "filename": path.name,
        "relative_path": relative_path,
        "extension": path.suffix.lower(),
        "file_size_bytes": size,
        "sha256": sha,
        "document_type": document_type.value if document_type else None,
        "classification_method": method.value,
        "isr_sequence": isr_sequence,
        "document_date": document_date.isoformat() if document_date else None,
        "date_basis": date_basis,
        "report_number": trusted.report_number if trusted else None,
        "raw_loan_number": raw_loan,
        "normalized_loan_number": normalize_loan_number(raw_loan),
        "manifest_status": status.value,
        "filename_rule": by_name.rule,
        "filename_document_type": by_name.document_type.value if by_name.document_type else None,
        "filename_isr_sequence": by_name.isr_sequence,
        "filename_document_date": (
            by_name.document_date.isoformat() if by_name.document_date else None
        ),
        "classification_conflicts": conflicts,
        "filename_project_ids": sorted(set(re.findall(r"P\d{6}", path.name))),
    }


# ---------------------------------------------------------------------------
# ISR completeness
# ---------------------------------------------------------------------------


class IsrCompleteness(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str
    expected_count: int | None
    found_count: int  # distinct sequence numbers
    isr_documents: int
    isr_without_sequence: int
    sequences: list[int]
    missing: list[int]
    duplicates: list[int]
    is_complete: bool


def check_isr_completeness(
    project_id: str,
    sequences: Iterable[int | None],
    expected_count: int | None,
) -> IsrCompleteness:
    """Report missing and duplicate ISR sequence numbers (expected run: 1..N)."""
    values = list(sequences)
    known = [s for s in values if s is not None]
    distinct = sorted(set(known))
    duplicates = sorted(s for s, n in Counter(known).items() if n > 1)
    upper = max(expected_count or 0, distinct[-1] if distinct else 0)
    present = set(distinct)
    missing = [i for i in range(1, upper + 1) if i not in present]
    without_sequence = len(values) - len(known)
    complete = (
        not missing
        and not duplicates
        and without_sequence == 0
        and (expected_count is None or len(distinct) == expected_count)
    )
    return IsrCompleteness(
        project_id=project_id,
        expected_count=expected_count,
        found_count=len(distinct),
        isr_documents=len(values),
        isr_without_sequence=without_sequence,
        sequences=distinct,
        missing=missing,
        duplicates=duplicates,
        is_complete=complete,
    )


def isr_completeness_by_project(
    inventory: BronzeTable, registry: ProjectRegistry
) -> dict[str, IsrCompleteness]:
    results = {}
    for project in registry.projects:
        sequences = [
            r["isr_sequence"]
            for r in inventory.for_project(project.project_id)
            if r["document_type"] == DocumentType.ISR.value
        ]
        results[project.project_id] = check_isr_completeness(
            project.project_id, sequences, project.expected_isr_count
        )
    return results
