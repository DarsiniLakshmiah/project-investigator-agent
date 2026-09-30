"""Source snapshot manifest and source-hash verification.

``configs/source_snapshot.json`` records every validated source file (3 structured
files + 53 project PDFs) with its path relative to the data root, size and SHA-256.
The ``snapshot_id`` is a hash over all (path, hash) pairs. The same manifest is
checked locally and in Databricks (against the Unity Catalog Volume): a missing
file or a hash mismatch is a hard failure and nothing is loaded.

Hashes are not secrets; the manifest is committed so both environments verify
against the same reviewed snapshot.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldbank_copilot.common.exceptions import SourceIntegrityError

MANIFEST_FILE = "source_snapshot.json"
SOURCE_SUFFIXES = {".pdf", ".xlsx", ".csv"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SourceFile:
    relative_path: str  # POSIX, relative to the data root
    sha256: str
    size_bytes: int
    kind: str  # structured | document
    project_id: str | None = None


@dataclass(frozen=True)
class SourceSnapshot:
    snapshot_id: str
    files: tuple[SourceFile, ...]

    @property
    def hashes(self) -> dict[str, str]:
        return {f.relative_path: f.sha256 for f in self.files}

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "file_count": len(self.files),
            "files": [f.__dict__ for f in self.files],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceSnapshot:
        files = tuple(SourceFile(**f) for f in data["files"])
        snapshot = cls(snapshot_identity(files), files)
        if snapshot.snapshot_id != data["snapshot_id"]:
            raise SourceIntegrityError("source manifest snapshot_id does not match its file list")
        return snapshot


def snapshot_identity(files: tuple[SourceFile, ...] | list[SourceFile]) -> str:
    lines = sorted(f"{f.relative_path}\t{f.sha256}" for f in files)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def build_snapshot(
    data_root: Path, structured: list[Path], documents: list[tuple[str, Path]]
) -> SourceSnapshot:
    """Snapshot of the given files (paths must lie under ``data_root``)."""
    root = Path(data_root)
    files = []
    for path in structured:
        files.append(
            SourceFile(
                Path(path).relative_to(root).as_posix(),
                sha256_file(path),
                Path(path).stat().st_size,
                "structured",
            )
        )
    for project_id, path in documents:
        files.append(
            SourceFile(
                Path(path).relative_to(root).as_posix(),
                sha256_file(path),
                Path(path).stat().st_size,
                "document",
                project_id,
            )
        )
    files.sort(key=lambda f: f.relative_path)
    return SourceSnapshot(snapshot_identity(files), tuple(files))


def load_snapshot(config_dir: Path) -> SourceSnapshot:
    path = Path(config_dir) / MANIFEST_FILE
    if not path.exists():
        raise SourceIntegrityError(
            f"{path} not found: run scripts/platformize.py --write-snapshot "
            "on the validated sources"
        )
    return SourceSnapshot.from_dict(json.loads(path.read_text(encoding="utf-8")))


def write_snapshot(snapshot: SourceSnapshot, config_dir: Path) -> Path:
    path = Path(config_dir) / MANIFEST_FILE
    path.write_text(json.dumps(snapshot.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


@dataclass(frozen=True)
class FileCheck:
    relative_path: str
    expected_sha256: str
    actual_sha256: str | None
    status: str  # MATCH | MISMATCH | MISSING


@dataclass(frozen=True)
class IntegrityReport:
    snapshot_id: str
    data_root: str
    checks: tuple[FileCheck, ...]
    unexpected_files: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return all(c.status == "MATCH" for c in self.checks)

    def counts(self) -> dict[str, int]:
        out = {"MATCH": 0, "MISMATCH": 0, "MISSING": 0}
        for c in self.checks:
            out[c.status] += 1
        return out


def verify_snapshot(snapshot: SourceSnapshot, data_root: Path | str) -> IntegrityReport:
    """Hash every expected file under ``data_root``; report extra source-like files."""
    root = Path(data_root)
    checks = []
    for f in snapshot.files:
        path = root / f.relative_path
        if not path.is_file():
            checks.append(FileCheck(f.relative_path, f.sha256, None, "MISSING"))
            continue
        actual = sha256_file(path)
        status = "MATCH" if actual == f.sha256 else "MISMATCH"
        checks.append(FileCheck(f.relative_path, f.sha256, actual, status))
    expected = {f.relative_path for f in snapshot.files}
    extra = sorted(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SOURCE_SUFFIXES
        and p.relative_to(root).as_posix() not in expected
    )
    return IntegrityReport(snapshot.snapshot_id, str(root), tuple(checks), tuple(extra))


def require_integrity(report: IntegrityReport) -> None:
    if not report.ok:
        bad = [
            f"{c.relative_path}: {c.status} (expected {c.expected_sha256[:12]}, "
            f"actual {(c.actual_sha256 or '-')[:12]})"
            for c in report.checks
            if c.status != "MATCH"
        ]
        raise SourceIntegrityError(
            f"{len(bad)} source file(s) differ from snapshot {report.snapshot_id[:16]} under "
            f"{report.data_root}: " + "; ".join(bad[:10])
        )
