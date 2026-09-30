"""Dataset profiles and reconciliation (validated data vs persisted Delta data).

A profile is computed from contract-shaped rows, whether built in memory or read
back from Delta: row count, content fingerprint (hash of the sorted record hashes,
recomputed from the row values), per-column null counts, distinct counts of
identity/provenance columns and the configured group counts.

Reconciliation compares two profiles field by field. Any difference is reported;
``require_match`` raises, because a mismatch means persistence changed the data.
The committed expected profiles (``configs/reconciliation/expected_profiles.json``)
describe the *current validated snapshot*. They are regenerated when the source
snapshot legitimately changes; they are not business rules.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from worldbank_copilot.common.exceptions import ReconciliationError
from worldbank_copilot.lakehouse.contracts import RECORD_HASH, RECORD_ID, Role, TableContract
from worldbank_copilot.lakehouse.identity import canonical, content_hash

EXPECTED_FILE = Path("reconciliation") / "expected_profiles.json"
_DISTINCT = (
    "project_id",
    "document_id",
    "source_document",
    "evidence_document_id",
    "indicator_key",
    "_source_file",
    "relative_path",
    "loan_number",
)


def _key(values: tuple) -> str:
    return "|".join("" if v is None else str(canonical(v)) for v in values)


def profile(contract: TableContract, rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = [c.name for c in contract.columns if c.role is not Role.OPERATIONAL]
    recomputed = [content_hash(contract, row) for row in rows]
    hashes = sorted(recomputed)
    # A stored record_hash that differs from the hash of the stored values means a value
    # changed in persistence (e.g. precision or date conversion).
    stored_mismatches = sum(
        row.get(RECORD_HASH) != h for row, h in zip(rows, recomputed, strict=True)
    )
    groups = {}
    for group in contract.profile_groups:
        counts = Counter(_key(tuple(row.get(g) for g in group)) for row in rows)
        groups["+".join(group)] = dict(sorted(counts.items()))
    return {
        "table": f"{contract.layer}.{contract.name}",
        "row_count": len(rows),
        "fingerprint": hashlib.sha256("\n".join(hashes).encode()).hexdigest(),
        "distinct_record_ids": len({row[RECORD_ID] for row in rows}),
        "stored_hash_mismatches": stored_mismatches,
        "null_counts": {n: sum(row.get(n) is None for row in rows) for n in names},
        "distinct_counts": {
            n: len({_key((row.get(n),)) for row in rows})
            for n in _DISTINCT
            if n in contract.column_names
        },
        "groups": groups,
    }


def compare(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Human-readable differences (empty list = reconciled)."""
    diffs = []
    for field in ("row_count", "fingerprint", "distinct_record_ids", "stored_hash_mismatches"):
        if expected.get(field) != actual.get(field):
            diffs.append(f"{field}: expected {expected.get(field)!r}, got {actual.get(field)!r}")
    for section in ("null_counts", "distinct_counts", "groups"):
        exp, act = expected.get(section, {}), actual.get(section, {})
        for key in sorted(set(exp) | set(act)):
            if exp.get(key) != act.get(key):
                diffs.append(f"{section}.{key}: expected {exp.get(key)!r}, got {act.get(key)!r}")
    return diffs


def require_match(table: str, expected: dict[str, Any], actual: dict[str, Any], label: str) -> None:
    diffs = compare(expected, actual)
    if diffs:
        raise ReconciliationError(
            f"{table} does not reconcile with {label}: " + "; ".join(diffs[:8])
        )


def load_expected(config_dir: Path) -> dict[str, Any] | None:
    path = Path(config_dir) / EXPECTED_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_expected(profiles: dict[str, Any], snapshot_id: str, config_dir: Path) -> Path:
    path = Path(config_dir) / EXPECTED_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"source_snapshot_id": snapshot_id, "profiles": profiles}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
