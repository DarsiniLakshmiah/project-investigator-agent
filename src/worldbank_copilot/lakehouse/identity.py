"""Deterministic record identities and content fingerprints.

* ``stable_id`` hashes a table name and natural-key values: the same record gets the
  same ``record_id`` in every run and every environment (no random UUIDs).
* ``content_hash`` hashes a row's business, provenance and quality columns in a
  canonical form. Decimals are rendered at the contract scale, so a value read back
  from Delta (``Decimal('100.000000')``) hashes like the value written
  (``Decimal('100.00')``). Operational load metadata is excluded, so reloading an
  unchanged snapshot leaves every hash and therefore every Delta row unchanged.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from worldbank_copilot.lakehouse.contracts import DECIMAL_SCALE, TableContract

_QUANTUM = Decimal(1).scaleb(-DECIMAL_SCALE)


def canonical(value: Any) -> Any:
    """JSON-safe canonical form used for hashing and comparison."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, Decimal):
        return format(value.quantize(_QUANTUM), "f")
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, list | tuple):
        return [canonical(v) for v in value]
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items())}
    raise TypeError(f"no canonical form for {type(value).__name__}")


def stable_id(table: str, *key: Any) -> str:
    payload = json.dumps([table, *[canonical(k) for k in key]], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def content_hash(contract: TableContract, row: dict[str, Any]) -> str:
    payload = {name: canonical(row.get(name)) for name in contract.hashed_columns}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    """Compact, key-sorted JSON for nested structures kept as text columns."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
