"""Named Phase 6 validation queries (``sql/phase6_validation.sql``), rendered from config.

These queries check the persisted data (ordering, provenance, quality, precision, no
Gold tables). They are validation, not Gold analytics.
"""

from __future__ import annotations

import re
from pathlib import Path

from worldbank_copilot.common.config import Settings

VALIDATION_FILE = Path("sql") / "phase6_validation.sql"
_NAME = re.compile(r"^-- name: (\S+)\s*$", re.M)


def load_queries(settings: Settings, path: Path | None = None) -> dict[str, str]:
    text = (path or settings.repo_root / VALIDATION_FILE).read_text(encoding="utf-8")
    values = {
        "catalog": settings.require("databricks.catalog"),
        "bronze": settings.require("databricks.bronze_schema"),
        "silver": settings.require("databricks.silver_schema"),
        "gold": settings.require("databricks.gold_schema"),
    }
    parts = _NAME.split(text)
    queries = {}
    for name, body in zip(parts[1::2], parts[2::2], strict=True):
        lines = [line for line in body.strip().splitlines() if not line.lstrip().startswith("--")]
        queries[name] = "\n".join(lines).strip().rstrip(";").format(**values)
    return queries
