"""Deterministic date parsing against explicitly declared formats.

Sources use different date conventions, so callers always say which formats a
field may use; nothing is guessed. In particular ``MM/DD/YYYY`` is only accepted
where a source contract declares it, never inferred from the value.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime

# Formats observed in the source files (Phase 1 inspection).
ISO_DATE = "%Y-%m-%d"  # workbook: 2016-08-22
ISO_DATETIME_Z = "%Y-%m-%dT%H:%M:%SZ"  # workbook: 2016-03-31T00:00:00Z
US_SLASH = "%m/%d/%Y"  # CSVs: 08/31/2026 (also accepts 4/10/2017)
DAY_MON_YEAR = "%d-%b-%Y"  # ISRs: 29-Nov-2017
MON_DAY_YEAR = "%b %d, %Y"  # ISRs: Aug 16, 2026
FILENAME_MM_DD_YYYY = "%m-%d-%Y"  # ISR-Disclosable-P130544-04-10-2017-...


class DateParseError(ValueError):
    """A non-blank value did not match any declared format."""

    def __init__(self, raw: str, formats: Sequence[str]):
        self.raw = raw
        self.formats = tuple(formats)
        super().__init__(f"Cannot parse date {raw!r} with formats {list(formats)}")


def parse_date(raw: str | None, formats: Sequence[str]) -> date | None:
    """Parse ``raw`` with the first matching format.

    Blank values return ``None`` (a missing value is not a parse failure).
    Non-blank values that match no format raise ``DateParseError``.
    """
    if not formats:
        raise ValueError("At least one date format must be declared")
    if raw is None or not raw.strip():
        return None
    text = raw.strip()
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise DateParseError(raw, formats)
