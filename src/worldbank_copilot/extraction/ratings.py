"""Controlled rating vocabulary. Raw values are kept; unknown values are never mapped.

Performance scale (PDO / implementation progress): Highly Satisfactory ... Highly
Unsatisfactory. Risk scale: Low, Moderate, Substantial, High. No numeric scores
here: ordering for rating-change rules belongs to the Gold phase.
"""

from __future__ import annotations

import re
from enum import StrEnum

from worldbank_copilot.extraction.provenance import ExtractionStatus


class RatingScale(StrEnum):
    PERFORMANCE = "PERFORMANCE"
    RISK = "RISK"


PERFORMANCE_RATINGS = (
    "Highly Satisfactory",
    "Satisfactory",
    "Moderately Satisfactory",
    "Moderately Unsatisfactory",
    "Unsatisfactory",
    "Highly Unsatisfactory",
)
RISK_RATINGS = ("Low", "Moderate", "Substantial", "High")
NOT_RATED = {"--", "-", "n/a", "na", "not rated", "not applicable", ""}

# Icon glyphs the PDFs print before ratings (private-use area, bullets, discs).
_GLYPHS = re.compile(r"[-●⚫•►➢]")
_LOOKUP = {r.casefold(): r for r in (*PERFORMANCE_RATINGS, *RISK_RATINGS)}
_SCALE = {
    **{r: RatingScale.PERFORMANCE for r in PERFORMANCE_RATINGS},
    **{r: RatingScale.RISK for r in RISK_RATINGS},
}

# Longest phrases first so "Moderately Satisfactory" wins over "Satisfactory".
RATING_PATTERN = (
    "|".join(re.escape(r) for r in sorted(_LOOKUP.values(), key=len, reverse=True)) + r"|--"
)


def strip_glyphs(text: str | None) -> str:
    return " ".join(_GLYPHS.sub(" ", text or "").split())


def normalize_rating(
    raw: str | None, expected: RatingScale | None = None
) -> tuple[str | None, ExtractionStatus]:
    """Map a printed rating to the controlled vocabulary.

    Returns (normalized, status): EXACT when the printed text already equals a
    vocabulary term, NORMALIZED when only glyphs/case/spacing differed, MISSING
    for explicit "not rated" markers, AMBIGUOUS for anything unrecognised or on
    the wrong scale (the raw value is kept by the caller).
    """
    cleaned = strip_glyphs(raw)
    if cleaned.casefold() in NOT_RATED:
        return None, ExtractionStatus.MISSING
    match = _LOOKUP.get(cleaned.casefold())
    if match is None:
        return None, ExtractionStatus.AMBIGUOUS
    if expected is not None and _SCALE[match] is not expected:
        return None, ExtractionStatus.AMBIGUOUS
    status = ExtractionStatus.EXACT if (raw or "").strip() == match else ExtractionStatus.NORMALIZED
    return match, status


def scale_of(normalized: str | None) -> RatingScale | None:
    return _SCALE.get(normalized) if normalized else None
