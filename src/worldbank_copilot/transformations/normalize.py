"""Reusable value normalisation for Silver.

Contract for every function here:

* a blank value (``None``, ``""``, whitespace only) becomes ``None``;
* a well-formed value becomes its normalised, typed form;
* a malformed value raises ``NormalizationError``. Callers record it as a data
  issue with provenance (see ``silver.RowNormalizer``); nothing is silently coerced.
"""

from __future__ import annotations

import html
import re
from collections.abc import Sequence
from datetime import date
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from html.parser import HTMLParser

from worldbank_copilot.common.dates import DateParseError, parse_date

_WHITESPACE = re.compile(r"\s+")
_DECIMAL = re.compile(r"^[+-]?\d+(\.\d+)?$")
_YEAR = re.compile(r"^\d{4}$")
_TAXONOMY_PREFIX = re.compile(r"^(FY\d{2}) - ")


class NormalizationError(ValueError):
    """A non-blank source value cannot be normalised."""


def is_blank(raw: object) -> bool:
    return raw is None or (isinstance(raw, str) and not raw.strip())


def blank_to_none(raw: str | None) -> str | None:
    """Return ``None`` for blank values; otherwise the value unchanged."""
    return None if is_blank(raw) else raw


def clean_text(raw: str | None) -> str | None:
    """Collapse all whitespace runs (incl. non-breaking spaces) to one space and trim."""
    if is_blank(raw):
        return None
    return _WHITESPACE.sub(" ", raw.replace(" ", " ")).strip()


class _TextExtractor(HTMLParser):
    _BLOCK_TAGS = {"p", "br", "div", "li", "ul", "ol", "tr", "td", "h1", "h2", "h3", "h4"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):  # noqa: ARG002
        if tag in self._BLOCK_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in self._BLOCK_TAGS:
            self.parts.append(" ")

    def handle_data(self, data):
        self.parts.append(data)


def strip_html(raw: str | None) -> str | None:
    """Remove markup and decode entities; block tags become spaces; then ``clean_text``."""
    if is_blank(raw):
        return None
    parser = _TextExtractor()
    parser.feed(raw)
    parser.close()
    return clean_text(html.unescape("".join(parser.parts)))


def normalize_category(raw: str | None) -> str | None:
    """Categorical text: whitespace-normalised, original casing kept (no vocabulary mapping)."""
    return clean_text(raw)


def normalize_identifier(raw: str | None, *, upper: bool = False) -> str | None:
    """Trim an identifier; internal whitespace is malformed (identifiers are not re-spaced)."""
    if is_blank(raw):
        return None
    value = raw.strip()
    if _WHITESPACE.search(value):
        raise NormalizationError(f"identifier contains internal whitespace: {raw!r}")
    return value.upper() if upper else value


def parse_decimal(raw: str | None) -> Decimal | None:
    """Exact decimal from plain notation (``-12.50``). Commas, exponents, text: malformed."""
    if is_blank(raw):
        return None
    text = raw.strip()
    if not _DECIMAL.match(text):
        raise NormalizationError(f"not a plain decimal number: {raw!r}")
    try:
        return Decimal(text)
    except InvalidOperation as exc:  # pragma: no cover - regex already guards
        raise NormalizationError(f"not a decimal number: {raw!r}") from exc


def parse_money(raw: str | None) -> Decimal | None:
    """Monetary value as ``Decimal`` (never float). Sign is kept; callers decide validity."""
    return parse_decimal(raw)


def parse_year(raw: str | None) -> int | None:
    if is_blank(raw):
        return None
    text = raw.strip()
    if not _YEAR.match(text):
        raise NormalizationError(f"not a four-digit year: {raw!r}")
    return int(text)


def parse_source_date(raw: str | None, formats: Sequence[str]) -> date | None:
    """Typed date using only the formats the source contract declares."""
    try:
        return parse_date(raw, formats)
    except DateParseError as exc:
        raise NormalizationError(str(exc)) from exc


def taxonomy_label(name: str | None) -> str | None:
    """The explicit taxonomy prefix of a sector/theme name (e.g. ``FY17``), if present."""
    if name is None:
        return None
    match = _TAXONOMY_PREFIX.match(name)
    return match.group(1) if match else None


def percentage(numerator: Decimal | None, denominator: Decimal | None) -> Decimal | None:
    """``numerator / denominator * 100`` rounded half-even to 2 dp; None if undefined."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return (numerator / denominator * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
