"""Helpers over cached ParsedDocuments: sections, tables, cells, values, dates."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from worldbank_copilot.extraction.ratings import strip_glyphs
from worldbank_copilot.parsing.models import ParsedDocument, ParsedTable, Section

DMY = r"\d{1,2}-[A-Za-z]{3}-\s?\d{4}"
MON_YEAR = r"[A-Za-z]{3}/\d{4}"
_DMY_RE = re.compile(rf"^{DMY}$")
_MON_YEAR_RE = re.compile(rf"^{MON_YEAR}$")
_NUMBER_RE = re.compile(r"^[-+]?\d[\d,]*(\.\d+)?%?$")
LOAN_ID_RE = re.compile(r"\b(IBRD|IDA|TF)-\s?(\d{5})(?:-\s?\d{3})?\b")


def clean_cell(text: str | None) -> str:
    return strip_glyphs(text).strip()


def section_title(doc: ParsedDocument, section_id: str | None) -> str | None:
    section = doc.section(section_id)
    return section.title if section else None


def sections_matching(doc: ParsedDocument, pattern: str) -> list[Section]:
    regex = re.compile(pattern, re.IGNORECASE)
    return [s for s in doc.sections if regex.search(s.title)]


def section_text(doc: ParsedDocument, section: Section) -> str:
    blocks = {b.block_id: b for b in doc.blocks}
    parts = [
        blocks[i].text_clean
        for i in section.block_ids
        if i in blocks and i != section.heading_block_id and blocks[i].text_clean
    ]
    return "\n".join(parts)


def tables_in(doc: ParsedDocument, sections: Iterable[Section]) -> list[ParsedTable]:
    ids = {t for s in sections for t in s.table_ids}
    return [t for t in doc.tables if t.table_id in ids]


def table_blob(table: ParsedTable, rows: int = 3) -> str:
    return " ".join(" ".join(clean_cell(c) for c in row) for row in table.rows[:rows])


def compact_number(text: str) -> str:
    """Remove spaces a line wrap injected inside a number: '24,069,800,0 00.00'."""
    return re.sub(r"(?<=[\d,.])\s+(?=[\d,.])", "", text.strip())


def parse_number(text: str | None) -> Decimal | None:
    if text is None:
        return None
    value = compact_number(clean_cell(text)).rstrip("%").replace(",", "")
    if not value or not _NUMBER_RE.match(value):
        return None
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def is_number(text: str | None) -> bool:
    return parse_number(text) is not None


def parse_dmy(text: str | None) -> date | None:
    value = clean_cell(text).replace("- ", "-")
    if not _DMY_RE.match(value):
        return None
    try:
        return datetime.strptime(value, "%d-%b-%Y").date()
    except ValueError:
        return None


def parse_month_year(text: str | None) -> tuple[int, int] | None:
    value = clean_cell(text)
    if not _MON_YEAR_RE.match(value):
        return None
    try:
        parsed = datetime.strptime(value, "%b/%Y")
    except ValueError:
        return None
    return parsed.year, parsed.month


def dates_in(text: str) -> list[date]:
    found = []
    for match in re.finditer(DMY, text):
        parsed = parse_dmy(match.group(0))
        if parsed:
            found.append(parsed)
    return found


def row_tokens(row: list[str]) -> list[str]:
    """All whitespace tokens of a row, in order (undoes merged cells like '2.00 98.00')."""
    return [t for cell in row for t in clean_cell(cell).split()]


def statement_loan_number(text: str) -> str | None:
    """'IBRD-86010' / 'IBRD-86010- 001' -> 'IBRD86010' (Statement of Loans notation)."""
    match = LOAN_ID_RE.search(text)
    return f"{match.group(1)}{match.group(2)}" if match else None
