"""Results-framework extraction (silver.project_results) from ISRs.

Two ISR layouts exist:

* BLOCK (older ISRs): one small table per indicator: optional IN-number, the
  indicator name ending in "(Unit, Type)", a Baseline / Actual (Previous) /
  Actual (Current) / End Target header, a Value row and a Date row.
* WIDE (newer ISRs): one row per indicator: name | baseline value | month/year |
  previous value | date | current value | date | closing-period value | month/year.
  DLI tables use a related 8-column layout.

Docling tables are parsed first and validated per indicator. A results page is
re-read from the PDF text layer (PDF_TEXT_FALLBACK) when a BLOCK table on it fails
validation (missing/merged cells, truncated name without its "(Unit, Type)"
suffix, wrong value count) or when the page's text coverage is low. Where Docling
and the fallback both produced an indicator and disagree, values are left NULL
with an EXTRACTION_CONFLICT issue. Nothing is guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from worldbank_copilot.common.quality import CheckCode
from worldbank_copilot.extraction.models import ResultObservation
from worldbank_copilot.extraction.provenance import (
    ExtractionIssue,
    ExtractionMethod,
    ExtractionStatus,
    evidence,
    issue,
)
from worldbank_copilot.extraction.ratings import strip_glyphs
from worldbank_copilot.extraction.tables import DMY, LOAN_ID_RE, MON_YEAR, clean_cell, section_title
from worldbank_copilot.extraction.text_source import LoggedTextSource
from worldbank_copilot.parsing.models import ParsedDocument, ParsedTable

LOW_COVERAGE = 0.95
_ID = re.compile(r"\b(IN\d{8})\b")
_UNIT = re.compile(r"\(([^()]+?)(?:,\s*([^()]+))?\)\s*$")
_VALUE = re.compile(r"^(?:[-+]?\d[\d,]*(?:\.\d+)?%?|--|Yes|No|N/A)$", re.I)
_DATE_CELL = re.compile(rf"^(?:{DMY}|{MON_YEAR}|--)?$")
_BLOCK_HEADER = re.compile(
    r"^Baseline\s+Actual\s*\(Previous\)\s+Actual\s*\(Current\)\s+End Target", re.I
)
_TEXT_RUNNING = re.compile(
    r"\(P\d{6}\)|Page \d+ of \d+|Implementation Status & Results Report|^The World Bank$|"
    r"^PHIND\w*$|^Public Disclosure"
)
_NAME_START = re.compile(r"^[►►]")
_PDO_HEADING = re.compile(r"PDO Indicators|Project Development Objective Indicators", re.I)
_IR_HEADING = re.compile(r"Intermediate Results Indicators", re.I)
_DLI_HEADING = re.compile(r"Disbursement Linked Indicators|DLI", re.I)
_AREA_LABEL = re.compile(r"^RA\s?\d+\s*[:\-–]\s*\S", re.I)  # "RA3: Improve ..." / "RA 1 - ..."


@dataclass
class RawIndicator:
    name: str
    values: list[str | None]  # baseline, previous, current, target
    dates: list[str | None]
    page: int
    method: ExtractionMethod
    layout: str
    source_id: str | None = None
    indicator_type: str | None = None
    result_area: str | None = None
    comments: str | None = None
    table_id: str | None = None
    row: int | None = None
    source_text: str | None = None
    section: str | None = None
    problems: list[str] = field(default_factory=list)
    name_source: tuple[int, str] | None = None  # (page, text) when completed from PDF text

    @property
    def valid(self) -> bool:
        return not self.problems


def normalize_indicator_name(name: str) -> str:
    """Case-folded, glyph-free, whitespace-collapsed name (units kept)."""
    text = strip_glyphs(name).replace("►", " ").replace("➢", " ")
    return " ".join(text.split()).casefold()


# Printed after the "(unit, type)" parenthetical; split off the name, never discarded.
_TAGS_AFTER_UNIT = re.compile(r"^(.*\))((?:\s+(?:DLI|CRI))+)\s*$")
_DLI_STATUS = re.compile(
    r"\s+(Not Due|Partially achieved|Not achieved|Achieved|Off-track|On-track)\s*$", re.I
)


def split_name_suffixes(name: str) -> tuple[str, list[str], str | None]:
    """'X (Number) DLI' -> ('X (Number)', ['DLI'], None); DLI status likewise."""
    status = None
    if (match := _DLI_STATUS.search(name)) and ")" in name[: match.start()]:
        status, name = match.group(1), name[: match.start()]
    tags: list[str] = []
    if match := _TAGS_AFTER_UNIT.match(name):
        name, tags = match.group(1), match.group(2).split()
    return name.strip(), tags, status


def split_unit(name: str) -> tuple[str | None, str | None]:
    match = _UNIT.search(name)
    if not match:
        return None, None
    return match.group(1).strip(), (match.group(2) or "").strip() or None


def _indicator_type(title: str | None) -> str | None:
    if not title:
        return None
    if _PDO_HEADING.search(title):
        return "PDO"
    if _IR_HEADING.search(title):
        return "INTERMEDIATE"
    if _DLI_HEADING.search(title):
        return "DLI"
    return None


def _clean(value: str | None) -> str | None:
    value = clean_cell(value)
    return value or None


# ---------------------------------------------------------------------------
# Docling: table classification and parsing
# ---------------------------------------------------------------------------


def _is_label_row(row: list[str]) -> bool:
    cells = [clean_cell(c) for c in row]
    return len(set(cells)) == 1 and bool(cells[0])


_GROUP_WORDS = (("baseline", 0), ("previous", 1), ("current", 2), ("closing", 3), ("end target", 3))
ColumnMap = dict[int, tuple[int, str]]  # column -> (0 baseline|1 previous|2 current|3 target, kind)


def _group_of(text: str) -> int | None:
    lowered = text.lower()
    return next((g for word, g in _GROUP_WORDS if word in lowered), None)


def header_map(top: list[str], sub: list[str]) -> ColumnMap | None:
    """Map columns from the two header rows (group row over Value/Date row).

    Works for 9-column ISRs (baseline, previous, current, closing), 7-column first
    ISRs (no previous) and DLI tables. Returns None unless every group found has a
    value column.
    """
    mapping: ColumnMap = {}
    for col in range(1, min(len(top), len(sub))):
        group = _group_of(clean_cell(top[col])) if clean_cell(top[col]) else None
        if group is None:
            group = _group_of(clean_cell(sub[col]))
        if group is None:
            continue
        label = clean_cell(sub[col]).lower()
        kind = "date" if ("date" in label or "month" in label) else "value"
        mapping[col] = (group, kind)
    groups = {g for g, k in mapping.values() if k == "value"}
    return mapping if mapping and groups >= {g for g, _ in mapping.values()} else None


def _find_header(rows: list[list[str]]) -> tuple[int, ColumnMap] | None:
    for i in range(len(rows) - 1):
        top, sub = rows[i], rows[i + 1]
        if any(_group_of(c) == 0 for c in top[1:]) and any(
            clean_cell(c).lower().endswith(("value", "date", "month/year")) for c in sub[1:]
        ):
            mapping = header_map(top, sub)
            if mapping:
                return i + 1, mapping
    return None


def is_loan_table(table: ParsedTable) -> bool:
    """Per-loan financial / key-date tables (never results continuations)."""
    return any(
        LOAN_ID_RE.search(clean_cell(r[0])) or "loan/credit" in clean_cell(r[0]).lower()
        for r in table.rows
        if r
    )


def classify(table: ParsedTable) -> str | None:
    blob = " ".join(clean_cell(c) for r in table.rows[:4] for c in r).lower()
    if "pbc" in blob and "baseline" in blob:
        return "DLI"
    if "indicator" in blob and "closing period" in blob:
        return "WIDE"
    if "baseline" in blob and ("end target" in blob or "actual (current)" in blob):
        return "BLOCK"
    return None


# Printed values of (Text) / (Yes/No) indicators, e.g. "Not started", "Incomplete".
_TEXT_VALUE = re.compile(r"^[A-Za-z][A-Za-z /\-]{0,40}$")


def _mapped_values(
    cells: list[str], mapping: ColumnMap, text_values: bool = False
) -> tuple[list, list] | None:
    """Values/dates per group from mapped cells; resplit merged cells if needed."""
    values: list[str | None] = [None] * 4
    dates: list[str | None] = [None] * 4
    columns = sorted(mapping)
    if all(c < len(cells) for c in columns):
        ok = True
        for col in columns:
            group, kind = mapping[col]
            text = clean_cell(cells[col])
            pattern = _DATE_CELL if kind == "date" else _VALUE
            if (
                text
                and not pattern.match(text)
                and not (text_values and kind != "date" and _TEXT_VALUE.match(text))
            ):
                ok = False
                break
            (dates if kind == "date" else values)[group] = text or None
        if ok:
            return values, dates
    # Undo merged cells (e.g. 'Feb/2026 33'): the tokens must line up 1:1 with columns.
    tokens = " ".join(clean_cell(cells[c]) for c in columns if c < len(cells)).split()
    if len(tokens) != len(columns):
        return None
    values, dates = [None] * 4, [None] * 4
    for col, token in zip(columns, tokens, strict=True):
        group, kind = mapping[col]
        if not (_DATE_CELL if kind == "date" else _VALUE).match(token):
            return None
        (dates if kind == "date" else values)[group] = token
    return values, dates


_SUBHEADER_WORDS = {
    "value",
    "date",
    "month/year",
    "name value",
    "result",
    "baseline",
    "closing period",
}


def _is_subheader(cells: list[str]) -> bool:
    labels = [c.lower() for c in cells[1:] if c]
    return bool(labels) and all(
        c in _SUBHEADER_WORDS or c.startswith(("actual (", "baseline")) for c in labels
    )


_COMMENT_LABEL = "comments on achieving"


def comment_cell(cells: list[str]) -> str | None:
    """Comment text of a WIDE 'Comments on achieving targets' row.

    Docling repeats a merged cell across the columns it spans, so the label itself can
    appear again after column 1. The comment is the merged text cell that follows the
    label. When the non-label cells differ, the paragraph was split into column
    fragments with interleaved words; it cannot be rebuilt reliably, so no comment is
    stored (NULL) rather than a fragment.
    """
    text = [c for c in cells if c and not c.lower().startswith(_COMMENT_LABEL)]
    return text[0] if text and len(set(text)) == 1 else None


@dataclass
class WideState:
    mapping: ColumnMap | None = None
    n_cols: int | None = None
    group: str | None = None


def parse_wide(
    doc: ParsedDocument, table: ParsedTable, state: WideState, layout: str
) -> list[RawIndicator]:
    """WIDE results rows and DLI rows, using the header map (carried across page breaks)."""
    out: list[RawIndicator] = []
    section = section_title(doc, table.section_id)
    kind = "DLI" if layout == "DLI" else (_indicator_type(section) or "UNKNOWN")
    rows = [[clean_cell(c) for c in r] for r in table.rows]
    found = _find_header(rows)
    if found:
        state.mapping, state.n_cols = found[1], table.n_cols
    if state.mapping is None or table.n_cols != state.n_cols:
        return out
    # Parse every row: header rows can also appear mid-table (next component's header),
    # and rows before them belong to the previous page's table.
    for index in range(len(rows)):
        cells = rows[index]
        if not any(cells):
            continue
        if _is_label_row(table.rows[index]):
            state.group = cells[0]
            continue
        if (
            _find_header(rows[index : index + 2])
            or cells[0].lower().rstrip(":") in ("indicator", "indicator name")
            or _is_subheader(cells)
        ):
            continue
        if layout == "DLI":
            if len(cells) < 2 or cells[1].lower() != "value":
                continue
            data = ["", *cells[1:]]
            data[1] = ""  # the 'Value' label column carries no value
        else:
            data = cells
            if len(cells) > 1 and cells[1].lower().startswith("comments on achieving"):
                if out and out[-1].name[:20] == cells[0][:20]:
                    out[-1].comments = comment_cell(cells[2:])
                continue
            if not any(cells[1:]):
                if out:  # a name fragment on its own row: the neighbouring name is incomplete
                    out[-1].problems.append("name continues on a separate row")
                continue
        unit, _ = split_unit(split_name_suffixes(cells[0])[0])
        parsed = _mapped_values(
            data, state.mapping, text_values=(unit or "").lower() in ("text", "yes/no")
        )
        raw = RawIndicator(
            name=cells[0],
            values=[None] * 4,
            dates=[None] * 4,
            page=table.page_number,
            method=ExtractionMethod.DOCLING_TABLE,
            layout="DLI" if layout == "DLI" else "WIDE",
            indicator_type=kind,
            result_area=state.group,
            table_id=table.table_id,
            row=index,
            section=section,
            source_text=" | ".join(cells),
        )
        if parsed is None:
            raw.problems.append("value/date cells do not match the header columns")
        else:
            raw.values, raw.dates = parsed
        out.append(raw)
    return out


def parse_block_table(doc: ParsedDocument, table: ParsedTable) -> list[RawIndicator]:
    """One BLOCK table -> indicators; flags anything it cannot validate."""
    section = section_title(doc, table.section_id)
    rows = [[clean_cell(c) for c in r] for r in table.rows]
    out: list[RawIndicator] = []
    # Docling misplaces IN-IDs into the neighbouring indicator's Value/Date rows, so
    # an ID is attributed only from the name row or the header rows that follow it.
    name, source_id = None, None
    for index, cells in enumerate(rows):
        joined = " ".join(cells)
        first = _ID.sub("", cells[0]).strip() if cells else ""
        on_value_row = first.lower().startswith(("value", "date"))
        found_id = None if on_value_row else _ID.search(joined)
        if _is_label_row(table.rows[index]) and not on_value_row:
            name = _ID.sub("", cells[0]).strip()
            source_id = found_id.group(1) if found_id else None
            continue
        if found_id and name:
            source_id = found_id.group(1)
        if any(c.lower().startswith("baseline") for c in cells[1:]) and first:
            name = name or first
            continue
        if first.lower().startswith("value"):
            tokens = [t for t in " ".join(cells[1:]).split() if t]
            extra = first[len("value") :].strip()
            raw = RawIndicator(
                name=name or "",
                values=[None] * 4,
                dates=[None] * 4,
                page=table.page_number,
                method=ExtractionMethod.DOCLING_TABLE,
                layout="BLOCK",
                source_id=source_id,
                indicator_type=_indicator_type(section),
                table_id=table.table_id,
                row=index,
                section=section,
                source_text=joined,
            )
            if extra or len(tokens) != 4 or not all(_VALUE.match(t) for t in tokens):
                raw.problems.append("Value row does not hold exactly four values")
            else:
                raw.values = [None if t == "--" else t for t in tokens]
            if not name or split_unit(name) == (None, None):
                raw.problems.append("indicator name missing or truncated (no unit suffix)")
            elif _AREA_LABEL.match(strip_glyphs(name)):
                raw.problems.append("result-area label taken as indicator name")
            out.append(raw)
        elif first.lower().startswith("date") and out:
            tokens = " ".join(cells[1:]).split()
            if len(tokens) == 4 and all(_DATE_CELL.match(t) for t in tokens):
                out[-1].dates = [None if t == "--" else t for t in tokens]
            else:
                out[-1].problems.append("Date row does not hold exactly four dates")
            name, source_id = None, None
    return out


# ---------------------------------------------------------------------------
# PDF text fallback (BLOCK grammar)
# ---------------------------------------------------------------------------


def parse_block_text(pages: dict[int, list[str]], initial_type: str | None) -> list[RawIndicator]:
    """Parse the BLOCK grammar from PDF text lines spanning several pages."""
    out: list[RawIndicator] = []
    kind, area = initial_type, None
    name_lines: list[str] = []
    source_id = None
    pending: RawIndicator | None = None
    comment: list[str] | None = None
    lines = [
        (p, strip_glyphs(ln) if not _NAME_START.match(ln) else ln)
        for p in sorted(pages)
        for ln in pages[p]
        if not _TEXT_RUNNING.search(ln)
    ]

    def close_comment():
        nonlocal comment
        if comment is not None and out:
            out[-1].comments = " ".join(comment).strip() or None
        comment = None

    for page, line in lines:
        text = strip_glyphs(line)
        if _AREA_LABEL.match(text):
            close_comment()
            area, name_lines = text, []
            continue
        if _PDO_HEADING.search(text):
            kind, area = "PDO", None
            continue
        if _IR_HEADING.search(text):
            kind, area = "INTERMEDIATE", None
            continue
        id_match = _ID.search(text)
        if id_match:
            before = text[: id_match.start()].strip()
            if comment is not None and before:
                comment.append(before)
            close_comment()
            source_id, name_lines = id_match.group(1), []
            rest = text[id_match.end() :].strip()
            if rest:
                name_lines.append(rest)
            continue
        if _NAME_START.match(line) or (
            name_lines and not _BLOCK_HEADER.match(text) and pending is None and comment is None
        ):
            if _NAME_START.match(line):
                close_comment()
                name_lines = []
            name_lines.append(text.lstrip("► ").strip())
            continue
        if _BLOCK_HEADER.match(text):
            pending = RawIndicator(
                name=" ".join(name_lines).strip(),
                values=[None] * 4,
                dates=[None] * 4,
                page=page,
                method=ExtractionMethod.PDF_TEXT_FALLBACK,
                layout="BLOCK",
                source_id=source_id,
                indicator_type=kind,
                result_area=area,
                source_text=" ".join(name_lines)[:200],
            )
            name_lines, source_id = [], None
            continue
        if pending is not None and text.lower().startswith("value"):
            tokens = text.split()[1:]
            pending.page = page
            if len(tokens) == 4 and all(_VALUE.match(t) for t in tokens):
                pending.values = [None if t == "--" else t for t in tokens]
            else:
                pending.problems.append(f"text Value line has {len(tokens)} tokens")
            pending.source_text = f"{pending.name[:120]} | {text}"
            continue
        if pending is not None and text.lower().startswith("date"):
            tokens = text.split()[1:]
            if len(tokens) == 4 and all(_DATE_CELL.match(t) for t in tokens):
                pending.dates = [None if t == "--" else t for t in tokens]
            else:
                pending.problems.append(f"text Date line has {len(tokens)} tokens")
            if not pending.name or split_unit(pending.name) == (None, None):
                pending.problems.append("indicator name without unit suffix in text layer")
            out.append(pending)
            pending, comment = None, []
            continue
        if comment is not None:
            if text.lower().startswith("comments"):
                text = text[len("comments") :].strip(" :")
            if text.lower().startswith("overall comments"):
                close_comment()
                continue
            if text:
                comment.append(text)
            continue
        if pending is None and not name_lines and text and not _VALUE.match(text):
            area = text  # result area / component label line
    close_comment()
    return out


# ---------------------------------------------------------------------------
# Orchestration per ISR
# ---------------------------------------------------------------------------


_RESULTS_SECTION = re.compile(r"results|indicator", re.I)
_END_OF_RESULTS = re.compile(
    r"financial performance|disbursements|key dates|restructuring history", re.I
)


def _results_pages(doc: ParsedDocument) -> list[int]:
    """Pages from the first results section up to the next non-results section."""
    pages: set[int] = set()
    inside = False
    for section in doc.sections:
        if _RESULTS_SECTION.search(section.title) and not _END_OF_RESULTS.search(section.title):
            inside = True
        elif inside and _END_OF_RESULTS.search(section.title):
            inside = False
        if inside:
            pages.update(range(section.start_page, section.end_page + 1))
    return sorted(pages)


def extract_results(
    doc: ParsedDocument,
    text: LoggedTextSource,
    observation_date,
) -> tuple[list[ResultObservation], list[ExtractionIssue], list[int]]:
    """Returns (observations, issues, pages that used the PDF text fallback)."""
    issues: list[ExtractionIssue] = []
    by_page: dict[int, list[RawIndicator]] = {}
    wide, dli = WideState(), WideState()
    results_pages = set(_results_pages(doc))
    orphan_pages: set[int] = set()
    for table in doc.tables:
        kind = classify(table)
        if (
            kind is None
            and table.page_number in results_pages
            and any(clean_cell(r[0]).lower().startswith(("value", "date")) for r in table.rows if r)
            and not (wide.mapping and table.n_cols == wide.n_cols)
        ):
            # Value/Date rows in a table Docling split from its header: the block is broken.
            orphan_pages.add(table.page_number)
            continue
        continuation = kind is None and not is_loan_table(table)
        if kind == "BLOCK":
            parsed = parse_block_table(doc, table)
        elif kind == "DLI" or (continuation and dli.mapping and table.n_cols == dli.n_cols):
            parsed = parse_wide(doc, table, dli, "DLI")
        elif kind == "WIDE" or (continuation and wide.mapping and table.n_cols == wide.n_cols):
            parsed = parse_wide(doc, table, wide, "WIDE")  # also header-less continuations
        else:
            continue
        by_page.setdefault(table.page_number, []).extend(parsed)

    coverage = {p.page_number: p.pdf_text_coverage for p in doc.pages}
    fallback_pages = sorted(
        page
        for page, items in by_page.items()
        if any(i.layout == "BLOCK" and not i.valid for i in items)
        or (
            coverage.get(page) is not None
            and coverage[page] < LOW_COVERAGE
            and any(i.layout == "BLOCK" for i in items)
        )
    )
    if any(i.layout == "BLOCK" for items in by_page.values() for i in items):
        # Block-layout ISR: result pages where Docling produced no results table at all,
        # or split a block from its header, are also read from the text layer.
        empty = [p for p in results_pages if not by_page.get(p)]
        fallback_pages = sorted(set(fallback_pages) | set(empty) | orphan_pages)

    final: list[RawIndicator] = [
        i for page, items in by_page.items() if page not in fallback_pages for i in items
    ]
    if fallback_pages:
        span = range(min(fallback_pages) - 1, max(fallback_pages) + 2)
        pages = {}
        for p in span:
            if 1 <= p <= doc.page_count:
                reason = "BLOCK results table failed validation or low coverage"
                lines = text.page_lines(p, element="results", reason=reason)
                if lines:
                    pages[p] = lines
        first_type = next(
            (i.indicator_type for i in by_page.get(fallback_pages[0], []) if i.indicator_type), None
        )
        from_text = [i for i in parse_block_text(pages, first_type) if i.page in fallback_pages]
        docling_valid = {
            normalize_indicator_name(i.name): i
            for p in fallback_pages
            for i in by_page.get(p, [])
            if i.valid
        }
        for item in from_text:
            other = docling_valid.get(normalize_indicator_name(item.name))
            if other and (other.values != item.values or other.dates != item.dates):
                item.problems.append("Docling table and PDF text disagree")
                item.values, item.dates = [None] * 4, [None] * 4
                issues.append(
                    issue(
                        CheckCode.EXTRACTION_CONFLICT,
                        "WARNING",
                        f"results: Docling and PDF text disagree for "
                        f"{item.name[:80]!r}; values left NULL",
                        evidence(
                            doc,
                            item.page,
                            ExtractionMethod.PDF_TEXT_FALLBACK,
                            text=item.source_text,
                        ),
                    )
                )
            final.append(item)
        issues.append(
            issue(
                CheckCode.PDF_TEXT_FALLBACK_USED,
                "INFO",
                f"results: PDF text fallback for pages {fallback_pages}",
                evidence(doc, fallback_pages[0], ExtractionMethod.PDF_TEXT_FALLBACK),
                pages=fallback_pages,
            )
        )
    completed = _complete_wide_names(doc, text, [i for i in final if i.layout == "WIDE"])
    if completed:
        issues.append(
            issue(
                CheckCode.PDF_TEXT_FALLBACK_USED,
                "INFO",
                f"results: {completed} wide-layout indicator name(s) completed from "
                "the PDF text layer",
                evidence(doc, final[0].page, ExtractionMethod.PDF_TEXT_FALLBACK),
            )
        )
    wide_problem_pages = sorted({i.page for i in final if i.layout != "BLOCK" and not i.valid})
    for page in wide_problem_pages:
        issues.append(
            issue(
                CheckCode.RESULTS_TABLE_INVALID,
                "WARNING",
                f"results: {sum(not i.valid for i in by_page[page])} wide/DLI row(s) "
                f"on page {page} could not be validated; values left NULL",
                evidence(doc, page, ExtractionMethod.DOCLING_TABLE),
            )
        )

    return (
        [_observation(doc, raw, observation_date) for raw in final if raw.name],
        issues,
        (fallback_pages),
    )


_NAME_PROBLEM = "name continues on a separate row"
_UNIT_CLOSE = re.compile(
    r"\((?:Number|Percentage|Text|Yes/No|Amount|Kilometers|Km|MLD|"
    r"Hectare|Hectares|Tons?|%)[^()]*\)",
    re.I,
)


def _complete_wide_names(
    doc: ParsedDocument, text: LoggedTextSource, items: list[RawIndicator]
) -> int:
    """Complete wrapped wide-layout names from the PDF text layer of the same page.

    Starting at the line containing the Docling fragment, lines are joined (value and
    date tokens removed) until a unit parenthetical closes, within 4 lines. Only a name
    problem is repaired; value problems are left as they are.
    """
    count = 0
    for item in items:
        if item.problems != [_NAME_PROBLEM]:
            continue
        lines = (
            text.page_lines(
                item.page,
                element=f"results name {item.name[:30]}",
                reason="wide-layout name truncated",
            )
            or []
        )
        fragment = " ".join(item.name.split())[:25]
        start = next((i for i, ln in enumerate(lines) if fragment and fragment in ln), None)
        if start is None:
            continue
        words: list[str] = []
        for line in lines[start : start + 4]:
            kept = [w for w in line.split() if not (_VALUE.match(w) or (_DATE_CELL.match(w) and w))]
            words.extend(kept)
            candidate = " ".join(words)
            match = _UNIT_CLOSE.search(candidate)
            if match:
                name = candidate[: match.end()]
                name = re.split(r"\s+Comments on", name)[0]
                item.name_source = (item.page, " | ".join(lines[start : start + 4]))
                item.name = name
                item.problems = []
                count += 1
                break
    return count


def _observation(doc: ParsedDocument, raw: RawIndicator, observation_date) -> ResultObservation:
    ref = evidence(
        doc,
        raw.page,
        raw.method,
        section=raw.section,
        table_id=raw.table_id,
        row=raw.row,
        text=raw.source_text,
    )
    problems = [issue(CheckCode.EXTRACTION_AMBIGUOUS, "WARNING", p, ref) for p in raw.problems]
    if raw.problems:
        values, dates = (
            ([None] * 4, [None] * 4)
            if any(
                "values" in p or "cells" in p or "disagree" in p or "tokens" in p
                for p in raw.problems
            )
            else (raw.values, raw.dates)
        )
        status = ExtractionStatus.AMBIGUOUS
    else:
        values, dates, status = raw.values, raw.dates, ExtractionStatus.EXACT
    name = " ".join(strip_glyphs(raw.name).replace("►", " ").split())
    name, tags, reported_status = split_name_suffixes(name)
    unit, measure_type = split_unit(name)
    return ResultObservation(
        project_id=doc.project_id,
        indicator_key="",  # assigned by identity resolution across ISRs
        indicator_id_source=raw.source_id,
        indicator_name_raw=name,
        indicator_name_normalized=normalize_indicator_name(name),
        indicator_type=raw.indicator_type,
        indicator_tags=tags,
        reported_status=reported_status,
        result_area=raw.result_area,
        unit=unit if unit and measure_type else unit,
        baseline_value=values[0],
        baseline_date=dates[0],
        previous_value=values[1],
        previous_date=dates[1],
        current_value=values[2],
        current_date=dates[2],
        target_value=values[3],
        target_date=dates[3],
        comments=raw.comments,
        isr_sequence=doc.isr_sequence,
        observation_date=observation_date,
        source_document=doc.filename,
        source_page=raw.page,
        source_table=raw.table_id,
        layout=raw.layout,
        extraction_method=raw.method,
        status=status,
        source_ref=ref,
        name_source_ref=(
            evidence(
                doc, raw.name_source[0], ExtractionMethod.PDF_TEXT_FALLBACK, text=raw.name_source[1]
            )
            if raw.name_source
            else None
        ),
        quality_issues=problems,
    )
