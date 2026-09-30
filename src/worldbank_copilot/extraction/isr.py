"""ISR snapshot extraction (silver.isr_snapshots): one row per ISR.

Each element is taken from the Docling table first and validated. If validation
fails, the PDF text layer of that page is used (logged, method
PDF_TEXT_FALLBACK). Ratings are only ever read where explicitly printed; nothing
is inferred from narrative.

ISR date policy (canonical_report_date):
* header/report date printed in the ISR running header, when identified;
* otherwise the "Archived on" date.
Both dates are always kept; a difference is reported as ISR_DATE_DIFFERENCE
(INFO up to 30 days, WARNING above).
"""

from __future__ import annotations

import re
from decimal import Decimal

from worldbank_copilot.common.quality import CheckCode
from worldbank_copilot.extraction.models import (
    ExtractedRating,
    IsrSnapshot,
    LoanDisbursement,
    LoanKeyDates,
    NarrativeSection,
    SortRating,
)
from worldbank_copilot.extraction.provenance import (
    EvidenceRef,
    ExtractionIssue,
    ExtractionMethod,
    ExtractionStatus,
    evidence,
    issue,
)
from worldbank_copilot.extraction.ratings import (
    RATING_PATTERN,
    RatingScale,
    normalize_rating,
    strip_glyphs,
)
from worldbank_copilot.extraction.tables import (
    DMY,
    clean_cell,
    dates_in,
    parse_dmy,
    parse_number,
    section_text,
    section_title,
    sections_matching,
    statement_loan_number,
    table_blob,
)
from worldbank_copilot.extraction.text_source import LoggedTextSource
from worldbank_copilot.parsing.models import ParsedDocument, ParsedTable

MATERIAL_DATE_DIFFERENCE_DAYS = 30

_RATING_ROWS = {
    "pdo": ("progress towards achievement", RatingScale.PERFORMANCE),
    "ip": ("overall implementation progress", RatingScale.PERFORMANCE),
    "risk": ("overall risk rating", RatingScale.RISK),
}
_TEXT_RATING_LINES = {
    "pdo": re.compile(
        rf"^Progress towards achievement of PDO\s+({RATING_PATTERN})"
        rf"(?:\s+({RATING_PATTERN}))?\s*$",
        re.I,
    ),
    "ip": re.compile(
        rf"^Overall Implementation Progress(?: \(IP\))?\s+({RATING_PATTERN})"
        rf"(?:\s+({RATING_PATTERN}))?\s*$",
        re.I,
    ),
    "risk": re.compile(
        rf"^Overall Risk Rating\s+({RATING_PATTERN})"
        rf"(?:\s+({RATING_PATTERN}))?\s*$",
        re.I,
    ),
}
_RESTRUCTURING_LINE = re.compile(rf"Restructuring Level \d+ Approved on\s*({DMY})", re.I)
_NARRATIVE = re.compile(
    r"implementation status|key issues|key decisions|^risks?$|overall comments", re.I
)
_CURRENCY = re.compile(r"^[A-Z]{3}$")


class IsrContext:
    def __init__(self, doc: ParsedDocument, text: LoggedTextSource):
        self.doc = doc
        self.text = text
        self.issues: list[ExtractionIssue] = []
        self.refs: list[EvidenceRef] = []

    def ev(self, page: int, method: ExtractionMethod, **kw) -> EvidenceRef:
        ref = evidence(self.doc, page, method, **kw)
        self.refs.append(ref)
        return ref

    def fallback_lines(self, page: int, element: str, reason: str) -> list[str]:
        lines = self.text.page_lines(page, element=element, reason=reason) or []
        self.issues.append(
            issue(
                CheckCode.PDF_TEXT_FALLBACK_USED,
                "INFO",
                f"{element}: PDF text fallback on page {page} ({reason})",
                evidence(self.doc, page, ExtractionMethod.PDF_TEXT_FALLBACK),
            )
        )
        return lines


def _rating(
    raw: str | None, scale: RatingScale, ref: EvidenceRef, ctx: IsrContext, label: str
) -> ExtractedRating:
    normalized, status = normalize_rating(raw, scale)
    if status is ExtractionStatus.AMBIGUOUS:
        ctx.issues.append(
            issue(
                CheckCode.UNKNOWN_RATING_VALUE,
                "WARNING",
                f"{label}: unrecognised rating {raw!r} kept raw",
                ref,
            )
        )
    return ExtractedRating(
        raw_rating=clean_cell(raw) or None,
        normalized_rating=normalized,
        status=status,
        evidence=ref,
    )


def _find_ratings_table(doc: ParsedDocument) -> ParsedTable | None:
    for table in doc.tables:
        if any(
            clean_cell(r[0]).lower().startswith("progress towards achievement")
            for r in table.rows
            if r
        ):
            return table
    return None


def _column(header: list[str], *words: str) -> int | None:
    for index, cell in enumerate(header):
        text = clean_cell(cell).lower()
        if any(w in text for w in words):
            return index
    return None


def extract_ratings(ctx: IsrContext) -> dict[str, tuple[ExtractedRating | None, ...]]:
    """Returns {key: (previous, current)} for pdo / ip / risk (risk may be missing)."""
    doc, found = ctx.doc, {}
    table = _find_ratings_table(doc)
    if table is not None:
        header = next(
            (r for r in table.rows if any("current" in clean_cell(c).lower() for c in r)), []
        )
        prev_col, cur_col = _column(header, "previous"), _column(header, "current")
        for index, row in enumerate(table.rows):
            name = clean_cell(row[0]).lower() if row else ""
            for key, (prefix, scale) in _RATING_ROWS.items():
                if name.startswith(prefix) and cur_col is not None and cur_col < len(row):

                    def ref(col, _row=row, _index=index):  # noqa: E306
                        return ctx.ev(
                            table.page_number,
                            ExtractionMethod.DOCLING_TABLE,
                            section=section_title(doc, table.section_id),
                            table_id=table.table_id,
                            row=_index,
                            column=col,
                            text=" | ".join(_row),
                        )

                    prev = (
                        _rating(row[prev_col], scale, ref(prev_col), ctx, f"{key} previous")
                        if prev_col is not None and prev_col < len(row)
                        else None
                    )
                    cur = _rating(row[cur_col], scale, ref(cur_col), ctx, f"{key} current")
                    found[key] = (prev, cur)
    valid = all(k in found and found[k][1].normalized_rating for k in ("pdo", "ip"))
    if not valid:
        page = table.page_number if table else _ratings_page(doc)
        reason = "no ratings table" if table is None else "ratings table incomplete"
        # The ratings block can continue onto the next page (e.g. P130544 ISR 21).
        lines = [strip_glyphs(ln) for ln in ctx.fallback_lines(page, "ratings", reason)]
        _ratings_from_label_block(ctx, lines, page, found)
        if not all(k in found and found[k][1].normalized_rating for k in ("pdo", "ip")):
            following = [
                strip_glyphs(ln)
                for ln in ctx.fallback_lines(page + 1, "ratings", "block continues")
            ]
            _ratings_from_label_block(ctx, following, page + 1, found)
            lines = lines + following
        for key, pattern in _TEXT_RATING_LINES.items():
            if key in found and found[key][1].normalized_rating:
                continue
            for line in lines:
                match = pattern.match(line)
                if not match:
                    continue
                ref = ctx.ev(page, ExtractionMethod.PDF_TEXT_FALLBACK, text=line)
                scale = _RATING_ROWS[key][1]
                if match.group(2) is None:
                    ctx.issues.append(
                        issue(
                            CheckCode.EXTRACTION_AMBIGUOUS,
                            "WARNING",
                            f"{key}: one rating printed; previous/current cannot be distinguished",
                            ref,
                        )
                    )
                    break
                found[key] = (
                    _rating(match.group(1), scale, ref, ctx, f"{key} previous"),
                    _rating(match.group(2), scale, ref, ctx, f"{key} current"),
                )
                break
    return found


_LABEL_KEYS = (
    ("progress towards achievement", "pdo"),
    ("overall implementation progress", "ip"),
    ("overall risk rating", "risk"),
)
_RATING_ONLY = re.compile(rf"^({RATING_PATTERN})$", re.I)


def _ratings_from_label_block(ctx: IsrContext, lines: list[str], page: int, found: dict):
    """Layout where the text layer prints the rating labels, then all values.

    With L labels and 2L values the two readings are column-major (all previous,
    then all current) and row-major (previous/current per label). A reading is
    accepted only if it is scale-valid (performance for PDO/IP, risk for overall
    risk) and the other reading is invalid or identical; otherwise AMBIGUOUS.
    """
    keys: list[str] = []
    start = None
    for index, line in enumerate(lines):
        lowered = line.lower()
        key = next(
            (
                k
                for prefix, k in _LABEL_KEYS
                if lowered == prefix
                or lowered.startswith(prefix)
                and not _TEXT_RATING_LINES[k].match(line)
                and len(line) < 45
            ),
            None,
        )
        if key and (start is None or index == start + len(keys)):
            start = index if start is None else start
            keys.append(key)
    if not keys:
        return
    values = []
    for line in lines[start + len(keys) :]:
        if _RATING_ONLY.match(line):
            values.append(line)
        elif values:
            break
    if len(values) != 2 * len(keys):
        return

    def reading(pairs):
        out = {}
        for key, (prev, cur) in zip(keys, pairs, strict=True):
            scale = _RATING_ROWS[key][1]
            for raw in (prev, cur):
                if normalize_rating(raw, scale)[1] is ExtractionStatus.AMBIGUOUS:
                    return None
            out[key] = (prev, cur)
        return out

    n = len(keys)
    column_major = reading([(values[i], values[n + i]) for i in range(n)])
    row_major = reading([(values[2 * i], values[2 * i + 1]) for i in range(n)])
    chosen = (
        column_major
        if row_major is None or row_major == column_major
        else (row_major if column_major is None else None)
    )
    text = " | ".join(lines[start : start + 3 * n])
    ref = ctx.ev(page, ExtractionMethod.PDF_TEXT_FALLBACK, text=text)
    if chosen is None:
        ctx.issues.append(
            issue(
                CheckCode.EXTRACTION_AMBIGUOUS,
                "WARNING",
                "rating labels and values printed separately; reading order is ambiguous",
                ref,
            )
        )
        return
    for key, (prev, cur) in chosen.items():
        if key in found and found[key][1].normalized_rating:
            continue
        scale = _RATING_ROWS[key][1]
        found[key] = (
            _rating(prev, scale, ref, ctx, f"{key} previous"),
            _rating(cur, scale, ref, ctx, f"{key} current"),
        )


def _ratings_page(doc: ParsedDocument) -> int:
    sections = sections_matching(doc, r"overall ratings")
    return sections[0].start_page if sections else 1


def extract_sort(ctx: IsrContext) -> list[SortRating]:
    """SORT table: per-category rating at approval / previous / current."""
    doc, results = ctx.doc, []
    active = None  # (n_cols, approval, previous, current) of the last SORT header seen
    for table in doc.tables:
        if "risk category" in table_blob(table, 1).lower():
            header = table.rows[0]
            active = (
                table.n_cols,
                _column(header, "approval"),
                _column(header, "previous", "last approved"),
                _column(header, "current", "proposed"),
            )
            body = list(enumerate(table.rows[1:], start=1))
        elif (
            active
            and table.n_cols == active[0]
            and table.rows
            and any(normalize_rating(c, RatingScale.RISK)[0] for c in table.rows[0][1:])
        ):
            body = list(enumerate(table.rows))  # continuation of a SORT split by a page break
        else:
            continue
        _, approval, previous, current = active
        reached_overall = False
        for index, row in body:
            category = re.sub(r"^\d+\.\s*", "", clean_cell(row[0])) if row else ""
            if not category:
                continue
            reached_overall = reached_overall or category.lower() == "overall"

            def rating(col, label, _row=row, _index=index, _table=table, _category=category):
                if col is None or col >= len(_row):
                    return None
                ref = ctx.ev(
                    _table.page_number,
                    ExtractionMethod.DOCLING_TABLE,
                    section=section_title(doc, _table.section_id),
                    table_id=_table.table_id,
                    row=_index,
                    column=col,
                    text=" | ".join(_row),
                )
                return _rating(_row[col], RatingScale.RISK, ref, ctx, f"SORT {_category} {label}")

            results.append(
                SortRating(
                    risk_category=category,
                    rating_at_approval=rating(approval, "approval"),
                    previous_rating=rating(previous, "previous"),
                    current_rating=rating(current, "current"),
                )
            )
        if reached_overall:
            active = None
    results = _dedupe(results, lambda s: s.risk_category.lower())
    if not results:
        results = _sort_from_text(ctx)
    return results


_RUNNING_HEADER = re.compile(
    r"\(P\d{6}\)|Page \d+ of \d+|Implementation Status & Results Report|^The World Bank$"
)
_SORT_LINE = re.compile(
    rf"^(?P<category>.+?)\s+(?P<ratings>(?:(?:{RATING_PATTERN})\s*){{3}})$", re.I
)


def _sort_from_text(ctx: IsrContext) -> list[SortRating]:
    """SORT printed as text lines: 'Category <approval> <previous> <current>'."""
    sections = sections_matching(ctx.doc, r"risk-?\s?rating tool|^risks?$")
    if not sections:
        return []
    page = sections[0].start_page
    lines = [
        strip_glyphs(ln)
        for p in (page, page + 1)  # the SORT can cross a page break
        for ln in ctx.fallback_lines(p, "SORT", "no SORT table")
        if not _RUNNING_HEADER.search(ln)
    ]
    start = next((i for i, ln in enumerate(lines) if ln.lower().startswith("risk category")), None)
    if start is None:
        return []
    results, carry = [], ""
    for line in lines[start + 1 :]:
        match = _SORT_LINE.match(line)
        if not match:
            carry = f"{carry} {line}".strip()
            if len(carry) > 120:
                break
            continue
        category = f"{carry} {match.group('category')}".strip()
        carry = ""
        ratings = re.findall(RATING_PATTERN, match.group("ratings"), re.I)
        ref = ctx.ev(page, ExtractionMethod.PDF_TEXT_FALLBACK, text=line)
        approval, previous, current = (
            _rating(r, RatingScale.RISK, ref, ctx, f"SORT {category}") for r in ratings
        )
        results.append(
            SortRating(
                risk_category=category,
                rating_at_approval=approval,
                previous_rating=previous,
                current_rating=current,
            )
        )
        if category.lower() == "overall":
            break
    return results


def _loan_financial(tokens: list[str], historical: bool) -> dict | None:
    """Parse '<status...> [CUR] original revised cancelled disbursed undisbursed ...' tokens.

    Anchored on the '% Disbursed' token: exactly 5 amounts precede it, or 6 when the
    header has a 'Historical Disbursed' column printed before the percentage (kept as
    printed in ``historical_disbursed_musd``). Anything else is not parsed.
    """
    status, currency, numbers = [], None, []
    for token in tokens:
        if token == "%":
            if numbers:
                numbers[-1] = numbers[-1] + "%"
        elif parse_number(token) is not None:
            numbers.append(token)
        elif not numbers and _CURRENCY.match(token):
            currency = token
        elif not numbers:
            status.append(token)
    pct_index = next((i for i, n in enumerate(numbers) if n.endswith("%")), None)
    if pct_index is None:
        return None
    before = numbers[:pct_index]
    historical_value = None
    if len(before) == 6 and historical:
        before, historical_value = before[:5], parse_number(before[5])
    if len(before) != 5:
        return None
    values = [parse_number(n) for n in before]
    return {
        "historical_disbursed_musd": historical_value,
        "loan_status": " ".join(status) or None,
        "currency": currency,
        "original_musd": values[0],
        "revised_musd": values[1],
        "cancelled_musd": values[2],
        "disbursed_musd": values[3],
        "undisbursed_musd": values[4],
        "disbursed_pct_reported": parse_number(numbers[pct_index]),
    }


def extract_disbursements(ctx: IsrContext) -> list[LoanDisbursement]:
    doc, results = ctx.doc, []
    for table in doc.tables:
        blob = table_blob(table, 1).lower()
        if "loan/credit/tf" not in blob or "disbursed" not in blob:
            continue
        historical = "historical" in blob
        for index, row in enumerate(table.rows):
            joined = " ".join(clean_cell(c) for c in row)
            loan = statement_loan_number(joined)
            if not loan or dates_in(joined):
                continue
            tokens = joined.split()
            start = next(i for i, t in enumerate(tokens) if t.startswith(("IBRD", "IDA", "TF")))
            parsed = _loan_financial(tokens[start + 1 :], historical)
            method, ref_text, page = ExtractionMethod.DOCLING_TABLE, joined, table.page_number
            if parsed is None:
                lines = ctx.fallback_lines(page, f"disbursement {loan}", "row did not parse")
                line = next(
                    (ln for ln in lines if statement_loan_number(ln) == loan and not dates_in(ln)),
                    None,
                )
                if line:
                    parsed = _loan_financial(line.split()[1:], historical)
                    method, ref_text = ExtractionMethod.PDF_TEXT_FALLBACK, line
            ref = ctx.ev(
                page,
                method,
                section=section_title(doc, table.section_id),
                table_id=table.table_id,
                row=index,
                text=ref_text,
            )
            if parsed is None:
                ctx.issues.append(
                    issue(
                        CheckCode.EXTRACTION_AMBIGUOUS,
                        "WARNING",
                        f"disbursement line for {loan} could not be parsed",
                        ref,
                    )
                )
                continue
            results.append(
                LoanDisbursement(
                    loan_number=loan, status=ExtractionStatus.EXACT, evidence=ref, **parsed
                )
            )
    return _dedupe(results, lambda d: d.loan_number)


_KEY_DATE_FIELDS = (
    ("approval", "approval_date"),
    ("signing", "signing_date"),
    ("effectiveness", "effectiveness_date"),
    ("orig", "original_closing_date"),
    ("rev", "revised_closing_date"),
)
_WRAPPED_HYPHEN = re.compile(r"(?<=[A-Za-z0-9])- (?=\d)")
_DATE_OR_PLACEHOLDER = re.compile(rf"^(?:{DMY}|--)$")


def _key_dates(values: list, loan: str, ref: EvidenceRef) -> LoanKeyDates:
    fields = {name: value for (_, name), value in zip(_KEY_DATE_FIELDS, values, strict=True)}
    return LoanKeyDates(loan_number=loan, status=ExtractionStatus.EXACT, evidence=ref, **fields)


def _header_columns(header: list[str]) -> list[int] | None:
    """Column index per key-date field, only if every header cell names one field."""
    columns = []
    for keyword, _ in _KEY_DATE_FIELDS:
        hits = [i for i, cell in enumerate(header) if keyword in clean_cell(cell).lower()]
        if len(hits) != 1:
            return None
        columns.append(hits[0])
    labels = [sum(k in clean_cell(header[c]).lower() for k, _ in _KEY_DATE_FIELDS) for c in columns]
    return columns if all(n == 1 for n in labels) else None


def _placeholder_tokens(text: str, loan: str) -> list[str] | None:
    """Date/'--' tokens after the loan id and status, e.g. from a PDF text line."""
    # Re-join wrapped tokens ('22-Nov- 2024', 'IBRD-86010- 001'), never '--' placeholders.
    tokens = _WRAPPED_HYPHEN.sub("-", text).split()
    start = next((i for i, t in enumerate(tokens) if statement_loan_number(t) == loan), None)
    if start is None:
        return None
    values = [t for t in tokens[start + 1 :] if _DATE_OR_PLACEHOLDER.match(t)]
    return values if len(values) == 5 else None


def extract_key_dates(ctx: IsrContext) -> list[LoanKeyDates]:
    doc, results, failed = ctx.doc, [], []
    header: list[str] | None = None
    for table in doc.tables:
        for index, row in enumerate(table.rows):
            joined = " ".join(clean_cell(c) for c in row)
            lowered = joined.lower()
            if "approval" in lowered and "closing" in lowered and not statement_loan_number(joined):
                header = row
                continue
            loan = statement_loan_number(joined)
            if not loan or len(dates_in(joined)) < 2:
                continue
            ref = ctx.ev(
                table.page_number,
                ExtractionMethod.DOCLING_TABLE,
                section=section_title(doc, table.section_id),
                table_id=table.table_id,
                row=index,
                text=joined,
            )
            columns = _header_columns(header) if header and len(header) == len(row) else None
            if columns is not None:
                cells = [clean_cell(row[c]) for c in columns]
                # A date in an unmapped column means the header split a label across
                # cells (e.g. 'Orig.' | 'Closing'): positions cannot be trusted.
                stray = any(parse_dmy(clean_cell(v)) for i, v in enumerate(row) if i not in columns)
                if not stray and all(not v or v == "--" or parse_dmy(v) for v in cells):
                    results.append(_key_dates([parse_dmy(v) for v in cells], loan, ref))
                    continue
            failed.append((loan, table.page_number))
    for loan, page in failed:
        lines = ctx.fallback_lines(page, f"key dates {loan}", "table columns not unambiguous")
        line = next(
            (
                ln
                for ln in lines
                if statement_loan_number(ln) == loan and _placeholder_tokens(ln, loan)
            ),
            None,
        )
        ref = ctx.ev(page, ExtractionMethod.PDF_TEXT_FALLBACK, text=line)
        if line is None:
            ctx.issues.append(
                issue(
                    CheckCode.EXTRACTION_AMBIGUOUS,
                    "WARNING",
                    f"key dates for {loan} could not be mapped to columns",
                    ref,
                )
            )
            continue
        values = [parse_dmy(t) for t in _placeholder_tokens(line, loan)]
        results.append(_key_dates(values, loan, ref))
    if not results and not failed:
        sections = sections_matching(doc, r"key dates \(by loan\)")
        if sections:
            page = sections[0].start_page
            for line in ctx.fallback_lines(page, "key dates by loan", "not parsed as a table"):
                loan = statement_loan_number(line)
                tokens = _placeholder_tokens(line, loan) if loan else None
                if tokens:
                    ref = ctx.ev(page, ExtractionMethod.PDF_TEXT_FALLBACK, text=line)
                    results.append(_key_dates([parse_dmy(t) for t in tokens], loan, ref))
    return _dedupe(results, lambda d: d.loan_number)


def _dedupe(items, key):
    seen, out = set(), []
    for item in items:
        if key(item) not in seen:
            seen.add(key(item))
            out.append(item)
    return out


def extract_narratives(ctx: IsrContext) -> list[NarrativeSection]:
    out = []
    for section in ctx.doc.sections:
        if not _NARRATIVE.search(section.title.strip()):
            continue
        text = section_text(ctx.doc, section)
        if not text.strip():
            continue
        ref = ctx.ev(
            section.start_page,
            ExtractionMethod.DOCLING_TEXT,
            section=section.title,
            block_id=section.heading_block_id,
            text=text,
        )
        out.append(
            NarrativeSection(
                title=section.title,
                text=text,
                start_page=section.start_page,
                end_page=section.end_page,
                evidence=ref,
            )
        )
    return out


def extract_restructuring_history(ctx: IsrContext) -> list[EvidenceRef]:
    refs = []
    for section in sections_matching(ctx.doc, r"restructuring history"):
        blocks = {b.block_id: b for b in ctx.doc.blocks}
        for block_id in section.block_ids:
            block = blocks.get(block_id)
            if block and _RESTRUCTURING_LINE.search(block.text_clean):
                refs.append(
                    ctx.ev(
                        block.page_number,
                        ExtractionMethod.DOCLING_TEXT,
                        section=section.title,
                        block_id=block_id,
                        text=block.text_clean,
                    )
                )
    return refs


def restructuring_dates(snapshot: IsrSnapshot) -> list[tuple]:
    out = []
    for ref in snapshot.restructuring_history:
        match = _RESTRUCTURING_LINE.search(ref.source_text or "")
        found = dates_in(match.group(1)) if match else []
        if found:
            out.append((found[0], ref))
    return out


def _named_text(narratives: list[NarrativeSection], pattern: str, exclude: str | None = None):
    regex = re.compile(pattern, re.I)
    for n in narratives:
        if regex.search(n.title) and not (exclude and re.search(exclude, n.title, re.I)):
            return n.text
    return None


def extract_isr_snapshot(doc: ParsedDocument, text: LoggedTextSource) -> IsrSnapshot:
    ctx = IsrContext(doc, text)
    header, archive = doc.report_date, doc.archive_date
    canonical, basis = (header, "header_date") if header else (archive, "archive_date")
    difference = (header - archive).days if header and archive else None
    if difference:
        severity = "WARNING" if abs(difference) > MATERIAL_DATE_DIFFERENCE_DAYS else "INFO"
        ctx.issues.append(
            issue(
                CheckCode.ISR_DATE_DIFFERENCE,
                severity,
                f"header date {header} vs archive date {archive} ({difference:+d} days); both kept",
                evidence(doc, 1, ExtractionMethod.DOCLING_TEXT),
                header_date=str(header),
                archive_date=str(archive),
                difference_days=difference,
            )
        )

    ratings = extract_ratings(ctx)
    sort = extract_sort(ctx)
    risk_prev, risk_cur = ratings.get("risk", (None, None))
    if risk_cur is None or risk_cur.normalized_rating is None:
        overall = next((s for s in sort if s.risk_category.lower() == "overall"), None)
        if overall and overall.current_rating:
            risk_prev, risk_cur = overall.previous_rating, overall.current_rating
    for key, label in (("pdo", "PDO rating"), ("ip", "implementation progress rating")):
        if key not in ratings or ratings[key][1].normalized_rating is None:
            ctx.issues.append(
                issue(
                    CheckCode.RATING_NOT_FOUND,
                    "WARNING",
                    f"{label} not found",
                    evidence(doc, _ratings_page(doc), ExtractionMethod.DOCLING_TABLE),
                )
            )
    if risk_cur is None or risk_cur.normalized_rating is None:
        ctx.issues.append(
            issue(CheckCode.RATING_NOT_FOUND, "WARNING", "overall risk rating not found")
        )

    loans = extract_disbursements(ctx)
    key_dates = extract_key_dates(ctx)
    narratives = extract_narratives(ctx)
    history = extract_restructuring_history(ctx)

    def total(attr):
        values = [getattr(loan, attr) for loan in loans]
        return sum(values, Decimal(0)) if values and all(v is not None for v in values) else None

    pdo_prev, pdo_cur = ratings.get("pdo", (None, None))
    ip_prev, ip_cur = ratings.get("ip", (None, None))
    return IsrSnapshot(
        project_id=doc.project_id,
        document_id=doc.document_id,
        isr_sequence=doc.isr_sequence,
        isr_number=doc.report_number,
        archive_date=archive,
        header_date=header,
        canonical_report_date=canonical,
        canonical_date_basis=basis if canonical else None,
        date_difference_days=difference,
        pdo_rating=pdo_cur,
        implementation_progress_rating=ip_cur,
        overall_risk_rating=risk_cur,
        previous_pdo_rating=pdo_prev,
        previous_implementation_rating=ip_prev,
        previous_overall_risk_rating=risk_prev,
        sort_ratings=sort,
        loan_disbursements=loans,
        loan_key_dates=key_dates,
        commitment_amount_musd=total("revised_musd"),
        disbursed_amount_musd=total("disbursed_musd"),
        disbursement_pct_reported=loans[0].disbursed_pct_reported if len(loans) == 1 else None,
        key_issues_text=_named_text(narratives, r"key issues"),
        key_decisions_text=_named_text(narratives, r"key decisions", exclude="implementation"),
        implementation_status_text=_named_text(narratives, r"implementation status"),
        narrative_sections=narratives,
        restructuring_history=history,
        source_document=doc.filename,
        source_pages=sorted({r.page_number for r in ctx.refs}),
        source_refs=ctx.refs,
        quality_issues=ctx.issues,
    )
