"""Stage 3: deterministic temporal-scope extraction (Phase 9B).

Supported forms (case-insensitive):

* LATEST         - latest / current / currently / most recent / last ISR / to date / now
* ISR_SEQUENCE   - "ISR 5", "ISR sequence 5", "ISR No. 5", "ISR 5 and ISR 6"
* ISR_RANGE      - "ISR 5 to 8", "ISR 5 through ISR 8", "ISRs 5-8", "between ISR 5 and 8"
* APPRAISAL      - "at appraisal", "during appraisal", "at the time of appraisal", "at approval"
* YEAR           - "in 2021" (any 19xx/20xx year not part of a date)
* DATE           - "31 March 2016", "March 31, 2016", "31-Mar-2016", "2016-03-31";
                   "March 2016" (a month)
* DATE_RANGE     - "between 2019 and 2021", "from 2019 to 2021", "since 2023",
                   "after/before/until <date or year>"
* HISTORY        - history / historical / over time / trend / evolution / throughout
* EVENT_ANCHORED - "since|after|before|until (the) [first|second|...|last|latest|2021]
                   restructuring|additional financing|cancellation|effectiveness|approval|
                   closing date change" - resolved later against SOURCE-STATED timeline
                   dates; several matching events are never guessed (UNRESOLVED)
* RELATIVE       - "last 12 months", "past year", "recently": no governed reference date
                   is defined for these in 9B, and the wall clock is never used ->
                   UNRESOLVED (explicitly, not silently ignored).

When several forms occur, the scope keeps every expression and selects the primary kind
by specificity: EVENT_ANCHORED > DATE_RANGE > DATE > YEAR > ISR_RANGE > ISR_SEQUENCE >
APPRAISAL > HISTORY > LATEST. Two different explicit kinds that cannot be combined
(e.g. an ISR and an unrelated year) are recorded as such; the primary is still the most
specific one. Defaults are applied later, per intent (``defaulted=True``).
"""

from __future__ import annotations

import calendar
import re
from datetime import date

from worldbank_copilot.routing.models import (
    TemporalExpression,
    TemporalKind,
    TemporalScope,
    TemporalStatus,
)

_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTHS |= {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
_MONTH = (
    r"(january|february|march|april|may|june|july|august|september|october|november|"
    r"december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
)
_YEAR = r"((?:19|20)\d{2})"
_ISR = r"isrs?\s*(?:sequence|seq\.?|no\.?|number|#)?\s*"

PRIORITY = [
    TemporalKind.EVENT_ANCHORED,
    TemporalKind.DATE_RANGE,
    TemporalKind.DATE,
    TemporalKind.YEAR,
    TemporalKind.ISR_RANGE,
    TemporalKind.ISR_SEQUENCE,
    TemporalKind.APPRAISAL,
    TemporalKind.HISTORY,
    TemporalKind.LATEST,
]
EVENT_TYPES = {
    "restructuring": "RESTRUCTURING",
    "additional financing": "ADDITIONAL_FINANCING",
    "cancellation": "CANCELLATION",
    "effectiveness": "EFFECTIVENESS",
    "approval": "APPROVAL",
    "closing date change": "CLOSING_DATE_CHANGE",
    "closing date extension": "CLOSING_DATE_CHANGE",
    "extension": "CLOSING_DATE_CHANGE",
}
_EVENT = re.compile(
    r"\b(since|after|before|until|prior to|following)\s+(?:the\s+)?"
    r"(?:(first|second|third|fourth|last|latest|most recent|original|" + _YEAR + r")\s+)?"
    r"(restructuring|additional financing|cancellation|effectiveness|approval|"
    r"closing date change|closing date extension|extension)\b",
    re.IGNORECASE,
)
_RELATIVE = re.compile(
    r"\b(last|past|previous)\s+(\d+\s+|few\s+|couple of\s+)?"
    r"(days?|weeks?|months?|years?|quarters?)\b"
    r"|(?<!most )\brecent(ly)?\b|\bthis year\b|\blast year\b",
    re.IGNORECASE,
)
_LATEST = re.compile(
    r"\b(latest|most recent|current(ly)?|to date|as of now|at present|now)\b"
    r"|\blast isr\b",
    re.IGNORECASE,
)
_HISTORY = re.compile(
    r"\b(history|historical(ly)?|over time|trends?|evolution|evolved|throughout|all isrs|"
    r"every isr|each isr)\b",
    re.IGNORECASE,
)
_APPRAISAL = re.compile(
    r"\b(at|during) (the )?(time of )?(appraisal|approval)\b|\bappraisal stage\b",
    re.IGNORECASE,
)
_ISR_RANGE = re.compile(
    r"\b(?:between\s+)?"
    + _ISR
    + r"(\d{1,2})\s*(?:to|through|thru|until|-|–|and)\s*(?:isr\s*)?(\d{1,2})\b",
    re.IGNORECASE,
)
_ISR_ONE = re.compile(r"\b" + _ISR + r"(\d{1,2})\b", re.IGNORECASE)
_ISO_DATE = re.compile(r"\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b")
_DMY = re.compile(r"\b(\d{1,2})[\s-]+" + _MONTH + r"\.?[\s-]+" + _YEAR + r"\b", re.IGNORECASE)
_MDY = re.compile(r"\b" + _MONTH + r"\.?\s+(\d{1,2}),?\s+" + _YEAR + r"\b", re.IGNORECASE)
_MY = re.compile(r"\b" + _MONTH + r"\.?\s+" + _YEAR + r"\b", re.IGNORECASE)
_YEAR_RANGE = re.compile(
    r"\b(between|from)\s+" + _YEAR + r"\s+(and|to|through|until)\s+" + _YEAR + r"\b", re.IGNORECASE
)
_YEAR_OPEN = re.compile(
    r"\b(since|after|from|before|until|prior to)\s+" + _YEAR + r"\b", re.IGNORECASE
)
_YEAR_ONE = re.compile(r"\b" + _YEAR + r"\b")


def _month(name: str) -> int:
    return _MONTHS[name.lower()[:3]]


def _overlaps(span: tuple[int, int], taken: list[tuple[int, int]]) -> bool:
    return any(span[0] < b and a < span[1] for a, b in taken)


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def extract(text: str) -> list[TemporalExpression]:
    """All temporal expressions, longest-first without overlaps, in text order."""
    found: list[TemporalExpression] = []
    taken: list[tuple[int, int]] = []

    def add(expr: TemporalExpression) -> None:
        if not _overlaps(expr.span, taken):
            found.append(expr)
            taken.append(expr.span)

    for m in _EVENT.finditer(text):
        direction = {"prior to": "before", "following": "after"}.get(
            m.group(1).lower(), m.group(1).lower()
        )
        qualifier = (m.group(2) or "").lower() or None
        add(
            TemporalExpression(
                kind=TemporalKind.EVENT_ANCHORED,
                span=m.span(),
                text=m.group(0),
                anchor_event_type=EVENT_TYPES[m.group(4).lower()],
                anchor_qualifier=qualifier,
                anchor_direction=direction,
            )
        )
    for m in _RELATIVE.finditer(text):
        add(TemporalExpression(kind=TemporalKind.RELATIVE, span=m.span(), text=m.group(0)))
    for m in _ISO_DATE.finditer(text):
        d = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        add(
            TemporalExpression(
                kind=TemporalKind.DATE, span=m.span(), text=m.group(0), date_from=d, date_to=d
            )
        )  # an invalid date (d is None) is kept, so it is reported, never reinterpreted
    for m in _DMY.finditer(text):
        d = _safe_date(int(m.group(3)), _month(m.group(2)), int(m.group(1)))
        add(
            TemporalExpression(
                kind=TemporalKind.DATE, span=m.span(), text=m.group(0), date_from=d, date_to=d
            )
        )  # an invalid date (d is None) is kept, so it is reported, never reinterpreted
    for m in _MDY.finditer(text):
        d = _safe_date(int(m.group(3)), _month(m.group(1)), int(m.group(2)))
        add(
            TemporalExpression(
                kind=TemporalKind.DATE, span=m.span(), text=m.group(0), date_from=d, date_to=d
            )
        )  # an invalid date (d is None) is kept, so it is reported, never reinterpreted
    for m in _MY.finditer(text):
        month = _month(m.group(1))
        year = int(m.group(2))
        last = calendar.monthrange(year, month)[1]
        add(
            TemporalExpression(
                kind=TemporalKind.DATE_RANGE,
                span=m.span(),
                text=m.group(0),
                date_from=date(year, month, 1),
                date_to=date(year, month, last),
            )
        )
    for m in _YEAR_RANGE.finditer(text):
        a, b = int(m.group(2)), int(m.group(4))
        if a <= b:
            add(
                TemporalExpression(
                    kind=TemporalKind.DATE_RANGE,
                    span=m.span(),
                    text=m.group(0),
                    date_from=date(a, 1, 1),
                    date_to=date(b, 12, 31),
                )
            )
    for m in _YEAR_OPEN.finditer(text):
        word, year = m.group(1).lower(), int(m.group(2))
        lo, hi = (
            (date(year, 1, 1), None)
            if word in ("since", "from")
            else (
                (date(year + 1, 1, 1), None) if word == "after" else (None, date(year - 1, 12, 31))
            )
        )
        if word == "until":
            lo, hi = None, date(year, 12, 31)
        add(
            TemporalExpression(
                kind=TemporalKind.DATE_RANGE,
                span=m.span(),
                text=m.group(0),
                date_from=lo,
                date_to=hi,
            )
        )
    for m in _ISR_RANGE.finditer(text):
        a, b = int(m.group(1)), int(m.group(2))
        is_and = re.search(r"\band\b", m.group(0), re.IGNORECASE) and not m.group(
            0
        ).lower().startswith("between")
        if is_and:
            add(
                TemporalExpression(
                    kind=TemporalKind.ISR_SEQUENCE,
                    span=m.span(),
                    text=m.group(0),
                    isr_sequences=tuple(sorted({a, b})),
                )
            )
        elif 1 <= a <= b:
            add(
                TemporalExpression(
                    kind=TemporalKind.ISR_RANGE,
                    span=m.span(),
                    text=m.group(0),
                    isr_sequences=(a, b),
                )
            )
    for m in _ISR_ONE.finditer(text):
        add(
            TemporalExpression(
                kind=TemporalKind.ISR_SEQUENCE,
                span=m.span(),
                text=m.group(0),
                isr_sequences=(int(m.group(1)),),
            )
        )
    for m in _APPRAISAL.finditer(text):
        add(TemporalExpression(kind=TemporalKind.APPRAISAL, span=m.span(), text=m.group(0)))
    for m in _YEAR_ONE.finditer(text):
        year = int(m.group(1))
        add(
            TemporalExpression(
                kind=TemporalKind.YEAR,
                span=m.span(),
                text=m.group(0),
                date_from=date(year, 1, 1),
                date_to=date(year, 12, 31),
            )
        )
    for m in _HISTORY.finditer(text):
        add(TemporalExpression(kind=TemporalKind.HISTORY, span=m.span(), text=m.group(0)))
    for m in _LATEST.finditer(text):
        add(TemporalExpression(kind=TemporalKind.LATEST, span=m.span(), text=m.group(0)))
    return sorted(found, key=lambda e: e.span)


def parse_temporal(text: str) -> TemporalScope:
    """The explicit temporal scope of a question (defaults are applied per intent later)."""
    expressions = tuple(extract(text))
    if not expressions:
        return TemporalScope(
            kind=TemporalKind.NONE,
            status=TemporalStatus.NOT_APPLICABLE,
            explicit=False,
            defaulted=False,
        )
    relative = [e for e in expressions if e.kind == TemporalKind.RELATIVE]
    if relative:
        return TemporalScope(
            kind=TemporalKind.RELATIVE,
            status=TemporalStatus.UNRESOLVED,
            explicit=True,
            defaulted=False,
            expressions=expressions,
            detail=f"relative period {relative[0].text!r} needs a governed reference date; "
            "the wall clock is never used",
        )
    invalid = [e for e in expressions if e.kind == TemporalKind.DATE and e.date_from is None]
    if invalid:
        return TemporalScope(
            kind=TemporalKind.DATE,
            status=TemporalStatus.UNRESOLVED,
            explicit=True,
            defaulted=False,
            expressions=expressions,
            detail=f"invalid date {invalid[0].text!r}",
        )
    kinds = {e.kind for e in expressions}
    primary_kind = next(k for k in PRIORITY if k in kinds)
    primaries = [e for e in expressions if e.kind == primary_kind]
    detail = None
    if len(primaries) > 1 and primary_kind not in (
        TemporalKind.ISR_SEQUENCE,
        TemporalKind.LATEST,
        TemporalKind.HISTORY,
    ):
        return TemporalScope(
            kind=primary_kind,
            status=TemporalStatus.UNRESOLVED,
            explicit=True,
            defaulted=False,
            expressions=expressions,
            detail=f"{len(primaries)} different {primary_kind.value} expressions",
        )
    if primary_kind == TemporalKind.EVENT_ANCHORED:
        return TemporalScope(
            kind=primary_kind,
            status=TemporalStatus.UNRESOLVED,
            explicit=True,
            defaulted=False,
            expressions=expressions,
            detail="event anchor not yet resolved against the timeline",
        )
    sequences = tuple(sorted({s for e in primaries for s in e.isr_sequences}))
    first = primaries[0]
    explicit_others = kinds - {primary_kind, TemporalKind.LATEST, TemporalKind.HISTORY}
    if explicit_others:
        detail = "also mentioned: " + ", ".join(sorted(k.value for k in explicit_others))
    return TemporalScope(
        kind=primary_kind,
        status=TemporalStatus.RESOLVED,
        explicit=True,
        defaulted=False,
        expressions=expressions,
        isr_sequences=sequences,
        date_from=first.date_from,
        date_to=first.date_to,
        detail=detail,
    )


def with_default(scope: TemporalScope, default: TemporalKind) -> TemporalScope:
    """Apply an intent's default when the question states no time (recorded)."""
    if scope.explicit or default == TemporalKind.NONE:
        return scope
    return scope.model_copy(
        update={
            "kind": default,
            "status": TemporalStatus.DEFAULTED,
            "defaulted": True,
            "detail": f"no time stated; intent default {default.value} applied",
        }
    )
