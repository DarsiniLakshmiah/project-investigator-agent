"""Phase 9B stage 3: deterministic temporal grammar."""

from datetime import date

import pytest

from worldbank_copilot.routing.models import TemporalKind as K
from worldbank_copilot.routing.models import TemporalStatus as S
from worldbank_copilot.routing.temporal import parse_temporal, with_default

CASES = [
    ("What is the latest rating?", K.LATEST, (), None, None),
    ("current disbursement", K.LATEST, (), None, None),
    ("in the most recent ISR", K.LATEST, (), None, None),
    ("What was the IP rating in ISR 4?", K.ISR_SEQUENCE, (4,), None, None),
    ("ISR sequence 12 results", K.ISR_SEQUENCE, (12,), None, None),
    ("ISR No. 7", K.ISR_SEQUENCE, (7,), None, None),
    ("problems reported in ISR 5 and ISR 6", K.ISR_SEQUENCE, (5, 6), None, None),
    ("ratings from ISR 5 to 8", K.ISR_RANGE, (5, 8), None, None),
    ("ISR 3 through ISR 9", K.ISR_RANGE, (3, 9), None, None),
    ("between ISR 2 and 4", K.ISR_RANGE, (2, 4), None, None),
    ("risk rating at appraisal", K.APPRAISAL, (), None, None),
    ("during appraisal", K.APPRAISAL, (), None, None),
    ("restructured in 2018", K.YEAR, (), date(2018, 1, 1), date(2018, 12, 31)),
    ("approved on 31 March 2016", K.DATE, (), date(2016, 3, 31), date(2016, 3, 31)),
    ("approved on March 31, 2016", K.DATE, (), date(2016, 3, 31), date(2016, 3, 31)),
    ("archived 29-Nov-2017", K.DATE, (), date(2017, 11, 29), date(2017, 11, 29)),
    ("as of 2026-08-31", K.DATE, (), date(2026, 8, 31), date(2026, 8, 31)),
    ("in February 2023", K.DATE_RANGE, (), date(2023, 2, 1), date(2023, 2, 28)),
    ("between 2019 and 2021", K.DATE_RANGE, (), date(2019, 1, 1), date(2021, 12, 31)),
    ("from 2019 to 2021", K.DATE_RANGE, (), date(2019, 1, 1), date(2021, 12, 31)),
    ("since 2023", K.DATE_RANGE, (), date(2023, 1, 1), None),
    ("after 2020", K.DATE_RANGE, (), date(2021, 1, 1), None),
    ("before 2019", K.DATE_RANGE, (), None, date(2018, 12, 31)),
    ("until 2021", K.DATE_RANGE, (), None, date(2021, 12, 31)),
    ("rating history", K.HISTORY, (), None, None),
    ("how has it evolved over time", K.HISTORY, (), None, None),
]


@pytest.mark.parametrize(("text", "kind", "isrs", "lo", "hi"), CASES)
def test_supported_forms(text, kind, isrs, lo, hi):
    scope = parse_temporal(text)
    assert (scope.kind, scope.status, scope.explicit, scope.defaulted) == (
        kind,
        S.RESOLVED,
        True,
        False,
    )
    assert (scope.isr_sequences, scope.date_from, scope.date_to) == (isrs, lo, hi)
    assert scope.expressions and all(
        text[a:b] == e.text for e in scope.expressions for a, b in [e.span]
    )


def test_no_time_is_not_applicable_and_defaults_are_recorded():
    scope = parse_temporal("What are the risks?")
    assert (scope.kind, scope.status, scope.explicit) == (K.NONE, S.NOT_APPLICABLE, False)
    defaulted = with_default(scope, K.LATEST)
    assert (defaulted.kind, defaulted.status, defaulted.defaulted) == (K.LATEST, S.DEFAULTED, True)
    assert with_default(parse_temporal("rating in ISR 4"), K.LATEST).kind == K.ISR_SEQUENCE
    assert with_default(scope, K.NONE) == scope


@pytest.mark.parametrize(
    "text", ["in the last 12 months", "over the past year", "recently", "last year", "this year"]
)
def test_relative_periods_are_unresolved_never_wall_clock(text):
    scope = parse_temporal(f"What changed {text}?")
    assert (scope.kind, scope.status) == (K.RELATIVE, S.UNRESOLVED)
    assert scope.date_from is None and scope.date_to is None
    assert "wall clock is never used" in scope.detail


@pytest.mark.parametrize(
    ("text", "event", "qualifier", "direction"),
    [
        ("since the restructuring", "RESTRUCTURING", None, "since"),
        ("after the 2021 restructuring", "RESTRUCTURING", "2021", "after"),
        ("before the first restructuring", "RESTRUCTURING", "first", "before"),
        ("since the latest restructuring", "RESTRUCTURING", "latest", "since"),
        ("since effectiveness", "EFFECTIVENESS", None, "since"),
        ("prior to the additional financing", "ADDITIONAL_FINANCING", None, "before"),
        ("following the cancellation", "CANCELLATION", None, "after"),
        ("until the closing date extension", "CLOSING_DATE_CHANGE", None, "until"),
    ],
)
def test_event_anchors_are_detected_and_left_unresolved_for_the_timeline(
    text, event, qualifier, direction
):
    scope = parse_temporal(f"Ratings {text}")
    assert (scope.kind, scope.status) == (K.EVENT_ANCHORED, S.UNRESOLVED)
    (expr,) = [e for e in scope.expressions if e.kind == K.EVENT_ANCHORED]
    assert (expr.anchor_event_type, expr.anchor_qualifier, expr.anchor_direction) == (
        event,
        qualifier,
        direction,
    )


def test_several_different_explicit_values_are_not_guessed():
    assert parse_temporal("in 2018 and in 2021").status == S.UNRESOLVED
    assert parse_temporal("ISR 2 to 4 and ISR 6 to 8").status == S.UNRESOLVED


def test_the_most_specific_kind_is_primary_and_others_are_recorded():
    scope = parse_temporal("latest ISR rating history in 2021")
    assert scope.kind == K.YEAR and {e.kind for e in scope.expressions} >= {K.LATEST, K.HISTORY}
    mixed = parse_temporal("ISR 18 in 2021")
    assert mixed.kind == K.YEAR and "ISR_SEQUENCE" in mixed.detail


def test_report_numbers_and_invalid_dates_are_not_time():
    assert parse_temporal("ISR07744 says").kind == K.NONE
    invalid = parse_temporal("on 31 February 2016")
    assert (invalid.kind, invalid.status) == (K.DATE, S.UNRESOLVED)  # never reinterpreted


def test_parsing_never_depends_on_today(monkeypatch):
    import worldbank_copilot.routing.temporal as t

    class Frozen(date):
        @classmethod
        def today(cls):
            raise AssertionError("wall clock used")

    monkeypatch.setattr(t, "date", Frozen)
    for text, *_ in CASES:
        parse_temporal(text)
