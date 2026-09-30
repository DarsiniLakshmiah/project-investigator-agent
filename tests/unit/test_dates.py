from datetime import date

import pytest

from worldbank_copilot.common.dates import (
    DAY_MON_YEAR,
    FILENAME_MM_DD_YYYY,
    ISO_DATE,
    ISO_DATETIME_Z,
    MON_DAY_YEAR,
    US_SLASH,
    DateParseError,
    parse_date,
)


@pytest.mark.parametrize(
    ("raw", "formats", "expected"),
    [
        ("2016-08-22", [ISO_DATE, ISO_DATETIME_Z], date(2016, 8, 22)),
        ("2016-03-31T00:00:00Z", [ISO_DATE, ISO_DATETIME_Z], date(2016, 3, 31)),
        ("08/31/2026", [US_SLASH], date(2026, 8, 31)),
        ("4/10/2017", [US_SLASH], date(2017, 4, 10)),
        ("29-Nov-2017", [DAY_MON_YEAR], date(2017, 11, 29)),
        ("Aug 16, 2026", [MON_DAY_YEAR], date(2026, 8, 16)),
        ("04-10-2017", [FILENAME_MM_DD_YYYY], date(2017, 4, 10)),
        ("  08/31/2026 ", [US_SLASH], date(2026, 8, 31)),
    ],
)
def test_declared_formats_parse(raw, formats, expected):
    assert parse_date(raw, formats) == expected


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_blank_is_missing_not_error(raw):
    assert parse_date(raw, [US_SLASH]) is None


def test_undeclared_format_is_rejected_not_guessed():
    # An ISO value in a field declared as MM/DD/YYYY must fail, not be guessed.
    with pytest.raises(DateParseError) as exc:
        parse_date("2016-08-22", [US_SLASH])
    assert exc.value.raw == "2016-08-22"


def test_impossible_date_fails():
    with pytest.raises(DateParseError):
        parse_date("02/30/2020", [US_SLASH])


def test_formats_are_required():
    with pytest.raises(ValueError):
        parse_date("2020-01-01", [])
