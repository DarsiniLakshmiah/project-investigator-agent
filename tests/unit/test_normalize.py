from datetime import date
from decimal import Decimal

import pytest

from worldbank_copilot.common.dates import ISO_DATE, ISO_DATETIME_Z, US_SLASH
from worldbank_copilot.transformations.normalize import (
    NormalizationError,
    blank_to_none,
    clean_text,
    normalize_category,
    normalize_identifier,
    parse_decimal,
    parse_money,
    parse_source_date,
    parse_year,
    percentage,
    strip_html,
    taxonomy_label,
)


@pytest.mark.parametrize("raw", [None, "", "   ", "\t\n"])
def test_blank_to_none(raw):
    assert blank_to_none(raw) is None
    assert clean_text(raw) is None


def test_blank_to_none_keeps_content_unchanged():
    assert blank_to_none(" USD ") == " USD "


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("USD  ", "USD"),
        (" Water  Institutions, Policies", "Water Institutions, Policies"),
        ("line\nbreak\tand nbsp", "line break and nbsp"),
    ],
)
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<p>The objective is to provide water.</p>", "The objective is to provide water."),
        ("<p>First.</p><p>Second &amp; third.</p>", "First. Second & third."),
        ("Line one<br/>line two", "Line one line two"),
        ("<ul><li>a</li><li>b</li></ul>", "a b"),
        ("No markup here", "No markup here"),
        ("<p>   </p>", None),
        (None, None),
    ],
)
def test_strip_html(raw, expected):
    assert strip_html(raw) == expected


def test_category_keeps_casing():
    assert normalize_category("  Program-for-Results   Financing ") == (
        "Program-for-Results Financing"
    )


def test_identifier():
    assert normalize_identifier(" 1657297 ") == "1657297"
    assert normalize_identifier("p130544", upper=True) == "P130544"
    assert normalize_identifier("") is None
    with pytest.raises(NormalizationError, match="internal whitespace"):
        normalize_identifier("16 57297")


def test_money_is_exact_decimal():
    value = parse_money("24937499.48")
    assert isinstance(value, Decimal) and value == Decimal("24937499.48")
    assert parse_money("100000000.0") == Decimal("100000000")
    assert parse_money("-2613739.59") == Decimal("-2613739.59")
    # Decimal sum is exact where float would not be.
    parts = [parse_money(x) for x in ("32713450.09", "92871483.5", "24937499.48")]
    assert sum(parts, Decimal(0)) == Decimal("150522433.07")


@pytest.mark.parametrize("raw", ["12,000", "1e5", "abc", "NaN", "Infinity", "$5", "5 000"])
def test_malformed_money_is_rejected_not_coerced(raw):
    with pytest.raises(NormalizationError):
        parse_money(raw)


def test_decimal_blank():
    assert parse_decimal("  ") is None


def test_year():
    assert parse_year("2021") == 2021
    with pytest.raises(NormalizationError):
        parse_year("FY21")


def test_dates():
    assert parse_source_date("2016-03-31T00:00:00Z", [ISO_DATE, ISO_DATETIME_Z]) == date(
        2016, 3, 31
    )
    assert parse_source_date("08/31/2026", [US_SLASH]) == date(2026, 8, 31)
    assert parse_source_date("", [US_SLASH]) is None
    with pytest.raises(NormalizationError):
        parse_source_date("31/08/2026", [US_SLASH])


def test_taxonomy_label():
    assert taxonomy_label("FY17 - Water Supply") == "FY17"
    assert taxonomy_label("Water Supply") is None
    assert taxonomy_label(None) is None


def test_percentage_rounding_and_undefined():
    assert percentage(Decimal("132713450.09"), Decimal("250000000")) == Decimal("53.09")
    assert percentage(Decimal("1"), Decimal("8")) == Decimal("12.50")
    assert percentage(Decimal("1"), Decimal("0")) is None
    assert percentage(None, Decimal("1")) is None
