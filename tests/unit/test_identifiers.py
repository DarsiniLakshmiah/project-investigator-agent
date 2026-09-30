import pytest

from worldbank_copilot.common.identifiers import (
    LoanLinkStatus,
    LoanNumberNotation,
    link_loan_reference,
    normalize_loan_number,
    parse_loan_number,
)


def test_statement_notation():
    parsed = parse_loan_number("IBRD86010")
    assert parsed.notation is LoanNumberNotation.STATEMENT
    assert (parsed.lender, parsed.base_number, parsed.suffix) == ("IBRD", "8601", "0")
    assert parsed.normalized == "IBRD-8601-0"
    assert parsed.raw == "IBRD86010"


def test_legal_notation():
    parsed = parse_loan_number("8601-IN")
    assert parsed.notation is LoanNumberNotation.LEGAL
    assert (parsed.base_number, parsed.country_code, parsed.lender) == ("8601", "IN", None)
    assert parsed.normalized == "8601-IN"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("IBRD86010", "IBRD-8601-0"),
        (" ibrd86010 ", "IBRD-8601-0"),
        ("IBRD 86010", "IBRD-8601-0"),
        ("IBRD13445", "IBRD-1344-5"),  # suffix is significant and kept
        ("8601-IN", "8601-IN"),
        ("8601 - in", "8601-IN"),
        ("IBRD 8601-IN", "IBRD-8601-IN"),
    ],
)
def test_normalization(raw, expected):
    assert normalize_loan_number(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [None, "", "   ", "860 I-IN", "IBRD8601", "IBRD1234A", "TF12345", "8601", "IDA86010"],
)
def test_unrecognised_values_are_not_corrected(raw):
    assert parse_loan_number(raw) is None
    assert normalize_loan_number(raw) is None


def test_legal_reference_links_to_single_project_loan():
    link = link_loan_reference("8601-IN", ["IBRD86010", "IBRD93240"])
    assert link.status is LoanLinkStatus.LINKED
    assert link.matched_raw == "IBRD86010"


def test_second_loan_of_same_project_links_independently():
    assert link_loan_reference("9324-IN", ["IBRD86010", "IBRD93240"]).matched_raw == "IBRD93240"


def test_ambiguous_suffixes_are_not_linked():
    # The real snapshot has bases with several suffixes (e.g. IBRD13440 and IBRD13445).
    link = link_loan_reference("1344-IN", ["IBRD13440", "IBRD13445"])
    assert link.status is LoanLinkStatus.AMBIGUOUS
    assert link.matched_raw is None
    assert link.candidates == ("IBRD13440", "IBRD13445")


def test_statement_to_statement_requires_same_suffix():
    assert link_loan_reference("IBRD13445", ["IBRD13440"]).status is LoanLinkStatus.NO_MATCH
    assert link_loan_reference("IBRD13445", ["IBRD13440", "IBRD13445"]).matched_raw == "IBRD13445"


def test_no_match_and_unrecognised():
    assert link_loan_reference("9999-IN", ["IBRD86010"]).status is LoanLinkStatus.NO_MATCH
    assert link_loan_reference("860 I-IN", ["IBRD86010"]).status is LoanLinkStatus.UNRECOGNIZED


def test_no_fuzzy_matching_on_near_miss():
    assert link_loan_reference("8610-IN", ["IBRD86010"]).status is LoanLinkStatus.NO_MATCH
