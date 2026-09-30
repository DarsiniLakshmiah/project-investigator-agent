"""Deterministic loan-number normalisation and linking.

Two notations refer to the same IBRD loan:

* Statement of Loans (CSV): ``IBRD86010``: lender, 4-digit loan number, 1-digit suffix.
* Legal documents: ``8601-IN``: 4-digit loan number, borrower country code.

The trailing suffix digit in the Statement notation is significant: the loans
snapshot contains bases with several suffixes (e.g. ``IBRD13440`` and
``IBRD13445``). A document reference such as ``8601-IN`` therefore links to a
statement loan only when exactly one loan with that base number exists within
the same project. Anything else is reported as ambiguous or unmatched. There
is no fuzzy matching: OCR-damaged values such as ``860 I-IN`` are unrecognised.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

_STATEMENT_PATTERN = re.compile(r"^(IBRD)(\d{4})(\d)$")
_LEGAL_PATTERN = re.compile(r"^(?:(IBRD)[\s-]*)?(\d{4})\s*-\s*([A-Z]{2})$")


class LoanNumberNotation(StrEnum):
    STATEMENT = "STATEMENT"  # IBRD86010
    LEGAL = "LEGAL"  # 8601-IN


@dataclass(frozen=True)
class LoanNumber:
    raw: str
    notation: LoanNumberNotation
    base_number: str
    lender: str | None = None
    suffix: str | None = None
    country_code: str | None = None

    @property
    def normalized(self) -> str:
        """Canonical text that keeps every component the source provided."""
        if self.notation is LoanNumberNotation.STATEMENT:
            return f"{self.lender}-{self.base_number}-{self.suffix}"
        prefix = f"{self.lender}-" if self.lender else ""
        return f"{prefix}{self.base_number}-{self.country_code}"

    @property
    def link_key(self) -> str:
        """The component shared by both notations."""
        return self.base_number


def parse_loan_number(raw: str | None) -> LoanNumber | None:
    """Parse a loan number in a known notation, or return ``None``.

    Only case and whitespace are normalised before matching; no other correction.
    """
    if raw is None:
        return None
    compact = raw.strip().upper()
    if not compact:
        return None
    match = _STATEMENT_PATTERN.match(re.sub(r"\s+", "", compact))
    if match:
        lender, base, suffix = match.groups()
        return LoanNumber(raw, LoanNumberNotation.STATEMENT, base, lender=lender, suffix=suffix)
    match = _LEGAL_PATTERN.match(compact)
    if match:
        lender, base, country = match.groups()
        return LoanNumber(raw, LoanNumberNotation.LEGAL, base, lender=lender, country_code=country)
    return None


def normalize_loan_number(raw: str | None) -> str | None:
    """``normalized_loan_number`` for a raw value; ``None`` if unrecognised."""
    parsed = parse_loan_number(raw)
    return parsed.normalized if parsed else None


class LoanLinkStatus(StrEnum):
    LINKED = "LINKED"
    NO_MATCH = "NO_MATCH"
    AMBIGUOUS = "AMBIGUOUS"
    UNRECOGNIZED = "UNRECOGNIZED"


@dataclass(frozen=True)
class LoanLink:
    status: LoanLinkStatus
    reference_raw: str | None
    matched_raw: str | None = None
    candidates: tuple[str, ...] = ()


def _compatible(a: LoanNumber, b: LoanNumber) -> bool:
    if a.link_key != b.link_key:
        return False
    if a.lender and b.lender and a.lender != b.lender:
        return False
    if a.suffix is not None and b.suffix is not None and a.suffix != b.suffix:
        return False
    return True


def link_loan_reference(reference_raw: str | None, project_loans_raw: Iterable[str]) -> LoanLink:
    """Link a loan reference to one of a single project's loans.

    ``project_loans_raw`` must be the raw loan numbers of the *same project*;
    scoping to the project is what makes base-number matching safe.
    """
    reference = parse_loan_number(reference_raw)
    if reference is None:
        return LoanLink(LoanLinkStatus.UNRECOGNIZED, reference_raw)
    matches = []
    for candidate_raw in project_loans_raw:
        candidate = parse_loan_number(candidate_raw)
        if candidate is not None and _compatible(reference, candidate):
            matches.append(candidate_raw)
    if len(matches) == 1:
        return LoanLink(LoanLinkStatus.LINKED, reference_raw, matches[0], tuple(matches))
    if not matches:
        return LoanLink(LoanLinkStatus.NO_MATCH, reference_raw)
    return LoanLink(LoanLinkStatus.AMBIGUOUS, reference_raw, None, tuple(matches))
