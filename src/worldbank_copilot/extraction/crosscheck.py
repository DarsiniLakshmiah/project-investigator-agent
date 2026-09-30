"""Cross-source validation: document-derived facts vs structured Silver.

Observations only. Neither source is overwritten, and disagreements are never
resolved here. Snapshot and document dates differ, so the observation date of
each side is always reported next to the values.

Comparable pairs (latest ISR per project vs silver_loans):
loan numbers, original principal, cancelled / disbursed amounts, board
approval, effectiveness and current closing dates. Document amounts in a
non-USD currency (e.g. the JPY cancellation of IBRD93240) are reported as
CROSS_SOURCE_NOT_COMPARABLE, never converted.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from worldbank_copilot.common.quality import CheckCode
from worldbank_copilot.extraction.models import IsrSnapshot, ProjectEvent
from worldbank_copilot.extraction.provenance import EvidenceRef, ExtractionIssue, issue

# ISR amounts are printed in US$ millions with 2 decimals: +/- US$5,000 is rounding.
AMOUNT_TOLERANCE_USD = Decimal("5000")


def _obs(code: str, severity: str, message: str, ref: EvidenceRef | None, **details):
    return issue(code, severity, message, ref, **details)


def _compare(
    project_id,
    loan,
    field,
    doc_value,
    doc_ref,
    doc_date,
    structured_value,
    snapshot_date,
    *,
    amount=False,
    expected_drift=False,
) -> ExtractionIssue | None:
    if doc_value is None or structured_value is None:
        return None
    details = {
        "field": field,
        "loan_number": loan,
        "document_value": str(doc_value),
        "document_as_of": str(doc_date),
        "structured_value": str(structured_value),
        "structured_as_of": str(snapshot_date),
        "structured_source": "silver_loans",
    }
    if amount:
        same = abs(Decimal(doc_value) - Decimal(structured_value)) <= AMOUNT_TOLERANCE_USD
    else:
        same = doc_value == structured_value
    if same:
        return _obs(
            CheckCode.CROSS_SOURCE_AGREEMENT,
            "INFO",
            f"{project_id} {loan} {field}: document and loan snapshot agree",
            doc_ref,
            **details,
        )
    # Amounts that move over time (disbursed/cancelled) differ between dates as expected.
    severity = "INFO" if expected_drift else "WARNING"
    note = " (values observed at different dates)" if expected_drift else ""
    return _obs(
        CheckCode.CROSS_SOURCE_DIFFERENCE,
        severity,
        f"{project_id} {loan} {field}: document {doc_value} vs loan snapshot "
        f"{structured_value}{note}",
        doc_ref,
        **details,
    )


def crosscheck_project(
    project_id: str,
    snapshots: list[IsrSnapshot],
    events: list[ProjectEvent],
    loans: list[Any],
) -> list[ExtractionIssue]:
    out: list[ExtractionIssue] = []
    project_loans = {loan.raw_loan_number: loan for loan in loans if loan.project_id == project_id}
    dated = [s for s in snapshots if s.isr_sequence is not None]
    if not dated:
        return out
    latest = max(dated, key=lambda s: s.isr_sequence)
    as_of = latest.canonical_report_date

    isr_loans = {d.loan_number for d in latest.loan_disbursements} | {
        k.loan_number for k in latest.loan_key_dates
    }
    if isr_loans != set(project_loans):
        out.append(
            _obs(
                CheckCode.CROSS_SOURCE_DIFFERENCE,
                "WARNING",
                f"{project_id}: loan numbers differ: latest ISR {sorted(isr_loans)} vs "
                f"loan snapshot {sorted(project_loans)}",
                latest.source_refs[0],
                field="loan_numbers",
            )
        )
    else:
        out.append(
            _obs(
                CheckCode.CROSS_SOURCE_AGREEMENT,
                "INFO",
                f"{project_id}: loan numbers agree {sorted(isr_loans)}",
                latest.source_refs[0],
                field="loan_numbers",
            )
        )

    million = Decimal(1_000_000)
    for disb in latest.loan_disbursements:
        loan = project_loans.get(disb.loan_number)
        if loan is None:
            continue
        for field, doc_musd, value, drift in (
            ("original_principal_usd", disb.original_musd, loan.original_principal_usd, False),
            ("cancelled_amount_usd", disb.cancelled_musd, loan.cancelled_amount_usd, True),
            ("disbursed_amount_usd", disb.disbursed_musd, loan.disbursed_amount_usd, True),
            (
                "disbursed_amount_usd (ISR 'Historical Disbursed' column)",
                disb.historical_disbursed_musd,
                loan.disbursed_amount_usd,
                True,
            ),
        ):
            doc_usd = doc_musd * million if doc_musd is not None else None
            found = _compare(
                project_id,
                disb.loan_number,
                field,
                doc_usd,
                disb.evidence,
                as_of,
                value,
                loan.snapshot_date,
                amount=True,
                expected_drift=drift,
            )
            if found:
                out.append(found)
    for kd in latest.loan_key_dates:
        loan = project_loans.get(kd.loan_number)
        if loan is None:
            continue
        for field, doc_value, value in (
            ("board_approval_date", kd.approval_date, loan.board_approval_date),
            ("agreement_signing_date", kd.signing_date, loan.agreement_signing_date),
            ("effective_date", kd.effectiveness_date, loan.effective_date),
            ("closing_date", kd.revised_closing_date, loan.closing_date),
        ):
            found = _compare(
                project_id,
                kd.loan_number,
                field,
                doc_value,
                kd.evidence,
                as_of,
                value,
                loan.snapshot_date,
            )
            if found:
                out.append(found)

    for event in events:
        loan = project_loans.get(event.loan_number or "")
        ref = event.source_refs[0]
        if event.event_type == "CANCELLATION" and loan is not None:
            if (event.cancelled_currency or "").upper() not in ("USD", "US$"):
                out.append(
                    _obs(
                        CheckCode.CROSS_SOURCE_NOT_COMPARABLE,
                        "INFO",
                        f"{project_id} {event.loan_number}: document cancellation "
                        f"{event.cancelled_amount} {event.cancelled_currency} is not "
                        f"comparable with snapshot cancelled_amount_usd "
                        f"{loan.cancelled_amount_usd} (currency; no conversion applied)",
                        ref,
                        field="cancelled_amount",
                        document_currency=event.cancelled_currency,
                        document_value=str(event.cancelled_amount),
                        structured_value=str(loan.cancelled_amount_usd),
                    )
                )
            else:
                found = _compare(
                    project_id,
                    event.loan_number,
                    "cancelled_amount_usd",
                    event.cancelled_amount,
                    ref,
                    event.event_date,
                    loan.cancelled_amount_usd,
                    loan.snapshot_date,
                    amount=True,
                    expected_drift=True,
                )
                if found:
                    out.append(found)
        if event.event_type == "ADDITIONAL_FINANCING":
            # The AF loan is the project loan approved after the original one(s).
            candidates = [
                loan
                for loan in project_loans.values()
                if event.additional_financing_amount is not None
                and loan.original_principal_usd is not None
                and abs(loan.original_principal_usd - event.additional_financing_amount * million)
                <= AMOUNT_TOLERANCE_USD
            ]
            if len(candidates) == 1:
                loan = candidates[0]
                out.append(
                    _obs(
                        CheckCode.CROSS_SOURCE_AGREEMENT,
                        "INFO",
                        f"{project_id}: AF amount US${event.additional_financing_amount}M "
                        f"matches original principal of {loan.raw_loan_number}",
                        ref,
                        field="additional_financing_amount",
                        loan_number=loan.raw_loan_number,
                    )
                )
                found = _compare(
                    project_id,
                    loan.raw_loan_number,
                    "board_approval_date (AF paper printed approval date)",
                    event.event_date,
                    ref,
                    event.source_refs[0].document_date,
                    loan.board_approval_date,
                    loan.snapshot_date,
                )
                if found:
                    out.append(found)
            else:
                out.append(
                    _obs(
                        CheckCode.CROSS_SOURCE_NOT_COMPARABLE,
                        "INFO",
                        f"{project_id}: AF amount could not be matched to exactly one "
                        f"loan ({len(candidates)} candidates)",
                        ref,
                        field="additional_financing_amount",
                    )
                )
    return out
