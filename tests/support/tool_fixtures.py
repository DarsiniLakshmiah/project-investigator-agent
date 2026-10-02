"""Synthetic governed rows for Phase 9 tool tests (structure only; no real values).

Rows use the real column names of the Gold/Silver contracts; unspecified columns are
NULL (the in-memory reader returns ``None`` for them, like a NULL column).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from tests.conftest import REPO_CONFIG_DIR

from worldbank_copilot.common import load_project_registry
from worldbank_copilot.intelligence.rules import load_rules, load_scales
from worldbank_copilot.tools.base import ToolContext
from worldbank_copilot.tools.reader import InMemoryReader

REGISTRY = load_project_registry(REPO_CONFIG_DIR)
RULES = load_rules(REPO_CONFIG_DIR)
SCALES = load_scales(REPO_CONFIG_DIR)
IPF, PFORR, OTHER = "P130544", "P179039", "P506272"


def gold(provenance_class: str = "DOCUMENTED_FINDING", **kw: Any) -> dict[str, Any]:
    return {
        "source_table": "silver.x",
        "source_record_id": f"src-{kw.get('record_id', 'r')}",
        "document_id": f"{kw.get('project_id', IPF)}-doc",
        "page_number": 3,
        "section": "Section",
        "extraction_method": "DOCLING_TABLE",
        "provenance_class": provenance_class,
        **kw,
    }


def project_360(pid: str, **kw: Any) -> dict[str, Any]:
    row = {
        "record_id": f"p360-{pid}",
        "project_id": pid,
        "project_name": f"Project {pid}",
        "instrument": "Investment Project Financing",
        "project_status": "Active",
        "approval_date": date(2016, 3, 31),
        "effectiveness_date": date(2016, 8, 1),
        "original_closing_date": date(2022, 11, 30),
        "original_closing_date_status": "AGREED",
        "current_closing_date": date(2027, 9, 30),
        "days_extended": 1765,
        "original_principal_usd": Decimal("100"),
        "cancelled_usd": Decimal("10"),
        "net_principal_usd": Decimal("90"),
        "disbursed_usd": Decimal("45"),
        "undisbursed_usd": Decimal("45"),
        "disbursement_pct_of_net_principal": Decimal("50"),
        "financial_snapshot_date": date(2026, 8, 31),
        "loan_count": 2,
        "loans_with_valuation_caveats": ["IBRD93240"],
        "latest_isr_sequence": 3,
        "latest_isr_date": date(2026, 1, 1),
        "latest_isr_date_basis": "header_date",
        "latest_do_rating": "Moderately Unsatisfactory",
        "latest_ip_rating": "Moderately Satisfactory",
        "latest_overall_risk_rating": "Substantial",
        "previous_do_rating": "Moderately Satisfactory",
        "previous_ip_rating": "Moderately Satisfactory",
        "do_rating_change": "DOWNGRADE",
        "ip_rating_change": "UNCHANGED",
        "number_of_isrs": 3,
        "number_of_project_events": 4,
        "number_of_restructurings": 1,
        "number_of_restructuring_papers": 1,
        "formal_risk_count": 2,
        "assessment_finding_count": 1,
        "number_of_result_indicators": 3,
        "latest_results_reporting_date": date(2026, 1, 1),
        "current_attention_signal_count": 2,
        "current_watch_signal_count": 1,
        "current_high_signal_count": 1,
        "source_record_ids": ["s1", "s2"],
    }
    row.update(kw)
    return row


def isr(pid: str, seq: int, pdo: str | None, ip: str | None, risk: str | None, **kw: Any):
    row = {
        "record_id": f"isr-{pid}-{seq}",
        "project_id": pid,
        "document_id": f"{pid}-isr{seq}",
        "isr_sequence": seq,
        "canonical_report_date": date(2024, seq, 1),
        "canonical_date_basis": "header_date",
    }
    for col, value in (
        ("pdo_rating", pdo),
        ("implementation_progress_rating", ip),
        ("overall_risk_rating", risk),
    ):
        row.update(
            {
                col: value,
                f"{col}_raw": value,
                f"{col}_status": "EXTRACTED" if value else "NOT_FOUND",
                f"{col}_page_number": 1,
                f"{col}_table_id": "t0001",
                f"{col}_extraction_method": "DOCLING_TABLE",
            }
        )
    row.update(kw)
    return row


def result(pid: str, rid: str, name: str, seq: int | None, status: str = "OK", **kw: Any):
    return gold(
        record_id=f"rp-{rid}-{seq}",
        project_id=pid,
        canonical_indicator_id=rid,
        indicator_name=name,
        indicator_type=kw.pop("indicator_type", "PDO"),
        unit="Number",
        layout="BLOCK",
        identity_review_status=kw.pop("identity_review_status", "EXACT_IDENTITY"),
        isr_sequence=seq,
        reporting_date=date(2024, seq or 1, 1),
        baseline_value="0",
        current_value="50",
        target_value="100",
        baseline_number=Decimal("0"),
        current_number=Decimal("50"),
        target_number=Decimal("100"),
        previous_number=Decimal("40") if seq and seq > 1 else None,
        absolute_change=Decimal("10") if seq and seq > 1 else None,
        progress_percentage=Decimal("50") if status == "OK" else None,
        target_gap=Decimal("50"),
        trend_direction="TOWARD_TARGET",
        target_status="NOT_MET",
        calculation_status=status,
        target_changed=False,
        extraction_status="EXTRACTED",
        **kw,
    )


def tables() -> dict[str, list[dict[str, Any]]]:
    return {
        "gold.project_360": [
            project_360(IPF),
            project_360(
                PFORR,
                instrument="Program-for-Results Financing",
                loans_with_valuation_caveats=[],
            ),
        ],
        "silver.projects": [
            {
                "record_id": f"sp-{p}",
                "project_id": p,
                "borrower": "Borrower",
                "implementing_agency": "Agency",
                "project_development_objective": "Objective text",
            }
            for p in (IPF, PFORR)
        ],
        "gold.project_timeline": [
            gold(
                "FACT",
                record_id="t1",
                project_id=IPF,
                timeline_event_id="e1",
                event_type="APPROVAL",
                event_title="Board approval",
                event_date=date(2016, 3, 31),
                event_date_status="SOURCE_STATED",
                event_sequence=1,
                date_sequence_anomaly=False,
            ),
            gold(
                record_id="t2",
                project_id=IPF,
                timeline_event_id="e2",
                event_type="ISR_REPORT",
                event_title="ISR 1",
                event_date=date(2017, 1, 5),
                event_date_status="SOURCE_STATED",
                event_sequence=2,
                isr_sequence=1,
                date_sequence_anomaly=True,
            ),
            gold(
                record_id="t3",
                project_id=IPF,
                timeline_event_id="e3",
                event_type="RESTRUCTURING",
                event_title="Restructuring paper",
                event_date=None,
                event_date_status="DERIVED_CANDIDATE",
                candidate_event_date=date(2018, 6, 1),
                candidate_date_status="CANDIDATE",
                event_sequence=3,
                date_sequence_anomaly=False,
            ),
            gold(
                record_id="t4",
                project_id=IPF,
                timeline_event_id="e4",
                event_type="ISR_REPORT",
                event_title="ISR 2",
                event_date=date(2016, 12, 1),
                event_date_status="SOURCE_STATED",
                event_sequence=4,
                isr_sequence=2,
                date_sequence_anomaly=False,
            ),
            gold(
                "FACT",
                record_id="t9",
                project_id=PFORR,
                timeline_event_id="e9",
                event_type="APPROVAL",
                event_title="Board approval",
                event_date=date(2023, 3, 28),
                event_date_status="SOURCE_STATED",
                event_sequence=1,
                date_sequence_anomaly=False,
            ),
        ],
        "silver.isr_snapshots": [
            isr(IPF, 1, "Satisfactory", "Satisfactory", "Moderate"),
            isr(IPF, 2, "Moderately Satisfactory", None, "Substantial"),
            isr(IPF, 3, "Moderately Unsatisfactory", "Moderately Satisfactory", "Substantial"),
            isr(PFORR, 1, "Satisfactory", "Satisfactory", "Substantial"),
        ],
        "silver.loans": [
            {
                "record_id": "l1",
                "project_id": IPF,
                "raw_loan_number": "IBRD86010",
                "original_principal_usd": Decimal("80"),
                "valuation_caveats": [],
            },
            {
                "record_id": "l2",
                "project_id": IPF,
                "raw_loan_number": "IBRD93240",
                "original_principal_usd": Decimal("20"),
                "valuation_caveats": ["JPY commitment; US$ value moves with exchange rates"],
            },
            {
                "record_id": "l3",
                "project_id": PFORR,
                "raw_loan_number": "IBRD94960",
                "original_principal_usd": Decimal("50"),
                "valuation_caveats": [],
            },
        ],
        "silver.isr_loan_disbursements": [
            {
                "record_id": f"d{s}{n}",
                "project_id": IPF,
                "isr_sequence": s,
                "loan_number": n,
                "disbursed_musd": Decimal(s),
                "status": "EXTRACTED",
                "evidence_document_id": f"{IPF}-isr{s}",
                "evidence_page_number": 8,
            }
            for s in (1, 2, 3)
            for n in ("IBRD86010", "IBRD93240")
            if not (s == 1 and n == "IBRD93240")
        ],
        "silver.project_events": [
            {
                "record_id": "ev1",
                "project_id": IPF,
                "event_id": "ev1",
                "event_type": "ADDITIONAL_FINANCING",
                "event_date": date(2019, 1, 1),
                "additional_financing_amount": Decimal("12000000000"),
                "additional_financing_currency": "JPY",
                "source_document": f"{IPF}-af",
                "source_page": 1,
                "extraction_method": "REGEX",
                "status": "EXTRACTED",
            },
            {
                "record_id": "ev2",
                "project_id": IPF,
                "event_id": "ev2",
                "event_type": "RESTRUCTURING",
                "event_date": None,
                "source_document": f"{IPF}-rp",
                "source_page": 2,
                "status": "EXTRACTED",
            },
        ],
        "gold.result_progress": [
            result(IPF, "ind-a", "Direct project beneficiaries", 1),
            result(IPF, "ind-a", "Direct project beneficiaries", 2),
            result(IPF, "ind-b", "Female beneficiaries", 2),
            result(
                IPF,
                "ind-c",
                "Female beneficiaries (number)",
                2,
                identity_review_status="PENDING_ALIAS_REVIEW",
            ),
            result(
                IPF,
                "ind-d",
                "DLI 1 achieved",
                2,
                status="DLI_LAYOUT_NOT_EVALUATED",
                indicator_type="DLI",
            ),
        ],
        "gold.risk_register": [
            gold(
                record_id="r1",
                project_id=IPF,
                risk_or_finding_id="r1",
                record_type="ASSESSMENT_FINDING",
                category="Environmental",
                title="Finding",
                resolution_status="NOT_STATED",
                observed_date=date(2016, 1, 1),
            ),
            gold(
                record_id="r2",
                project_id=IPF,
                risk_or_finding_id="r2",
                record_type="FORMAL_RISK_RATING",
                category="Fiduciary",
                title="Fiduciary",
                rating="Substantial",
                rating_rank=3,
                resolution_status="NOT_STATED",
            ),
            gold(
                record_id="r3",
                project_id=IPF,
                risk_or_finding_id="r3",
                record_type="FORMAL_RISK_RATING",
                category="Political",
                title="Political",
                rating="High",
                rating_rank=4,
                resolution_status="NOT_STATED",
            ),
            gold(
                record_id="r4",
                project_id=IPF,
                risk_or_finding_id="r4",
                record_type="FORMAL_RISK_RATING",
                category="Other",
                title="Unrated",
                rating=None,
                rating_rank=None,
                resolution_status="NOT_STATED",
            ),
        ],
        "gold.attention_signals": [
            gold(
                "SYSTEM_DERIVED_SIGNAL",
                record_id="s1",
                project_id=IPF,
                signal_id="s1",
                rule_id="RESULT_NO_CHANGE",
                rule_version="RESULT_NO_CHANGE@v1",
                signal_category="RESULTS",
                signal_type="t",
                subject="ind-a",
                signal_title="No change",
                signal_description="d",
                severity="INFO",
                signal_status="CURRENT",
                observed_date=date(2024, 2, 1),
                rule_description="rule",
                supporting_record_ids=["x"],
                caveats=[],
            ),
            gold(
                "SYSTEM_DERIVED_SIGNAL",
                record_id="s2",
                project_id=IPF,
                signal_id="s2",
                rule_id="RATING_DOWNGRADE",
                rule_version="RATING_DOWNGRADE@v1",
                signal_category="IMPLEMENTATION_RATING",
                signal_type="t",
                subject="PDO",
                signal_title="Downgrade",
                signal_description="d",
                severity="HIGH",
                signal_status="CURRENT",
                observed_date=date(2024, 3, 1),
                rule_description="rule",
                supporting_record_ids=["y"],
                caveats=["c"],
            ),
            gold(
                "SYSTEM_DERIVED_SIGNAL",
                record_id="s3",
                project_id=IPF,
                signal_id="s3",
                rule_id="RATING_DOWNGRADE",
                rule_version="RATING_DOWNGRADE@v1",
                signal_category="IMPLEMENTATION_RATING",
                signal_type="t",
                subject="IP",
                signal_title="Old downgrade",
                signal_description="d",
                severity="WATCH",
                signal_status="HISTORICAL",
                observed_date=date(2020, 3, 1),
                rule_description="rule",
                supporting_record_ids=["z"],
                caveats=[],
            ),
        ],
    }


def context(reader: InMemoryReader | None = None, **kw: Any) -> ToolContext:
    return ToolContext(
        reader=reader or InMemoryReader(tables()),
        registry=REGISTRY,
        scales=SCALES,
        rules=RULES,
        request_id="req-test",
        **kw,
    )
