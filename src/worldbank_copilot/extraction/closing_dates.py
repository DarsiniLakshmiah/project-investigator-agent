"""Original closing date reconciliation and explicit enrichment of silver.projects.

No structured source carries an original closing date, so Phase 3 left
``silver_projects.original_closing_date`` NULL. Documents do carry one:

* ISR "Key Dates (by loan)" -> "Orig. Closing Date" (every ISR);
* restructuring / AF paper "Loan Closing" tables -> "Original Closing".

Rules:

1. Per loan: collect every explicit candidate. If at least one exists and all
   agree, the value is ESTABLISHED (status DERIVED_FROM_EXPLICIT_SOURCE, all
   evidence kept). If candidates disagree -> CONFLICT, value NULL. No
   candidates -> unresolved, value NULL.
2. Per project: the original closing date of the project's first loan (earliest
   board approval in silver_loans), taken only if that loan is ESTABLISHED.
   Loans added later (e.g. additional financing) never define the project's
   original closing date.

``silver_projects`` is never mutated. ``apply_project_enrichment`` returns an
enriched *copy* plus lineage, which later layers may choose to use.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from worldbank_copilot.common.quality import CheckCode
from worldbank_copilot.extraction.events import loan_closing_rows
from worldbank_copilot.extraction.models import IsrSnapshot, ProjectEnrichment
from worldbank_copilot.extraction.provenance import (
    EvidenceRef,
    ExtractionIssue,
    ExtractionStatus,
    issue,
)
from worldbank_copilot.parsing.models import ParsedDocument

LOAN_RULE = "all explicit 'Original Closing' values for the loan agree (ISR key dates + papers)"
PROJECT_RULE = "original closing date of the project's first loan (earliest board approval)"


def _candidates(snapshots: list[IsrSnapshot], papers: list[ParsedDocument]):
    found: dict[str, list[tuple[Any, EvidenceRef, str]]] = defaultdict(list)
    for snap in snapshots:
        for kd in snap.loan_key_dates:
            if kd.original_closing_date:
                found[kd.loan_number].append(
                    (kd.original_closing_date, kd.evidence, f"ISR {snap.isr_sequence} key dates")
                )
    for doc in papers:
        for row in loan_closing_rows(doc):
            if row.original:
                found[row.loan_number].append(
                    (row.original, row.ref, f"{doc.document_type} loan closing table")
                )
    return found


def reconcile_original_closing(
    project_id: str,
    snapshots: list[IsrSnapshot],
    papers: list[ParsedDocument],
    loans: list[Any],
) -> tuple[list[ProjectEnrichment], list[ExtractionIssue]]:
    """Return per-loan and per-project enrichment records (never applied here)."""
    records: list[ProjectEnrichment] = []
    issues: list[ExtractionIssue] = []
    candidates = _candidates(snapshots, papers)
    project_loans = sorted(
        (loan for loan in loans if loan.project_id == project_id),
        key=lambda loan: (loan.board_approval_date is None, loan.board_approval_date),
    )
    loan_numbers = [loan.raw_loan_number for loan in project_loans]
    by_loan: dict[str, ProjectEnrichment] = {}
    for loan in sorted(set(loan_numbers) | set(candidates)):
        seen = candidates.get(loan, [])
        values = sorted({value for value, _, _ in seen})
        summary = [{"value": str(v), "source": src, "label": ref.label} for v, ref, src in seen]
        if len(values) == 1:
            status, value = ExtractionStatus.DERIVED_FROM_EXPLICIT_SOURCE, values[0]
        elif len(values) > 1:
            status, value = ExtractionStatus.CONFLICT, None
            issues.append(
                issue(
                    CheckCode.EXTRACTION_CONFLICT,
                    "WARNING",
                    f"{project_id} {loan}: original closing candidates disagree: "
                    f"{[str(v) for v in values]}",
                    seen[0][1],
                )
            )
        else:
            status, value = ExtractionStatus.MISSING, None
        # Keep the evidence compact but complete across sources: first and last
        # occurrence of each distinct value per source kind (ISRs, each paper type).
        refs: list[EvidenceRef] = []
        for distinct in values:
            by_kind: dict[str, list[EvidenceRef]] = defaultdict(list)
            for v, ref, src in seen:
                if v == distinct:
                    by_kind["ISR" if src.startswith("ISR") else ref.document_id].append(ref)
            for hits in by_kind.values():
                refs += [hits[0]] + ([hits[-1]] if len(hits) > 1 else [])
        record = ProjectEnrichment(
            project_id=project_id,
            target_table="silver_loans",
            target_field="original_closing_date",
            loan_number=loan,
            value=value,
            status=status,
            rule=LOAN_RULE,
            evidence=refs,
            candidates=summary[:3] + ([{"more": len(summary) - 3}] if len(summary) > 3 else []),
        )
        by_loan[loan] = record
        records.append(record)

    first = loan_numbers[0] if loan_numbers else None
    base = by_loan.get(first) if first else None
    if base is not None and base.value is not None:
        records.append(
            ProjectEnrichment(
                project_id=project_id,
                target_table="silver_projects",
                target_field="original_closing_date",
                loan_number=first,
                value=base.value,
                status=ExtractionStatus.DERIVED_FROM_EXPLICIT_SOURCE,
                rule=PROJECT_RULE,
                evidence=base.evidence,
            )
        )
        issues.append(
            issue(
                CheckCode.ORIGINAL_CLOSING_DATE_ESTABLISHED,
                "INFO",
                f"{project_id}: original closing date {base.value} established from "
                f"{first} ({len(base.candidates)} candidate group(s), all agree)",
                base.evidence[0] if base.evidence else None,
            )
        )
    else:
        records.append(
            ProjectEnrichment(
                project_id=project_id,
                target_table="silver_projects",
                target_field="original_closing_date",
                loan_number=first,
                value=None,
                status=base.status if base else ExtractionStatus.MISSING,
                rule=PROJECT_RULE,
                evidence=base.evidence if base else [],
            )
        )
        issues.append(
            issue(
                CheckCode.ORIGINAL_CLOSING_DATE_UNRESOLVED,
                "WARNING",
                f"{project_id}: original closing date not established",
            )
        )
    return records, issues


def apply_project_enrichment(
    projects: list[Any], enrichments: list[ProjectEnrichment]
) -> tuple[list[Any], list[dict]]:
    """Enriched copies of silver_projects rows (inputs untouched) plus lineage rows."""
    chosen = {
        (e.project_id, e.target_field): e
        for e in enrichments
        if e.target_table == "silver_projects" and e.value is not None
    }
    enriched, lineage = [], []
    for project in projects:
        updates = {}
        for (project_id, field), record in chosen.items():
            if project_id != project.project_id:
                continue
            current = getattr(project, field)
            if current is not None and current != record.value:
                lineage.append(
                    {
                        "project_id": project_id,
                        "field": field,
                        "applied": False,
                        "reason": "structured value present and different; not overwritten",
                        "structured_value": str(current),
                        "document_value": str(record.value),
                    }
                )
                continue
            updates[field] = record.value
            lineage.append(
                {
                    "project_id": project_id,
                    "field": field,
                    "applied": True,
                    "value": str(record.value),
                    "rule": record.rule,
                    "status": record.status.value,
                    "evidence": [ref.label for ref in record.evidence],
                }
            )
        enriched.append(project.model_copy(update=updates))
    return enriched, lineage
