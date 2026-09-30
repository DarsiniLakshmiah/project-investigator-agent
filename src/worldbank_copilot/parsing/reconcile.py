"""Reconcile manifest/inventory metadata with values found in document content.

Per field:

* neither source has a value            -> UNKNOWN (resolved NULL)
* only the manifest/inventory has one   -> MANIFEST_ONLY (resolved = manifest)
* only the document has one             -> DOCUMENT_ONLY (resolved = document)
* both agree                            -> CONFIRMED
* both differ, document value confident -> CORRECTED_FROM_DOCUMENT (resolved = document;
                                           manifest value kept and reported)
* both differ, document value uncertain -> CONFLICT (resolved NULL)

Document content outranks the manifest only when the text clearly identifies the
value. Nothing is replaced silently: every field records manifest_value,
document_value, resolved_value and the resolution reason.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from worldbank_copilot.common.identifiers import normalize_loan_number
from worldbank_copilot.parsing.metadata import DocumentEvidence
from worldbank_copilot.parsing.models import ExtractedValue, FieldValidation, ValidationStatus

RECONCILED_FIELDS = (
    "project_id",
    "document_type",
    "document_date",
    "isr_sequence",
    "report_number",
    "loan_number",
)


def _comparable(field: str, value: Any) -> Any:
    if value is None:
        return None
    if field == "loan_number":
        return normalize_loan_number(str(value)) or str(value).upper()
    if field == "document_date" and isinstance(value, str):
        return date.fromisoformat(value)
    if isinstance(value, str):
        return value.strip().upper()
    return value


def _jsonable(value: Any) -> Any:
    """Store values in their JSON form so saved and reloaded validations are identical."""
    return value.isoformat() if isinstance(value, date) else value


def reconcile_field(
    field: str, manifest_value: Any, document: ExtractedValue | None
) -> FieldValidation:
    manifest_value = _jsonable(manifest_value)
    if document is not None:
        document = document.model_copy(update={"value": _jsonable(document.value)})
    document_value = document.value if document else None
    m, d = _comparable(field, manifest_value), _comparable(field, document_value)
    common = {"field": field, "manifest_value": manifest_value,
              "document_value": document_value, "evidence": document}  # fmt: skip
    if m is None and d is None:
        return FieldValidation(**common, status=ValidationStatus.UNKNOWN, resolved_value=None,
                               resolution_reason="not in manifest/inventory and not found in "
                               "document text")  # fmt: skip
    if d is None:
        return FieldValidation(**common, status=ValidationStatus.MANIFEST_ONLY,
                               resolved_value=manifest_value,
                               resolution_reason="not identifiable in document text; "
                               "manifest/inventory value kept")  # fmt: skip
    if m is None:
        return FieldValidation(**common, status=ValidationStatus.DOCUMENT_ONLY,
                               resolved_value=document_value,
                               resolution_reason=f"found in document ({document.method}, page "
                               f"{document.page_number}); no manifest value")  # fmt: skip
    if m == d:
        return FieldValidation(**common, status=ValidationStatus.CONFIRMED,
                               resolved_value=document_value,
                               resolution_reason=f"document text agrees ({document.method}, "
                               f"page {document.page_number})")  # fmt: skip
    if document.confident:
        return FieldValidation(**common, status=ValidationStatus.CORRECTED_FROM_DOCUMENT,
                               resolved_value=document_value,
                               resolution_reason=f"document text clearly states "
                               f"{document_value!r} ({document.method}, page "
                               f"{document.page_number}); manifest value {manifest_value!r} "
                               "superseded but recorded")  # fmt: skip
    return FieldValidation(**common, status=ValidationStatus.CONFLICT, resolved_value=None,
                           resolution_reason="manifest and uncertain document value disagree; "
                           "left unresolved")  # fmt: skip


def reconcile(manifest: dict[str, Any], evidence: DocumentEvidence) -> list[FieldValidation]:
    """``manifest`` holds the inventory/manifest values keyed by RECONCILED_FIELDS."""
    return [
        reconcile_field(field, manifest.get(field), getattr(evidence, field))
        for field in RECONCILED_FIELDS
    ]


def resolved(validations: list[FieldValidation], field: str) -> Any:
    return next((v.resolved_value for v in validations if v.field == field), None)
