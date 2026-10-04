"""Synthetic governed-evidence entries for offline integrity tests (no I/O)."""

from worldbank_copilot.investigation.evidence_models import EvidenceReference, OperationType
from worldbank_copilot.investigation.models import fingerprint
from worldbank_copilot.investigation.synthesis import source_identity
from worldbank_copilot.tools.models import ProvenanceClass, SourceRef


def entry(kind, record):
    identity = {
        "project_id": "P000001",
        "table": "synthetic.gold",
        "record_id": record,
        "field": "implementation_status",
    }
    ref = EvidenceReference(
        evidence_id=fingerprint("ev", identity),
        project_id="P000001",
        source_type=OperationType.STRUCTURED,
        identity=identity,
        provenance=ProvenanceClass(kind),
        payload={
            "value": None if kind == "UNKNOWN" else "Delay observed",
            "unknown_reason": "Not supplied" if kind == "UNKNOWN" else None,
        },
        source=SourceRef(table="synthetic.gold", record_id=record),
        citation_support="SOURCE_REFERENCE",
    )
    data = ref.model_dump(mode="json")
    data["source_identity"] = source_identity(ref)
    return data
