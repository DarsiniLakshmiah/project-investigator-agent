"""Synthetic, project-scoped 10D capability inputs; no source/database execution."""

from __future__ import annotations

from worldbank_copilot.investigation.assembly import package_identity
from worldbank_copilot.investigation.claims import CandidateClaim, SynthesisOutput
from worldbank_copilot.investigation.evidence_models import (
    EvidencePackage,
    EvidenceReference,
    MechanicalAssessment,
    OperationType,
    RequirementSummary,
    SnapshotManifest,
)
from worldbank_copilot.investigation.models import AssessmentStatus, fingerprint
from worldbank_copilot.investigation.synthesis import ApprovedContext, source_identity
from worldbank_copilot.routing.models import TemporalKind, TemporalScope, TemporalStatus
from worldbank_copilot.tools.models import ProvenanceClass, SourceRef


def fixture(kind="FACT"):
    scope = TemporalScope(
        kind=TemporalKind.HISTORY, status=TemporalStatus.RESOLVED, explicit=False, defaulted=True
    )
    identity = {
        "project_id": "P000001",
        "table": "synthetic.gold",
        "record_id": "record1",
        "field": "implementation_status",
    }
    provenance = ProvenanceClass(kind)
    ref = EvidenceReference(
        evidence_id=fingerprint("ev", identity),
        project_id="P000001",
        source_type=OperationType.STRUCTURED,
        identity=identity,
        provenance=provenance,
        payload={
            "value": None if kind == "UNKNOWN" else "Delay observed",
            "unknown_reason": "Not supplied" if kind == "UNKNOWN" else None,
        },
        source=SourceRef(table="synthetic.gold", record_id="record1"),
        citation_support="SOURCE_REFERENCE",
    )
    summary = RequirementSummary(
        requirement_id="req_synthetic",
        objective="Describe the supplied status",
        required=True,
        operation_ids=(),
        evidence_ids=(ref.evidence_id,),
        assessment=MechanicalAssessment(
            requirement_id="req_synthetic", status=AssessmentStatus.SATISFIED
        ),
    )
    package = EvidencePackage(
        request_id="synthetic",
        project_id="P000001",
        question=summary.objective,
        temporal_scope=scope,
        requirement_summaries=(summary,),
        operations=(),
        evidence_index=(ref,),
        conflicts=(),
        missing_requirements=(),
        failed_requirements=(),
        snapshots=SnapshotManifest(
            table_versions={"synthetic.gold": 1},
            retrieval_profile=None,
            corpus_identity=None,
            index_identity=None,
        ),
        warnings=(),
        package_fingerprint="pending",
    )
    data = package.model_dump(mode="json")
    data["package_fingerprint"] = fingerprint("package", package_identity(data))
    package = EvidencePackage.model_validate(data)
    entry = ref.model_dump(mode="json")
    entry["source_identity"] = source_identity(ref)
    return ApprovedContext(
        request_id="synthetic",
        project_id="P000001",
        plan_id="plan_synthetic",
        package_fingerprint=package.package_fingerprint,
        temporal_scope=scope.model_dump(mode="json"),
        question=summary.objective,
        requirements=(
            {
                "requirement_id": summary.requirement_id,
                "objective": summary.objective,
                "required": True,
                "evidence_ids": [ref.evidence_id],
            },
        ),
        evidence=(entry,),
        omitted_evidence_ids=(),
        limitations=(
            "Synthetic capability evidence only.",
            "Cross-source atomicity is NOT_ESTABLISHED.",
        ),
    )


def claim(context, **changes):
    entry = context.evidence[0]
    data = dict(
        claim_id="C1",
        claim_text="The supplied record states that a delay was observed.",
        claim_type="ASSERTION",
        provenance_label=entry["provenance"],
        evidence_ids=[entry["evidence_id"]],
        requirement_ids=["req_synthetic"],
        citations=[
            {"evidence_id": entry["evidence_id"], "source_identity": entry["source_identity"]}
        ],
        project_id=context.project_id,
        temporal_scope=context.temporal_scope,
        status="CANDIDATE",
    )
    data.update(changes)
    return CandidateClaim.model_validate(data)


def draft(context, **changes):
    return SynthesisOutput(
        candidate_claims=(claim(context, **changes),),
        insufficient_evidence=False,
        limitations=(),
        summary_claim_ids=("C1",),
    )
