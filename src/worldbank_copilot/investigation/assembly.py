"""Deterministic evidence assembly and byte-bounded future context projections."""

from __future__ import annotations

import json
from enum import StrEnum

from pydantic import BaseModel, JsonValue

from worldbank_copilot.investigation.evidence_models import (
    SPECS,
    EvidenceLimits,
    EvidencePackage,
    EvidenceReference,
    MechanicalAssessment,
    MechanicalConflict,
    OperationRecord,
    OperationStatus,
    OperationType,
    RequirementSummary,
    SnapshotManifest,
)
from worldbank_copilot.investigation.models import AssessmentStatus, InvestigationState, fingerprint
from worldbank_copilot.investigation.policy import Contract
from worldbank_copilot.retrieval.contract import (
    HINTS_NOT_APPLIED,
    NO_EVIDENCE_MEANING,
    RetrievalProfile,
)
from worldbank_copilot.tools.models import Fact, ProvenanceClass, SourceRef


class AssemblyIntegrityError(ValueError):
    """No partial retention of a component that violates source integrity."""


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _walk(value, path="items"):
    if isinstance(value, Fact):
        yield path, value
    elif isinstance(value, BaseModel):
        if (
            hasattr(value, "source")
            and isinstance(value.source, SourceRef)
            and hasattr(value, "provenance_class")
        ):
            yield path, value
        else:
            for name in type(value).model_fields:
                yield from _walk(getattr(value, name), f"{path}.{name}")
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            yield from _walk(item, f"{path}[{i}]")
    elif isinstance(value, dict):
        for name, item in sorted(value.items()):
            yield from _walk(item, f"{path}.{name}")


def references(record: OperationRecord):
    output, links = [], []
    from worldbank_copilot.investigation.evidence_models import EvidenceLink

    if record.structured_result is not None:
        result = record.structured_result
        for path, node in _walk(result.items):
            source = node.source
            identity = {
                "project_id": record.project_id,
                "operation_id": record.operation_id,
                "snapshots": {
                    table: result.data_snapshot.get(table)
                    for table in SPECS[record.tool_or_profile].tables
                },
                "field_path": path,
                "source": None if source is None else source.model_dump(mode="json"),
            }
            if source is None and not result.data_snapshot:
                raise AssemblyIntegrityError("SOURCE_IDENTITY_MISSING")
            ref = EvidenceReference(
                evidence_id=fingerprint("ev", identity),
                project_id=record.project_id,
                source_type=OperationType.STRUCTURED,
                identity=identity,
                provenance=node.provenance_class,
                payload=node.model_dump(mode="json"),
                source=source,
                citation_support="SOURCE_REFERENCE",
            )
            output.append(ref)
            links.append(EvidenceLink(evidence_id=ref.evidence_id))
    if record.retrieval_result is not None:
        for item in record.retrieval_result.evidence:
            evidence = item.evidence
            if not evidence.chunk_id or not evidence.source_hash:
                raise AssemblyIntegrityError("SOURCE_IDENTITY_MISSING")
            identity = {
                "project_id": record.project_id,
                "chunk_id": evidence.chunk_id,
                "source_hash": evidence.source_hash,
            }
            citation = evidence.citation
            complete = (
                bool(citation.document_label)
                and all(p > 0 for p in citation.pages)
                and citation.document_id == evidence.document_id
                and citation.source_file == evidence.source_file
            )
            ref = EvidenceReference(
                evidence_id=fingerprint("ev", identity),
                project_id=record.project_id,
                source_type=OperationType.DOCUMENT,
                identity=identity,
                provenance=item.provenance_class,
                payload={
                    "text": evidence.text,
                    "context_text": evidence.context_text,
                    "document_id": evidence.document_id,
                    "document_date": evidence.document_date.isoformat()
                    if evidence.document_date
                    else None,
                    "isr_sequence": evidence.isr_sequence,
                    "source_file": evidence.source_file,
                },
                citation=citation,
                citation_support="COMPLETE" if complete else "LIMITED",
            )
            output.append(ref)
            links.append(EvidenceLink(evidence_id=ref.evidence_id, local_id=evidence.evidence_id))
    return tuple(output), tuple(links)


def merge_references(existing, incoming, limit):
    out = {r.evidence_id: r for r in existing}
    for ref in incoming:
        old = out.get(ref.evidence_id)
        if old is not None and old != ref:
            raise AssemblyIntegrityError("EVIDENCE_IDENTITY_COLLISION")
        out[ref.evidence_id] = ref
    if len(out) > limit:
        raise AssemblyIntegrityError("REFERENCE_BOUND_EXCEEDED")
    return tuple(out[k] for k in sorted(out))


def _conflicts(refs):
    groups = {}
    for ref in refs:
        payload, source = ref.payload, ref.source
        if (
            ref.source_type != OperationType.STRUCTURED
            or ref.provenance != ProvenanceClass.FACT
            or source is None
            or not source.record_id
            or payload.get("derivation") is not None
            or "name" not in payload
            or "value" not in payload
        ):
            continue
        version = ref.identity.get("snapshots", {}).get(source.table)
        if version is None:
            continue  # unknown snapshots do not establish equal temporal basis
        key = {
            "project_id": ref.project_id,
            "table": source.table,
            "record_id": source.record_id,
            "field": payload["name"],
            "unit": payload.get("unit"),
            "table_version": version,
        }
        groups.setdefault(canonical_json(key), []).append(ref)
    return tuple(
        MechanicalConflict(
            comparison_key=json.loads(key), evidence_ids=tuple(sorted(r.evidence_id for r in group))
        )
        for key, group in sorted(groups.items())
        if len({canonical_json(r.payload["value"]) for r in group}) > 1
    )


def package_identity(data):
    """Content identity excludes latency, transport timing and execution scheduling."""
    operations = []
    for record in data["operations"]:
        operations.append(
            {
                k: record[k]
                for k in (
                    "operation_id",
                    "requirement_ids",
                    "project_id",
                    "operation_type",
                    "tool_or_profile",
                    "status",
                    "reason_code",
                    "attempted",
                    "evidence_links",
                    "error_class",
                )
            }
        )
    return {
        **{k: v for k, v in data.items() if k not in ("package_fingerprint", "operations")},
        "operations": sorted(operations, key=lambda op: op["operation_id"]),
    }


def assemble(
    state: InvestigationState,
    records: tuple[OperationRecord, ...],
    refs: tuple[EvidenceReference, ...],
    profile: RetrievalProfile | None,
    limits: EvidenceLimits,
):
    if any(r.project_id != state.project_id for r in refs) or any(
        o.project_id != state.project_id for o in records
    ):
        raise AssemblyIntegrityError("PACKAGE_SCOPE_MISMATCH")
    refs = merge_references((), refs, limits.max_references)
    snapshots = {}
    warnings = {
        "Mechanical execution is recorded; semantic evidence sufficiency is NOT_ASSESSED.",
        "Contextual snapshots do not establish historical sufficiency.",
    }
    for record in records:
        if record.structured_result is not None:
            result = record.structured_result
            for table, version in result.data_snapshot.items():
                if table in snapshots and snapshots[table] != version:
                    raise AssemblyIntegrityError("SNAPSHOT_DRIFT")
                snapshots[table] = version
            warnings.update(result.caveats)
            warnings.update(result.notices)
        if record.retrieval_result is not None:
            warnings.update(record.retrieval_result.warnings)
            if not record.retrieval_result.recorded_hints.applied:
                warnings.add(HINTS_NOT_APPLIED)
        if record.status == OperationStatus.NO_EVIDENCE:
            warnings.add(NO_EVIDENCE_MEANING)
    conflicts = _conflicts(refs)
    conflicting_ids = {e for c in conflicts for e in c.evidence_ids}
    summaries, missing, failed = [], [], []
    for req in state.requirements:
        operations = [o for o in records if req.requirement_id in o.requirement_ids]
        ids = tuple(sorted({link.evidence_id for o in operations for link in o.evidence_links}))
        errors = [
            o
            for o in operations
            if o.status not in (OperationStatus.OK, OperationStatus.NO_EVIDENCE)
        ]
        if not operations:
            status = AssessmentStatus.PENDING
            reasons = ()
        elif conflicting_ids & set(ids):
            status = AssessmentStatus.CONFLICTING
            reasons = ("MECHANICAL_SOURCE_VALUE_CONFLICT",)
        elif errors:
            status = AssessmentStatus.PARTIAL if ids else AssessmentStatus.ERROR
            reasons = tuple(sorted({o.reason_code for o in errors}))
        elif not ids:
            status = AssessmentStatus.UNSATISFIED
            reasons = ("NO_EVIDENCE_FROM_COMPLETED_OPERATIONS",)
        else:
            status = AssessmentStatus.PARTIAL
            reasons = ("SEMANTIC_SUFFICIENCY_NOT_ASSESSED",)
        if req.required and (not ids or len(operations) < len(req.operation_ids)):
            missing.append(req.requirement_id)
        if errors:
            failed.append(req.requirement_id)
        assessment = MechanicalAssessment(
            requirement_id=req.requirement_id,
            status=status,
            supporting_evidence_ids=ids[:100],
            reason_codes=reasons,
            unsupported_aspects=()
            if status == AssessmentStatus.PENDING
            else ("Semantic support is not assessed in Phase 10C.",),
        )
        summaries.append(
            RequirementSummary(
                requirement_id=req.requirement_id,
                objective=req.objective,
                required=req.required,
                operation_ids=req.operation_ids,
                evidence_ids=ids,
                assessment=assessment,
            )
        )
    manifest = SnapshotManifest(
        table_versions=snapshots,
        retrieval_profile=state.retrieval_profile_id,
        corpus_identity=None,
        chunk_strategy_identity=None
        if profile is None
        else f"{profile.chunk_strategy}@{profile.chunk_strategy_version}",
        index_identity=None
        if profile is None
        else f"{profile.vector_search_endpoint}/{profile.vector_search_index}",
    )
    warnings.add(manifest.limitation)
    data = dict(
        request_id=state.request_id,
        project_id=state.project_id,
        question=state.question,
        temporal_scope=state.temporal_scope,
        requirement_summaries=tuple(summaries),
        operations=records,
        evidence_index=refs,
        conflicts=conflicts,
        missing_requirements=tuple(missing),
        failed_requirements=tuple(failed),
        snapshots=manifest,
        warnings=tuple(sorted(warnings)),
        package_fingerprint="pending",
    )
    preliminary = EvidencePackage(**data)
    data["package_fingerprint"] = fingerprint(
        "package", package_identity(preliminary.model_dump(mode="json"))
    )
    package = EvidencePackage(**data)
    if len(package.model_dump_json().encode()) > limits.max_package_bytes:
        raise AssemblyIntegrityError("PACKAGE_BOUND_EXCEEDED")
    return package


class ProjectionRole(StrEnum):
    INVESTIGATOR = "INVESTIGATOR"
    SYNTHESIS = "SYNTHESIS"
    CRITIC = "CRITIC"


class EvidenceProjection(Contract):
    role: ProjectionRole
    request_id: str
    project_id: str
    package_fingerprint: str
    requirements: tuple[dict[str, JsonValue], ...]
    evidence: tuple[EvidenceReference, ...]
    warnings: tuple[str, ...]
    conflicts: tuple[MechanicalConflict, ...]
    omitted_evidence_ids: tuple[str, ...]
    omitted_requirement_ids: tuple[str, ...] = ()
    truncated: bool
    trust: str = "UNTRUSTED_EVIDENCE"


def project_context(
    package: EvidencePackage, policy, role: ProjectionRole, *, limits=None, requirement_ids=None
):
    limits = limits or EvidenceLimits()
    if package.package_fingerprint != fingerprint(
        "package", package_identity(package.model_dump(mode="json"))
    ):
        raise AssemblyIntegrityError("PACKAGE_FINGERPRINT_MISMATCH")
    token_bound = {
        ProjectionRole.INVESTIGATOR: policy.max_investigator_context_tokens,
        ProjectionRole.SYNTHESIS: policy.max_synthesis_context_tokens,
        ProjectionRole.CRITIC: policy.max_critic_context_tokens,
    }[role]
    byte_bound = token_bound * limits.projection_bytes_per_policy_token
    allowed = {s.requirement_id for s in package.requirement_summaries}
    selected = allowed if requirement_ids is None else set(requirement_ids)
    if selected - allowed:
        raise ValueError("unknown requirement selection")
    summaries = tuple(
        s.model_dump(mode="json")
        for s in package.requirement_summaries
        if s.requirement_id in selected
    )
    ids = {
        e
        for s in package.requirement_summaries
        if s.requirement_id in selected
        for e in s.evidence_ids
    }
    evidence = tuple(e for e in package.evidence_index if e.evidence_id in ids)
    # Whole references only: never cut a citation, value or source path midway.
    kept = []
    omitted = [ref.evidence_id for ref in evidence]
    base = dict(
        role=role,
        request_id=package.request_id,
        project_id=package.project_id,
        package_fingerprint=package.package_fingerprint,
        requirements=summaries,
        warnings=package.warnings,
        conflicts=package.conflicts,
        omitted_evidence_ids=(),
        truncated=False,
    )
    for ref in evidence:
        remaining = tuple(e for e in omitted if e != ref.evidence_id)
        trial = EvidenceProjection(
            **{
                **base,
                "evidence": tuple((*kept, ref)),
                "omitted_evidence_ids": remaining,
                "truncated": bool(remaining),
            }
        )
        if len(trial.model_dump_json().encode()) <= byte_bound:
            kept.append(ref)
            omitted.remove(ref.evidence_id)
    projection = EvidenceProjection(
        **{
            **base,
            "evidence": tuple(kept),
            "omitted_evidence_ids": tuple(omitted),
            "truncated": bool(omitted),
        }
    )
    if len(projection.model_dump_json().encode()) > byte_bound:
        raise ValueError("projection metadata exceeds byte bound; select fewer requirements")
    return projection
