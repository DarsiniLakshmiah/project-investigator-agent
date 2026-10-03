"""Explicit application context projection; original evidence report is immutable."""

import json

from worldbank_copilot.investigation.assembly import package_identity
from worldbank_copilot.investigation.claims import CriticOutput, Failure, NodeError, SynthesisOutput
from worldbank_copilot.investigation.models import fingerprint
from worldbank_copilot.investigation.synthesis import (
    CRITIC_INSTRUCTIONS,
    INSTRUCTIONS,
    build_context,
)


def bounded_report(report):
    """Keep a deterministic subset, retaining every requirement and declaring omissions.

    Validate with the actual reviewed context reader and unchanged byte bound.
    Missing references never become satisfied requirements or new evidence.
    """
    policy = report.investigation.policy
    overhead = max(
        len(INSTRUCTIONS.encode())
        + len(
            json.dumps(
                SynthesisOutput.model_json_schema(), sort_keys=True, separators=(",", ":")
            ).encode()
        ),
        len(CRITIC_INSTRUCTIONS.encode())
        + len(
            json.dumps(
                CriticOutput.model_json_schema(), sort_keys=True, separators=(",", ":")
            ).encode()
        ),
    )
    bound = (
        min(policy.max_synthesis_context_tokens, policy.max_critic_context_tokens) - overhead - 2000
    )
    try:
        context = build_context(report, byte_limit=bound)
        if not {ref for conflict in report.package.conflicts for ref in conflict.evidence_ids} <= {
            e["evidence_id"] for e in context.evidence
        }:
            raise NodeError(Failure.INSUFFICIENT_EVIDENCE)
        return report
    except NodeError as exc:
        if exc.category.value != "CONTEXT_BUDGET_EXCEEDED":
            raise
    package = report.package
    ordered = []
    for summary in package.requirement_summaries:
        if summary.evidence_ids:
            ordered.append(summary.evidence_ids[0])
    ordered.extend(e.evidence_id for e in package.evidence_index)
    ordered = list(dict.fromkeys(ordered))
    for limit in (16, 8, 4, 2, 1, 0):
        ids = set(ordered[:limit]) | {
            ref for conflict in package.conflicts for ref in conflict.evidence_ids
        }
        data = package.model_dump(mode="python")
        data["evidence_index"] = tuple(e for e in package.evidence_index if e.evidence_id in ids)
        data["operations"] = tuple(
            type(o)(
                **{
                    **o.model_dump(mode="python"),
                    "evidence_links": tuple(
                        link for link in o.evidence_links if link.evidence_id in ids
                    ),
                }
            )
            for o in package.operations
        )
        summaries = []
        for summary in package.requirement_summaries:
            kept = tuple(e for e in summary.evidence_ids if e in ids)
            assessment = type(summary.assessment)(
                **{
                    **summary.assessment.model_dump(mode="python"),
                    "supporting_evidence_ids": kept[:100],
                }
            )
            summaries.append(
                type(summary)(
                    **{
                        **summary.model_dump(mode="python"),
                        "evidence_ids": kept,
                        "assessment": assessment,
                    }
                )
            )
        data["requirement_summaries"] = tuple(summaries)
        data["conflicts"] = tuple(c for c in package.conflicts if set(c.evidence_ids) <= ids)
        data["missing_requirements"] = tuple(
            sorted(
                set(package.missing_requirements)
                | {s.requirement_id for s in summaries if s.required and not s.evidence_ids}
            )
        )
        omitted = len(package.evidence_index) - len(data["evidence_index"])
        data["warnings"] = (
            *package.warnings,
            f"Application context projection omits {omitted} "
            "references; omission is not evidence of absence.",
        )
        data["package_fingerprint"] = "pending"
        projected = type(package)(**data)
        data["package_fingerprint"] = fingerprint(
            "package", package_identity(projected.model_dump(mode="json"))
        )
        projected = type(package)(**data)
        trial = type(report)(**{**report.model_dump(mode="python"), "package": projected})
        try:
            context = build_context(trial, byte_limit=bound)
            if not {ref for conflict in package.conflicts for ref in conflict.evidence_ids} <= {
                e["evidence_id"] for e in context.evidence
            }:
                raise NodeError(Failure.INSUFFICIENT_EVIDENCE)
            return trial
        except NodeError as exc:
            if exc.category.value != "CONTEXT_BUDGET_EXCEEDED":
                raise
    raise NodeError(Failure.CONTEXT_BUDGET_EXCEEDED)
