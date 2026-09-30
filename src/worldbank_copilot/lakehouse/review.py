"""Indicator alias review artefact (human review of POSSIBLE_INDICATOR_MATCH pairs).

Each candidate pair from Phase 5 identity resolution becomes one row with both
indicators' names, units, types, ISR coverage and source references. The
``review_status`` is ``PENDING_REVIEW`` unless the pair is already covered by a
reviewed entry in ``configs/indicator_aliases.yaml`` (``APPROVED_ALIAS``).

Nothing here writes ``indicator_aliases.yaml``: approving an alias is a human
decision recorded in that file (see README "Indicator alias review").
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from worldbank_copilot.common.quality import CheckCode
from worldbank_copilot.extraction.models import ResultObservation
from worldbank_copilot.extraction.provenance import ExtractionIssue
from worldbank_copilot.lakehouse.contracts import Column, DType, Role, TableContract, build_contract
from worldbank_copilot.lakehouse.records import Dataset, LoadContext, finalize

REVIEW_STATUSES = ("PENDING_REVIEW", "APPROVED_ALIAS", "REJECTED")
REVIEW_FILE = Path("review") / "indicator_alias_candidates.csv"


def _side(prefix: str) -> list[Column]:
    return [
        Column(f"indicator_key_{prefix}", DType.STRING, False, Role.KEY),
        Column(f"indicator_name_raw_{prefix}", DType.STRING, False),
        Column(f"indicator_name_normalized_{prefix}", DType.STRING, False),
        Column(f"unit_{prefix}", DType.STRING, True),
        Column(f"indicator_types_{prefix}", DType.ARRAY_STRING, False),
        Column(f"first_isr_sequence_{prefix}", DType.BIGINT, True),
        Column(f"last_isr_sequence_{prefix}", DType.BIGINT, True),
        Column(f"isr_sequences_{prefix}", DType.ARRAY_BIGINT, False),
        Column(f"observation_count_{prefix}", DType.BIGINT, False),
        Column(f"source_refs_{prefix}", DType.ARRAY_STRING, False, Role.PROVENANCE),
    ]


def candidates_contract() -> TableContract:
    body = [
        Column("project_id", DType.STRING, False),
        *_side("a"),
        *_side("b"),
        Column("candidate_reason", DType.STRING, False),
        Column("review_status", DType.STRING, False, Role.QUALITY, REVIEW_STATUSES),
        Column(
            "alias_config_entry",
            DType.STRING,
            True,
            description="Canonical name when an approved alias covers the pair.",
        ),
    ]
    return build_contract(
        "indicator_match_candidates",
        "silver",
        "indicator_match_candidates",
        body,
        ("project_id", "indicator_key_a", "indicator_key_b"),
        "Near-identical indicator pairs kept separate pending human review. They never "
        "establish identity.",
        profile_groups=[("project_id",), ("review_status",)],
    )


def _side_values(prefix: str, key: str, observations: list[ResultObservation]) -> dict[str, Any]:
    members = sorted(
        (o for o in observations if o.indicator_key == key),
        key=lambda o: (o.isr_sequence is None, o.isr_sequence or 0),
    )
    if not members:
        raise KeyError(f"no observations for indicator {key}")
    sequences = sorted({o.isr_sequence for o in members if o.isr_sequence is not None})
    refs = [members[0].source_ref.label] + (
        [members[-1].source_ref.label] if len(members) > 1 else []
    )
    return {
        f"indicator_key_{prefix}": key,
        f"indicator_name_raw_{prefix}": members[0].indicator_name_raw,
        f"indicator_name_normalized_{prefix}": members[0].indicator_name_normalized,
        f"unit_{prefix}": members[0].unit,
        f"indicator_types_{prefix}": sorted({o.indicator_type or "UNKNOWN" for o in members}),
        f"first_isr_sequence_{prefix}": sequences[0] if sequences else None,
        f"last_isr_sequence_{prefix}": sequences[-1] if sequences else None,
        f"isr_sequences_{prefix}": sequences,
        f"observation_count_{prefix}": len(members),
        f"source_refs_{prefix}": refs,
    }


def candidate_bodies(
    issues: list[ExtractionIssue],
    observations: list[ResultObservation],
    aliases: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    bodies = []
    for issue in issues:
        if issue.code != CheckCode.POSSIBLE_INDICATOR_MATCH:
            continue
        a, b = sorted(issue.details["indicator_keys"])
        project_id = next(o.project_id for o in observations if o.indicator_key == a)
        if (project_id, a, b) in seen:
            continue
        seen.add((project_id, a, b))
        body = {
            "project_id": project_id,
            **_side_values("a", a, observations),
            **_side_values("b", b, observations),
        }
        project_aliases = aliases.get(project_id, {})
        pair = {body["indicator_name_normalized_a"], body["indicator_name_normalized_b"]}
        entry = next((c for alias, c in project_aliases.items() if {alias, c} == pair), None)
        status = "APPROVED_ALIAS" if entry else "PENDING_REVIEW"
        body.update(
            candidate_reason=issue.details.get("reason", ""),
            review_status=status,
            alias_config_entry=entry,
        )
        bodies.append(body)
    bodies.sort(
        key=lambda b: (
            b["project_id"],
            b["indicator_name_normalized_a"],
            b["indicator_name_normalized_b"],
        )
    )
    return bodies


def candidates_dataset(
    issues: list[ExtractionIssue],
    observations: list[ResultObservation],
    aliases: dict[str, dict[str, str]],
    ctx: LoadContext,
) -> Dataset:
    return finalize(candidates_contract(), candidate_bodies(issues, observations, aliases), ctx)


REVIEW_COLUMNS = (
    "candidate_id",
    "project_id",
    "review_status",
    "reviewer_decision",
    "reviewer",
    "review_notes",
    "candidate_reason",
    "indicator_name_raw_a",
    "indicator_name_raw_b",
    "unit_a",
    "unit_b",
    "indicator_types_a",
    "indicator_types_b",
    "first_isr_sequence_a",
    "last_isr_sequence_a",
    "first_isr_sequence_b",
    "last_isr_sequence_b",
    "source_refs_a",
    "source_refs_b",
    "indicator_name_normalized_a",
    "indicator_name_normalized_b",
    "indicator_key_a",
    "indicator_key_b",
)


def write_review_csv(dataset: Dataset, path: Path) -> Path:
    """Review sheet: reviewer columns are empty; decisions go into indicator_aliases.yaml."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in dataset.rows:
            out = {k: row.get(k) for k in REVIEW_COLUMNS}
            out.update(
                candidate_id=row["record_id"], reviewer_decision="", reviewer="", review_notes=""
            )
            for k in ("indicator_types_a", "indicator_types_b", "source_refs_a", "source_refs_b"):
                out[k] = " | ".join(row[k])
            writer.writerow(out)
    return path
