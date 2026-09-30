"""Indicator identity across ISRs (silver.project_results.indicator_key).

Order of rules:

1. Exact normalized name (case-folded, glyph-free, whitespace-collapsed, unit
   and type suffix kept) -> one key.
2. Documented alias (``configs/indicator_aliases.yaml``) -> the alias's
   canonical key. The file starts empty: no alias is asserted without review.
3. Explicit source IDs: the ``IN########`` IDs printed in ISRs were checked
   and are *report-specific* (every ISR prints new IDs for the same
   indicator). They are kept on each observation but never used to link
   observations across ISRs.

Near-identical names (same name once the unit/type parenthetical is removed,
or one name a long prefix of another) are *not* merged. They are reported as
POSSIBLE_INDICATOR_MATCH for review. No fuzzy or embedding matching is used.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path

import yaml

from worldbank_copilot.common.quality import CheckCode
from worldbank_copilot.extraction.models import ResultObservation
from worldbank_copilot.extraction.provenance import ExtractionIssue, issue

MIN_PREFIX_CHARS = 30
_SUFFIX = re.compile(r"\s*\([^()]*\)\s*$")
_PUNCT = re.compile(r"[^\w%]+")


def indicator_key(project_id: str, normalized_name: str) -> str:
    return f"{project_id}-IND-{hashlib.sha1(normalized_name.encode()).hexdigest()[:10]}"


def base_name(normalized_name: str) -> str:
    """Name without trailing unit/type parentheticals and punctuation."""
    text = normalized_name
    while _SUFFIX.search(text):
        text = _SUFFIX.sub("", text)
    return " ".join(_PUNCT.sub(" ", text).split())


def load_aliases(path: Path | None) -> dict[str, dict[str, str]]:
    """{project_id: {alias normalized name: canonical normalized name}}."""
    if path is None or not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, dict[str, str]] = {}
    for project_id, entries in (data.get("projects") or {}).items():
        for entry in entries or []:
            canonical = " ".join(str(entry["canonical"]).split()).casefold()
            for alias in entry.get("aliases") or []:
                out.setdefault(project_id, {})[" ".join(str(alias).split()).casefold()] = canonical
    return out


def resolve_identities(
    observations: list[ResultObservation],
    aliases: dict[str, dict[str, str]] | None = None,
) -> list[ExtractionIssue]:
    """Assign indicator_key/identity_basis in place; return review observations."""
    aliases = aliases or {}
    issues: list[ExtractionIssue] = []
    for obs in observations:
        canonical = aliases.get(obs.project_id, {}).get(obs.indicator_name_normalized)
        if canonical:
            obs.indicator_key = indicator_key(obs.project_id, canonical)
            obs.identity_basis = "ALIAS"
        else:
            obs.indicator_key = indicator_key(obs.project_id, obs.indicator_name_normalized)
            obs.identity_basis = "EXACT_NAME"

    by_project: dict[str, dict[str, list[ResultObservation]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for obs in observations:
        by_project[obs.project_id][obs.indicator_key].append(obs)

    for project_id, groups in sorted(by_project.items()):
        for key, members in groups.items():
            types = sorted({m.indicator_type for m in members if m.indicator_type})
            if len(types) > 1:
                issues.append(
                    issue(
                        CheckCode.EXTRACTION_AMBIGUOUS,
                        "INFO",
                        f"{project_id} indicator '{members[0].indicator_name_raw[:80]}' "
                        f"is typed {types} in different ISRs",
                        members[0].source_ref,
                        indicator_key=key,
                    )
                )
        names = {key: members[0].indicator_name_normalized for key, members in groups.items()}
        keys = sorted(names)
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                reason = _possible_match(names[a], names[b])
                if reason:
                    issues.append(
                        issue(
                            CheckCode.POSSIBLE_INDICATOR_MATCH,
                            "INFO",
                            f"{project_id}: possible same indicator, kept separate ({reason})",
                            groups[a][0].source_ref,
                            indicator_keys=[a, b],
                            names=[names[a][:160], names[b][:160]],
                            isr_sequences=[
                                sorted(
                                    {
                                        m.isr_sequence
                                        for m in groups[k]
                                        if m.isr_sequence is not None
                                    }
                                )
                                for k in (a, b)
                            ],
                            reason=reason,
                        )
                    )
    return issues


def _possible_match(a: str, b: str) -> str | None:
    base_a, base_b = base_name(a), base_name(b)
    if not base_a or not base_b:
        return None
    if base_a == base_b:
        return "same name apart from unit/type suffix or punctuation"
    short, long_ = sorted((base_a, base_b), key=len)
    if len(short) >= MIN_PREFIX_CHARS and long_.startswith(short):
        return "one name is a prefix of the other (possible truncation or revision)"
    return None
