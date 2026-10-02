"""Phase 9C human-review artefact: composition, baseline comparison and decisions (Markdown).

Pure rendering over the DRAFT dataset and a recorded 9B baseline; nothing here labels,
reviews or tunes anything.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from worldbank_copilot.common.project_registry import ProjectRegistry
from worldbank_copilot.routing.config import AliasConfig, normalize_phrase
from worldbank_copilot.routing.evaluation import (
    BaselineResult,
    RoutingCase,
    RoutingDataset,
    compare,
    metrics,
    similarity,
)

DIFFICULT_REFS = ("q16", "q22", "q33", "q44", "q46", "q28", "q30", "q43")
SECURITY_CODES = {
    "AMBIGUOUS_PROJECT_REFERENCE",
    "CROSS_PROJECT",
    "MULTI_PROJECT_NOT_SUPPORTED",
    "UNSUPPORTED_PROJECT",
    "NOT_AUTHORIZED",
    "PROJECT_REQUIRED",
    "INVALID_REQUEST",
    "PREDICTION_NOT_SUPPORTED",
    "READ_ONLY",
    "OUT_OF_DOMAIN",
}


def _cell(text: Any) -> str:
    return str(text if text not in (None, "") else "–").replace("|", "\\|").replace("\n", " ")


def _project(c: RoutingCase) -> str:
    return c.expected.project_id or c.active_project_id or "none"


def _human(c: RoutingCase) -> str:
    intents = "+".join(i.value for i in c.expected.intents)
    return f"{c.expected.route} / {c.expected.reason_code}" + (f" ({intents})" if intents else "")


def _base(b: BaselineResult) -> str:
    return f"{b.route} / {b.reason_code}"


def _temporal(c: RoutingCase) -> str:
    t = c.expected.temporal
    if t is None:
        return "n/a"
    values = list(t.isr_sequences) or [v for v in (t.date_from, t.date_to) if v]
    return f"{t.kind.value}{' (implicit)' if not t.explicit else ''}" + (
        f" {', '.join(map(str, values))}" if values else ""
    )


def composition(dataset: RoutingDataset, baseline: dict[str, BaselineResult]) -> dict[str, Counter]:
    out: dict[str, Counter] = defaultdict(Counter)
    for c in dataset.cases:
        b = baseline[c.case_id]
        out["project"][_project(c)] += 1
        out["expected route"][c.expected.route] += 1
        for i in c.expected.intents or ("(none: stopped before intent)",):
            out["expected intent"][getattr(i, "value", i)] += 1
        t = c.expected.temporal
        out["temporal category"][t.kind.value if t else "n/a"] += 1
        out["explicit vs implicit time"][
            "n/a" if t is None else ("explicit" if t.explicit else "implicit/default")
        ] += 1
        out["source"]["Phase 8-derived" if c.source == "PHASE8" else "newly authored"] += 1
        out["9B deterministic vs semantic"][
            "semantic needed" if b.route == "SEMANTIC_CLASSIFICATION_REQUIRED" else "deterministic"
        ] += 1
        out[f"split x route ({c.split})"][c.expected.route] += 1
        out[f"split x project ({c.split})"][_project(c)] += 1
    return out


def alias_rows(config: AliasConfig, registry: ProjectRegistry) -> list[tuple[str, str, str]]:
    rows = []
    for pid, aliases in config.aliases.items():
        own = normalize_phrase(registry.get(pid).name)
        others = set()
        for p in registry.projects:
            if p.project_id != pid:
                others |= set(normalize_phrase(p.name).split())
        for alias in aliases:
            norm = normalize_phrase(alias)
            if norm == own or f"in {norm}" == own:
                why = "registry project name" + (" without the 'IN' prefix" if norm != own else "")
            else:
                unique = sorted(set(norm.split()) - others)
                why = f"fragment of the registry name; distinguishing word(s) {unique}"
            rows.append((alias, pid, why))
    return rows


def render(
    dataset: RoutingDataset,
    baseline_list: list[BaselineResult],
    aliases: list[tuple[str, str, str]],
    reviewed_by: str | None,
    original: dict[str, Any] | None = None,
    router_version: str = "",
) -> str:
    base = {b.case_id: b for b in baseline_list}
    m = metrics(dataset, baseline_list)
    cmp = {c.case_id: compare(c, base[c.case_id]) for c in dataset.cases}
    L: list[str] = []
    w = L.append
    reviewed = dataset.review_status == "REVIEWED"
    w("# Phase 9C — routing evaluation dataset: human label review")
    w("")
    if reviewed:
        w(
            f"**All {len(dataset.cases)} labels are REVIEWED** (human review "
            f"{dataset.reviewed_on}); the split is {dataset.split_status}. Labels state what "
            'the system should do. The "9B" columns show what the deterministic router does. '
            "A difference is a measured baseline error or a semantic-classification gap, not "
            "a labelling error. No rule was tuned against these labels."
        )
    else:
        w(
            '**Every label is DRAFT.** Labels state what the system should do. The "9B" '
            "columns show what the current deterministic router does; a difference is "
            "expected and is not a labelling error. No rule was tuned against these labels."
        )
    w("")
    w(
        f"{len(dataset.cases)} cases · split ({dataset.split_status.lower()}): "
        f"{sum(c.split == 'dev' for c in dataset.cases)} dev / "
        f"{sum(c.split == 'test' for c in dataset.cases)} test."
    )
    w("")
    if dataset.semantic_rules:
        w("## Human semantic routing rules")
        w("")
        for rule in dataset.semantic_rules:
            w(f"- {rule}")
        w("")

    def table(cases: list[RoutingCase], extra: bool = False) -> None:
        w("| Case | Project | Question | Label | Time | Expected tools | 9B result | Note |")
        w("|---|---|---|---|---|---|---|---|")
        for c in cases:
            tools = ", ".join(
                t.tool + (f"({', '.join(f'{k}={v}' for k, v in t.args.items())})" if t.args else "")
                for t in c.expected.tools
            )
            note = c.review_note or c.label_rationale if extra else c.label_rationale
            ref = f" [{c.phase8_ref}]" if c.phase8_ref else ""
            w(
                f"| {c.case_id}{ref} | {_project(c)} | {_cell(c.question)} | {_cell(_human(c))} | "
                f"{_cell(_temporal(c))} | {_cell(tools)} | {_cell(_base(base[c.case_id]))} "
                f"| {_cell(note)} |"
            )
        w("")

    difficult = [
        c
        for c in dataset.cases
        if c.alternatives
        or c.phase8_ref in DIFFICULT_REFS
        or {
            "WHY_BOUNDARY",
            "RATINGS_APPRAISAL_RISKS",
            "MULTI_SUBJECT",
            "AMBIGUOUS_INFORMATION_NEED",
        }
        & set(c.review_flags)
    ]
    p179039 = [c for c in dataset.cases if "P179039_LIMITATION" in c.review_flags]
    security = [
        c
        for c in dataset.cases
        if c.expected.reason_code in SECURITY_CODES or "SECURITY" in c.review_flags
    ]
    semantic = [
        c for c in dataset.cases if base[c.case_id].route == "SEMANTIC_CLASSIFICATION_REQUIRED"
    ]
    used = {c.case_id for c in difficult + p179039 + security + semantic}
    straightforward = [c for c in dataset.cases if c.case_id not in used]

    w("## 1. Straightforward labels")
    w("")
    table(straightforward)
    w("## 2. Semantic-classification cases (9B returns SEMANTIC_CLASSIFICATION_REQUIRED)")
    w("")
    w(
        "These are the cases 9D may need to resolve. The label is the human target, "
        "not a rule change."
    )
    w("")
    table([c for c in semantic if c.case_id not in {x.case_id for x in difficult + p179039}])

    def detail(c: RoutingCase) -> None:
        b = base[c.case_id]
        ref = f" (Phase 8 {c.phase8_ref})" if c.phase8_ref else ""
        w(f"### {c.case_id}{ref} — {_project(c)}")
        w("")
        w(f"- **Question:** {c.question}")
        w(f"- **Label:** {_human(c)}; time {_temporal(c)}")
        if c.human_decision:
            w(f"- **Human decision:** {c.human_decision}")
        for alt in c.alternatives:
            intents = "+".join(i.value for i in alt.intents)
            w(
                f"- **Alternative{' (considered)' if c.human_decision else ''}:** "
                f"{alt.route}{f' ({intents})' if intents else ''} — {alt.rationale}"
            )
        if not c.alternatives:
            w("- **Alternative:** none proposed")
        w(f"- **Rationale:** {c.label_rationale}")
        if c.decision_information and not c.human_decision:
            w(f"- **What would change the decision:** {c.decision_information}")
        if c.review_note:
            w(f"- **Review note:** {c.review_note}")
        w(f"- **Flags:** {', '.join(c.review_flags) or '–'}")
        w(
            f"- **Current 9B:** {_base(b)} (intent {b.intent or '–'}, "
            f"tools {list(b.tools) or '–'}); "
            f"comparison: {cmp[c.case_id]['category']}"
        )
        w("")

    decided = [c for c in difficult if c.human_decision]
    remaining = [c for c in difficult if not c.human_decision]
    w("## 3a. Human decisions on ambiguous labels")
    w("")
    for c in decided:
        detail(c)
    w("## 3b. Remaining ambiguous labels (no human decision yet)")
    w("")
    if not remaining:
        w("None.")
        w("")
    for c in remaining:
        detail(c)
    w("## 4. Security, isolation and refusal cases")
    w("")
    table(security)
    w(
        "Execution check. Of the cases whose human label refuses or clarifies at the project "
        "or input stage, the number for which 9B executed nothing is "
        f"**{m['isolation_zero_execution']} of {m['isolation_cases']}**."
    )
    w("")
    for c in security:
        b = base[c.case_id]
        if b.executed_anything and c.expected.reason_code in SECURITY_CODES:
            w(
                f"- **Exception {c.case_id}:** 9B resolved the scope to {b.project_id} "
                f"({b.project_basis}) and executed {list(b.executed_tools)} for {b.project_id}."
            )
    w("")
    w("## 5. P179039 cases")
    w("")
    w(
        "q28, q29, q30, q35 and q36 stay DOCUMENT as agreed. Routing does not fix the "
        "P179039 retrieval limitation, and the Phase 8 retrieval labels are unchanged. q28 "
        'stays a label-review question; q30 waits for a decision on what "programme size" means.'
    )
    w("")
    table(p179039, extra=True)

    w("## 6. Project aliases (reviewed_by: " + _cell(reviewed_by) + ")")
    w("")
    w(
        "Approved list: exactly these aliases, each pointing to one project."
        if reviewed_by
        else "These are not reviewed yet. Each alias must point to exactly one project."
    )
    w("")
    w("| Alias | Project | Rationale |")
    w("|---|---|---|")
    for alias, pid, why in aliases:
        w(f"| {alias} | {pid} | {_cell(why)} |")
    w("")

    w("## 7. Dataset composition")
    w("")
    for name, counter in composition(dataset, base).items():
        w(f"- **{name}:** " + ", ".join(f"{k} {v}" for k, v in sorted(counter.items())))
    w("")
    w(f"## 8. Development/test grouping ({dataset.split_status.lower()})")
    w("")
    w(
        "Questions are grouped into families before splitting. A family keeps together "
        "near-duplicate questions, questions with the same underlying evidence (for example "
        "q03 and r050; q04, q10 and RES00355), and identical wording asked under a different "
        "scope (for example r029 and r059, or r037, r056 and r078). A whole family goes to "
        "one split, so no wording or evidence leaks from dev to test. The test split covers "
        "all three projects and all five route classes."
    )
    w("")
    families: dict[str, list[str]] = defaultdict(list)
    for c in dataset.cases:
        families[f"{c.split}: {c.family}"].append(c.case_id)
    for key in sorted(families):
        w(f"- {key}: {', '.join(families[key])}")
    dev = [c for c in dataset.cases if c.split == "dev"]
    test = [c for c in dataset.cases if c.split == "test"]
    closest = max(
        ((similarity(a.question, b.question), a.case_id, b.case_id) for a in dev for b in test),
        default=(0, "", ""),
    )
    w("")
    w(
        f"Most similar dev/test pair by token Jaccard: {closest[1]} / {closest[2]} "
        f"= {closest[0]:.2f}."
    )
    w("")

    w(
        f"## 9. Deterministic baseline, router {router_version} "
        + (
            "(against the REVIEWED labels; deterministic router only, no semantic classifier)"
            if reviewed
            else "(DRAFT labels; not the final routing accuracy)"
        )
    )
    w("")
    if original:
        w(
            "The original 9B.1 baseline was recorded before the approved r073 fix, against the "
            "first DRAFT labels; it is kept unchanged in "
            "evaluation/routing_baseline_9B.1.yaml. The rows below compare it with the "
            "current baseline. No rule was tuned on these numbers."
        )
        w("")
        w(f"| Metric | 9B.1 original | {router_version} |")
        w("|---|---|---|")
        for key in (
            "deterministic_coverage",
            "exact_route_agreement",
            "exact_route_and_reason_agreement",
            "agreement_on_resolved",
            "semantic_required_rate",
            "clarify_rate",
            "refuse_rate",
            "project_resolution_accuracy",
            "temporal_exact_match",
            "isolation_cases",
            "isolation_zero_execution",
        ):
            w(f"| {key.replace('_', ' ')} | {original.get(key)} | {m[key]} |")
        w("")
    for key in (
        "cases",
        "deterministic_coverage",
        "exact_route_agreement",
        "exact_route_and_reason_agreement",
        "agreement_on_resolved",
        "semantic_required_rate",
        "clarify_rate",
        "refuse_rate",
        "project_resolution_accuracy",
        "temporal_exact_match",
        "temporal_compared",
    ):
        w(f"- {key.replace('_', ' ')}: {m[key]}")
    w("")
    w("Route confusion where 9B resolved the case (human → 9B):")
    w("")
    for pair, k in m["route_confusion_on_resolved"].items():
        w(f"- {pair}: {k}")
    w("")
    w("### Mismatch report (human expected vs 9B), grouped by reason")
    w("")
    groups: dict[str, list[str]] = defaultdict(list)
    for c in dataset.cases:
        groups[cmp[c.case_id]["category"]].append(c.case_id)
    for cat in sorted(groups):
        w(f"- **{cat}** ({len(groups[cat])}): {', '.join(groups[cat])}")
    w("")
    return "\n".join(L) + "\n"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
