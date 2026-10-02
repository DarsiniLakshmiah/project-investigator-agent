"""Phase 9D development report (Markdown) over the recorded development experiments.

The false-deterministic-resolution analysis reads the RULE baseline recorded in 9C
(evaluation/routing_baseline_9B.2.yaml); no classifier is run on TEST here. The
categories are analysis for a future harness revision - they change no label and no rule.
"""

from __future__ import annotations

from typing import Any

from worldbank_copilot.routing.evaluation import BaselineResult, RoutingDataset, compare

# Analysis of every final 9B.2 mismatch that the rules RESOLVED (not semantic-needed).
MISMATCH_ANALYSIS: dict[str, tuple[str, str, str]] = {
    # case: (category, evidence, recommendation for a future bounded harness revision)
    "r003": (
        "deterministic lexical-rule error",
        "DOCUMENT_ACCORDING_TO fires on 'according to ISR 24', which only gives source "
        "context for an extracted value (human rule 2).",
        "Subject x source decision table: a document reference plus an extracted "
        "structured subject (ISR disbursement, ratings) stays STRUCTURED.",
    ),
    "r011": (
        "deterministic lexical-rule error",
        "DOCUMENT_IN_NAMED_DOCUMENT fires on 'identified in the fiduciary systems "
        "assessment'; the findings are extracted in the risk register (rule 2).",
        "Same table: assessment findings named by their source document -> RISKS with "
        "a source_document_type filter.",
    ),
    "r044": (
        "deterministic lexical-rule error",
        "'from the ESSA' triggers document scope for a request to LIST extracted "
        "findings (rule 2).",
        "Same table; 'list' + extracted findings -> RISKS.",
    ),
    "r008": (
        "semantic ambiguity",
        "'main environmental risks' needs the document's own prioritisation (rule 1); "
        "the rules only see the subject 'risks'.",
        "Salience qualifiers (main/key/principal) on risks -> DOCUMENT, or leave to a "
        "validated semantic layer.",
    ),
    "r027": (
        "deterministic lexical-rule error",
        "Explanatory + decision subject -> EXPLANATION, but 'which restructuring ... "
        "and why' must first establish the event (rule 4).",
        "Encode rule 4: 'which <decision/event> ... why' -> INVESTIGATION.",
    ),
    "r033": (
        "tool capability limitation",
        "get_rating_history has no date filter, so a YEAR scope is CLARIFY "
        "(TEMPORAL_NOT_SUPPORTED) - intentionally conservative.",
        "Add an ISR-date filter to the rating-history tool (ISR dates are governed).",
    ),
    "r055": (
        "temporal capability limitation",
        "The event-anchor grammar reads 'before the restructuring, additional financing "
        "and cancellation events' as one ambiguous anchor -> CLARIFY.",
        "Event-set anchors ('before the X, Y and Z events') -> the history of those "
        "event types, not a single anchor.",
    ),
    "r060": (
        "temporal capability limitation",
        "Relative periods are UNRESOLVED in 9B.2; the reviewed contract resolves them "
        "against the loan-statement as-of date (2026-08-31). The finance tool also has "
        "no date interval.",
        "Governed as-of resolution per intent (finance: statement snapshot) and a "
        "finance-history capability (ISR-printed values by date).",
    ),
    "r040": (
        "deterministic lexical-rule error",
        "Event types are taken from the whole question, so the anchor phrase 'since the "
        "2021 restructuring' adds RESTRUCTURING to the requested CLOSING_DATE_CHANGE.",
        "Exclude the temporal-anchor span from event-type extraction.",
    ),
    "r002": (
        "deterministic lexical-rule error",
        "'current value' is read as an explicit LATEST time (the route is right).",
        "Treat 'current value' as a results field name, not a time word.",
    ),
    "r025": (
        "deterministic lexical-rule error",
        "'trends' is read as HISTORY although it is the subject matter (route right).",
        "Remove 'trends' from the HISTORY lexicon or require a time context.",
    ),
    "r007": (
        "intentional conservative behavior",
        "Route and reason agree (CLARIFY TIME_REQUIRED); only the recorded time kind "
        "differs (defaulted LATEST vs labelled NONE).",
        "None needed; optionally record no default when TIME_REQUIRED fires.",
    ),
    "r053": (
        "semantic ambiguity",
        "'during appraisal' qualifies the risks, not the period of the investigation "
        "(labelled HISTORY implicit).",
        "Distinguish qualifier vs scope for appraisal wording inside investigations.",
    ),
    "r056": (
        "intentional conservative behavior",
        "Investigation default time HISTORY vs labelled LATEST; the plan already asks "
        "for CURRENT signals.",
        "Per-investigation-kind default time (attention -> LATEST).",
    ),
}


def _table(rows: list[list[Any]], header: list[str]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
    return out


def render(
    log: dict[str, Any],
    decision: dict[str, Any],
    dataset: RoutingDataset,
    baseline: list[BaselineResult],
) -> str:
    L: list[str] = []
    w = L.append
    w("# Phase 9D — bounded semantic routing: development report (first stop)")
    w("")
    w(
        "**Development split only.** The 51 frozen TEST cases have not been used by any "
        "classifier, embedding, threshold or selection step "
        f"(`test_evaluated: {str(log['test_evaluated']).lower()}`; dataset sha256 "
        f"`{log['dataset_sha256'][:16]}…`)."
    )
    w("")
    w("## Provisional decision")
    w("")
    w(f"- **Selected:** {decision['selected_candidate']} (status {decision['status']})")
    w(f"- **Reason:** {decision['reason']}")
    for name, state in decision["candidates"].items():
        w(f"- **{name}:** {state}")
    w(f"- **Freeze rule:** {decision['freeze_rule']}")
    w("")
    rule = log["protocol"]["selection"]
    for run_name, run in log["runs"].items():
        lat = run["classifier_latency_ms"]
        w(f"## {run_name}: leave-one-family-out on {len(run['examples'])} development examples")
        w("")
        w(
            f"Embedder `{run['embedder']}`. Majority-route baseline: always "
            f"{run['majority_route_baseline']['route']} = "
            f"{run['majority_route_baseline']['accuracy']} route accuracy."
        )
        w("")
        unthresholded = [
            r
            for r in run["experiments"]
            if r["config"]["min_similarity"] == 0 and r["config"]["min_vote_share"] == 0
        ]
        w("### Unthresholded configurations (no abstention)")
        w("")
        L.extend(
            _table(
                [
                    [
                        r["name"],
                        f"{r['metrics']['answered']}/{r['metrics']['n']}",
                        r["metrics"]["route_accuracy_all"],
                        r["metrics"]["intent_accuracy_all"],
                    ]
                    for r in unthresholded
                ],
                ["config", "answered", "route accuracy", "intent accuracy"],
            )
        )
        w("")
        best = sorted(
            run["experiments"],
            key=lambda r: (-(r["metrics"]["route_precision"] or 0), -r["metrics"]["coverage"]),
        )[:6]
        w("### Highest route precision among answered predictions (any coverage)")
        w("")
        L.extend(
            _table(
                [
                    [
                        r["name"],
                        f"{r['metrics']['answered']}/{r['metrics']['n']}",
                        r["metrics"]["coverage"],
                        r["metrics"]["route_precision"],
                        r["metrics"]["intent_precision"],
                    ]
                    for r in best
                ],
                ["config", "answered", "coverage", "route precision", "intent precision"],
            )
        )
        w("")
        w(
            f"Selection rule: route precision >= {rule['min_route_precision']} "
            f"at coverage >= {rule['min_coverage']}. "
            f"Configurations evaluated: {len(run['experiments'])}. Qualifying: "
            f"{'none' if not run['selected'] else run['selected']['name']}."
        )
        w("")
        w(f"### Coverage vs accuracy (base config {run['base_config']}, ranked by confidence)")
        w("")
        L.extend(
            _table(
                [
                    [
                        r["kept"],
                        r["coverage"],
                        r["min_confidence"],
                        r["route_accuracy"],
                        r["intent_accuracy"],
                    ]
                    for r in run["coverage_curve"]
                ],
                ["kept", "coverage", "min confidence", "route accuracy", "intent accuracy"],
            )
        )
        w("")
        w("### Per-example leave-one-family-out predictions (base config)")
        w("")
        L.extend(
            _table(
                [
                    [
                        p["example_id"],
                        p["gold_route"],
                        p["gold_intent"],
                        p["route"],
                        p["intent"],
                        round(p["top_similarity"], 3),
                        p["confidence"],
                    ]
                    for p in run["base_lofo"]
                ],
                [
                    "case",
                    "expected route",
                    "expected intent",
                    "predicted route",
                    "predicted intent",
                    "top similarity",
                    "confidence",
                ],
            )
        )
        w("")
        rules = run["rules_only_dev"]
        hyp = run["hypothetical_fallback_dev"]
        w("### End-to-end on DEVELOPMENT (29 cases)")
        w("")
        w(
            f"- Candidate A, rules only (semantic-needed -> CLARIFY): route accuracy "
            f"{rules['route_accuracy']}; semantic-needed {rules['semantic_needed']}."
        )
        w(
            f"- Hypothetical fallback with the rejected base config: route accuracy "
            f"{hyp['route_accuracy']}; semantic invocations {hyp['semantic_invocations']}; "
            f"semantic correct {hyp['semantic_correct']}; abstained {hyp['semantic_abstained']}."
        )
        w(
            f"- Fallback-eligible development cases: "
            f"{', '.join(r['case_id'] for r in hyp['rows'] if r['via'] != 'rules')} "
            "(denominator 2 - too small to support any accuracy claim)."
        )
        w(
            f"- Classifier latency (in-process): p50 {lat['p50']} ms, p95 {lat['p95']} ms "
            f"(n={lat['n']})."
        )
        w("")
    w("## False deterministic resolutions and other 9B.2 mismatches")
    w("")
    w(
        "Read from the recorded 9C rule baseline (no classifier involved). "
        "DETERMINISTIC_FALSE_RESOLUTION = the rules confidently chose a wrong executing route; "
        "those are harness/rule problems, not classifier territory, and are NOT repaired here."
    )
    w("")
    by_id = {b.case_id: b for b in baseline}
    rows = []
    for case in dataset.cases:
        cmp = compare(case, by_id[case.case_id])
        if cmp["category"] == "AGREES" or cmp["category"].startswith("SEMANTIC_NEEDED"):
            continue
        b = by_id[case.case_id]
        false_resolution = (
            b.route in ("STRUCTURED", "DOCUMENT", "INVESTIGATION")
            and b.route != case.expected.route
        )
        cat, evidence, rec = MISMATCH_ANALYSIS.get(case.case_id, ("unanalysed", "", ""))
        rows.append(
            [
                case.case_id,
                case.split,
                f"{case.expected.route}",
                f"{b.route} / {b.reason_code}",
                "DETERMINISTIC_FALSE_RESOLUTION" if false_resolution else cmp["category"],
                cat,
                evidence,
                rec,
            ]
        )
    L.extend(
        _table(
            rows,
            [
                "case",
                "split",
                "human",
                "9B.2",
                "type",
                "category",
                "evidence",
                "recommendation (future harness revision)",
            ],
        )
    )
    w("")
    return "\n".join(L) + "\n"
