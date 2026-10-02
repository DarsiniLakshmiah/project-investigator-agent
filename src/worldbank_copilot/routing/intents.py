"""Stage 4: deterministic intent rules (Phase 9B).

Rules (configs/routing/intents.yaml) only recognise wording and record WHY they fired
(rule id, group, precedence, span, pattern index). A hit fully contained in a longer
hit is suppressed (recorded), e.g. "risk" inside "overall risk rating", or the decision
word "restructuring" inside "according to the restructuring paper".

Combination (fixed, in this order):

1. refusal hit                -> that refusal intent (rule order decides between refusals)
2. investigation hit          -> CHANGE_INVESTIGATION (its investigation kind)
3. document-scope hit         -> EXPLANATION if also explanatory, else DOCUMENT_CONTENT
4. explanatory + subjects:
   - only structured-change subjects (RATINGS/FINANCE/RESULTS/ATTENTION)
                              -> CHANGE_INVESTIGATION (a change must be established first)
   - only documented subjects (DECISION/RISKS/TIMELINE)
                              -> EXPLANATION (the documented rationale of a decision)
   - both, or neither         -> SEMANTIC_CLASSIFICATION_REQUIRED
5. no modifier: one subject family -> its structured intent; RATINGS+RISKS -> RISKS;
   a decision word alone, several families, or no hit -> SEMANTIC_CLASSIFICATION_REQUIRED.

RATINGS is refined by the explicit time: appraisal -> RISKS (ratings given at appraisal
are risk ratings); ISR / range / history -> RATING_HISTORY; otherwise CURRENT_RATINGS.
There is no generic "why => INVESTIGATION" rule and no forced default intent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from worldbank_copilot.routing.config import IntentRule, IntentRules
from worldbank_copilot.routing.models import (
    Intent,
    IntentDecision,
    IntentStatus,
    RuleHit,
    TemporalKind,
    TemporalScope,
)

STRUCTURED_CHANGE = frozenset({"RATINGS", "FINANCE", "RESULTS", "ATTENTION"})
DOCUMENTED = frozenset({"DECISION", "RISKS", "TIMELINE"})
FAMILY_INTENT = {
    "FINANCE": Intent.FINANCIAL_STATUS,
    "RESULTS": Intent.RESULTS_PROGRESS,
    "RISKS": Intent.RISKS,
    "ATTENTION": Intent.ATTENTION,
    "OVERVIEW": Intent.PROJECT_OVERVIEW,
    "TIMELINE": Intent.TIMELINE_EVENTS,
}
HISTORY_KINDS = {TemporalKind.ISR_SEQUENCE, TemporalKind.ISR_RANGE, TemporalKind.HISTORY}


@dataclass(frozen=True)
class _Compiled:
    rule: IntentRule
    order: int
    patterns: tuple[re.Pattern[str], ...]


class IntentEngine:
    def __init__(self, rules: IntentRules):
        self.rules = rules
        self.doc_alternation = _doc_alternation(rules.document_types)
        self.compiled = [
            _Compiled(
                rule,
                order,
                tuple(
                    re.compile(p.replace("{doc}", self.doc_alternation), re.IGNORECASE)
                    for p in rule.patterns
                ),
            )
            for order, rule in enumerate(rules.rules)
        ]

    def hits(self, text: str) -> tuple[list[RuleHit], list[RuleHit]]:
        """(kept, suppressed) rule hits in text order."""
        raw: list[tuple[RuleHit, int]] = []
        for c in self.compiled:
            for index, pattern in enumerate(c.patterns):
                for m in pattern.finditer(text):
                    doc_type = None
                    if c.rule.group == "document_scope":
                        doc_type = next(
                            (n[3:] for n, v in m.groupdict().items() if v and n.startswith("dt_")),
                            None,
                        )
                        if c.rule.non_isr_only and doc_type == "ISR":
                            continue
                    raw.append(
                        (
                            RuleHit(
                                rule_id=c.rule.id,
                                group=c.rule.group,
                                precedence=c.rule.precedence,
                                span=m.span(),
                                text=m.group(0),
                                pattern_index=index,
                                intent=c.rule.intent.value if c.rule.intent else None,
                                subject=c.rule.subject,
                                rating_type=c.rule.rating_type,
                                document_type=doc_type,
                                investigation=c.rule.investigation,
                            ),
                            c.order,
                        )
                    )
        kept, suppressed = [], []
        for hit, order in raw:
            a, b = hit.span
            contained = any(
                (o.span[0] <= a and b <= o.span[1]) and (o.span[1] - o.span[0]) > (b - a)
                for o, _ in raw
            )
            (suppressed if contained else kept).append((hit, order))
        unique = {}
        for hit, order in sorted(kept, key=lambda h: (h[0].span, h[1], h[0].pattern_index)):
            unique.setdefault((hit.rule_id, hit.span), (hit, order))
        ordered = [h for h, _ in sorted(unique.values(), key=lambda h: (h[0].span, h[1]))]
        return ordered, [h for h, _ in suppressed]

    def decide(self, text: str, temporal: TemporalScope) -> IntentDecision:
        hits, suppressed = self.hits(text)
        order = {r.id: i for i, r in enumerate(self.rules.rules)}
        subjects = sorted({h.subject for h in hits if h.subject})
        rating_types = sorted({h.rating_type for h in hits if h.rating_type})
        doc_types = sorted({h.document_type for h in hits if h.document_type})
        explanatory = any(h.group == "explanatory" for h in hits)
        doc_scoped = any(h.group == "document_scope" for h in hits)

        def decision(intent: Intent | None, decided_by, reason: str, investigation=None):
            return IntentDecision(
                status=IntentStatus.RESOLVED
                if intent
                else IntentStatus.SEMANTIC_CLASSIFICATION_REQUIRED,
                intent=intent,
                method="RULE" if intent else "NONE",
                decided_by=tuple(decided_by),
                reason=reason,
                hits=tuple(hits),
                suppressed_hits=tuple(suppressed),
                subjects=tuple(subjects),
                rating_types=tuple(rating_types),
                document_types=tuple(doc_types),
                explanatory=explanatory,
                document_scoped=doc_scoped,
                investigation=investigation,
                rules_version=self.rules.version,
            )

        def ids(group: str) -> list[str]:
            return sorted({h.rule_id for h in hits if h.group == group}, key=order.get)

        refusals = sorted((h for h in hits if h.group == "refusal"), key=lambda h: order[h.rule_id])
        if refusals:
            first = refusals[0]
            return decision(Intent(first.intent), [first.rule_id], f"refusal rule {first.rule_id}")
        investigations = sorted(
            (h for h in hits if h.group == "investigation"), key=lambda h: order[h.rule_id]
        )
        if investigations:
            first = investigations[0]
            return decision(
                Intent.CHANGE_INVESTIGATION,
                [first.rule_id],
                f"explicit investigation wording ({first.investigation})",
                investigation=first.investigation,
            )
        if doc_scoped:
            if explanatory:
                return decision(
                    Intent.EXPLANATION,
                    ids("document_scope") + ids("explanatory"),
                    "asks for the rationale stated in a named document",
                )
            return decision(
                Intent.DOCUMENT_CONTENT, ids("document_scope"), "asks what a named document says"
            )
        families = set(subjects)
        if explanatory:
            change = families & STRUCTURED_CHANGE
            documented = families & DOCUMENTED
            if change and not documented:
                return decision(
                    Intent.CHANGE_INVESTIGATION,
                    ids("explanatory") + ids("subject"),
                    "explanation of a structured change: the change must be established "
                    "from structured data and explained from documents",
                    investigation="+".join(sorted(change)),
                )
            if documented and not change:
                return decision(
                    Intent.EXPLANATION,
                    ids("explanatory") + ids("subject"),
                    "documented rationale of a decision or rated risk",
                )
            reason = (
                "explanatory question mixing structured-change and documented-decision subjects"
                if change and documented
                else "explanatory question without a deterministic subject"
            )
            return decision(None, [], reason)
        if "DECISION" in families and "TIMELINE" in families:
            families.discard("DECISION")
        if families == {"DECISION"}:
            return decision(
                None,
                [],
                "a decision is mentioned but not whether its timeline "
                "or its documented content is wanted",
            )
        families.discard("DECISION")
        if families == {"RATINGS", "RISKS"}:
            families = {"RISKS"}
        if len(families) > 1:
            return decision(None, [], f"several subject families {sorted(families)}")
        if not families:
            return decision(None, [], "no deterministic rule matched")
        (family,) = families
        subject_ids = ids("subject")
        if family == "RATINGS":
            if temporal.explicit and temporal.kind == TemporalKind.APPRAISAL:
                return decision(
                    Intent.RISKS, subject_ids, "a rating given at appraisal is a risk rating"
                )
            if temporal.explicit and temporal.kind in HISTORY_KINDS:
                return decision(
                    Intent.RATING_HISTORY, subject_ids, f"ratings for {temporal.kind.value}"
                )
            return decision(
                Intent.CURRENT_RATINGS, subject_ids, "ratings without a historical scope"
            )
        return decision(FAMILY_INTENT[family], subject_ids, f"subject {family}")


def _doc_alternation(document_types: dict[str, tuple[str, ...]]) -> str:
    groups = []
    for doc_type, phrases in sorted(document_types.items(), key=lambda kv: -max(map(len, kv[1]))):
        words = "|".join(
            r"\s+".join(re.escape(w) for w in p.split())
            for p in sorted(phrases, key=len, reverse=True)
        )
        groups.append(f"(?P<dt_{doc_type}>{words})")
    return "(?:" + "|".join(groups) + r")\b"
