"""Application evaluation metrics: pure functions over one persisted case record.

A record holds the raw ``InvestigationResult`` (JSON), what the models received (captured
by the runner's pass-through recorder) and any runtime error. Metrics are grouped into
layers that are never collapsed into one score:

A retrieval   - Phase 8 relevance (``matched_items``/``ranking_metrics``) over the document
                passages the investigation retrieved, in retrieval order.
B context     - the same judgments over the evidence actually shown to the Synthesizer.
C generation  - final published claims: Critic support labels, golden-fact recall, F1.
D citations   - deterministic identity/project/temporal checks; entailment = Critic.
E agent       - task success, routing, tools, isolation, temporal, refusal, abstention.
F operations  - latency, model calls, tokens, reliability, graceful degradation.

A metric that is not valid for a case is ``None`` (reported as NOT_APPLICABLE), never a
forced number. Semantic dimensions (relevance, completeness, conciseness, citation
entailment) come only from an optional human review; no LLM judges identity, dates,
values, isolation or refusal.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import date
from typing import Any

from worldbank_copilot.evaluation.app_cases import CATEGORIES, Case
from worldbank_copilot.investigation.evidence import EvidenceExecutionError, _owned
from worldbank_copilot.retrieval.evaluation import matched_items, normalize, ranking_metrics

NA = None
KS = (5, 10)
_DATE_KEYS = ("document_date", "event_date", "observed_date", "report_date", "reporting_date")
_MODEL_FAILURES = {"MODEL_CALL_FAILED", "OUTPUT_PARSE_FAILED", "SCHEMA_INVALID"}
_PARSE_OUTCOMES = {"MODEL_OUTPUT_INVALID", "SCHEMA_VALIDATION_FAILED", "OUTPUT_BUDGET_EXCEEDED"}
_SECURITY = {
    "PROJECT_ISOLATION_VIOLATION",
    "EVIDENCE_REFERENCE_INVALID",
    "CITATION_VALIDATION_FAILED",
    "TEMPORAL_SCOPE_VIOLATION",
    "REQUIREMENT_REFERENCE_INVALID",
    "SCHEMA_VALIDATION_FAILED",
}
_PROJECT_ID = re.compile(r"\bP\d{6}\b")
BUCKETS = (
    "ROUTING",
    "INVESTIGATOR_PLANNING",
    "TOOL_EXECUTION",
    "RETRIEVAL_COVERAGE",
    "CONTEXT_PACKING",
    "SYNTHESIS",
    "CITATION",
    "TEMPORAL_INTEGRITY",
    "CRITIC",
    "FINALIZATION",
    "PROJECT_ISOLATION",
    "REFUSAL",
    "RUNTIME",
    "ANSWER_RELEVANCE",
    "ANSWER_COMPLETENESS",
)


# -- small helpers ----------------------------------------------------------------------
def ratio(num: float, den: float) -> float | None:
    return num / den if den else NA


def _published_text(result: dict) -> str:
    return " ".join([result.get("message") or "", *(c["text"] for c in result.get("claims", ()))])


def _evidence_by_id(result: dict) -> dict[str, dict]:
    return {e["evidence_id"]: e for e in result.get("evidence", ()) if e.get("evidence_id")}


def _cited(result: dict) -> list[str]:
    return list(dict.fromkeys(i for c in result.get("claims", ()) for i in c["evidence_ids"]))


def evidence_day(payload: dict) -> date | None:
    for key in _DATE_KEYS:
        value = payload.get(key)
        if value:
            try:
                return date.fromisoformat(str(value)[:10])
            except ValueError:
                return None
    return None


def _flatten(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [s for v in value.values() for s in _flatten(v)]
    if isinstance(value, list | tuple):
        return [s for v in value for s in _flatten(v)]
    return [] if value is None else [str(value)]


# -- A/B: document relevance rows ------------------------------------------------------
def retrieved_rows(result: dict) -> list[dict]:
    """Document passages the investigation retrieved, in retrieval order (Phase 8 rows)."""
    rows = []
    for e in result.get("evidence", ()):
        if e.get("source_type") != "DOCUMENT":
            continue
        citation, payload = e.get("citation") or {}, e.get("payload") or {}
        rows.append(
            _row(
                citation.get("document_id") or payload.get("document_id"),
                citation.get("pages") or (),
                payload,
            )
        )
    return rows


def context_rows(shown: list[dict]) -> list[dict]:
    """Document passages shown to the Synthesizer (from the captured model input)."""
    rows = []
    for e in shown:
        if e.get("source_type") != "DOCUMENT":
            continue
        content, source = e.get("content") or {}, e.get("source") or {}
        rows.append(_row(content.get("document_id"), source.get("pages") or (), content))
    return rows


def _row(document_id, pages, payload) -> dict:
    text = " ".join(str(payload.get(k) or "") for k in ("text", "context_text"))
    return {
        "document_id": document_id or "",
        "page_numbers": list(pages),
        "chunk_text": text,
        "project_id": (document_id or "").split("-")[0],
    }


def retrieval_metrics(rows: list[dict], questions: list) -> dict:
    """Phase 8 metrics averaged over the case's linked (answerable) questions."""
    answerable = [q for q in questions if q.answerable]
    if not answerable:
        return {k: NA for k in _RETRIEVAL_KEYS}
    per = []
    for q in answerable:
        base = ranking_metrics(rows, q, KS)  # Phase 8: recall@k, mrr (binary relevance)
        hits = [bool(matched_items(r, q)) for r in rows]
        per.append(
            {
                "recall_at_5": base["recall_at_5"],
                "recall_at_10": base["recall_at_10"],
                "precision_at_5": sum(hits[:5]) / 5,
                "precision_at_10": sum(hits[:10]) / 10,
                "hit_at_5": float(any(hits[:5])),
                "hit_at_10": float(any(hits[:10])),
                "mrr": base["mrr"],
            }
        )
    out = {k: statistics.fmean(p[k] for p in per) for k in per[0]}
    out["ndcg_at_10"] = NA  # only binary Phase 8 labels exist: no graded relevance
    return out


_RETRIEVAL_KEYS = (
    "recall_at_5",
    "recall_at_10",
    "precision_at_5",
    "precision_at_10",
    "hit_at_5",
    "hit_at_10",
    "mrr",
    "ndcg_at_10",
)


def context_metrics(retrieved: list[dict], shown: list[dict] | None, questions: list) -> dict:
    """Did packing keep the relevant evidence retrieval found, and how noisy is it."""
    answerable = [q for q in questions if q.answerable]
    if not answerable or shown is None:
        return {"context_recall": NA, "context_precision": NA}
    found, kept = set(), set()
    for q in answerable:
        found |= {(q.id, i) for r in retrieved for i in matched_items(r, q)}
        kept |= {(q.id, i) for r in shown for i in matched_items(r, q)}
    relevant_shown = sum(1 for r in shown if any(matched_items(r, q) for q in answerable))
    return {
        "context_recall": ratio(len(kept & found), len(found)),
        "context_precision": ratio(relevant_shown, len(shown)),  # labelled documents only
    }


# -- C: golden facts --------------------------------------------------------------------
_MONTHS = (
    "January February March April May June July August September October November December"
).split()


def _date_forms(value: str) -> list[str]:
    try:
        d = date.fromisoformat(value[:10])
    except ValueError:
        return [value]
    month = _MONTHS[d.month - 1]
    return [
        d.isoformat(),
        f"{month} {d.day}, {d.year}",
        f"{d.day} {month} {d.year}",
        f"{month[:3]} {d.day}, {d.year}",
        f"{d.day} {month[:3]} {d.year}",
    ]


def _numbers(text: str) -> list[float]:
    return [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*\.?\d*", text)]


def fact_represented(kind: str, value: str, surface: str, values: list[str]) -> bool:
    """Is one golden value present in the answer surface (exact, format-tolerant)?"""
    low = surface.lower()
    if kind == "date":
        return any(f.lower() in low for f in _date_forms(value)) or value in values
    if kind == "number":
        try:
            golden = float(value)
        except ValueError:
            return value in values
        return any(math.isclose(n, golden, rel_tol=5e-3) for n in _numbers(surface)) or any(
            v == value for v in values
        )
    if kind == "id":
        token = normalize(value).replace(" ", "")
        return token in normalize(surface).replace(" ", "") or value in values
    return normalize(value) in normalize(surface) or value in values


def answer_surface(result: dict) -> tuple[str, list[str]]:
    """What the user was shown: claims (+ cited evidence), or deterministic records."""
    evidence = _evidence_by_id(result)
    if result.get("status") == "EVIDENCE_ONLY":
        records = [e.get("payload") or {} for e in result.get("evidence", ())]
    else:
        records = [(evidence.get(i) or {}).get("payload") or {} for i in _cited(result)]
    values = [v for p in records for v in _flatten(p)]
    return " ".join([*(c["text"] for c in result.get("claims", ())), *values]), values


def fact_recall(case: Case, golden: dict, result: dict) -> tuple[float | None, list[dict]]:
    facts = [golden.get(s.fact_id) for s in case.expected.required_facts]
    facts = [f for f in facts if f and f["status"] == "RESOLVED"]
    if not facts:
        return NA, []
    surface, values = answer_surface(result)
    hit = total = 0
    detail = []
    for f in facts:
        found = [v for v in f["values"] if fact_represented(f["kind"], v, surface, values)]
        if f["match"] == "any":
            total, hit = total + 1, hit + (1 if found else 0)
        else:
            total, hit = total + len(f["values"]), hit + len(found)
        detail.append({"fact_id": f["fact_id"], "values": len(f["values"]), "found": len(found)})
    return ratio(hit, total), detail


# -- D: citations ------------------------------------------------------------------------
def citation_metrics(case: Case, result: dict, anchor_day: date | None) -> dict:
    claims = result.get("claims", ())
    evidence = _evidence_by_id(result)
    handles = [(c, i) for c in claims for i in c["evidence_ids"]]
    valid = [
        (c, i)
        for c, i in handles
        if i in evidence
        and any(r["evidence_id"] == i and r["source_identity"] for r in c["citations"])
    ]
    complete = [
        c
        for c in claims
        if c["citations"]
        and all(i in evidence for i in c["evidence_ids"])
        and {r["evidence_id"] for r in c["citations"]} <= set(c["evidence_ids"])
    ]
    cited = [evidence[i] for _, i in valid]
    temporal = temporal_citations(claims, evidence, anchor_day)
    return {
        "citation_validity": ratio(len(valid), len(handles)),
        "citation_completeness": ratio(len(complete), len(claims))
        if case.expected.citation_required
        else NA,
        "citation_project_consistency": ratio(
            sum(
                _owned_ok([e], case.project_id) and _doc_in_project(e, case.project_id)
                for e in cited
            ),
            len(cited),
        ),
        "citation_temporal_consistency": temporal,
        # Semantic entailment: the Critic's per-claim verdict (SUPPORTED only).
        "citation_entailment_critic": ratio(
            sum(c["support"] == "SUPPORTED" for c in claims), len(claims)
        ),
    }


def temporal_citations(claims, evidence: dict, anchor_day: date | None) -> float | None:
    """BEFORE/AFTER/ACROSS claims checked against the governed anchor date (independent)."""
    timed = [c for c in claims if c.get("temporal_relation", "NONE") != "NONE"]
    if not timed or anchor_day is None:
        return NA
    ok = 0
    for c in timed:
        days = [
            evidence_day((evidence.get(i) or {}).get("payload") or {}) for i in c["evidence_ids"]
        ]
        sides = {
            None
            if d is None
            else "BEFORE"
            if d < anchor_day
            else "AFTER"
            if d > anchor_day
            else "EVENT"
            for d in days
        }
        relation = c["temporal_relation"]
        if relation == "ACROSS":
            ok += {"BEFORE", "AFTER"} <= sides and None not in sides
        else:
            ok += sides == {relation}
    return ok / len(timed)


def _owned_ok(items, project_id) -> bool:
    try:
        _owned([i.get("payload") or {} for i in items], project_id)
    except EvidenceExecutionError:
        return False
    return True


def _doc_in_project(item: dict, project_id: str) -> bool:
    document = (item.get("citation") or {}).get("document_id") or (item.get("payload") or {}).get(
        "document_id"
    )
    return not document or document.startswith(project_id)


def project_isolation(case: Case, result: dict) -> bool:
    """CRITICAL: nothing published or returned belongs to another project."""
    if result.get("project_id") != case.project_id:
        return False
    items = list(result.get("evidence", ()))
    if not _owned_ok(items, case.project_id) or not all(
        _doc_in_project(e, case.project_id) for e in items
    ):
        return False
    # Published claims are the answer; a refusal message naming the requested project is not
    # leaked evidence, so runtime messages are not scanned.
    text = " ".join(c["text"] for c in result.get("claims", ()))
    mentioned = set(_PROJECT_ID.findall(text)) - {case.project_id}
    foreign = set(case.expected.foreign_projects)
    payload_projects = {
        str(v)
        for e in items
        for v in _flatten(e.get("payload") or {})
        if _PROJECT_ID.fullmatch(str(v))
    }
    return not (mentioned & foreign) and not (payload_projects - {case.project_id})


# -- E: agent / system ------------------------------------------------------------------
def task_success(case: Case, result: dict) -> str:
    status = result.get("status")
    exp = case.expected
    published = _published_text(result)
    if any(
        re.search(p, " ".join(c["text"] for c in result.get("claims", ())))
        for p in exp.forbidden_claims
    ):
        return "FAIL"
    if exp.prediction_must_be_refused and (status != "REFUSE" or result.get("claims")):
        return "FAIL"
    if (
        exp.forbidden_claims
        and any(re.search(p, published) for p in exp.forbidden_claims)
        and status != "REFUSE"
    ):
        return "FAIL"
    if status in exp.allowed_statuses:
        return "PASS"
    return "PARTIAL" if status in exp.acceptable_statuses else "FAIL"


def investigator_tools(captured: dict) -> list[str]:
    return [a.get("tool") for d in captured.get("decisions", ()) for a in d.get("actions", ())]


def score_case(
    case: Case, record: dict, golden: dict, questions: dict, review: dict | None = None
) -> dict:
    """All per-case metrics, the verdict and failure buckets (pure, re-runnable)."""
    error, result = record.get("error"), record.get("result")
    captured = record.get("captured") or {}
    if result is None:  # runtime exception: nothing else is measurable
        return {
            "case_id": case.case_id,
            "category": case.category,
            "verdict": "FAIL",
            "runtime_success": False,
            "buckets": ["RUNTIME"],
            "error": error,
        }
    exp = case.expected
    linked = [questions[q] for q in exp.relevant_evidence if q in questions]
    activity = result.get("activity") or {}
    validation = result.get("validation") or {}
    calls = result.get("model_calls") or ()
    claims = result.get("claims") or ()
    shown = captured.get("synthesis_evidence")
    anchor_golden = (
        golden.get(exp.anchor.event_date.fact_id) if exp.anchor and exp.anchor.event_date else None
    )
    anchor_day = (
        date.fromisoformat(anchor_golden["values"][0][:10])
        if anchor_golden and len(anchor_golden["values"]) == 1
        else None
    )

    retrieved = retrieved_rows(result)
    retrieval = retrieval_metrics(retrieved, linked)
    context = context_metrics(retrieved, context_rows(shown) if shown is not None else None, linked)
    supports = Counter(c["support"] for c in claims)
    strict = ratio(supports["SUPPORTED"], len(claims))
    recall, facts = fact_recall(case, golden, result)
    f1 = (
        2 * strict * recall / (strict + recall)
        if strict is not None and recall is not None and strict + recall > 0
        else (0.0 if strict == 0 and recall == 0 else NA)
    )
    citations = citation_metrics(case, result, anchor_day)
    isolated = project_isolation(case, result)
    task = task_success(case, result)
    status = result.get("status")
    removed = validation.get("claims_removed") or {}
    roles = Counter(c["role"] for c in calls)
    failed_calls = [c for c in calls if c.get("outcome") != "COMPLETED"]
    planning = activity.get("planning_failure")
    runtime_ok = error is None and not (
        status == "FAIL_CLOSED" and (failed_calls or planning in _MODEL_FAILURES)
    )
    tools = investigator_tools(captured)
    periods = {e.get("period") for e in (shown or ())}

    temporal = {}
    if exp.anchor:
        resolution = activity.get("anchor_resolution")
        temporal["anchor_resolution_ok"] = resolution in exp.anchor.resolutions
        temporal["anchor_relation_ok"] = (
            activity.get("temporal_anchor") in exp.anchor.relations
            if exp.anchor.relations and resolution == "RESOLVED"
            else NA
        )
        relation = activity.get("temporal_anchor")
        need = {"BEFORE": {"BEFORE"}, "AFTER": {"AFTER"}, "COMPARE": {"BEFORE", "AFTER"}}.get(
            relation
        )
        temporal["temporal_context_coverage"] = (
            float(need <= periods)
            if need and shown is not None and resolution == "RESOLVED"
            else NA
        )
        temporal["anchor_golden"] = anchor_golden["status"] if anchor_golden else NA

    shown_types = {e.get("source_type") for e in (shown or ())}
    answer_types = (
        {"STRUCTURED"}
        if status == "EVIDENCE_ONLY"
        else {(_evidence_by_id(result).get(i) or {}).get("source_type") for i in _cited(result)}
    )
    should_refuse = exp.allowed_statuses == ["REFUSE"]  # refusal is the only correct outcome
    answer_expected = "ANSWER" in exp.allowed_statuses or "EVIDENCE_ONLY" in exp.allowed_statuses
    no_answer_case = "INSUFFICIENT_EVIDENCE" in exp.allowed_statuses and not answer_expected
    metrics = {
        "case_id": case.case_id,
        "category": case.category,
        "project_id": case.project_id,
        "status": status,
        "route": result.get("route"),
        "reason_code": result.get("reason_code"),
        "trace_id": result.get("trace_id"),
        "runtime_success": runtime_ok,
        # A retrieval / B context
        **retrieval,
        **context,
        "evidence_retrieved": activity.get("evidence_retrieved"),
        "evidence_shown": activity.get("evidence_shown"),
        "evidence_not_shown": (
            activity["evidence_retrieved"] - activity["evidence_shown"]
            if activity.get("evidence_shown") is not None
            and activity.get("evidence_retrieved") is not None
            else NA
        ),
        "required_source_coverage": (
            float(set(exp.required_source_types) <= answer_types)
            if exp.required_source_types and claims
            else NA
        ),
        "context_source_coverage": (
            float(set(exp.required_source_types) <= shown_types)
            if exp.required_source_types and shown is not None
            else NA
        ),
        **{k: v for k, v in temporal.items()},
        # C generation
        "published_claims": len(claims),
        "support": dict(supports),
        "claim_precision_strict": strict,
        "fact_recall": recall,
        "fact_detail": facts,
        "answer_f1": f1,
        "relevance": (review or {}).get("relevance"),
        "completeness": (review or {}).get("completeness"),
        "conciseness": (review or {}).get("conciseness"),
        # D citations
        **citations,
        "citation_entailment_human": (review or {}).get("citation_entailment"),
        # E agent
        "task_success": task,
        "routing_ok": result.get("route") in exp.acceptable_routes if exp.acceptable_routes else NA,
        "tool_selection_ok": (
            bool(set(tools) & set(exp.useful_tools))
            if exp.useful_tools and roles["INVESTIGATOR"]
            else NA
        ),
        "tools_proposed": tools,
        "tool_calls": activity.get("tool_calls", 0),
        "tool_errors": sum(
            1 for c in captured.get("calls_made", ()) if c.get("status") not in ("OK", "EMPTY")
        ),
        "rejected_actions": activity.get("rejected_actions", 0),
        "rejected_by_reason": activity.get("rejected_by_reason") or {},
        "decision_rounds": activity.get("decision_rounds", 0),
        "search_actions": tools.count("search_documents"),
        "project_isolation": isolated,
        "clarification_ok": status == "CLARIFY" if exp.clarification_expected else NA,
        "refusal_ok": (status == "REFUSE" and not claims) if should_refuse else NA,
        "false_refusal": status == "REFUSE"
        and "REFUSE" not in {*exp.allowed_statuses, *exp.acceptable_statuses},
        "abstention_ok": (
            status in ("INSUFFICIENT_EVIDENCE", "CLARIFY", "REFUSE") or task == "PASS"
        )
        if no_answer_case
        else NA,
        "unnecessary_abstention": answer_expected and status == "INSUFFICIENT_EVIDENCE",
        "unsupported_published": sum(
            c["support"] in ("UNSUPPORTED", "CONTRADICTED") for c in claims
        ),
        "unverified_published": supports["NOT_ASSESSED"],
        "claims_removed": removed,
        "limitations_withheld": validation.get("limitations_withheld", 0),
        # F operations
        "latency_ms": result.get("latency_ms"),
        "stage_latency_ms": result.get("stage_latency_ms") or {},
        "model_calls": dict(roles),
        "model_call_failures": len(failed_calls),
        "parse_failures": sum(c.get("outcome") in _PARSE_OUTCOMES for c in calls),
        "tokens": _tokens(calls),
        "fail_closed": status == "FAIL_CLOSED",
        "critic_status": validation.get("critic_status"),
        "graceful_degradation": validation.get("critic_status") == "FAILED" and status == "ANSWER",
        "planning_failure": planning,
    }
    metrics["escaped_unsafe_claims"] = _escaped(metrics, claims)
    metrics["verdict"], metrics["buckets"] = _verdict(
        case, result, metrics, validation, failed_calls
    )
    return metrics


def _tokens(calls) -> dict:
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for c in calls:
        for key in ("input_tokens", "output_tokens", "reasoning_tokens"):
            if isinstance(c.get(key), int):
                out[c["role"]][key] += c[key]
    return {r: dict(v) for r, v in out.items()}


def _escaped(m: dict, claims) -> int:
    """Published claims that a deterministic evaluation check would reject."""
    if not claims:
        return 0
    bad = (m["citation_validity"] not in (None, 1.0)) or (
        m["citation_project_consistency"] not in (None, 1.0)
    )
    bad = (
        bad or (m["citation_temporal_consistency"] not in (None, 1.0)) or not m["project_isolation"]
    )
    return len(claims) if bad else m["unsupported_published"]


def _verdict(case, result, m, validation, failed_calls) -> tuple[str, list[str]]:
    exp, status = case.expected, result.get("status")
    buckets: set[str] = set()
    fail = partial = False
    if not m["runtime_success"]:
        fail = True
        buckets.add("RUNTIME")
    if not m["project_isolation"]:
        fail = True
        buckets.add("PROJECT_ISOLATION")
    if m["citation_validity"] not in (None, 1.0) or m["citation_project_consistency"] not in (
        None,
        1.0,
    ):
        fail = True
        buckets.add("CITATION")
    if m["citation_temporal_consistency"] not in (None, 1.0):
        fail = True
        buckets.add("TEMPORAL_INTEGRITY")
    if m["task_success"] == "FAIL":
        fail = True
    elif m["task_success"] == "PARTIAL":
        partial = True
    if m.get("anchor_resolution_ok") is False or m.get("anchor_relation_ok") is False:
        partial = True
        buckets.add("TEMPORAL_INTEGRITY")
    if m.get("temporal_context_coverage") == 0.0:
        partial = True
        buckets.add("TEMPORAL_INTEGRITY")
    if m["fact_recall"] is not None and m["fact_recall"] < 1:
        partial = True
        buckets.add("ANSWER_COMPLETENESS")
    if m["required_source_coverage"] == 0.0:
        partial = True
    if m["graceful_degradation"]:
        partial = True
        buckets.add("CRITIC")
    if m["citation_completeness"] not in (None, 1.0):
        partial = True
        buckets.add("CITATION")
    # Diagnostics for every non-pass outcome (what failed, never a fix).
    if fail or partial:
        if m["false_refusal"] or (exp.prediction_must_be_refused and status != "REFUSE"):
            buckets.add("REFUSAL")
        if m["refusal_ok"] is False or (m["abstention_ok"] is False):
            buckets.add("REFUSAL")
        if (
            status in ("REFUSE", "CLARIFY")
            and result.get("route") in ("REFUSE", "CLARIFY")
            and status not in exp.allowed_statuses
        ):
            buckets.add("ROUTING")
        if m["routing_ok"] is False:
            buckets.add("ROUTING")
        if m["planning_failure"] or m["tool_selection_ok"] is False:
            buckets.add("INVESTIGATOR_PLANNING")
        if m["tool_errors"] or (m["rejected_actions"] and not m["evidence_retrieved"]):
            buckets.add("TOOL_EXECUTION")
        if m["recall_at_10"] == 0.0 or (
            m["unnecessary_abstention"] and not m["evidence_retrieved"]
        ):
            buckets.add("RETRIEVAL_COVERAGE")
        if m["context_recall"] == 0.0 or m["context_source_coverage"] == 0.0:
            buckets.add("CONTEXT_PACKING")
        if any(c["role"] == "SYNTHESIZER" for c in failed_calls):
            buckets.add("SYNTHESIS")
        removed = m["claims_removed"]
        if status == "INSUFFICIENT_EVIDENCE" and removed and not m["published_claims"]:
            critic = sum(
                v
                for k, v in removed.items()
                if k in ("UNSUPPORTED", "CONTRADICTED", "NOT_REVIEWED")
            )
            buckets.add("CRITIC" if critic else "SYNTHESIS")
        if validation.get("critic_status") == "FAILED":
            buckets.add("CRITIC")
        if status == "FAIL_CLOSED" and set(validation.get("failures") or ()) & _SECURITY:
            buckets.add("FINALIZATION")
    if m["relevance"] == 0:
        buckets.add("ANSWER_RELEVANCE")
    if m["completeness"] == 0:
        buckets.add("ANSWER_COMPLETENESS")
    verdict = "FAIL" if fail else "PARTIAL" if partial else "PASS"
    return verdict, sorted(buckets, key=BUCKETS.index)


# -- aggregate --------------------------------------------------------------------------
def _mean(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return (statistics.fmean(float(v) for v in values), len(values)) if values else (NA, 0)


def _rate(rows, key, value=True):
    applicable = [r for r in rows if r.get(key) is not None]
    return ratio(sum(1 for r in applicable if r[key] == value), len(applicable)), len(applicable)


def _pct(values, q):
    if not values:
        return NA
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def aggregate(scored: list[dict]) -> dict:
    """Layered summary; every value carries its denominator (n); NOT_APPLICABLE = None."""
    ok = [s for s in scored if s.get("status") is not None]
    claims = sum(s["published_claims"] for s in ok)
    support = sum((Counter(s["support"]) for s in ok), Counter())
    latencies = [s["latency_ms"] for s in ok if s.get("latency_ms") is not None]
    removed = sum((Counter(s["claims_removed"]) for s in ok), Counter())
    roles = sum((Counter(s["model_calls"]) for s in ok), Counter())
    tokens: dict[str, Counter] = defaultdict(Counter)
    for s in ok:
        for role, counts in s["tokens"].items():
            tokens[role].update(counts)
    stages: dict[str, list[float]] = defaultdict(list)
    for s in ok:
        for name, ms in s["stage_latency_ms"].items():
            stages[name].append(ms)
    all_calls = sum(roles.values())

    def m(key):
        value, n = _mean(ok, key)
        return {"value": value, "n": n}

    def r(key, value=True):
        rate, n = _rate(ok, key, value)
        return {"value": rate, "n": n}

    return {
        "cases": len(scored),
        "verdicts": dict(Counter(s["verdict"] for s in scored)),
        "retrieval": {k: m(k) for k in _RETRIEVAL_KEYS},
        "context": {
            "context_recall": m("context_recall"),
            "context_precision": m("context_precision"),
            "context_source_coverage": m("context_source_coverage"),
            "temporal_context_coverage": m("temporal_context_coverage"),
            "evidence_retrieved_mean": m("evidence_retrieved"),
            "evidence_shown_mean": m("evidence_shown"),
        },
        "generation": {
            "claim_precision_strict": {"value": ratio(support["SUPPORTED"], claims), "n": claims},
            "support_distribution": dict(support),
            "fact_recall": m("fact_recall"),
            "answer_f1": m("answer_f1"),
            "relevance": m("relevance"),
            "completeness": m("completeness"),
            "conciseness": m("conciseness"),
        },
        "citations": {
            k: m(k)
            for k in (
                "citation_validity",
                "citation_completeness",
                "citation_project_consistency",
                "citation_temporal_consistency",
                "citation_entailment_critic",
                "citation_entailment_human",
            )
        },
        "agent": {
            "task_success": r("task_success", "PASS"),
            "task_success_or_partial": {
                "value": ratio(
                    sum(s.get("task_success") in ("PASS", "PARTIAL") for s in ok), len(ok)
                ),
                "n": len(ok),
            },
            "routing_ok": r("routing_ok"),
            "tool_selection_ok": r("tool_selection_ok"),
            "project_isolation": r("project_isolation"),
            "anchor_resolution_ok": r("anchor_resolution_ok"),
            "anchor_relation_ok": r("anchor_relation_ok"),
            "clarification_ok": r("clarification_ok"),
            "refusal_ok": r("refusal_ok"),
            "false_refusal_rate": r("false_refusal"),
            "abstention_ok": r("abstention_ok"),
            "unnecessary_abstention_rate": r("unnecessary_abstention"),
            "hallucination_rate": {
                "value": ratio(sum(s["unsupported_published"] for s in ok), claims),
                "n": claims,
            },
            "unverified_published_rate": {
                "value": ratio(support["NOT_ASSESSED"], claims),
                "n": claims,
            },
            "candidate_claims_removed": dict(removed),
            "escaped_unsafe_claims": sum(s["escaped_unsafe_claims"] for s in ok),
            "tool_calls_mean": m("tool_calls"),
            "decision_rounds_mean": m("decision_rounds"),
            "search_actions_mean": m("search_actions"),
            "rejected_actions": sum(s["rejected_actions"] for s in ok),
            "tool_errors": sum(s["tool_errors"] for s in ok),
        },
        "operations": {
            "runtime_success": {
                "value": ratio(sum(s["runtime_success"] for s in scored), len(scored)),
                "n": len(scored),
            },
            "latency_ms": {
                "median": statistics.median(latencies) if latencies else NA,
                "p90": _pct(latencies, 0.9),
                "max": max(latencies) if latencies else NA,
            },
            "stage_latency_ms_median": {k: statistics.median(v) for k, v in sorted(stages.items())},
            "model_calls_by_role": dict(roles),
            "model_call_failure_rate": {
                "value": ratio(sum(s["model_call_failures"] for s in ok), all_calls),
                "n": all_calls,
            },
            "parse_failure_rate": {
                "value": ratio(sum(s["parse_failures"] for s in ok), all_calls),
                "n": all_calls,
            },
            "fail_closed_rate": r("fail_closed"),
            "graceful_degradation_cases": sum(s["graceful_degradation"] for s in ok),
            "tokens_by_role": {k: dict(v) for k, v in tokens.items()},
            "cost_usd": "UNAVAILABLE (endpoint pricing not exposed; tokens and calls reported)",
        },
        "by_category": {
            c: {
                "cases": len(rows),
                "verdicts": dict(Counter(s["verdict"] for s in rows)),
                "median_latency_ms": statistics.median(lat)
                if (lat := [s["latency_ms"] for s in rows if s.get("latency_ms") is not None])
                else NA,
            }
            for c in CATEGORIES
            if (rows := [s for s in scored if s["category"] == c])
        },
        "failure_buckets": dict(
            Counter(b for s in scored if s["verdict"] != "PASS" for b in s["buckets"])
        ),
    }


# -- report -----------------------------------------------------------------------------
def _fmt(entry) -> str:
    if isinstance(entry, dict) and "value" in entry:
        value, n = entry["value"], entry.get("n")
        if value is None:
            return "NOT_APPLICABLE"
        return f"{value:.2f} (n={n})" if isinstance(value, float) else f"{value} (n={n})"
    return "NOT_APPLICABLE" if entry is None else str(entry)


HEADLINE = (
    ("Retrieval Recall@10", ("retrieval", "recall_at_10")),
    ("Retrieval Precision@10", ("retrieval", "precision_at_10")),
    ("Grounded claim precision (strict)", ("generation", "claim_precision_strict")),
    ("Citation validity", ("citations", "citation_validity")),
    ("Project isolation", ("agent", "project_isolation")),
    ("Task success (PASS)", ("agent", "task_success")),
    ("Prediction/out-of-scope refusal", ("agent", "refusal_ok")),
)


def render_report(summary: dict, scored: list[dict], run: dict) -> str:
    lines = [
        f"# Application evaluation — run {run.get('run_id')}",
        "",
        f"Revision: {run.get('revision')} · started {run.get('started_at')} · "
        f"{summary['cases']} cases · verdicts {summary['verdicts']}",
        "",
        "## Headline (slide)",
        "",
    ]
    for label, (group, key) in HEADLINE:
        lines.append(f"- {label}: {_fmt(summary[group][key])}")
    ops = summary["operations"]["latency_ms"]
    lines += [f"- Latency median / p90: {ops['median']} / {ops['p90']} ms", ""]
    titles = {
        "retrieval": "A. Retrieval (Phase 8 relevance, binary)",
        "context": "B. Context",
        "generation": "C. Generation",
        "citations": "D. Citations",
        "agent": "E. Agent / system",
        "operations": "F. Operations",
    }
    for group, title in titles.items():
        lines += [f"## {title}", ""]
        lines += [f"- {k}: {_fmt(v)}" for k, v in summary[group].items()]
        lines.append("")
    lines += [
        "## G. Category breakdown",
        "",
        "| category | cases | verdicts | median latency ms |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {c} | {v['cases']} | {v['verdicts']} | {v['median_latency_ms']} |"
        for c, v in summary["by_category"].items()
    ]
    lines += [
        "",
        "## H. Failure analysis",
        "",
        f"Buckets (non-PASS cases): {summary['failure_buckets']}",
        "",
        "| case | category | verdict | status | buckets | trace |",
        "|---|---|---|---|---|---|",
    ]
    for s in scored:
        if s["verdict"] != "PASS":
            lines.append(
                f"| {s['case_id']} | {s['category']} | {s['verdict']} | {s.get('status')} | "
                f"{', '.join(s['buckets'])} | {s.get('trace_id')} |"
            )
    lines += [
        "",
        "## Per-case results",
        "",
        "| case | verdict | runtime | isolation | citations | task | "
        "relevance | latency ms | trace |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in scored:
        lines.append(
            f"| {s['case_id']} | {s['verdict']} | {s['runtime_success']} | "
            f"{s.get('project_isolation')} | {_fmt(s.get('citation_validity'))} | "
            f"{s.get('task_success')} | "
            f"{_fmt(s.get('relevance'))} | {s.get('latency_ms')} | {s.get('trace_id')} |"
        )
    lines += [
        "",
        "Semantic dimensions (relevance, completeness, conciseness, human citation "
        "entailment) are NOT_APPLICABLE until a human review file is supplied. nDCG is "
        "NOT_APPLICABLE: only binary relevance labels exist. Cost is UNAVAILABLE.",
    ]
    return "\n".join(lines) + "\n"
