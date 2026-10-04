"""Presentation models for a ``copilot.investigate`` result (pure; no I/O, no model calls).

Input is the JSON form of ``InvestigationResult`` returned by the backend job. This module
only formats what the Copilot decided: it never routes, retrieves, summarizes, validates
or finalizes. Canonical identifiers, source identities and local handles are never shown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

PROJECTS = {
    "P130544": "Karnataka Urban Water Supply Modernization Project",
    "P179039": "Karnataka Sustainable Rural Water Supply Program",
    "P506272": "Karnataka Water Security and Resilience Program",
}
DEFAULT_PROJECT = "P130544"
EXAMPLES = (
    "What deserves my attention?",
    "Show me the restructuring timeline.",
    "Why did the PDO rating change?",
    "What implementation challenges are documented?",
)

STATUS_MESSAGES = {
    "INSUFFICIENT_EVIDENCE": (
        "The available evidence was not sufficient to support a publishable answer."
    ),
    "CLARIFY": (
        "I couldn't determine the intended analysis from that question. Try asking about "
        "implementation progress, ratings, results, financing, restructurings or project "
        "documents."
    ),
    "REFUSE": (
        "This Copilot provides evidence-grounded implementation intelligence and does not "
        "predict whether a project will succeed or fail."
    ),
    "FAIL_CLOSED": (
        "The answer could not be published because it did not pass the Copilot's validation checks."
    ),
}
GOVERNED_NOTICE = "Governed data — no generative model used"
SIGNAL_NOTICE = (
    "Attention signals are deterministic conditions derived from governed project data. "
    "They are not predictions of project success or failure."
)

# Fixed provenance labels and badge colors (Streamlit color names).
PROVENANCE = {
    "FACT": ("FACT", "blue"),
    "DOCUMENTED_FINDING": ("DOCUMENTED FINDING", "violet"),
    "SYSTEM_DERIVED_SIGNAL": ("SYSTEM-DERIVED SIGNAL", "orange"),
    "AI_INTERPRETATION": ("AI INTERPRETATION", "green"),
    "UNKNOWN": ("UNKNOWN — not available", "gray"),
}
PUBLISHABLE = {"FACT", "DOCUMENTED_FINDING", "SYSTEM_DERIVED_SIGNAL", "AI_INTERPRETATION"}

# Engineering notes kept for Technical Details, not the user-facing limitations list.
_INTERNAL_NOTES = re.compile(
    r"NOT_ESTABLISHED|NOT_ASSESSED|^Semantic support requires|^Contextual snapshots|"
    r"^Per-table snapshots|^deferred rule candidate|recorded, not applied as retrieval|"
    r"^Application context projection|^Context omitted evidence|^Synthetic capability|"
    r"^SYSTEM_DERIVED_SIGNAL: each item"
)
# Record fields never displayed: identifiers, lineage internals, handles.
_HIDDEN_FIELDS = re.compile(
    r"(_id|_ids|_hash|identity)$|^(source|provenance_class|supporting_record_ids|rule_version|"
    r"extraction_status|date_used_for_filter|source_file|chunk_type|chunk_strategy|"
    r"retrieval_method|rank|scope|rerank_decision|reranker|candidate_k|final_k|"
    r"first_stage_candidates|first_stage_ms|latency_ms|retrieval_notes|query)$"
)
_TABLE_NAMES = {
    "gold.attention_signals": "Attention signals",
    "gold.project_timeline": "Project timeline",
    "gold.project_360": "Project overview",
    "gold.result_progress": "Results progress",
    "gold.risk_register": "Risk register",
}
MAX_SOURCES = 25
MAX_EXCERPT = 600


@dataclass(frozen=True)
class Badge:
    label: str
    color: str


@dataclass(frozen=True)
class ClaimView:
    text: str
    badge: Badge
    sources: tuple[str, ...]


@dataclass(frozen=True)
class SourceView:
    title: str
    details: tuple[str, ...]
    badge: Badge
    excerpt: str | None = None


@dataclass(frozen=True)
class Table:
    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class SignalView:
    title: str
    description: str
    severity: str
    status: str
    observed: str | None
    values: str | None
    caveats: tuple[str, ...]


@dataclass(frozen=True)
class ResultView:
    status: str
    headline: str | None
    claims: tuple[ClaimView, ...] = ()
    governed_notice: str | None = None
    tables: tuple[Table, ...] = ()
    passages: tuple[SourceView, ...] = ()
    signal_groups: tuple[tuple[str, tuple[SignalView, ...]], ...] = ()
    sources: tuple[SourceView, ...] = ()
    hidden_sources: int = 0
    limitations: tuple[str, ...] = ()
    technical: tuple[tuple[str, str], ...] = ()
    stage_latency: tuple[tuple[str, str], ...] = ()
    model_calls: tuple[tuple[tuple[str, str], ...], ...] = ()
    technical_notes: tuple[str, ...] = ()
    withheld_unknown_claims: int = 0


def present(result: dict[str, Any]) -> ResultView:
    """Build the view for one result. Status semantics come from the Copilot, unchanged."""
    status = result["status"]
    limitations, notes = _split_limitations(result.get("limitations") or ())
    claims, withheld = _claims(result) if status == "ANSWER" else ((), 0)
    tables, passages = _governed(result) if status == "EVIDENCE_ONLY" else ((), ())
    sources = tuple(_source(item) for item in result.get("evidence") or ())
    sources = tuple(s for s in sources if s is not None)
    headline = None if status in ("ANSWER", "EVIDENCE_ONLY") else STATUS_MESSAGES.get(status)
    if status == "ANSWER" and not claims:  # defensive: nothing authoritative to show
        headline = STATUS_MESSAGES["INSUFFICIENT_EVIDENCE"]
    return ResultView(
        status=status,
        headline=headline,
        claims=claims,
        governed_notice=GOVERNED_NOTICE if status == "EVIDENCE_ONLY" else None,
        tables=tables,
        passages=passages,
        signal_groups=signal_groups(result.get("attention_signals") or ()),
        sources=sources[:MAX_SOURCES],
        hidden_sources=max(0, len(sources) - MAX_SOURCES),
        limitations=limitations if status != "REFUSE" else (),
        technical=_technical(result),
        stage_latency=tuple(
            (stage, f"{ms:,.0f} ms") for stage, ms in (result.get("stage_latency_ms") or {}).items()
        ),
        model_calls=tuple(_model_call(call) for call in result.get("model_calls") or ()),
        technical_notes=notes,
        withheld_unknown_claims=withheld,
    )


def badge(provenance: str) -> Badge:
    label, color = PROVENANCE.get(provenance, PROVENANCE["UNKNOWN"])
    return Badge(label, color)


def signal_groups(signals) -> tuple[tuple[str, tuple[SignalView, ...]], ...]:
    """Signals grouped by their backend category; severity is shown as provided."""
    groups: dict[str, list[SignalView]] = {}
    for s in signals:
        values = " → ".join(v for v in (s.get("comparison_value"), s.get("current_value")) if v)
        groups.setdefault(_humanize(s.get("category") or "Other"), []).append(
            SignalView(
                title=s.get("title") or "Attention signal",
                description=s.get("description") or "",
                severity=s.get("severity") or "",
                status=s.get("status") or "",
                observed=s.get("observed_date"),
                values=values or None,
                caveats=tuple(s.get("caveats") or ()),
            )
        )
    return tuple((name, tuple(items)) for name, items in sorted(groups.items()))


# -- answer --------------------------------------------------------------------------
def _claims(result) -> tuple[tuple[ClaimView, ...], int]:
    views, withheld = [], 0
    for claim in result.get("claims") or ():
        if claim.get("provenance") not in PUBLISHABLE:  # UNKNOWN is never authoritative
            withheld += 1
            continue
        views.append(
            ClaimView(
                text=claim["text"],
                badge=badge(claim["provenance"]),
                sources=tuple(dict.fromkeys(_citation(c) for c in claim.get("citations") or ())),
            )
        )
    return tuple(views), withheld


def _citation(citation: dict) -> str:
    if citation.get("document_label"):
        pages = citation.get("pages") or ()
        return citation["document_label"] + (f", p. {_pages(pages)}" if pages else "")
    if citation.get("table"):
        return f"{_table_name(citation['table'])} (governed data)"
    return "Governed project record"


# -- evidence-only (deterministic routes) --------------------------------------------
def _governed(result) -> tuple[tuple[Table, ...], tuple[SourceView, ...]]:
    by_tool: dict[str, list[dict]] = {}
    for item in result.get("evidence") or ():
        if item.get("tool"):
            by_tool.setdefault(item["tool"], []).append(item["payload"])
    tables, passages = [], []
    for tool, payloads in by_tool.items():
        if tool == "get_attention_signals":
            continue  # rendered as signal cards from result.attention_signals
        if tool == "get_project_timeline":
            tables.append(_timeline(payloads))
        elif tool == "search_project_documents":
            passages.extend(_passages(payloads))
        else:
            tables.append(_records(_humanize(tool.removeprefix("get_")), payloads))
    return tuple(tables), tuple(passages)


def _timeline(events) -> Table:
    def date_of(e):
        if e.get("event_date"):
            return e["event_date"], e["event_date"]
        if e.get("candidate_event_date"):
            return e["candidate_event_date"], f"{e['candidate_event_date']} (estimated)"
        return "9999", "Date not stated"

    rows = []
    for e in sorted(events, key=lambda e: (date_of(e)[0], e.get("event_sequence") or 0)):
        rows.append(
            (
                date_of(e)[1],
                _humanize(e.get("event_type") or ""),
                e.get("event_title") or "",
                e.get("event_description") or "",
                "" if e.get("isr_sequence") is None else f"ISR {e['isr_sequence']}",
            )
        )
    return Table("Timeline", ("Date", "Type", "Event", "Description", "ISR"), tuple(rows))


def _passages(results) -> list[SourceView]:
    views = []
    for search in results:
        for item in search.get("evidence") or ():
            e = item.get("evidence") or {}
            views.append(
                SourceView(
                    title=e.get("document_label") or "Project document",
                    details=_document_details(
                        e.get("document_type"),
                        e.get("document_date"),
                        e.get("isr_sequence"),
                        e.get("page_numbers") or (),
                        e.get("section"),
                    ),
                    badge=badge(item.get("provenance_class") or "DOCUMENTED_FINDING"),
                    excerpt=_excerpt(e.get("text")),
                )
            )
    return views


def _records(title: str, payloads) -> Table:
    columns: list[str] = []
    rows = []
    for payload in payloads:
        flat = _flatten(payload)
        columns.extend(k for k in flat if k not in columns)
        rows.append(flat)
    return Table(
        title,
        tuple(_humanize(c) for c in columns),
        tuple(tuple(row.get(c, "") for c in columns) for row in rows),
    )


def _flatten(payload: dict) -> dict[str, str]:
    """Scalar fields and nested facts (name/value/unit) as display strings."""
    out = {}
    for key, value in payload.items():
        if _HIDDEN_FIELDS.search(key):
            continue
        if isinstance(value, dict) and "value" in value:
            out[key] = _fact_value(value)
        elif isinstance(value, list | tuple) and value and isinstance(value[0], dict):
            out[key] = f"{len(value)} items"
        elif not isinstance(value, dict):
            out[key] = _scalar(value)
    return out


# -- evidence & sources --------------------------------------------------------------
def _source(item: dict) -> SourceView | None:
    payload, provenance = item.get("payload") or {}, (item.get("provenance") or ("UNKNOWN",))[0]
    if item.get("tool"):  # deterministic-route record: name its governed source only
        return SourceView(
            title=_humanize(item["tool"].removeprefix("get_")),
            details=("Governed project data",),
            badge=badge(provenance),
        )
    citation = item.get("citation") or {}
    if item.get("source_type") == "DOCUMENT" or citation:
        return SourceView(
            title=citation.get("document_label") or "Project document",
            details=_document_details(
                citation.get("document_type"),
                citation.get("document_date") or payload.get("document_date"),
                payload.get("isr_sequence"),
                citation.get("pages") or (),
                citation.get("section"),
            ),
            badge=badge(provenance),
            excerpt=_excerpt(payload.get("text")),
        )
    source = item.get("source") or {}
    name = _humanize(payload.get("name") or "record")
    value = _fact_value(payload) if "value" in payload else None
    return SourceView(
        title=f"{name}: {value}" if value else name,
        details=tuple(
            d
            for d in (
                f"{_table_name(source['table'])} (governed data)" if source.get("table") else None,
                f"page {source['page_number']}" if source.get("page_number") else None,
            )
            if d
        ),
        badge=badge(provenance),
    )


def _document_details(doc_type, date, isr, pages, section) -> tuple[str, ...]:
    return tuple(
        d
        for d in (
            _humanize(doc_type) if doc_type else None,
            str(date) if date else None,
            f"ISR {isr}" if isr is not None else None,
            f"p. {_pages(pages)}" if pages else None,
            section,
        )
        if d
    )


# -- technical details ---------------------------------------------------------------
def _technical(result) -> tuple[tuple[str, str], ...]:
    validation = result.get("validation") or {}
    claims = result.get("claims") or ()
    rows = (
        ("Route", result.get("route")),
        ("Intent", result.get("intent")),
        ("Routing reason", result.get("reason_code")),
        ("Result status", result.get("status")),
        ("Validation disposition", validation.get("disposition")),
        ("Mechanical validity", validation.get("mechanical_validity")),
        ("Semantic support", validation.get("semantic_support")),
        ("Critic status", validation.get("critic_status")),
        ("Validation failures", ", ".join(validation.get("failures") or ()) or None),
        ("Evidence records", str(len(result.get("evidence") or ()))),
        ("Published claims", str(len(claims))),
        ("Citations", str(sum(len(c.get("citations") or ()) for c in claims))),
        ("Attention signals", str(len(result.get("attention_signals") or ()))),
        ("Model calls", str(len(result.get("model_calls") or ()))),
        ("Total latency", f"{result.get('latency_ms', 0):,.0f} ms"),
        ("MLflow trace ID", result.get("trace_id")),
        ("Request ID", result.get("request_id")),
        ("Finalizer branch", "See the MLflow trace (finalization span)"),
        ("Model note", result.get("model_capability_note")),
    )
    return tuple((label, value) for label, value in rows if value)


def _model_call(call: dict) -> tuple[tuple[str, str], ...]:
    rows = (
        ("Role", call.get("role")),
        ("Endpoint", call.get("endpoint")),
        ("Model", call.get("model_identity")),
        ("Outcome", call.get("outcome")),
        ("Diagnostic", call.get("diagnostic_reason")),
        ("Input tokens", call.get("input_tokens")),
        ("Output tokens", call.get("output_tokens")),
        ("Latency", f"{call.get('latency_ms', 0):,.0f} ms"),
    )
    return tuple((label, str(value)) for label, value in rows if value is not None)


# -- helpers -------------------------------------------------------------------------
def _split_limitations(items) -> tuple[tuple[str, ...], tuple[str, ...]]:
    shown = tuple(dict.fromkeys(i for i in items if not _INTERNAL_NOTES.search(i)))
    hidden = tuple(dict.fromkeys(i for i in items if _INTERNAL_NOTES.search(i)))
    return shown, hidden


def _fact_value(fact: dict) -> str:
    if fact.get("value") is None:
        reason = fact.get("unknown_reason")
        return f"not available ({reason})" if reason else "not available"
    unit = fact.get("unit")
    return f"{_scalar(fact['value'])} {unit}".strip() if unit else _scalar(fact["value"])


def _scalar(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, list | tuple):
        return ", ".join(_scalar(v) for v in value)
    return str(value)


def _excerpt(text: str | None) -> str | None:
    if not text:
        return None
    text = " ".join(text.split())
    return text if len(text) <= MAX_EXCERPT else text[: MAX_EXCERPT - 1].rstrip() + "…"


def _pages(pages) -> str:
    return ", ".join(str(p) for p in pages)


def _table_name(table: str) -> str:
    return _TABLE_NAMES.get(table, _humanize(table.split(".")[-1]))


_ACRONYMS = re.compile(r"\b(isr|pdo|pforr|sort|ibrd|ida)\b", re.IGNORECASE)


def _humanize(value: str) -> str:
    text = str(value).replace("_", " ").strip()
    if text.isupper() or "_" in str(value):
        text = text[:1].upper() + text[1:].lower()
    return _ACRONYMS.sub(lambda m: m.group(1).upper(), text)
