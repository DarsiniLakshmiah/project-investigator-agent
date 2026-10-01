"""Deterministic query processing (Phase 8): no model, no routing.

* whitespace normalisation;
* project-id recognition: a question naming another project than the active scope is
  refused (ScopeViolation) instead of being answered from the wrong project;
* acronym expansion (the acronym is kept, the expansion appended);
* ISR references: "ISR 23" / "ISR sequence 23" -> isr_sequence 23; "latest ISR" ->
  resolved by the retriever to the project's highest ISR sequence;
* document-type hints (recorded; applied as filters only when the experiment says so).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from worldbank_copilot.retrieval.config import QueryConfig
from worldbank_copilot.retrieval.models import ScopeViolation

_PROJECT = re.compile(r"\bP\d{6}\b")
_ISR_SEQ = re.compile(r"\bISR\s*(?:sequence|seq\.?|no\.?|#)?\s*(\d{1,2})\b", re.IGNORECASE)
_LATEST = re.compile(r"\b(latest|most recent|last|current)\s+ISR\b", re.IGNORECASE)


@dataclass(frozen=True)
class ProcessedQuery:
    original: str
    text: str  # normalised
    expanded: str  # normalised + acronym expansions (lexical / dense input)
    projects_mentioned: tuple[str, ...]
    isr_sequences: tuple[int, ...]
    latest_isr: bool
    document_type_hints: tuple[str, ...]
    expansions: tuple[str, ...] = field(default=())


def process_query(question: str, project_id: str, config: QueryConfig) -> ProcessedQuery:
    text = " ".join((question or "").split())
    if not text:
        raise ValueError("empty question")
    mentioned = tuple(sorted(set(_PROJECT.findall(text))))
    foreign = [p for p in mentioned if p != project_id]
    if foreign:
        raise ScopeViolation(
            f"question refers to {foreign} but the retrieval scope is {project_id}; "
            "cross-project retrieval is not permitted"
        )
    expansions = []
    for acronym, expansion in sorted(config.acronyms.items()):
        if re.search(rf"(?<![\w&]){re.escape(acronym)}(?![\w&])", text) and (
            expansion.lower() not in text.lower()
        ):
            expansions.append(expansion)
    lower = text.lower()
    hints = tuple(
        sorted(
            doc_type
            for doc_type, words in config.document_type_hints.items()
            if any(re.search(rf"\b{re.escape(w)}\b", lower) for w in words)
        )
    )
    sequences = tuple(sorted({int(n) for n in _ISR_SEQ.findall(text)}))
    expanded = f"{text} ({'; '.join(expansions)})" if expansions else text
    return ProcessedQuery(
        original=question,
        text=text,
        expanded=expanded,
        projects_mentioned=mentioned,
        isr_sequences=sequences,
        latest_isr=bool(_LATEST.search(text)),
        document_type_hints=hints,
        expansions=tuple(expansions),
    )
