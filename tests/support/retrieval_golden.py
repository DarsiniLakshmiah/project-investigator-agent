"""Golden reference for the Phase 9 Retriever split (Decision 1).

``phase8_retrieve`` is a verbatim copy of ``Retriever.retrieve`` as validated in Phase 8
(git 351b5a0), before it was split into ``first_stage`` / ``finish``. It uses only the
retriever's unchanged building blocks (process_query, scope_for, candidates, dedupe,
rerank, evidence), so comparing it with the split ``retrieve`` proves the refactor is
behaviour-preserving. Do not edit it to make a comparison pass.

Also: deterministic stand-ins for the dense index and the CrossEncoder, so equivalence
can be checked locally on every retrieval method and both reranking modes.
"""

from __future__ import annotations

import re
import time
from typing import Any

from worldbank_copilot.retrieval.models import EvidenceSet, ScopeViolation
from worldbank_copilot.retrieval.query import process_query


def phase8_retrieve(
    self,
    question,
    project_id,
    *,
    strategy,
    method,
    reranker,
    candidate_k,
    final_k,
    use_type_filters=False,
) -> EvidenceSet:
    started = time.perf_counter()
    query = process_query(question, project_id, self.settings.query)
    scope = self.scope_for(query, project_id, strategy, use_type_filters)
    ranked = self.rerank(
        query, self.dedupe(self.candidates(query, scope, method, candidate_k)), reranker
    )
    top = ranked[:final_k]
    notes: list[str] = []
    status = "OK"
    threshold = self.settings.retrieval.min_rerank_score
    if not top:
        status, notes = "INSUFFICIENT_EVIDENCE", ["no chunk matched within the scope"]
    elif (
        reranker.name != "none" and threshold is not None and (top[0].rerank_score or 0) < threshold
    ):
        status = "INSUFFICIENT_EVIDENCE"
        notes = [f"best rerank score {top[0].rerank_score:.3f} below {threshold}"]
    evidence = [self.evidence(c, i) for i, c in enumerate(top, 1)] if status == "OK" else []
    for item in evidence:  # final output check: cited sources belong to the scope
        if item.project_id != project_id:
            raise ScopeViolation(f"evidence {item.chunk_id} belongs to {item.project_id}")
    return EvidenceSet(
        query=question,
        scope=scope,
        status=status,
        evidence=evidence,
        retrieval_method=method,
        reranker=reranker.name,
        candidate_k=candidate_k,
        final_k=final_k,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
        notes=notes,
    )


def comparable(result: EvidenceSet) -> dict[str, Any]:
    """Everything except wall-clock latency (evidence ids/order, scope, citations, scores)."""
    return result.model_dump(mode="json", exclude={"latency_ms"})


def outcome(call) -> tuple[str, Any]:
    """('ok', comparable result) or ('error', (exception type, message))."""
    try:
        return "ok", comparable(call())
    except Exception as exc:  # compared, never swallowed
        return "error", (type(exc).__name__, str(exc))


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


class TextEmbeddings:
    """Encodes the query text losslessly as the 'vector' (deterministic, no model)."""

    model, dimension = "golden-text", 0

    def embed(self, texts):
        return [[float(ord(ch)) for ch in text] for text in texts]


class ScopedDense:
    """Vector Search stand-in: token-overlap scoring over RETRIEVAL rows matching filters."""

    def __init__(self, rows):
        self.rows = [r for r in rows if r["chunk_role"] == "RETRIEVAL"]
        self.tokens = {r["chunk_id"]: _tokens(r["search_text"]) for r in self.rows}

    def search(self, vector, filters, k):
        query = _tokens("".join(chr(int(x)) for x in vector))
        hits = []
        for row in self.rows:
            if any(
                (row.get(col) not in value) if isinstance(value, list) else row.get(col) != value
                for col, value in filters.items()
            ):
                continue
            words = self.tokens[row["chunk_id"]]
            overlap = len(query & words)
            if overlap:
                hits.append((row["chunk_id"], overlap / (len(query | words) or 1)))
        hits.sort(key=lambda h: (-h[1], h[0]))
        return hits[:k]


class OverlapCrossEncoder:
    """CrossEncoder stand-in: deterministic, reorders candidates, can score negative."""

    name = "cross_encoder"

    def score(self, query, texts):
        words = _tokens(query)
        return [len(words & _tokens(t)) - 2.0 - len(t) / 10_000 for t in texts]
