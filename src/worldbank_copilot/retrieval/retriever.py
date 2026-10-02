"""Scoped retrieval: query -> filters -> candidates -> guardrails -> rerank -> evidence.

Order of operations (Claude.md §12, §22):

1. the project must be one of the approved projects and the strategy must exist;
2. the question is processed deterministically (a foreign project id is refused);
3. metadata filters are built (project and strategy always; ISR sequence when the
   question names one; document types only when ``use_type_filters``);
4. candidates come ONLY from rows/vectors inside that scope (lexical: filtered before
   indexing; dense: Vector Search filters);
5. every candidate is re-checked against the governed corpus: an id outside the scope or
   missing from the corpus raises (ERROR), it is never silently dropped;
6. duplicates (same document, same text) are suppressed; the reranker reorders;
7. the top ``final_k`` become citation-ready ``Evidence``; nothing qualifying ->
   INSUFFICIENT_EVIDENCE.

Phase 9 integration split (no methodology change): steps 1-6a (up to and including
de-duplication) are ``first_stage``; reranking and evidence assembly are ``finish``.
``retrieve`` is exactly ``finish(first_stage(...))``, so a rerank policy can inspect the
first-stage candidates before deciding whether to rerank.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from worldbank_copilot.retrieval.config import RetrievalSettings
from worldbank_copilot.retrieval.embeddings import EmbeddingProvider
from worldbank_copilot.retrieval.lexical import BM25, reciprocal_rank_fusion
from worldbank_copilot.retrieval.models import (
    Candidate,
    Citation,
    Evidence,
    EvidenceSet,
    RetrievalError,
    RetrievalScope,
    ScopeViolation,
)
from worldbank_copilot.retrieval.query import ProcessedQuery, process_query
from worldbank_copilot.retrieval.rerank import Reranker, rerank_order
from worldbank_copilot.retrieval.vector_search import DenseSearcher

REQUIRED_COLUMNS = (
    "chunk_id",
    "chunk_strategy",
    "chunk_role",
    "chunk_type",
    "parent_chunk_id",
    "project_id",
    "document_id",
    "document_type",
    "document_label",
    "document_date",
    "isr_sequence",
    "source_file",
    "source_hash",
    "page_number",
    "page_numbers",
    "section_title",
    "element_ids",
    "chunk_text",
    "search_text",
)


class ChunkStore:
    """Governed chunk rows (from silver.document_chunks) indexed by chunk_id."""

    def __init__(self, rows: Iterable[dict[str, Any]]):
        self.rows: dict[str, dict[str, Any]] = {}
        for row in rows:
            missing = [c for c in REQUIRED_COLUMNS if c not in row]
            if missing:
                raise RetrievalError(f"chunk row lacks {missing} (malformed corpus)")
            if not row["project_id"] or row["page_number"] is None or not row["document_id"]:
                raise RetrievalError(f"chunk {row['chunk_id']}: missing citation metadata")
            if row["chunk_id"] in self.rows:
                raise RetrievalError(f"duplicate chunk_id {row['chunk_id']}")
            self.rows[row["chunk_id"]] = row
        self._lexical: dict[RetrievalScope, BM25] = {}
        self.strategies = {r["chunk_strategy"] for r in self.rows.values()}

    @classmethod
    def from_table(cls, spark: Any, table: str) -> ChunkStore:
        """Load the governed corpus (bounded: a few MB per strategy) to the driver."""
        frame = spark.table(table).select(*REQUIRED_COLUMNS)
        return cls(row.asDict(recursive=True) for row in frame.collect())

    def get(self, chunk_id: str) -> dict[str, Any] | None:
        return self.rows.get(chunk_id)

    def latest_isr(self, project_id: str) -> int | None:
        seqs = [
            r["isr_sequence"]
            for r in self.rows.values()
            if r["project_id"] == project_id and r["isr_sequence"] is not None
        ]
        return max(seqs) if seqs else None

    def lexical(self, scope: RetrievalScope, k1: float, b: float) -> BM25:
        if scope not in self._lexical:
            rows = sorted(
                (r for r in self.rows.values() if scope.admits(r)), key=lambda r: r["chunk_id"]
            )
            self._lexical[scope] = BM25(
                [r["chunk_id"] for r in rows], [r["search_text"] for r in rows], k1, b
            )
        return self._lexical[scope]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


@dataclass(frozen=True)
class FirstStage:
    """Scoped, guarded, de-duplicated candidates in retrieval order (before reranking)."""

    question: str
    project_id: str
    query: ProcessedQuery
    scope: RetrievalScope
    method: str
    candidate_k: int
    candidates: tuple[Candidate, ...]
    started: float  # time.perf_counter() when the request started
    first_stage_ms: float


class Retriever:
    def __init__(
        self,
        store: ChunkStore,
        settings: RetrievalSettings,
        allowed_projects: Sequence[str],
        embeddings: EmbeddingProvider | None = None,
        dense: DenseSearcher | None = None,
    ):
        self.store = store
        self.settings = settings
        self.allowed = set(allowed_projects)
        self.embeddings = embeddings
        self.dense = dense
        self._vectors: dict[str, list[float]] = {}

    # -- scope ---------------------------------------------------------------------

    def scope_for(
        self, query: ProcessedQuery, project_id: str, strategy: str, use_type_filters: bool = False
    ) -> RetrievalScope:
        if project_id not in self.allowed:
            raise ScopeViolation(f"project {project_id!r} is not in the approved corpus")
        if strategy not in self.store.strategies:
            raise RetrievalError(f"chunk strategy {strategy!r} is not in the corpus")
        sequences = query.isr_sequences
        if not sequences and query.latest_isr:
            latest = self.store.latest_isr(project_id)
            sequences = (latest,) if latest is not None else ()
        types = query.document_type_hints if use_type_filters else ()
        if sequences and "ISR" not in types:
            types = ()  # an ISR sequence already implies the ISR document type
        return RetrievalScope(
            project_id=project_id,
            chunk_strategy=strategy,
            document_types=types,
            isr_sequences=sequences,
        )

    # -- candidates ------------------------------------------------------------------

    def _vector(self, text: str) -> list[float]:
        if self.embeddings is None:
            raise RetrievalError("dense retrieval needs an embedding provider")
        if text not in self._vectors:
            self._vectors[text] = self.embeddings.embed([text])[0]
        return self._vectors[text]

    def candidates(
        self, query: ProcessedQuery, scope: RetrievalScope, method: str, k: int
    ) -> list[Candidate]:
        if method == "lexical":
            cfg = self.settings.retrieval.lexical
            hits = self.store.lexical(scope, cfg.k1, cfg.b).search(query.expanded, k)
        elif method == "dense":
            if self.dense is None:
                raise RetrievalError("dense retrieval needs a Vector Search index")
            hits = self.dense.search(self._vector(query.expanded), scope.vector_filters(), k)
        elif method == "hybrid":
            lexical = [c.chunk_id for c in self.candidates(query, scope, "lexical", k)]
            dense = [c.chunk_id for c in self.candidates(query, scope, "dense", k)]
            hits = reciprocal_rank_fusion(
                [lexical, dense], self.settings.retrieval.hybrid.rrf_k, limit=k
            )
        else:
            raise RetrievalError(f"unknown retrieval method {method!r}")
        out = [
            Candidate(chunk_id=i, method=method, score=s, rank=r)
            for r, (i, s) in enumerate(hits, 1)
        ]
        self.guard(out, scope)
        return out

    def guard(self, candidates: Sequence[Candidate], scope: RetrievalScope) -> None:
        """ERROR if any candidate is unknown to the governed corpus or out of scope."""
        for candidate in candidates:
            row = self.store.get(candidate.chunk_id)
            if row is None:
                raise RetrievalError(
                    f"chunk {candidate.chunk_id} is not in the governed corpus "
                    "(index out of sync with silver.document_chunks)"
                )
            if not scope.admits(row):
                raise ScopeViolation(
                    f"chunk {candidate.chunk_id} ({row['project_id']}, "
                    f"{row['chunk_strategy']}) is outside the scope {scope.vector_filters()}"
                )

    def dedupe(self, candidates: Sequence[Candidate]) -> list[Candidate]:
        seen: set[tuple[str, str]] = set()
        out = []
        for candidate in candidates:
            row = self.store.rows[candidate.chunk_id]
            key = (row["document_id"], _norm(row["chunk_text"]))
            if key not in seen:
                seen.add(key)
                out.append(candidate)
        return out

    def rerank(
        self, query: ProcessedQuery, candidates: Sequence[Candidate], reranker: Reranker
    ) -> list[Candidate]:
        if reranker.name == "none" or not candidates:
            return list(candidates)
        texts = [self.store.rows[c.chunk_id]["search_text"] for c in candidates]
        order = rerank_order([c.chunk_id for c in candidates], reranker.score(query.text, texts))
        by_id = {c.chunk_id: c for c in candidates}
        return [by_id[i].model_copy(update={"rerank_score": s}) for i, s, _ in order]

    # -- evidence ------------------------------------------------------------------

    def evidence(self, candidate: Candidate, rank: int) -> Evidence:
        row = self.store.rows[candidate.chunk_id]
        parent = self.store.get(row["parent_chunk_id"]) if row["parent_chunk_id"] else None
        citation = Citation(
            project_id=row["project_id"],
            document_id=row["document_id"],
            document_label=row["document_label"],
            document_type=row["document_type"],
            document_date=row["document_date"],
            pages=list(row["page_numbers"]),
            section=row["section_title"],
            source_file=row["source_file"],
        )
        return Evidence(
            evidence_id=f"E{rank}",
            rank=rank,
            project_id=row["project_id"],
            document_id=row["document_id"],
            document_type=row["document_type"],
            document_label=row["document_label"],
            document_date=row["document_date"],
            isr_sequence=row["isr_sequence"],
            page_number=row["page_number"],
            page_numbers=list(row["page_numbers"]),
            section=row["section_title"],
            chunk_id=row["chunk_id"],
            chunk_type=row["chunk_type"],
            chunk_strategy=row["chunk_strategy"],
            text=row["chunk_text"],
            context_text=parent["chunk_text"] if parent else None,
            parent_chunk_id=row["parent_chunk_id"],
            retrieval_method=candidate.method,
            retrieval_score=candidate.score,
            rerank_score=candidate.rerank_score,
            source_file=row["source_file"],
            source_hash=row["source_hash"],
            element_ids=list(row["element_ids"]),
            citation=citation,
        )

    def retrieve(
        self,
        question: str,
        project_id: str,
        *,
        strategy: str,
        method: str,
        reranker: Reranker,
        candidate_k: int,
        final_k: int,
        use_type_filters: bool = False,
    ) -> EvidenceSet:
        stage = self.first_stage(
            question,
            project_id,
            strategy=strategy,
            method=method,
            candidate_k=candidate_k,
            use_type_filters=use_type_filters,
        )
        return self.finish(stage, reranker=reranker, final_k=final_k)

    def first_stage(
        self,
        question: str,
        project_id: str,
        *,
        strategy: str,
        method: str,
        candidate_k: int,
        use_type_filters: bool = False,
    ) -> FirstStage:
        """Query processing, scope, candidates, guardrails and de-duplication."""
        started = time.perf_counter()
        query = process_query(question, project_id, self.settings.query)
        scope = self.scope_for(query, project_id, strategy, use_type_filters)
        candidates = self.dedupe(self.candidates(query, scope, method, candidate_k))
        return FirstStage(
            question=question,
            project_id=project_id,
            query=query,
            scope=scope,
            method=method,
            candidate_k=candidate_k,
            candidates=tuple(candidates),
            started=started,
            first_stage_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def finish(self, stage: FirstStage, *, reranker: Reranker, final_k: int) -> EvidenceSet:
        """Rerank (or not), select the top ``final_k`` and build citation-ready evidence."""
        question, project_id, query = stage.question, stage.project_id, stage.query
        scope, method, candidate_k, started = (
            stage.scope,
            stage.method,
            stage.candidate_k,
            stage.started,
        )
        ranked = self.rerank(query, list(stage.candidates), reranker)
        top = ranked[:final_k]
        notes: list[str] = []
        status = "OK"
        threshold = self.settings.retrieval.min_rerank_score
        if not top:
            status, notes = "INSUFFICIENT_EVIDENCE", ["no chunk matched within the scope"]
        elif (
            reranker.name != "none"
            and threshold is not None
            and (top[0].rerank_score or 0) < threshold
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
