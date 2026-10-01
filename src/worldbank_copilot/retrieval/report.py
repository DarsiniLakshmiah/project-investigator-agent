"""Isolation checks and human-readable reports for notebook 07 (no business logic)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from worldbank_copilot.retrieval.evaluation import Question, RunResult, StageDecision
from worldbank_copilot.retrieval.models import EvidenceSet, ScopeViolation
from worldbank_copilot.retrieval.rerank import Reranker
from worldbank_copilot.retrieval.retriever import Retriever


@dataclass
class IsolationCheck:
    name: str
    passed: bool
    detail: str


def isolation_checks(
    retriever: Retriever,
    questions: Sequence[Question],
    projects: Sequence[str],
    strategy: str,
    method: str,
    reranker: Reranker,
    candidate_k: int,
) -> list[IsolationCheck]:
    """Cross-project leakage tests (all must pass; a failure is an ERROR).

    1. every question asked under EVERY project scope returns only that project's chunks;
    2. a question naming another project is refused;
    3. a project outside the approved corpus is refused.
    """
    checks: list[IsolationCheck] = []
    for project in projects:
        seen: Counter[str] = Counter()
        for q in questions:
            text = q.question
            result = retriever.retrieve(
                text,
                project,
                strategy=strategy,
                method=method,
                reranker=reranker,
                candidate_k=candidate_k,
                final_k=candidate_k,
            )
            seen.update(e.project_id for e in result.evidence)
        foreign = {p: n for p, n in seen.items() if p != project}
        checks.append(
            IsolationCheck(
                f"scope {project}: {len(questions)} questions ({method})",
                not foreign,
                f"evidence by project {dict(seen)}" + (f"; LEAKED {foreign}" if foreign else ""),
            )
        )
    for project in projects:
        other = next(p for p in projects if p != project)
        try:
            retriever.retrieve(
                f"What did {other} report in its latest ISR?",
                project,
                strategy=strategy,
                method=method,
                reranker=reranker,
                candidate_k=candidate_k,
                final_k=5,
            )
            checks.append(IsolationCheck(f"{project} asked about {other}", False, "not refused"))
        except ScopeViolation as exc:
            checks.append(IsolationCheck(f"{project} asked about {other}", True, f"refused: {exc}"))
    try:
        retriever.retrieve(
            "latest ISR",
            "P000000",
            strategy=strategy,
            method=method,
            reranker=reranker,
            candidate_k=candidate_k,
            final_k=5,
        )
        checks.append(IsolationCheck("unsupported project P000000", False, "not refused"))
    except ScopeViolation as exc:
        checks.append(IsolationCheck("unsupported project P000000", True, f"refused: {exc}"))
    return checks


def decisions_table(decisions: Sequence[StageDecision]) -> list[dict[str, Any]]:
    rows = []
    for decision in decisions:
        for run in decision.runs:
            summary = run.summary
            rows.append(
                {
                    "stage": decision.stage,
                    "configuration": run.config.name,
                    "status": run.status,
                    "winner": run.config == decision.winner,
                    **{
                        k: summary.get(k)
                        for k in (
                            "recall_at_5",
                            "recall_at_10",
                            "mrr",
                            "ndcg_at_5",
                            "precision_at_5",
                            "latency_ms_p50",
                            "latency_ms_p95",
                        )
                    },
                    "note": run.error or "",
                }
            )
    return rows


def breakdown(run: RunResult) -> list[dict[str, Any]]:
    rows = []
    for label, key in (
        ("project", lambda q: q.project_id),
        ("category", lambda q: q.category),
        ("kind", lambda q: q.kind),
    ):
        for group, metrics in run.aggregate(key).items():
            rows.append({"by": label, "group": group, **metrics})
    return rows


def format_evidence(result: EvidenceSet, width: int = 220) -> str:
    lines = [
        f"Q: {result.query}",
        f"   scope {result.scope.vector_filters()}  status {result.status}"
        f"  {result.retrieval_method}/{result.reranker} k={result.candidate_k}"
        f"  {result.latency_ms} ms",
    ]
    for item in result.evidence:
        score = (
            f"rerank {item.rerank_score:.3f}"
            if item.rerank_score is not None
            else (f"score {item.retrieval_score:.4f}")
        )
        lines.append(f"  [{item.evidence_id}] {item.citation.text}  ({score})")
        lines.append(f"       {item.text[:width]!r}")
    lines.extend(f"   note: {n}" for n in result.notes)
    return "\n".join(lines)
