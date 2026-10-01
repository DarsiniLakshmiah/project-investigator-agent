"""Retrieval evaluation (Phase 8): questions, relevance, metrics, staged experiments.

Relevance is judged against source evidence (evaluation/retrieval_questions.yaml), never
against the retriever's own output: a chunk is relevant to an evidence item when it comes
from the same document, shares a page, and contains the item's verbatim phrase (compared
on alphanumeric tokens, so rating glyphs and punctuation do not matter).

Metrics per answerable question (ranked list = candidates after reranking):
* recall@k   - share of evidence items covered in the top k ("any" mode: 1 if one is);
* mrr        - 1 / rank of the first relevant chunk (0 if none in the list);
* ndcg@5     - binary gains; ideal = min(#evidence items, 5) relevant chunks;
* precision@5- relevant chunks in the top 5 / 5;
* leakage@k  - chunks from another project in the top k (must be 0; ERROR otherwise).
No-answer questions are excluded from ranking metrics; their best rerank scores are used
to calibrate abstention.
"""

from __future__ import annotations

import math
import re
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from worldbank_copilot.retrieval.config import EvaluationConfig
from worldbank_copilot.retrieval.models import Candidate, RetrievalError, ScopeViolation
from worldbank_copilot.retrieval.query import process_query
from worldbank_copilot.retrieval.rerank import Reranker
from worldbank_copilot.retrieval.retriever import Retriever


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    pages: list[int] = Field(min_length=1)
    contains: str = Field(min_length=3)


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    project_id: str
    category: str
    kind: str
    question: str
    answerable: bool = True
    evidence_mode: Literal["all", "any"] = "any"
    evidence: list[EvidenceRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> Question:
        if self.answerable and not self.evidence:
            raise ValueError(f"{self.id}: answerable question without evidence")
        if not self.answerable and self.evidence:
            raise ValueError(f"{self.id}: no-answer question with evidence")
        wrong = [
            e.document_id for e in self.evidence if not e.document_id.startswith(self.project_id)
        ]
        if wrong:
            raise ValueError(f"{self.id}: evidence from another project {wrong}")
        return self


def load_questions(path: Path) -> list[Question]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    questions = [Question.model_validate(q) for q in data.get("questions", [])]
    ids = [q.id for q in questions]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"duplicate question ids {dup}")
    return questions


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def matched_items(row: dict[str, Any], question: Question) -> set[int]:
    """Indexes of the question's evidence items this chunk row supports."""
    text = normalize(row["chunk_text"])
    pages = set(row["page_numbers"])
    return {
        i
        for i, ev in enumerate(question.evidence)
        if row["document_id"] == ev.document_id
        and pages & set(ev.pages)
        and normalize(ev.contains) in text
    }


def ranking_metrics(
    rows: Sequence[dict[str, Any]], question: Question, ks: Sequence[int]
) -> dict[str, float]:
    matches = [matched_items(r, question) for r in rows]
    n_items = len(question.evidence)
    out: dict[str, float] = {}
    for k in ks:
        covered = set().union(*matches[:k]) if matches[:k] else set()
        if question.evidence_mode == "any":
            out[f"recall_at_{k}"] = 1.0 if covered else 0.0
        else:
            out[f"recall_at_{k}"] = len(covered) / n_items
        out[f"leakage_at_{k}"] = float(
            sum(1 for r in rows[:k] if r["project_id"] != question.project_id)
        )
    first = next((i for i, m in enumerate(matches, 1) if m), None)
    out["mrr"] = 1.0 / first if first else 0.0
    gains = [1.0 if m else 0.0 for m in matches[:5]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(n_items, 5)))
    out["ndcg_at_5"] = dcg / ideal if ideal else 0.0
    out["precision_at_5"] = sum(gains) / 5.0
    return out


# ---------------------------------------------------------------------------
# Experiment configurations and runs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    chunk_strategy: str
    retrieval: str
    reranker: str
    candidate_k: int
    use_type_filters: bool = False

    @property
    def name(self) -> str:
        filters = "+typefilter" if self.use_type_filters else ""
        return (
            f"{self.chunk_strategy}|{self.retrieval}|{self.reranker}|k{self.candidate_k}{filters}"
        )

    def replace(self, **changes: Any) -> RunConfig:
        values = {**self.__dict__, **changes}
        return RunConfig(**values)


@dataclass
class QuestionResult:
    question_id: str
    project_id: str
    category: str
    kind: str
    answerable: bool
    ranked_chunk_ids: list[str]
    metrics: dict[str, float]
    top_rerank_score: float | None
    retrieval_ms: float
    rerank_ms: float
    missed: list[int] = field(default_factory=list)  # evidence item indexes not in top 10


@dataclass
class RunResult:
    config: RunConfig
    questions: list[QuestionResult]
    status: str = "OK"  # OK | UNAVAILABLE
    error: str | None = None

    def answerable(self) -> list[QuestionResult]:
        return [q for q in self.questions if q.answerable]

    def aggregate(self, by: Callable[[QuestionResult], str] | None = None) -> dict[str, dict]:
        groups: dict[str, list[QuestionResult]] = defaultdict(list)
        for q in self.answerable():
            groups[by(q) if by else "all"].append(q)
        out = {}
        for key, items in sorted(groups.items()):
            names = sorted({m for q in items for m in q.metrics})
            out[key] = {m: round(statistics.fmean(q.metrics[m] for q in items), 4) for m in names}
            out[key]["questions"] = len(items)
        return out

    @property
    def summary(self) -> dict[str, float]:
        if self.status != "OK":
            return {}
        agg = self.aggregate().get("all", {})
        lat = [q.retrieval_ms + q.rerank_ms for q in self.questions]
        agg["latency_ms_p50"] = round(statistics.median(lat), 1) if lat else 0.0
        agg["latency_ms_p95"] = (
            round(sorted(lat)[max(0, math.ceil(0.95 * len(lat)) - 1)], 1) if lat else 0.0
        )
        return agg


def run_config(
    retriever: Retriever,
    questions: Sequence[Question],
    config: RunConfig,
    rerankers: dict[str, Reranker],
    ks: Sequence[int],
) -> RunResult:
    """Evaluate one configuration over all questions (the list is candidate_k long)."""
    reranker = rerankers.get(config.reranker)
    if reranker is None:
        return RunResult(config, [], "UNAVAILABLE", f"reranker {config.reranker} unavailable")
    results = []
    try:
        for q in questions:
            query = process_query(q.question, q.project_id, retriever.settings.query)
            scope = retriever.scope_for(
                query, q.project_id, config.chunk_strategy, config.use_type_filters
            )
            t0 = time.perf_counter()
            candidates = retriever.dedupe(
                retriever.candidates(query, scope, config.retrieval, config.candidate_k)
            )
            t1 = time.perf_counter()
            ranked: list[Candidate] = retriever.rerank(query, candidates, reranker)
            t2 = time.perf_counter()
            rows = [retriever.store.rows[c.chunk_id] for c in ranked]
            metrics = ranking_metrics(rows, q, ks) if q.answerable else {}
            if any(metrics.get(f"leakage_at_{k}", 0) for k in ks):
                raise ScopeViolation(f"{q.id}: cross-project chunk in results ({config.name})")
            covered = set().union(*(matched_items(r, q) for r in rows[:10])) if rows else set()
            if q.evidence_mode == "any" and covered:  # one item answers the question
                covered = set(range(len(q.evidence)))
            results.append(
                QuestionResult(
                    q.id,
                    q.project_id,
                    q.category,
                    q.kind,
                    q.answerable,
                    [c.chunk_id for c in ranked],
                    metrics,
                    ranked[0].rerank_score
                    if ranked and ranked[0].rerank_score is not None
                    else None,
                    round((t1 - t0) * 1000, 1),
                    round((t2 - t1) * 1000, 1),
                    [i for i in range(len(q.evidence)) if i not in covered],
                )
            )
    except ScopeViolation:
        raise
    except RetrievalError as exc:
        return RunResult(config, results, "UNAVAILABLE", str(exc))
    return RunResult(config, results)


@dataclass
class StageDecision:
    stage: str
    vary: str
    runs: list[RunResult]
    winner: RunConfig
    reason: str


def choose(runs: Sequence[RunResult], metric: str, min_improvement: float) -> tuple[RunResult, str]:
    """Simplest configuration unless a later (more complex) one beats it by min_improvement."""
    ok = [r for r in runs if r.status == "OK"]
    if not ok:
        raise RetrievalError("no configuration of this stage could run")
    best = ok[0]
    reasons = []
    for run in ok[1:]:
        gain = run.summary.get(metric, 0) - best.summary.get(metric, 0)
        if gain >= min_improvement:
            reasons.append(f"{run.config.name} beats {best.config.name} by {gain:+.3f} {metric}")
            best = run
        else:
            reasons.append(
                f"{run.config.name} not preferred ({gain:+.3f} {metric} < "
                f"{min_improvement} over {best.config.name})"
            )
    skipped = [f"{r.config.name}: {r.error}" for r in runs if r.status != "OK"]
    if skipped:
        reasons.append("unavailable: " + "; ".join(skipped))
    return best, "; ".join(reasons) or "single configuration"


def run_stages(
    retriever: Retriever,
    questions: Sequence[Question],
    config: EvaluationConfig,
    rerankers: dict[str, Reranker],
    baseline: RunConfig,
    progress: Callable[[str], None] | None = None,
) -> list[StageDecision]:
    """Staged experiments: each stage varies one dimension around the current winner."""
    say = progress or (lambda _m: None)
    current = baseline
    decisions = []
    for stage in config.stages:
        runs = []
        for value in stage.values:
            cfg = current.replace(**stage.fixed, **{stage.vary: value})
            say(f"[{stage.name}] {cfg.name}")
            runs.append(run_config(retriever, questions, cfg, rerankers, config.metrics_k))
        best, reason = choose(runs, config.primary_metric, config.min_improvement)
        current = best.config
        decisions.append(StageDecision(stage.name, stage.vary, runs, current, reason))
    return decisions


# ---------------------------------------------------------------------------
# Failure analysis and abstention calibration
# ---------------------------------------------------------------------------


def classify_misses(
    retriever: Retriever, questions: Sequence[Question], run: RunResult
) -> list[dict[str, Any]]:
    """Deterministic first-pass cause of each missed evidence item (top 10).

    * bad_chunking     - no chunk of this strategy contains the evidence phrase on its page
                         (the phrase was split across chunks or dropped);
    * metadata_filter  - matching chunks exist but the query scope excluded them;
    * ranking          - a matching chunk was retrieved but ranked below 10;
    * retrieval_miss   - matching chunks exist in scope but were not retrieved
                         (lexical or semantic mismatch; see the retrieval method).
    """
    by_id = {q.id: q for q in questions}
    out = []
    for result in run.answerable():
        q = by_id[result.question_id]
        query = process_query(q.question, q.project_id, retriever.settings.query)
        scope = retriever.scope_for(
            query, q.project_id, run.config.chunk_strategy, run.config.use_type_filters
        )
        for index in result.missed:
            ev = q.evidence[index]
            pool = [
                r
                for r in retriever.store.rows.values()
                if r["chunk_strategy"] == run.config.chunk_strategy
                and r["chunk_role"] == "RETRIEVAL"
                and index in matched_items(r, q)
            ]
            ranked = {cid: rank for rank, cid in enumerate(result.ranked_chunk_ids, 1)}
            if not pool:
                cause = "bad_chunking"
            elif not any(scope.admits(r) for r in pool):
                cause = "metadata_filter"
            elif any(r["chunk_id"] in ranked for r in pool):
                cause = "ranking"
            else:
                cause = "retrieval_miss"
            out.append(
                {
                    "question_id": q.id,
                    "project_id": q.project_id,
                    "category": q.category,
                    "question": q.question,
                    "evidence_document": ev.document_id,
                    "evidence_pages": ev.pages,
                    "evidence_phrase": ev.contains,
                    "cause": cause,
                    "evidence_mode": q.evidence_mode,
                }
            )
    return out


def abstention_calibration(run: RunResult) -> dict[str, Any]:
    """Best rerank scores of answerable vs no-answer questions and the best threshold."""
    answerable = [
        q.top_rerank_score for q in run.questions if q.answerable and q.top_rerank_score is not None
    ]
    unanswerable = [
        q.top_rerank_score
        for q in run.questions
        if not q.answerable and q.top_rerank_score is not None
    ]
    if not answerable or not unanswerable:
        return {"available": False, "reason": "needs reranker scores for both kinds of question"}
    candidates = sorted(set(answerable + unanswerable))
    best = max(
        candidates,
        key=lambda t: (sum(s >= t for s in answerable) + sum(s < t for s in unanswerable), -t),
    )
    correct = sum(s >= best for s in answerable) + sum(s < best for s in unanswerable)
    return {
        "available": True,
        "answerable_scores": {"min": min(answerable), "median": statistics.median(answerable)},
        "no_answer_scores": {"max": max(unanswerable), "median": statistics.median(unanswerable)},
        "best_threshold": best,
        "accuracy_at_threshold": round(correct / (len(answerable) + len(unanswerable)), 3),
        "note": "Separation, not a guarantee: the threshold is set in configs/retrieval/"
        "retrieval.yaml (min_rerank_score) only after review.",
    }


def log_to_mlflow(
    decisions: Sequence[StageDecision], params: dict[str, Any], experiment_name: str | None
) -> list[str]:
    """Log every experiment run (params + metrics) to MLflow; returns run ids."""
    import mlflow

    if experiment_name:
        mlflow.set_experiment(experiment_name)
    run_ids = []
    for decision in decisions:
        for run in decision.runs:
            with mlflow.start_run(run_name=f"{decision.stage}:{run.config.name}") as active:
                mlflow.log_params(
                    {**params, "stage": decision.stage, **run.config.__dict__, "status": run.status}
                )
                metrics = {k: v for k, v in run.summary.items() if isinstance(v, int | float)}
                if metrics:
                    mlflow.log_metrics(metrics)
                mlflow.set_tag("winner", str(run.config == decision.winner))
                run_ids.append(active.info.run_id)
    return run_ids
