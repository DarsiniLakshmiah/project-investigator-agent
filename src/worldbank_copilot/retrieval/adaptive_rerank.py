"""Phase 9E adaptive rerank policies P2-P6 (DIAGNOSTIC; none is a production default).

A policy decides, from first-stage signals only, whether the CrossEncoder reranks the k50
hybrid candidates. Policies see a `TriggerFeatures` object and nothing else: it carries no
question kind/category, evidence, answerability, expected answer, relevance labels or
counterfactual class. Evaluation metadata lives in `adaptive_eval` only.

* P2 top-1 disagreement   rerank iff lexical top-1 != dense top-1
* P3 head overlap (tau)   rerank iff |L10 & D10| / 10 < tau
* P4 head diversity (m)   rerank iff distinct document_id among fused top-5 >= m
* P5 lexical coverage(tau) rerank iff |C(q) & T(top-1 chunk_text)| / |C(q)| < tau
* P6 anchoring heuristic  rerank iff the query has no ISR / latest-ISR / document-type /
                          year / digit anchor (a hypothesis to test, not an assumption)

With no candidates nothing is reranked. `configs/retrieval/retrieval.yaml` `production`
is not read or changed here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

from worldbank_copilot.retrieval.query import ProcessedQuery
from worldbank_copilot.retrieval.rerank_policy import RerankDecision
from worldbank_copilot.retrieval.retriever import FirstStage, Retriever

POLICY_VERSION = "9e1"
_TERM = re.compile(r"[a-z0-9]+")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PolicySpec(_Model):
    id: Literal["P2", "P3", "P4", "P5", "P6"]
    family: str
    rule: str
    thresholds: tuple[float, ...] = ()


class AnchoringSpec(_Model):
    year_regex: str
    digit_regex: str


class FeatureSpec(_Model):
    overlap_depth: int
    diversity_depth: int
    min_term_length: int
    stopwords: tuple[str, ...]
    anchoring: AnchoringSpec
    no_candidates: Literal["never_rerank"]


class AdaptiveRerankConfig(_Model):
    version: int
    phase: Literal["9E"]
    status: str
    run_ids: dict[str, str]
    artifact_dir: str
    questions_file: str
    stack: dict[str, Any]
    expected: dict[str, Any]
    policies: tuple[PolicySpec, ...]
    features: FeatureSpec
    classification: dict[str, Any]
    random: dict[str, Any]
    oracle: dict[str, Any]
    pareto: dict[str, Any]
    drift: dict[str, Any]
    passes: dict[str, int]
    consistency: dict[str, float]
    live: dict[str, Any]
    scope: dict[str, bool]


def load_adaptive_config(config_dir: Path) -> AdaptiveRerankConfig:
    path = Path(config_dir) / "retrieval" / "adaptive_rerank.yaml"
    return AdaptiveRerankConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


# -- features (the ONLY policy input) ---------------------------------------------------------


@dataclass(frozen=True)
class TriggerFeatures:
    """First-stage signals. Deliberately has no label, metadata or evaluation field."""

    n_candidates: int
    top1_agree: bool
    overlap_at_10: float
    distinct_docs_top5: int
    lexical_coverage_top1: float
    anchored: bool


FEATURE_FIELDS = tuple(f.name for f in fields(TriggerFeatures))


def content_terms(text: str, spec: FeatureSpec) -> set[str]:
    stop = set(spec.stopwords)
    return {
        t
        for t in _TERM.findall((text or "").lower())
        if len(t) >= spec.min_term_length and t not in stop
    }


def is_anchored(query: ProcessedQuery, spec: FeatureSpec) -> bool:
    return bool(
        query.isr_sequences
        or query.latest_isr
        or query.document_type_hints
        or re.search(spec.anchoring.year_regex, query.text)
        or re.search(spec.anchoring.digit_regex, query.text)
    )


def compute_features(
    query: ProcessedQuery,
    lexical: Sequence[str],
    dense: Sequence[str],
    fused: Sequence[str],
    rows: dict[str, dict[str, Any]],
    spec: FeatureSpec,
) -> TriggerFeatures:
    """Features from the lexical / dense / fused (deduplicated) chunk-id lists."""
    depth = spec.overlap_depth
    terms = content_terms(query.text, spec)
    top_terms = content_terms(rows[fused[0]]["chunk_text"], spec) if fused else set()
    return TriggerFeatures(
        n_candidates=len(fused),
        top1_agree=bool(lexical and dense and lexical[0] == dense[0]),
        overlap_at_10=len(set(lexical[:depth]) & set(dense[:depth])) / depth,
        distinct_docs_top5=len({rows[c]["document_id"] for c in fused[: spec.diversity_depth]}),
        lexical_coverage_top1=len(terms & top_terms) / len(terms) if terms else 0.0,
        anchored=is_anchored(query, spec),
    )


# -- policies -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class AdaptivePolicy:
    """One preregistered operating point (policy id + threshold)."""

    policy_id: Literal["P2", "P3", "P4", "P5", "P6"]
    threshold: float | None = None

    @property
    def name(self) -> str:
        return self.policy_id if self.threshold is None else f"{self.policy_id}({self.threshold:g})"

    version = POLICY_VERSION

    def trigger(self, f: TriggerFeatures) -> bool:
        """The whole decision. Reads TriggerFeatures only."""
        if not isinstance(f, TriggerFeatures):
            raise TypeError("policies accept TriggerFeatures only")
        if f.n_candidates == 0:
            return False
        if self.policy_id == "P2":
            return not f.top1_agree
        if self.policy_id == "P3":
            return f.overlap_at_10 < self.threshold
        if self.policy_id == "P4":
            return f.distinct_docs_top5 >= self.threshold
        if self.policy_id == "P5":
            return f.lexical_coverage_top1 < self.threshold
        return not f.anchored  # P6

    def decision(self, f: TriggerFeatures) -> RerankDecision:
        rerank = self.trigger(f)
        return RerankDecision(
            policy=self.name,
            policy_version=self.version,
            rerank=rerank,
            reason=f"{self.name}: {'rerank' if rerank else 'no rerank'}",
            features={k: getattr(f, k) for k in FEATURE_FIELDS},
        )


def preregistered_policies(config: AdaptiveRerankConfig) -> list[AdaptivePolicy]:
    """Exactly the 13 frozen operating points, in report order."""
    out: list[AdaptivePolicy] = []
    for spec in config.policies:
        if spec.thresholds:
            out += [AdaptivePolicy(spec.id, t) for t in spec.thresholds]
        else:
            out.append(AdaptivePolicy(spec.id))
    return out


# -- live execution (07d): RerankPolicy over a FirstStage, no extra index query -----------------


class RecordingDense:
    """Wraps the dense searcher and keeps the most recent hits, so the live policy can read
    the dense list the first stage already fetched (no second AI Search query)."""

    def __init__(self, inner: Any):
        self.inner = inner
        self.last: list[tuple[str, float]] | None = None
        self.calls = 0

    def search(self, vector: Any, filters: dict[str, Any], k: int) -> list[tuple[str, float]]:
        self.last = list(self.inner.search(vector, filters, k))
        self.calls += 1
        return self.last


class LivePolicy:
    """`RerankPolicy` adapter: features from the FirstStage (+ recorded dense list + a local
    BM25 recomputation, which is deterministic and in-process), then the frozen trigger."""

    def __init__(
        self,
        policy: AdaptivePolicy,
        retriever: Retriever,
        recorder: RecordingDense,
        spec: FeatureSpec,
    ):
        self.policy, self.retriever, self.recorder, self.spec = policy, retriever, recorder, spec
        self.name, self.version = policy.name, policy.version

    def features(self, stage: FirstStage) -> TriggerFeatures:
        if self.recorder.last is None:
            raise RuntimeError("no dense hits recorded for this first stage")
        lexical = self.retriever.candidates(stage.query, stage.scope, "lexical", stage.candidate_k)
        return compute_features(
            stage.query,
            [c.chunk_id for c in lexical],
            [chunk_id for chunk_id, _ in self.recorder.last],
            [c.chunk_id for c in stage.candidates],
            self.retriever.store.rows,
            self.spec,
        )

    def decide(self, stage: FirstStage) -> RerankDecision:
        return self.policy.decision(self.features(stage))
