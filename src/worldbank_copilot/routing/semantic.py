"""Bounded semantic routing fallback (Phase 9D, Candidate B: nearest-neighbour similarity).

Contract
--------
* Invoked ONLY by the harness, ONLY when the deterministic rules returned
  SEMANTIC_CLASSIFICATION_REQUIRED, and ONLY after project scope, authorisation and every
  pre-execution refusal have been decided (``SemanticQueryContext`` is built from a
  RESOLVED scope).
* Input: the normalised question text. Output: ``SemanticDecision`` - one of the
  non-refusal intents (route derived from configs/routing/requirements.yaml), a
  confidence, the nearest DEVELOPMENT examples, or ABSTAIN. No project, no tool, no
  retrieval, no answer, no free text beyond a short reason.
* Examples come from the frozen DEVELOPMENT split only; test cases are rejected.
* Abstention is a valid outcome (the harness turns it into CLARIFY).

Embedders
---------
* ``LexicalEmbedder`` - deterministic hashed word + character n-grams. Runs anywhere; it
  is a real (lexical) candidate and the local plumbing stand-in. It says nothing about
  the quality of a learned embedding model.
* ``ProviderEmbedder`` - wraps the validated Phase 8 embedding provider (Qwen on Databricks
  Model Serving) with its OWN cache namespace ("routing-question"); it never reads or writes
  the document-chunk embedding cache or the AI Search index.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from worldbank_copilot.routing.models import (
    REFUSAL_INTENTS,
    Intent,
    SemanticDecision,
    SemanticNeighbour,
)

SEMANTIC_INTENTS = tuple(i for i in Intent if i not in REFUSAL_INTENTS)
CACHE_NAMESPACE = "routing-question"
Vector = dict[int, float] | list[float]


@dataclass(frozen=True)
class SemanticQueryContext:
    """What the fallback may see: scope already validated by the deterministic harness."""

    request_id: str
    question: str  # normalised
    project_id: str  # RESOLVED scope (audit only; never used to choose anything)
    temporal_kind: str
    rule_hits: tuple[str, ...]


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> list[Vector]: ...


def cosine(a: Vector, b: Vector) -> float:
    if isinstance(a, dict) and isinstance(b, dict):
        small, large = (a, b) if len(a) <= len(b) else (b, a)
        return sum(v * large.get(k, 0.0) for k, v in small.items())
    return sum(x * y for x, y in zip(a, b, strict=True))  # vectors are L2-normalised


_STOP = frozenset(
    "the a an of in on to for by with and or is are was were be been has have had do does did "
    "what which how why when who this that these those it its as at from according".split()
)


def _hash(feature: str, dims: int) -> int:
    return int.from_bytes(hashlib.sha1(feature.encode("utf-8")).digest()[:4], "big") % dims


class LexicalEmbedder:
    """Deterministic sparse embedding: content words + character 3/4-grams, L2-normalised."""

    def __init__(self, dims: int = 2**18, char_ngrams: tuple[int, ...] = (3, 4)):
        self.dims, self.char_ngrams = dims, char_ngrams
        self.name = f"lexical-hash-v1(dims={dims},char={char_ngrams})"

    def _one(self, text: str) -> dict[int, float]:
        words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP]
        counts: dict[int, float] = defaultdict(float)
        for w in words:
            counts[_hash(f"w:{w}", self.dims)] += 1.0
            padded = f"#{w}#"
            for n in self.char_ngrams:
                for i in range(max(len(padded) - n + 1, 0)):
                    counts[_hash(f"c{n}:{padded[i : i + n]}", self.dims)] += 0.5
        norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
        return {k: v / norm for k, v in counts.items()}

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        return [self._one(t) for t in texts]


@dataclass
class ProviderEmbedder:
    """Phase 8 embedding provider (e.g. Qwen) with a separate routing-question cache."""

    provider: Any  # retrieval.embeddings.EmbeddingProvider
    cache_path: Path | None = None  # optional JSON artefact; never the chunk-embedding table
    cache: dict[str, list[float]] = field(default_factory=dict)
    call_latency_ms: list[float] = field(default_factory=list)  # endpoint calls only

    def __post_init__(self) -> None:
        self.name = f"provider:{self.provider.model}"
        if self.cache_path and Path(self.cache_path).exists():
            self.cache.update(json.loads(Path(self.cache_path).read_text(encoding="utf-8")))

    def key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{CACHE_NAMESPACE}/{self.provider.model}/{digest}"

    def embed(self, texts: Sequence[str]) -> list[Vector]:
        missing = [t for t in dict.fromkeys(texts) if self.key(t) not in self.cache]
        if missing:
            started = time.perf_counter()
            vectors = self.provider.embed(missing)
            self.call_latency_ms.append(round((time.perf_counter() - started) * 1000, 1))
            for text, vector in zip(missing, vectors, strict=True):
                norm = math.sqrt(sum(x * x for x in vector)) or 1.0
                self.cache[self.key(text)] = [x / norm for x in vector]
            if self.cache_path:
                Path(self.cache_path).write_text(json.dumps(self.cache), encoding="utf-8")
        return [self.cache[self.key(t)] for t in texts]


@dataclass(frozen=True)
class Example:
    example_id: str
    family: str
    question: str
    intent: Intent


class ExampleStore:
    """Labelled examples from the frozen DEVELOPMENT split only."""

    def __init__(self, examples: Sequence[Example]):
        self.examples = tuple(examples)

    @classmethod
    def from_dataset(cls, dataset: Any) -> ExampleStore:
        out = []
        for case in dataset.cases:
            if case.split != "dev":
                continue  # frozen TEST cases are never examples
            e = case.expected
            if e.route in ("STRUCTURED", "DOCUMENT", "INVESTIGATION") and len(e.intents) == 1:
                out.append(Example(case.case_id, case.family, case.question, e.intents[0]))
        return cls(out)

    def assert_development_only(self, dataset: Any) -> None:
        split = {c.case_id: c.split for c in dataset.cases}
        leaked = [e.example_id for e in self.examples if split.get(e.example_id) != "dev"]
        if leaked:
            raise ValueError(f"non-development examples in the semantic store: {leaked}")


@dataclass(frozen=True)
class KnnConfig:
    k: int = 3
    weighting: str = "similarity"  # similarity | uniform
    min_similarity: float = 0.0  # abstain if the nearest example is less similar
    min_vote_share: float = 0.0  # abstain if the winning intent's share is lower

    @property
    def name(self) -> str:
        return (
            f"k={self.k},w={self.weighting},sim>={self.min_similarity:.2f},"
            f"share>={self.min_vote_share:.2f}"
        )


class KnnSemanticClassifier:
    """Nearest-neighbour intent classifier with abstention (Candidate B)."""

    version = "9D.B1"

    def __init__(
        self,
        embedder: Embedder,
        store: ExampleStore,
        config: KnnConfig,
        route_of: dict[Intent, str],
    ):
        self.embedder, self.store, self.config, self.route_of = embedder, store, config, route_of
        self.name = f"knn[{embedder.name}]"
        self._vectors = dict(
            zip(
                (e.example_id for e in store.examples),
                embedder.embed([e.question for e in store.examples]),
                strict=True,
            )
        )

    def scored(self, question: str, exclude_families: frozenset[str] = frozenset()):
        (query,) = self.embedder.embed([question])
        rows = [
            (cosine(query, self._vectors[e.example_id]), e)
            for e in self.store.examples
            if e.family not in exclude_families
        ]
        return sorted(rows, key=lambda r: (-r[0], r[1].example_id))

    def classify(
        self, context: SemanticQueryContext, exclude_families: frozenset[str] = frozenset()
    ) -> SemanticDecision:
        started = time.perf_counter()
        rows = self.scored(context.question, exclude_families)
        top = rows[: self.config.k]
        neighbours = tuple(
            SemanticNeighbour(
                example_id=e.example_id, intent=e.intent.value, similarity=round(s, 4)
            )
            for s, e in top
        )

        def decision(**kw) -> SemanticDecision:
            return SemanticDecision(
                classifier=self.name,
                version=self.version,
                neighbours=neighbours,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                **kw,
            )

        if not top:
            return decision(abstain=True, reason="no development examples available")
        votes: dict[Intent, float] = defaultdict(float)
        for s, e in top:
            votes[e.intent] += max(s, 0.0) if self.config.weighting == "similarity" else 1.0
        total = sum(votes.values()) or 1.0
        intent, weight = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0].value))[0]
        share = weight / total
        confidence = round(share * max(top[0][0], 0.0), 4)
        if top[0][0] < self.config.min_similarity:
            return decision(
                abstain=True,
                confidence=confidence,
                reason=f"nearest example similarity {top[0][0]:.3f} below "
                f"{self.config.min_similarity:.2f}",
            )
        if share < self.config.min_vote_share:
            return decision(
                abstain=True,
                confidence=confidence,
                reason=f"vote share {share:.2f} below {self.config.min_vote_share:.2f}",
            )
        return decision(
            abstain=False,
            intent=intent,
            route=self.route_of[intent],
            confidence=confidence,
            reason=f"{len(top)} nearest development examples; share {share:.2f}",
        )


def route_map(requirements: Any) -> dict[Intent, str]:
    """Intent -> route from the reviewed requirements configuration."""
    return {intent: req.route for intent, req in requirements.intents.items()}
