"""Rerankers (Phase 8): none, or a sentence-transformers CrossEncoder.

The CrossEncoder runs on the driver CPU (a MiniLM-sized model scores ~50 pairs in well
under a second). It is loaded lazily; if the package or model is unavailable the
reranker reports that instead of silently falling back.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from worldbank_copilot.retrieval.config import RerankerConfig
from worldbank_copilot.retrieval.models import RetrievalError


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class NoReranker:
    name = "none"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        return [0.0] * len(texts)


class CrossEncoderReranker:
    name = "cross_encoder"

    def __init__(self, config: RerankerConfig, model: Any | None = None):
        self.config = config
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RetrievalError(
                    "cross_encoder reranker needs sentence-transformers "
                    "(pip install sentence-transformers)"
                ) from exc
            self._model = CrossEncoder(self.config.model, max_length=self.config.max_length)
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        scores = self.model.predict(pairs, batch_size=self.config.batch_size)
        return [float(s) for s in scores]


def rerank_order(ids: Sequence[str], scores: Sequence[float]) -> list[tuple[str, float, int]]:
    """(id, score, previous_rank) sorted by score desc; ties keep the retrieval order."""
    ranked = [(i, s, r) for r, (i, s) in enumerate(zip(ids, scores, strict=True), 1)]
    return sorted(ranked, key=lambda x: (-x[1], x[2]))


def reranker_for(name: str, config: RerankerConfig) -> Reranker:
    if name == "none":
        return NoReranker()
    if name == "cross_encoder":
        return CrossEncoderReranker(config)
    raise RetrievalError(f"unknown reranker {name!r}")
