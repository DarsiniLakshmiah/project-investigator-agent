"""BM25 lexical retrieval and rank fusion (pure Python, deterministic).

The lexical index is built per request scope from governed chunk rows that already
passed the scope filter, so out-of-scope text is never scored. The corpus of one
project and strategy is a few thousand chunks (a few MB), small enough for an in-memory
index on the driver; this is a bounded, documented choice (Databricks Vector Search
serves the dense side).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

_TOKEN = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to "
    "was were what which with how why when did does do there their been any".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercased alphanumeric tokens; keeps ids like p130544, ibrd-86010, 2026-08-16."""
    tokens = []
    for token in _TOKEN.findall(text.lower()):
        if token in STOPWORDS:
            continue
        tokens.append(token)
        if any(c in token for c in "./-"):  # also index the parts: "ibrd-86010" -> ibrd, 86010
            tokens.extend(p for p in re.split(r"[./-]", token) if p and p not in STOPWORDS)
    return tokens


class BM25:
    def __init__(self, ids: Sequence[str], texts: Sequence[str], k1: float = 1.5, b: float = 0.75):
        if len(ids) != len(texts):
            raise ValueError("ids and texts differ in length")
        self.ids = list(ids)
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(doc.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        terms = [t for t in dict.fromkeys(tokenize(query)) if t in self.idf]
        if not terms or k <= 0:
            return []
        scores = []
        for index, doc in enumerate(self.docs):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / (self.avg or 1))
            for term in terms:
                tf = doc.get(term)
                if tf:
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
            if score > 0:
                scores.append((score, index))
        scores.sort(key=lambda s: (-s[0], self.ids[s[1]]))  # deterministic ties
        return [(self.ids[i], s) for s, i in scores[:k]]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60, limit: int | None = None
) -> list[tuple[str, float]]:
    """RRF: score(d) = sum over rankings of 1 / (k + rank). Ties broken by id."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return fused[:limit] if limit is not None else fused
