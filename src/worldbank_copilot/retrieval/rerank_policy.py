"""Rerank policies: decide, from first-stage evidence, whether to run the reranker (Phase 9).

Phase 9A implements only the two MEASURED bounds of Phase 8:

* ``NeverRerank``  - fixed + hybrid, no reranker (measured baseline at k10);
* ``AlwaysRerank`` - fixed + hybrid + CrossEncoder (measured quality ceiling at k50).

Adaptive candidate policies (P2-P6) live in ``retrieval/adaptive_rerank.py`` and are
evaluated by the 9E diagnostic (``retrieval/adaptive_eval.py``). No policy is a production
default: ``retrieval.yaml`` ``production`` stays null until an adaptive policy has been
evaluated and explicitly accepted.
The policy decides; deterministic harness code (``DocumentSearch``) executes.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from worldbank_copilot.retrieval.retriever import FirstStage


class RerankDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: str
    policy_version: str
    rerank: bool
    reason: str
    features: dict[str, float | int | str | bool] = Field(default_factory=dict)


class RerankPolicy(Protocol):
    name: str
    version: str

    def decide(self, stage: FirstStage) -> RerankDecision: ...


def _basic_features(stage: FirstStage) -> dict[str, float | int | str | bool]:
    return {"candidates": len(stage.candidates), "first_stage_ms": stage.first_stage_ms}


class NeverRerank:
    name, version = "never", "1"

    def decide(self, stage: FirstStage) -> RerankDecision:
        return RerankDecision(
            policy=self.name,
            policy_version=self.version,
            rerank=False,
            reason="NEVER bound: no query is reranked",
            features=_basic_features(stage),
        )


class AlwaysRerank:
    name, version = "always", "1"

    def decide(self, stage: FirstStage) -> RerankDecision:
        return RerankDecision(
            policy=self.name,
            policy_version=self.version,
            rerank=bool(stage.candidates),
            reason="ALWAYS bound: every query with candidates is reranked",
            features=_basic_features(stage),
        )
