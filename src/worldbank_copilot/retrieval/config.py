"""Retrieval configuration (configs/retrieval/*.yaml), validated on load."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from worldbank_copilot.common.exceptions import ConfigurationError

FOLDER = Path("retrieval")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StrategyConfig(_Model):
    kind: Literal["fixed", "structure", "parent_child"]
    max_chars: int | None = None
    overlap_chars: int = 0
    table_max_chars: int | None = None
    child_max_chars: int | None = None
    parent_max_chars: int | None = None

    @model_validator(mode="after")
    def _sizes(self) -> StrategyConfig:
        need = {
            "fixed": ("max_chars",),
            "structure": ("max_chars", "table_max_chars"),
            "parent_child": ("child_max_chars", "table_max_chars", "parent_max_chars"),
        }[self.kind]
        missing = [n for n in need if getattr(self, n) is None]
        if missing:
            raise ValueError(f"{self.kind} strategy needs {missing}")
        if self.max_chars is not None and self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")
        return self

    @property
    def retrieval_max_chars(self) -> int:
        """Largest retrieval chunk this strategy may produce (text or table)."""
        sizes = [self.max_chars, self.table_max_chars, self.child_max_chars]
        return max(s for s in sizes if s is not None)


class ChunkingConfig(_Model):
    strategies: dict[str, StrategyConfig]
    min_chars: int = Field(ge=0)
    context_header: bool = True

    def version(self, strategy: str) -> str:
        """Short hash of a strategy's settings (part of every chunk id)."""
        payload = {
            "strategy": self.strategies[strategy].model_dump(),
            "min_chars": self.min_chars,
            "context_header": self.context_header,
        }
        text = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:12]


class EmbeddingConfig(_Model):
    provider: Literal["databricks_serving"]
    endpoint: str
    expected_dimension: int = Field(gt=0)
    batch_size: int = Field(gt=0, le=2048)
    timeout_seconds: int = Field(gt=0)
    max_retries: int = Field(ge=0)


class VectorSearchConfig(_Model):
    endpoint: str
    index_table: str
    index_name: str
    pipeline_type: Literal["TRIGGERED", "CONTINUOUS"] = "TRIGGERED"
    sync_timeout_seconds: int = Field(gt=0)
    query_timeout_seconds: int = Field(gt=0)


class LexicalConfig(_Model):
    k1: float = Field(gt=0)
    b: float = Field(ge=0, le=1)


class HybridConfig(_Model):
    rrf_k: int = Field(gt=0)


class RerankerConfig(_Model):
    model: str
    max_length: int = Field(gt=0)
    batch_size: int = Field(gt=0)


class ProductionConfig(_Model):
    chunk_strategy: str | None = None
    retrieval: Literal["lexical", "dense", "hybrid"] | None = None
    reranker: Literal["none", "cross_encoder"] | None = None
    candidate_k: int | None = None
    final_k: int = Field(gt=0)


class RetrievalConfig(_Model):
    vector_search: VectorSearchConfig
    lexical: LexicalConfig
    hybrid: HybridConfig
    reranker: RerankerConfig
    required_filters: list[str]
    min_rerank_score: float | None = None
    production: ProductionConfig

    @model_validator(mode="after")
    def _scope(self) -> RetrievalConfig:
        if "project_id" not in self.required_filters:
            raise ValueError("project_id must be a required retrieval filter")
        return self


class QueryConfig(_Model):
    acronyms: dict[str, str]
    document_type_hints: dict[str, list[str]]


class StageConfig(_Model):
    name: str
    vary: Literal["chunk_strategy", "retrieval", "reranker", "candidate_k", "use_type_filters"]
    values: list[Any]
    fixed: dict[str, Any] = Field(default_factory=dict)


class MlflowConfig(_Model):
    enabled: bool = True
    experiment_name: str | None = None


class EvaluationConfig(_Model):
    questions_file: str
    metrics_k: list[int]
    primary_metric: str
    stages: list[StageConfig]
    min_improvement: float = Field(ge=0)
    mlflow: MlflowConfig = MlflowConfig()


class RetrievalSettings(_Model):
    chunking: ChunkingConfig
    embeddings: EmbeddingConfig
    retrieval: RetrievalConfig
    query: QueryConfig
    evaluation: EvaluationConfig


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"{path} not found")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_retrieval_settings(config_dir: Path) -> RetrievalSettings:
    folder = Path(config_dir) / FOLDER
    try:
        return RetrievalSettings(
            chunking=ChunkingConfig.model_validate(_read(folder / "chunking.yaml")),
            embeddings=EmbeddingConfig.model_validate(_read(folder / "embeddings.yaml")),
            retrieval=RetrievalConfig.model_validate(_read(folder / "retrieval.yaml")),
            query=QueryConfig.model_validate(_read(folder / "query.yaml")),
            evaluation=EvaluationConfig.model_validate(_read(folder / "evaluation.yaml")),
        )
    except ValueError as exc:
        raise ConfigurationError(f"invalid retrieval configuration: {exc}") from exc
