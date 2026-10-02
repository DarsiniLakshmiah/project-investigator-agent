"""Phase 10 retrieval contract: a stable, profile-owned document-evidence interface (Phase 9F).

Agents ask for evidence; the retrieval layer owns how it is retrieved. Phase 10 code
receives a ``DocumentRetrieval`` (built by platform wiring) and calls ``retrieve``. It never
touches ``Retriever``, ``DenseSearcher``, a Vector Search client or the CrossEncoder, and it
cannot choose candidate_k, final_k, RRF k, the reranker, the embedding model, the chunk
strategy or any threshold: ``RetrievalRequest`` has no such field (extra fields are refused).

The only profile is the Phase 10 QUALITY BASELINE ``phase8_quality_baseline``, defined once
in ``configs/phase9_closure.yaml``. It is NOT the production configuration:
``configs/retrieval/retrieval.yaml`` ``production`` stays null.

Execution reuses the allowlisted ``search_project_documents`` tool through the
``ToolExecutor``, so project scope, authorisation and error mapping are the same code path.

Truths the result keeps explicit:

* ``temporal_scope`` and ``document_type_hints`` are RECORDED, NOT APPLIED as filters
  (Phase 8 decision); ``filters_applied`` shows the scope actually used;
* NO_EVIDENCE means "relevant evidence was not found in the retrieved candidates", never
  "the documents do not contain relevant evidence";
* every evidence item is DOCUMENTED_FINDING (source provenance only; relevance not verified)
  and carries citation metadata.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from worldbank_copilot.common.exceptions import ConfigurationError
from worldbank_copilot.common.project_registry import validate_project_id_format
from worldbank_copilot.retrieval.config import RetrievalSettings
from worldbank_copilot.retrieval.rerank import Reranker
from worldbank_copilot.retrieval.rerank_policy import AlwaysRerank, RerankDecision
from worldbank_copilot.retrieval.retriever import Retriever
from worldbank_copilot.routing.models import Intent, RetrievalSpec, TemporalScope
from worldbank_copilot.tools.base import ToolContext
from worldbank_copilot.tools.documents import DocumentEvidence, DocumentSearch, DocumentSearchConfig
from worldbank_copilot.tools.executor import ToolExecutor
from worldbank_copilot.tools.models import ProvenanceClass, ToolStatus
from worldbank_copilot.tools.registry import default_executor

MANIFEST = "phase9_closure.yaml"
DOCUMENT_TOOL = "search_project_documents"
NO_EVIDENCE_MEANING = (
    "Relevant evidence was not found in the retrieved candidates. This does not establish "
    "that the documents do not contain relevant evidence."
)
HINTS_NOT_APPLIED = (
    "temporal_scope and document_type_hints are recorded, not applied as retrieval filters "
    "(Phase 8 decision); filters_applied shows the scope actually used."
)


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RetrievalProfile(_Frozen):
    """The Phase 10 quality baseline (read from the closure manifest; immutable)."""

    name: Literal["phase8_quality_baseline"]
    version: str
    role: str
    chunk_strategy: str
    chunk_strategy_version: str
    method: Literal["hybrid"]
    bm25_k1: float
    bm25_b: float
    min_rerank_score: Literal[None]
    query_config_sha256: str
    vector_search_endpoint: str
    vector_search_index: str
    rrf_k: int
    candidate_k: int
    final_k: int
    rerank_policy: Literal["always"]
    reranker_model: str
    reranker_max_length: int
    reranker_batch_size: int
    embedding_endpoint: str
    embedding_dimension: int
    use_type_filters: Literal[False]
    temporal_filters_applied: Literal[False]
    production: Literal[False]

    @property
    def profile_id(self) -> str:
        return f"{self.name}@{self.version}"


def profile_mismatches(profile: RetrievalProfile, rs: RetrievalSettings) -> list[str]:
    """Differences between the profile and the governed retrieval configuration."""
    checks = [
        ("chunk_strategy", profile.chunk_strategy in rs.chunking.strategies, True),
        (
            "chunk_strategy_version",
            rs.chunking.version(profile.chunk_strategy)
            if profile.chunk_strategy in rs.chunking.strategies
            else None,
            profile.chunk_strategy_version,
        ),
        ("bm25_k1", rs.retrieval.lexical.k1, profile.bm25_k1),
        ("bm25_b", rs.retrieval.lexical.b, profile.bm25_b),
        ("min_rerank_score", rs.retrieval.min_rerank_score, profile.min_rerank_score),
        (
            "query_config_sha256",
            hashlib.sha256(json.dumps(rs.query.model_dump(), sort_keys=True).encode()).hexdigest(),
            profile.query_config_sha256,
        ),
        (
            "vector_search_endpoint",
            rs.retrieval.vector_search.endpoint,
            profile.vector_search_endpoint,
        ),
        ("vector_search_index", rs.retrieval.vector_search.index_name, profile.vector_search_index),
        ("rrf_k", rs.retrieval.hybrid.rrf_k, profile.rrf_k),
        ("reranker_model", rs.retrieval.reranker.model, profile.reranker_model),
        ("reranker_max_length", rs.retrieval.reranker.max_length, profile.reranker_max_length),
        ("reranker_batch_size", rs.retrieval.reranker.batch_size, profile.reranker_batch_size),
        ("embedding_endpoint", rs.embeddings.endpoint, profile.embedding_endpoint),
        ("embedding_dimension", rs.embeddings.expected_dimension, profile.embedding_dimension),
    ]
    production = rs.retrieval.production.model_dump()
    checks.append(
        (
            "production_selection",
            all(v is None for k, v in production.items() if k != "final_k"),
            True,
        )
    )
    checks.append(("production_final_k", production["final_k"], profile.final_k))
    return [f"{name}: differs from baseline" for name, got, want in checks if got != want]


def load_quality_baseline(config_dir: Path, rs: RetrievalSettings) -> RetrievalProfile:
    """The Phase 10 quality baseline; fails closed if it disagrees with the configuration."""
    path = Path(config_dir) / MANIFEST
    if not path.is_file():
        raise ConfigurationError(f"{path} not found")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        profile = RetrievalProfile.model_validate(data["retrieval"]["phase10_quality_baseline"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid Phase 10 retrieval profile: {exc}") from exc
    if mismatches := profile_mismatches(profile, rs):
        raise ConfigurationError(f"Phase 10 retrieval profile mismatch: {mismatches}")
    return profile


def _search_mismatches(search: DocumentSearch, profile: RetrievalProfile) -> list[str]:
    """One effective invariant, used at construction and before every retrieval.

    Config identities are checked without loading models or contacting services. Provider
    model/dimension and reranker config are required; optional adapter/transport identities
    and cached BM25 parameters are checked where exposed. Protocols do not expose remote
    weights, index contents, corpus chunk versions, or arbitrary injected implementation
    behavior. Those remain platform-wiring responsibilities, not locally attestable facts.
    """
    out = profile_mismatches(profile, search.retriever.settings)

    def check(name: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            out.append(f"{name}: differs from baseline")

    for name in ("chunk_strategy", "method", "candidate_k", "final_k"):
        check(name, getattr(search.config, name, None), getattr(profile, name))
    check("rerank policy", type(search.policy), AlwaysRerank)
    check("profile_id", search.profile_id, profile.profile_id)
    check("reranker_name", getattr(search.cross_encoder, "name", None), "cross_encoder")
    config = getattr(search.cross_encoder, "config", None)
    for name in ("model", "max_length", "batch_size"):
        check(f"reranker_{name}", getattr(config, name, None), getattr(profile, f"reranker_{name}"))
    # Never access CrossEncoderReranker.model: that property loads the real model lazily.
    model = getattr(search.cross_encoder, "_model", None)
    if model is not None and hasattr(model, "max_length"):
        check("loaded_reranker_max_length", model.max_length, profile.reranker_max_length)

    provider = search.retriever.embeddings
    check("embedding_provider_model", getattr(provider, "model", None), profile.embedding_endpoint)
    check(
        "embedding_provider_dimension",
        getattr(provider, "dimension", None),
        profile.embedding_dimension,
    )
    config = getattr(provider, "config", None)
    if config is not None:
        check(
            "embedding_provider_endpoint",
            getattr(config, "endpoint", None),
            profile.embedding_endpoint,
        )
        check(
            "embedding_provider_config_dimension",
            getattr(config, "expected_dimension", None),
            profile.embedding_dimension,
        )
    transport = getattr(provider, "transport", None)
    if transport is not None and hasattr(transport, "endpoint"):
        check("embedding_transport_endpoint", transport.endpoint, profile.embedding_endpoint)

    dense = search.retriever.dense
    if dense is None:
        out.append("dense_search: missing")
    config = getattr(dense, "config", None)
    if config is not None:
        check("dense_endpoint", getattr(config, "endpoint", None), profile.vector_search_endpoint)
        check(
            "dense_config_index", getattr(config, "index_name", None), profile.vector_search_index
        )
    if hasattr(dense, "index_name"):
        check("dense_index", dense.index_name.rsplit(".", 1)[-1], profile.vector_search_index)
    if hasattr(dense, "dimension"):
        check("dense_dimension", dense.dimension, profile.embedding_dimension)
    for bm25 in search.retriever.store._lexical.values():
        check("cached_bm25_k1", bm25.k1, profile.bm25_k1)
        check("cached_bm25_b", bm25.b, profile.bm25_b)
    return sorted(set(out))


def build_document_search(
    profile: RetrievalProfile, retriever: Retriever, cross_encoder: Reranker
) -> DocumentSearch:
    """Platform wiring only: the profile's document search (AlwaysRerank, profile depths)."""
    search = DocumentSearch(
        retriever=retriever,
        config=DocumentSearchConfig(
            chunk_strategy=profile.chunk_strategy,
            method=profile.method,
            candidate_k=profile.candidate_k,
            final_k=profile.final_k,
        ),
        policy=AlwaysRerank(),
        cross_encoder=cross_encoder,
        profile_id=profile.profile_id,
    )
    if mismatches := _search_mismatches(search, profile):
        raise ConfigurationError(f"document search does not match the profile: {mismatches}")
    return search


# -- request / result ------------------------------------------------------------------------


class RetrievalRequest(_Frozen):
    """What a Phase 10 caller may ask. No retrieval parameter is caller-tunable."""

    project_id: str
    query: str = Field(min_length=1, max_length=1000)
    intent: Intent | None = None
    temporal_scope: TemporalScope | None = None  # recorded, not applied
    document_type_hints: tuple[str, ...] = ()  # recorded, not applied
    require_citations: Literal[True] = True

    @field_validator("project_id")
    @classmethod
    def _pid(cls, value: str) -> str:
        return validate_project_id_format(value)

    @classmethod
    def from_spec(
        cls,
        project_id: str,
        spec: RetrievalSpec,
        *,
        intent: Intent | None = None,
        temporal_scope: TemporalScope | None = None,
    ) -> RetrievalRequest:
        """A request for a routing / investigation-plan retrieval requirement."""
        return cls(
            project_id=project_id,
            query=spec.query,
            intent=intent,
            temporal_scope=temporal_scope,
            document_type_hints=spec.document_type_hints,
        )


class RetrievalStatus(StrEnum):
    OK = "OK"
    NO_EVIDENCE = "NO_EVIDENCE"
    RETRIEVAL_ERROR = "RETRIEVAL_ERROR"
    SCOPE_REFUSED = "SCOPE_REFUSED"


class RecordedHints(_Frozen):
    temporal_scope: TemporalScope | None
    document_type_hints: tuple[str, ...]
    applied: Literal[False] = False


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: RetrievalStatus
    project_id: str
    evidence: list[DocumentEvidence] = Field(default_factory=list)
    query_used: str
    filters_applied: dict[str, Any] | None = None  # the scope actually used (None if unknown)
    recorded_hints: RecordedHints
    candidate_count: int | None = None  # first-stage candidates (None if not reported)
    top_k: int
    profile: str
    rerank_decision: RerankDecision | None = None
    warnings: list[str] = Field(default_factory=list)
    latency_ms: float | None = None
    first_stage_ms: float | None = None
    error: str | None = None

    @model_validator(mode="after")
    def _contract(self) -> RetrievalResult:
        if (self.status == RetrievalStatus.OK) != bool(self.evidence):
            raise ValueError(f"status {self.status} with {len(self.evidence)} evidence items")
        for item in self.evidence:
            e = item.evidence
            if e.project_id != self.project_id or e.citation.project_id != self.project_id:
                raise ValueError(f"evidence {e.chunk_id} is outside project {self.project_id}")
            if item.provenance_class != ProvenanceClass.DOCUMENTED_FINDING:
                raise ValueError(f"evidence {e.chunk_id} is {item.provenance_class}")
            if item.relevance_verified:
                raise ValueError("retrieval never verifies relevance (Phase 10)")
            if not (e.citation.document_id and e.citation.pages and e.citation.source_file):
                raise ValueError(f"evidence {e.chunk_id} lacks citation metadata")
        if len(self.evidence) > self.top_k:
            raise ValueError("more evidence than the profile's top_k")
        return self


# -- the interface Phase 10 receives -----------------------------------------------------------


@dataclass(frozen=True)
class DocumentRetrieval:
    """The document-evidence interface Phase 10 receives (profile-owned, immutable)."""

    profile: RetrievalProfile
    search: DocumentSearch
    executor: ToolExecutor = field(default_factory=default_executor)

    def __post_init__(self) -> None:
        if mismatches := _search_mismatches(self.search, self.profile):
            raise ConfigurationError(f"document search does not match the profile: {mismatches}")

    def retrieve(
        self,
        request: RetrievalRequest,
        ctx: ToolContext,
        *,
        authorized_projects: Iterable[str],
    ) -> RetrievalResult:
        """Project-scoped evidence for ``request`` under the profile (never raises on failure)."""
        if mismatches := _search_mismatches(self.search, self.profile):  # tampered after build
            return self._result(request, RetrievalStatus.RETRIEVAL_ERROR, error=str(mismatches))
        ctx.documents = self.search
        res = self.executor.run(
            DOCUMENT_TOOL,
            {"project_id": request.project_id, "query": request.query},
            ctx,
            scope_project_id=request.project_id,
            authorized_projects=tuple(authorized_projects),
        )
        if res.status == ToolStatus.OK:
            (item,) = res.items
            return self._result(
                request,
                RetrievalStatus.OK,
                evidence=list(item.evidence),
                filters_applied=item.scope,
                candidate_count=item.first_stage_candidates,
                rerank_decision=item.rerank_decision,
                warnings=list(item.retrieval_notes),
                latency_ms=item.latency_ms,
                first_stage_ms=item.first_stage_ms,
            )
        if res.status == ToolStatus.INSUFFICIENT_EVIDENCE:
            return self._result(
                request,
                RetrievalStatus.NO_EVIDENCE,
                warnings=[NO_EVIDENCE_MEANING] + [m.detail for m in res.mechanical],
                latency_ms=res.latency_ms,
            )
        status = (
            RetrievalStatus.SCOPE_REFUSED
            if res.status == ToolStatus.SCOPE_REFUSED
            else RetrievalStatus.RETRIEVAL_ERROR
        )
        return self._result(
            request, status, error=res.error or res.status.value, latency_ms=res.latency_ms
        )

    def _result(
        self, request: RetrievalRequest, status: RetrievalStatus, **kw: Any
    ) -> RetrievalResult:
        hints = RecordedHints(
            temporal_scope=request.temporal_scope, document_type_hints=request.document_type_hints
        )
        warnings = kw.pop("warnings", [])
        if request.temporal_scope is not None or request.document_type_hints:
            warnings = [HINTS_NOT_APPLIED, *warnings]
        return RetrievalResult(
            status=status,
            project_id=request.project_id,
            query_used=request.query,
            recorded_hints=hints,
            top_k=self.profile.final_k,
            profile=self.profile.profile_id,
            warnings=warnings,
            **kw,
        )
