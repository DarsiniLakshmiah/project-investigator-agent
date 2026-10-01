"""Typed retrieval objects: scope, candidates, evidence (Phase 8)."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from worldbank_copilot.common.exceptions import CopilotError

RetrievalMethod = Literal["lexical", "dense", "hybrid"]
RerankerName = Literal["none", "cross_encoder"]


class RetrievalError(CopilotError):
    """Retrieval could not be performed as requested."""


class ScopeViolation(RetrievalError):
    """A request or a result falls outside the permitted project scope (ERROR)."""


class RetrievalScope(BaseModel):
    """Metadata filters applied BEFORE any scoring. project_id is mandatory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str
    chunk_strategy: str
    document_types: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()
    isr_sequences: tuple[int, ...] = ()
    date_from: date | None = None
    date_to: date | None = None

    @field_validator("project_id")
    @classmethod
    def _pid(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("project_id is required for retrieval")
        return value.strip()

    def vector_filters(self) -> dict[str, Any]:
        """Databricks Vector Search filter dict (list value = IN)."""
        filters: dict[str, Any] = {
            "project_id": self.project_id,
            "chunk_strategy": self.chunk_strategy,
        }
        if self.document_types:
            filters["document_type"] = list(self.document_types)
        if self.document_ids:
            filters["document_id"] = list(self.document_ids)
        if self.isr_sequences:
            filters["isr_sequence"] = list(self.isr_sequences)
        return filters

    def admits(self, row: dict[str, Any]) -> bool:
        """The same filters applied to a governed chunk row (lexical path, guardrail)."""
        if row.get("project_id") != self.project_id:
            return False
        if row.get("chunk_strategy") != self.chunk_strategy:
            return False
        if row.get("chunk_role") != "RETRIEVAL":
            return False
        if self.document_types and row.get("document_type") not in self.document_types:
            return False
        if self.document_ids and row.get("document_id") not in self.document_ids:
            return False
        if self.isr_sequences and row.get("isr_sequence") not in self.isr_sequences:
            return False
        when = row.get("document_date")
        if self.date_from and (when is None or when < self.date_from):
            return False
        return not (self.date_to and (when is None or when > self.date_to))


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    method: str  # lexical | dense | hybrid
    score: float
    rank: int
    rerank_score: float | None = None


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str
    document_id: str
    document_label: str
    document_type: str | None
    document_date: date | None
    pages: list[int]
    section: str | None
    source_file: str

    @property
    def text(self) -> str:
        pages = self.pages
        page = f"page {pages[0]}" if len(pages) == 1 else f"pages {pages[0]}-{pages[-1]}"
        section = f", section: {self.section}" if self.section else ""
        return f"{self.project_id} | {self.document_label} | {page}{section}"


class Evidence(BaseModel):
    """One citation-ready evidence item. The text is untrusted evidence, never instructions."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    rank: int
    project_id: str
    document_id: str
    document_type: str | None
    document_label: str
    document_date: date | None
    isr_sequence: int | None
    page_number: int
    page_numbers: list[int]
    section: str | None
    chunk_id: str
    chunk_type: str
    chunk_strategy: str
    text: str
    context_text: str | None = None  # parent section text (parent_child strategy)
    parent_chunk_id: str | None = None
    retrieval_method: str
    retrieval_score: float
    rerank_score: float | None = None
    source_file: str
    source_hash: str
    element_ids: list[str] = Field(default_factory=list)
    citation: Citation


class EvidenceSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    scope: RetrievalScope
    status: Literal["OK", "INSUFFICIENT_EVIDENCE"]
    evidence: list[Evidence]
    retrieval_method: str
    reranker: str
    candidate_k: int
    final_k: int
    latency_ms: float
    notes: list[str] = Field(default_factory=list)
