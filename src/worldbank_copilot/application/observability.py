"""Allowlisted request metrics; no prompts, documents, credentials or reasoning."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from pydantic import Field

from worldbank_copilot.investigation.policy import Contract


class ModelMetric(Contract):
    role: Literal["INVESTIGATOR", "SYNTHESIZER", "CRITIC"]
    model: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_./:-]{1,200}$")
    latency_ms: float = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    critic_codes: tuple[str, ...] = ()
    outcome: str = Field(pattern=r"^[A-Z_]+$")


class TraceRecord(Contract):
    request_id: str
    session_id: str
    project_id: str
    route: str
    latency_ms: float = Field(ge=0)
    model_calls: int = Field(ge=0)
    retrieval_calls: int = Field(ge=0)
    investigator_used: bool
    investigator_actions: int = Field(ge=0, le=1)
    critic_used: bool
    claim_count: int = Field(ge=0)
    required_total: int = Field(default=0, ge=0)
    required_with_evidence: int = Field(default=0, ge=0)
    evidence_ids: tuple[str, ...] = ()
    outcome: str
    stages_ms: dict[str, float] = Field(default_factory=dict)
    model_metrics: tuple[ModelMetric, ...] = ()
    configuration: Literal["copilot_e2e@1"] = "copilot_e2e@1"
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    mlflow_trace_id: str | None = None


@dataclass
class Recorder:
    mlflow_enabled: bool = False
    stages: dict = field(default_factory=dict)
    trace_id: str | None = None

    @contextmanager
    def span(self, name):
        started = time.monotonic()
        if self.mlflow_enabled:
            import mlflow

            with mlflow.start_span(name=name) as span:
                self.trace_id = span.trace_id
                yield span
        else:
            yield None
        self.stages[name] = max(0, time.monotonic() - started) * 1000

    def publish(self, span, record):
        if span is not None:
            # Exact contract whitelist, never arbitrary request/model dictionaries.
            span.set_attributes(record.model_dump(mode="json"))
