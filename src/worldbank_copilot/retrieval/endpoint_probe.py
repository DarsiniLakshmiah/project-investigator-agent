"""Load probe for a candidate embedding endpoint (Phase 8 experiment; writes nothing).

Three steps, one HTTP attempt each (no retries), separated by a controlled pause:

1. one input;
2. a small batch;
3. another small batch after the pause.

Each step records HTTP status, Databricks error code, Retry-After, latency, the number of
vectors, their dimension(s) and whether all values are finite. The verdict is USABLE only
if every step returned HTTP 200 with one finite vector per input and a single dimension.
The probe never caches vectors and never changes the production configuration.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from worldbank_copilot.retrieval.embeddings import (
    DatabricksHttpTransport,
    RetryableEmbeddingError,
    Transport,
    error_code,
    parse_retry_after,
)

CANDIDATES_FILE = Path("retrieval") / "embedding_candidates.yaml"


class ProbeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    endpoint: str
    request_timeout_seconds: float = Field(gt=0)
    batch_size: int = Field(gt=0, le=16)
    pause_seconds: float = Field(ge=0)


def load_probe_config(config_dir: Path) -> ProbeConfig:
    data = yaml.safe_load((Path(config_dir) / CANDIDATES_FILE).read_text(encoding="utf-8"))
    return ProbeConfig.model_validate(data["probe"])


@dataclass
class StepResult:
    step: str
    inputs: int
    status: int | None  # None: no HTTP response (timeout / connection error)
    error_code: str | None
    retry_after: float | None
    latency_ms: float
    vectors: int
    dimensions: list[int]
    finite: bool
    error: str | None = None

    @property
    def ok(self) -> bool:
        return (
            self.status == 200
            and self.vectors == self.inputs
            and len(set(self.dimensions)) == 1
            and self.finite
        )


@dataclass
class ProbeReport:
    endpoint: str
    steps: list[StepResult] = field(default_factory=list)

    @property
    def dimension(self) -> int | None:
        dims = {d for s in self.steps if s.ok for d in s.dimensions}
        return dims.pop() if len(dims) == 1 else None

    @property
    def verdict(self) -> str:
        if self.steps and all(s.ok for s in self.steps) and self.dimension is not None:
            return "USABLE"
        if any(s.status == 429 for s in self.steps):
            return "RATE_LIMITED"
        if any(s.status in (401, 403, 404) for s in self.steps):
            return "UNAVAILABLE"
        if any(s.status == 200 for s in self.steps):
            return "PARTIAL"  # some steps worked; see the failing ones
        return "FAILED"

    def rows(self) -> list[dict[str, Any]]:
        return [
            {
                "endpoint": self.endpoint,
                "step": s.step,
                "inputs": s.inputs,
                "status": s.status,
                "error_code": s.error_code,
                "retry_after": s.retry_after,
                "latency_ms": s.latency_ms,
                "vectors": s.vectors,
                "dimensions": sorted(set(s.dimensions)),
                "finite": s.finite,
                "ok": s.ok,
                "error": s.error,
            }
            for s in self.steps
        ]

    def format(self) -> str:
        lines = [f"endpoint {self.endpoint}: verdict {self.verdict}, dimension {self.dimension}"]
        for s in self.steps:
            lines.append(
                f"  [{'OK' if s.ok else 'FAIL'}] {s.step}: inputs={s.inputs} status={s.status} "
                f"code={s.error_code} retry_after={s.retry_after} latency={s.latency_ms:.0f}ms "
                f"vectors={s.vectors} dims={sorted(set(s.dimensions))} finite={s.finite}"
                + (f" error={s.error}" if s.error else "")
            )
        return "\n".join(lines)


def _step(
    name: str,
    transport: Transport,
    texts: Sequence[str],
    timeout: float,
    clock: Callable[[], float],
) -> StepResult:
    started = clock()
    try:
        reply = transport.post(texts, timeout)
    except RetryableEmbeddingError as exc:
        return StepResult(
            name,
            len(texts),
            None,
            None,
            None,
            round((clock() - started) * 1000, 1),
            0,
            [],
            False,
            str(exc),
        )
    latency = round((clock() - started) * 1000, 1)
    if reply.status != 200:
        return StepResult(
            name,
            len(texts),
            reply.status,
            error_code(reply.body),
            parse_retry_after(reply.headers),
            latency,
            0,
            [],
            False,
            str(reply.body)[:300],
        )
    data = reply.body.get("data") if isinstance(reply.body, dict) else None
    data = data if isinstance(data, list) else []
    vectors = [item.get("embedding") for item in data if isinstance(item, dict)]
    vectors = [v for v in vectors if isinstance(v, list)]
    finite = bool(vectors) and all(
        isinstance(x, int | float) and math.isfinite(x) for v in vectors for x in v
    )
    return StepResult(
        name, len(texts), 200, None, None, latency, len(vectors), [len(v) for v in vectors], finite
    )


def run_probe(
    config: ProbeConfig,
    sample_texts: Sequence[str],
    transport: Transport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
    log: Callable[[str], None] | None = None,
) -> ProbeReport:
    """One input, a batch, then another batch after ``pause_seconds``. No retries."""
    needed = 1 + 2 * config.batch_size
    if len(sample_texts) < needed:
        raise ValueError(f"the probe needs {needed} sample texts, got {len(sample_texts)}")
    say = log or (lambda _m: None)
    transport = transport or DatabricksHttpTransport(config.endpoint)
    report = ProbeReport(config.endpoint)
    plan = [
        ("single input", list(sample_texts[:1])),
        ("batch 1", list(sample_texts[1 : 1 + config.batch_size])),
        ("batch 2 (after pause)", list(sample_texts[1 + config.batch_size : needed])),
    ]
    for index, (name, texts) in enumerate(plan):
        if index:
            say(f"pause {config.pause_seconds:.0f}s before {name}")
            sleep(config.pause_seconds)
        result = _step(name, transport, texts, config.request_timeout_seconds, clock)
        report.steps.append(result)
        say(report.format().splitlines()[-1])
    return report
