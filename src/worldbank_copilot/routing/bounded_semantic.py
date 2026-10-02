"""Candidate C_DATABRICKS_BOUNDED_CLASSIFIER as the RoutingService semantic hook (Phase 9D).

`BoundedSemanticClassifier.classify(context)` is what `RoutingService(semantic=...)` calls -
only when the deterministic rules return SEMANTIC_CLASSIFICATION_REQUIRED, i.e. after input
validation, project resolution, authorisation and every refusal have been decided. It sends
the frozen model-facing contract (bounded_classifier.build_request) and returns a
`SemanticDecision`: one allowed intent (route = requirements[intent]) or ABSTAIN.

Every call is classified into exactly ONE failure kind, so no failure can be counted twice:

* ``NONE``        - HTTP 200 and the reply satisfies the frozen output contract;
* ``OPERATIONAL`` - no HTTP 200: transport error / timeout (status 0), 429, 5xx, other HTTP
                    error - infrastructure, never a classifier-quality judgement;
* ``CONTRACT``    - HTTP 200 but the reply violates the frozen contract/schema - a
                    classifier-quality failure.

Both failure kinds fail closed (abstain -> CLARIFY, nothing executed). Failure reasons never
quote model text, and no reasoning content is read beyond its type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from worldbank_copilot.routing.bounded_classifier import (
    CANDIDATE,
    BoundedClassifierConfig,
    InvalidClassifierOutput,
    build_request,
    contract_sha256,
    numeric_usage,
    parse_response,
    to_decision,
)
from worldbank_copilot.routing.bounded_classifier_probe import (
    TOOL_KEYS,
    ChatResponse,
    ChatTransport,
    content_shape,
    first_choice,
    retry_after_body,
    retry_after_header,
)
from worldbank_copilot.routing.models import Intent, SemanticDecision
from worldbank_copilot.routing.semantic import SemanticQueryContext

FailureKind = Literal["NONE", "OPERATIONAL", "CONTRACT"]


@dataclass(frozen=True)
class ClassificationCall:
    """What one classification call did - status, label, failure kind; never model text."""

    label: str | None  # an allowed intent or ABSTAIN; None on any failure
    failure_kind: FailureKind
    fail_reason: str | None
    status: int
    latency_s: float
    transport_error: str | None
    retry_after_header: str | None
    retry_after_body: Any
    model: str | None
    usage: dict[str, Any] = field(default_factory=dict)
    content_part_types: tuple[str, ...] = ()
    answer_text_chars: int | None = None
    answer_text_sha256: str | None = None
    tool_keys_in_request: tuple[str, ...] = ()


class BoundedSemanticClassifier:
    """Production semantic fallback backed by the bounded Databricks classifier."""

    name = CANDIDATE

    def __init__(
        self,
        config: BoundedClassifierConfig,
        transport: ChatTransport,
        route_of: dict[Intent, str],
        *,
        version: str | None = None,
    ):
        self.config, self.transport, self.route_of = config, transport, route_of
        self.version = version or contract_sha256(config)[:12]

    def request_body(self, context: SemanticQueryContext) -> dict[str, Any]:
        return build_request(
            self.config, context.question, context.project_id, context.temporal_kind
        )

    def call(self, context: SemanticQueryContext) -> tuple[SemanticDecision, ClassificationCall]:
        body = self.request_body(context)
        response = self.transport.post(body, self.config.request.timeout_seconds)
        return self.interpret(response, tuple(sorted(TOOL_KEYS & set(body))))

    def classify(self, context: SemanticQueryContext) -> SemanticDecision:
        return self.call(context)[0]

    def interpret(
        self, response: ChatResponse, tool_keys: tuple[str, ...] = ()
    ) -> tuple[SemanticDecision, ClassificationCall]:
        body = response.body if isinstance(response.body, dict) else {}
        types, chars, digest = content_shape(first_choice(response.body))
        latency_ms = round(response.latency_s * 1000, 3)
        label, kind, reason = None, "NONE", None
        try:
            output = parse_response(response.status, response.body, self.config)
        except InvalidClassifierOutput as exc:
            kind = "OPERATIONAL" if response.status != 200 else "CONTRACT"
            reason = "TRANSPORT_ERROR" if response.status == 0 else exc.reason.value
            decision = to_decision(
                None, self.route_of, version=self.version, latency_ms=latency_ms, failure=exc
            )
        else:
            label = output.label
            decision = to_decision(
                output, self.route_of, version=self.version, latency_ms=latency_ms
            )
        call = ClassificationCall(
            label=label,
            failure_kind=kind,
            fail_reason=reason,
            status=response.status,
            latency_s=round(response.latency_s, 4),
            transport_error=response.transport_error,
            retry_after_header=retry_after_header(response.headers),
            retry_after_body=retry_after_body(response.body),
            model=body.get("model"),
            usage=numeric_usage(body.get("usage")),
            content_part_types=tuple(types),
            answer_text_chars=chars,
            answer_text_sha256=digest,
            tool_keys_in_request=tool_keys,
        )
        return decision, call
