"""Sanitized diagnostic reason codes for model-call failures (prototype runtime only).

``DatabricksModelAdapter`` (pinned by the Phase 10D @4 lock) collapses several envelope
problems into one authoritative ``Failure`` category. ``DiagnosedModelAdapter`` wraps it
unchanged and, through the adapter's existing transport injection point, records only a
bounded reason code. No response body, model text, prompt, evidence, reasoning content
or header is retained.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from worldbank_copilot.investigation.claims import Failure, ModelReply, ModelRequest
from worldbank_copilot.investigation.model_adapter import DatabricksModelAdapter
from worldbank_copilot.routing.bounded_classifier import message_text
from worldbank_copilot.routing.bounded_classifier_probe import (
    ChatResponse,
    DatabricksChatTransport,
)


class DiagnosticReason(StrEnum):
    NO_RESPONSE = "NO_RESPONSE"  # timeout / connection error before any HTTP response
    REQUEST_REJECTED = "REQUEST_REJECTED"  # HTTP 400
    HTTP_ERROR = "HTTP_ERROR"  # any other non-200 status
    MALFORMED_RESPONSE = "MALFORMED_RESPONSE"  # not exactly one choice / no message object
    FINISH_REASON_LENGTH = "FINISH_REASON_LENGTH"  # output token limit reached
    FINISH_REASON_OTHER = "FINISH_REASON_OTHER"  # any other non-"stop" finish reason
    REFUSAL = "REFUSAL"
    TOOL_CALL = "TOOL_CALL"
    FUNCTION_CALL = "FUNCTION_CALL"
    EMPTY_CONTENT = "EMPTY_CONTENT"  # no final answer text (e.g. reasoning-only content)
    JSON_PARSE_FAILED = "JSON_PARSE_FAILED"  # final text is not valid JSON
    SCHEMA_VALIDATION_FAILED = "SCHEMA_VALIDATION_FAILED"  # JSON violates the strict contract
    OUTPUT_TOO_LARGE = "OUTPUT_TOO_LARGE"


# Reasons for failures raised by the unchanged ``parse_output`` (after a valid envelope).
PARSE_REASONS = {
    Failure.MODEL_OUTPUT_INVALID: DiagnosticReason.JSON_PARSE_FAILED,
    Failure.SCHEMA_VALIDATION_FAILED: DiagnosticReason.SCHEMA_VALIDATION_FAILED,
    Failure.OUTPUT_BUDGET_EXCEEDED: DiagnosticReason.OUTPUT_TOO_LARGE,
}


def classify_envelope(status: int, body: Any) -> DiagnosticReason | None:
    """The reason the adapter will reject this response, in the adapter's check order.

    None means the envelope is acceptable. Reads only structure and finish metadata.
    """
    if status == 0:
        return DiagnosticReason.NO_RESPONSE
    if status != 200:
        return DiagnosticReason.REQUEST_REJECTED if status == 400 else DiagnosticReason.HTTP_ERROR
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return DiagnosticReason.MALFORMED_RESPONSE
    choice = choices[0]
    finish = choice.get("finish_reason")
    if finish != "stop":
        return (
            DiagnosticReason.FINISH_REASON_LENGTH
            if finish == "length"
            else DiagnosticReason.FINISH_REASON_OTHER
        )
    message = choice.get("message")
    if not isinstance(message, dict):
        return DiagnosticReason.MALFORMED_RESPONSE
    if message.get("refusal"):
        return DiagnosticReason.REFUSAL
    if message.get("tool_calls"):
        return DiagnosticReason.TOOL_CALL
    if message.get("function_call"):
        return DiagnosticReason.FUNCTION_CALL
    if not message_text(message.get("content")).strip():
        return DiagnosticReason.EMPTY_CONTENT
    return None


class _DiagnosticTransport:
    """Pass-through transport that keeps only the last envelope's reason code."""

    def __init__(self, inner):
        self.inner = inner
        self.last_reason: DiagnosticReason | None = None

    def post(self, body: dict, timeout: float) -> ChatResponse:
        reply = self.inner.post(body, timeout)
        self.last_reason = classify_envelope(reply.status, reply.body)
        return reply


class DiagnosedModelAdapter:
    """``ModelAdapter`` delegating to the unchanged ``DatabricksModelAdapter``.

    Failures and replies are exactly the adapter's; ``last_reason`` adds a safe code.
    """

    def __init__(self, endpoint: str, *, transport=None):
        self._transport = _DiagnosticTransport(transport or DatabricksChatTransport(endpoint))
        self._adapter = DatabricksModelAdapter(endpoint, transport=self._transport)
        self.last_reason: DiagnosticReason | None = None

    def invoke(self, request: ModelRequest) -> ModelReply:
        self._transport.last_reason = self.last_reason = None
        try:
            return self._adapter.invoke(request)
        finally:
            self.last_reason = self._transport.last_reason
