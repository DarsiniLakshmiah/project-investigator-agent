"""Explicitly invoked Databricks chat adapter. Imports perform no live work."""

from __future__ import annotations

import re

from worldbank_copilot.investigation.claims import Failure, ModelReply, NodeError
from worldbank_copilot.routing.bounded_classifier import message_text
from worldbank_copilot.routing.bounded_classifier_probe import DatabricksChatTransport


class DatabricksModelAdapter:
    def __init__(self, endpoint, *, transport=None, reasoning_effort=None):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", endpoint):
            raise ValueError("invalid endpoint name")
        if reasoning_effort not in (None, "low", "medium", "high"):
            raise ValueError("invalid explicit reasoning setting")
        self.endpoint = endpoint
        self.transport = transport or DatabricksChatTransport(endpoint)
        self.reasoning_effort = reasoning_effort

    def invoke(self, request):
        body = {
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.context_json},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "phase10d_" + request.role.lower(),
                    "strict": True,
                    "schema": _strict_schema(request.output_schema),
                },
            },
        }
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        reply = self.transport.post(body, request.timeout_seconds)
        if reply.status == 0:
            raise NodeError(
                Failure.MODEL_TIMEOUT
                if reply.transport_error and "Timeout" in reply.transport_error
                else Failure.MODEL_UNAVAILABLE
            )
        if reply.status != 200:
            # Request rejection is distinct from availability; never retain response bodies.
            raise NodeError(
                Failure.MODEL_REQUEST_INVALID if reply.status == 400 else Failure.MODEL_UNAVAILABLE
            )
        choices = reply.body.get("choices") if isinstance(reply.body, dict) else None
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise NodeError(Failure.MODEL_OUTPUT_INVALID)
        choice = choices[0]
        message = choice.get("message")
        if (
            choice.get("finish_reason") != "stop"
            or not isinstance(message, dict)
            or message.get("refusal")
            or message.get("tool_calls")
            or message.get("function_call")
        ):
            raise NodeError(Failure.MODEL_OUTPUT_INVALID)
        text = message_text(message.get("content"))
        if not text.strip():
            raise NodeError(Failure.MODEL_OUTPUT_INVALID)
        usage = reply.body.get("usage") or {}

        def count(name):
            value = usage.get(name) if isinstance(usage, dict) else None
            return value if type(value) is int and value >= 0 else None

        # Never read or retain reasoning_content, hidden transcripts, headers or raw response.
        return ModelReply(
            text=text,
            model_identity=(
                reply.body.get("model")
                if isinstance(reply.body.get("model"), str)
                and re.fullmatch(r"[A-Za-z0-9_./:-]{1,200}", reply.body["model"])
                else self.endpoint
            ),
            input_tokens=count("prompt_tokens"),
            output_tokens=count("completion_tokens"),
        )


def _strict_schema(value):
    """Copy the Databricks generation schema; authoritative validation stays unchanged.

    Remove defaults and the proven unsupported pattern keyword only. Preserve
    all other constraints and the existing strict-object normalization.
    """
    if isinstance(value, list):
        return [_strict_schema(v) for v in value]
    if not isinstance(value, dict):
        return value
    result = {k: _strict_schema(v) for k, v in value.items() if k not in ("default", "pattern")}
    if "properties" in result:
        result["required"] = list(result["properties"])
        result["additionalProperties"] = False
    return result
