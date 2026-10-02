"""Candidate C_DATABRICKS_BOUNDED_CLASSIFIER - bounded intent-classification contract (9D).

A small Databricks Foundation Model API chat endpoint classifies the information need of an
UNRESOLVED semantic-routing request. The model output is constrained (strict JSON schema)
to ``{"intent": <one reviewed non-refusal intent | ABSTAIN>}``. This module:

* builds the chat request: fixed instructions, read-only request context as data, strict
  response schema; no chain-of-thought is requested and none is persisted;
* parses the response and FAILS CLOSED on anything outside the contract (HTTP error,
  missing choice, truncation, refusal, non-JSON, extra keys, out-of-enum value);
* maps the result to the existing semantic hook: abstain = (intent == ABSTAIN); route =
  requirements[intent] (never generated); confidence only if derivable (not self-reported).

Authorization, project_id, SQL, tool arguments, document ids, retrieval queries, answers and
execution stay in deterministic code. No network call is made here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from worldbank_copilot.routing.models import Intent, SemanticDecision
from worldbank_copilot.routing.semantic import SEMANTIC_INTENTS

CANDIDATE = "C_DATABRICKS_BOUNDED_CLASSIFIER"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EndpointConfig(_Model):
    preferred: str
    small_model_markers: tuple[str, ...]


class RequestConfig(_Model):
    timeout_seconds: float
    max_tokens: int
    retries: Literal[0]
    schema_name: str
    strict: bool
    required_parameters: dict[str, Any]  # optional chat params sent on every classification


class BoundedClassifierConfig(_Model):
    version: int
    candidate: Literal["C_DATABRICKS_BOUNDED_CLASSIFIER"]
    status: str
    experiment_history: tuple[dict[str, Any], ...]  # append-only record; not in the contract
    endpoint: EndpointConfig
    request: RequestConfig
    abstain_label: Literal["ABSTAIN"]
    intent_descriptions: dict[str, str]
    instructions: str
    capability: dict[str, Any]
    dev_evaluation: dict[str, Any]  # Phase 9D DEV protocol; not part of the model contract

    @model_validator(mode="after")
    def _labels(self) -> BoundedClassifierConfig:
        expected = [i.value for i in SEMANTIC_INTENTS] + [self.abstain_label]
        if list(self.intent_descriptions) != expected:
            raise ValueError("intent_descriptions: reviewed intents (in order) + ABSTAIN")
        return self

    @property
    def labels(self) -> list[str]:
        return list(self.intent_descriptions)


def load_bounded_classifier_config(config_dir: Path) -> BoundedClassifierConfig:
    path = Path(config_dir) / "routing" / "bounded_classifier.yaml"
    return BoundedClassifierConfig.model_validate(yaml.safe_load(path.read_text("utf-8")))


# -- request ----------------------------------------------------------------------------------


def output_schema(config: BoundedClassifierConfig) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"intent": {"type": "string", "enum": config.labels}},
        "required": ["intent"],
        "additionalProperties": False,
    }


def response_format(config: BoundedClassifierConfig) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": config.request.schema_name,
            "schema": output_schema(config),
            "strict": config.request.strict,
        },
    }


def system_prompt(config: BoundedClassifierConfig) -> str:
    lines = [f"- {label}: {text}" for label, text in config.intent_descriptions.items()]
    return config.instructions + "\nAllowed intents:\n" + "\n".join(lines)


def build_request(
    config: BoundedClassifierConfig,
    question: str,
    project_id: str,
    time_scope: str,
    *,
    structured: bool = True,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Chat-completions body. The request is passed as JSON data, never as instructions."""
    state = {"request": question, "resolved_project": project_id, "time_scope": time_scope}
    body: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_prompt(config)},
            {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
        ],
        "max_tokens": config.request.max_tokens,
        "response_format": response_format(config) if structured else {"type": "json_object"},
    }
    body.update(config.request.required_parameters)
    body.update(extra_params or {})
    return body


def contract_sha256(config: BoundedClassifierConfig) -> str:
    """Hash of everything that defines the model-facing contract (prompt + schema + limits)."""
    payload = {
        "endpoint": config.endpoint.preferred,
        "system_prompt": system_prompt(config),
        "response_format": response_format(config),
        "max_tokens": config.request.max_tokens,
        "required_parameters": config.request.required_parameters,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# -- response parsing (fail closed) -----------------------------------------------------------


class FailReason(StrEnum):
    HTTP_ERROR = "HTTP_ERROR"
    RATE_LIMITED = "RATE_LIMITED"
    NO_CHOICE = "NO_CHOICE"
    TRUNCATED = "TRUNCATED"
    REFUSAL = "REFUSAL"
    EMPTY = "EMPTY"
    NOT_JSON = "NOT_JSON"
    WRONG_SHAPE = "WRONG_SHAPE"
    OUT_OF_ENUM = "OUT_OF_ENUM"


class InvalidClassifierOutput(ValueError):
    def __init__(self, reason: FailReason, detail: str = ""):
        super().__init__(f"{reason.value}: {detail}".rstrip(": "))
        self.reason = reason


@dataclass(frozen=True)
class ClassifierOutput:
    label: str  # an allowed intent or ABSTAIN
    model: str | None  # model version reported by the endpoint
    usage: dict[str, Any]

    @property
    def abstain(self) -> bool:
        return self.label == "ABSTAIN"


def message_text(content: Any) -> str:
    """Final answer text only. Reasoning parts (list-form content, e.g. GPT OSS) and any
    other message fields (e.g. reasoning_content) are never read beyond their type."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def parse_response(status: int, body: Any, config: BoundedClassifierConfig) -> ClassifierOutput:
    if status == 429:
        raise InvalidClassifierOutput(FailReason.RATE_LIMITED, "HTTP 429")
    if status != 200:
        raise InvalidClassifierOutput(FailReason.HTTP_ERROR, f"HTTP {status}")
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise InvalidClassifierOutput(FailReason.NO_CHOICE, "expected exactly one choice")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise InvalidClassifierOutput(FailReason.TRUNCATED, str(choice.get("finish_reason")))
    message = choice.get("message") or {}
    if message.get("refusal"):
        raise InvalidClassifierOutput(FailReason.REFUSAL)
    text = message_text(message.get("content")).strip()
    if not text:
        raise InvalidClassifierOutput(FailReason.EMPTY)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidClassifierOutput(FailReason.NOT_JSON) from exc
    # Failure details never quote model-generated text (keys, values or prose).
    if not isinstance(data, dict) or set(data) != {"intent"}:
        shape = f"{len(data)} keys" if isinstance(data, dict) else type(data).__name__
        raise InvalidClassifierOutput(FailReason.WRONG_SHAPE, shape)
    label = data["intent"]
    if not isinstance(label, str) or label not in config.labels:
        raise InvalidClassifierOutput(FailReason.OUT_OF_ENUM, type(label).__name__)
    return ClassifierOutput(label, body.get("model"), numeric_usage(body.get("usage")))


def numeric_usage(usage: Any) -> dict[str, Any]:
    """Token counts only (numbers, nested one level); any text is dropped."""
    if not isinstance(usage, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in usage.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            out[key] = value
        elif isinstance(value, dict):
            nested = {
                k: v
                for k, v in value.items()
                if isinstance(v, int | float) and not isinstance(v, bool)
            }
            if nested:
                out[key] = nested
    return out


# -- harness mapping --------------------------------------------------------------------------


def to_decision(
    output: ClassifierOutput | None,
    route_of: dict[Intent, str],
    *,
    version: str,
    latency_ms: float = 0.0,
    failure: InvalidClassifierOutput | None = None,
) -> SemanticDecision:
    """Decision for the existing semantic hook: ABSTAIN / failure -> CLARIFY, no execution.

    Confidence stays 0.0: the endpoint supplies no calibrated score and self-reported
    confidence is not requested. Route comes from the reviewed requirements, never the model.
    """
    if output is None or failure is not None:
        return SemanticDecision(
            classifier=CANDIDATE,
            version=version,
            abstain=True,
            reason=str(failure or "NO_OUTPUT"),
            latency_ms=latency_ms,
        )
    if output.abstain:
        return SemanticDecision(
            classifier=CANDIDATE,
            version=version,
            abstain=True,
            reason="MODEL_ABSTAINED",
            latency_ms=latency_ms,
        )
    intent = Intent(output.label)
    return SemanticDecision(
        classifier=CANDIDATE,
        version=version,
        abstain=False,
        intent=intent,
        route=route_of[intent],
        reason="ACCEPTED",
        latency_ms=latency_ms,
    )


# Malformed / out-of-contract responses that must ALL be rejected (used by tests and re-run
# in the Databricks runtime by the capability notebook).
def malformed_fixtures() -> list[tuple[str, int, Any]]:
    def chat(content: Any, finish: str = "stop", **message: Any) -> dict[str, Any]:
        return {"choices": [{"finish_reason": finish, "message": {"content": content, **message}}]}

    return [
        ("http_500", 500, {"error_code": "INTERNAL_ERROR"}),
        ("http_429", 429, {"error_code": "REQUEST_LIMIT_EXCEEDED"}),
        ("no_choices", 200, {"choices": []}),
        ("two_choices", 200, {"choices": [chat("{}")["choices"][0]] * 2}),
        ("truncated", 200, chat('{"intent": "RISKS"', finish="length")),
        ("refusal", 200, chat(None, refusal="I can't help with that.")),
        ("empty", 200, chat("")),
        ("not_json", 200, chat("RISKS")),
        ("prose_around_json", 200, chat('Sure: {"intent": "RISKS"}')),
        ("array", 200, chat('["RISKS"]')),
        ("extra_key", 200, chat('{"intent": "RISKS", "reasoning": "because"}')),
        ("route_key", 200, chat('{"intent": "RISKS", "route": "STRUCTURED"}')),
        ("missing_intent", 200, chat('{"label": "RISKS"}')),
        ("out_of_enum", 200, chat('{"intent": "DELETE_ALL_DATA"}')),
        ("refusal_intent", 200, chat('{"intent": "WRITE_REQUEST"}')),
        ("lowercase", 200, chat('{"intent": "risks"}')),
        ("non_string", 200, chat('{"intent": 7}')),
        ("null_intent", 200, chat('{"intent": null}')),
        ("not_a_dict_body", 200, ["unexpected"]),
    ]
