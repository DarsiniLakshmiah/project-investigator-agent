"""Sanitized model-failure reason codes: offline fakes only, no endpoint or network."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from tests.unit.test_copilot_runtime import (
    INVESTIGATION,
    PROJECT,
    Critic,
    Synthesizer,
    copilot,
)

from worldbank_copilot.copilot import ResultStatus
from worldbank_copilot.copilot.model_diagnostics import DiagnosedModelAdapter, classify_envelope
from worldbank_copilot.investigation.claims import Failure, ModelRequest, NodeError
from worldbank_copilot.investigation.model_adapter import DatabricksModelAdapter
from worldbank_copilot.routing.bounded_classifier_probe import ChatResponse

SECRET = "SECRET-MODEL-TEXT-7c1f"  # planted in every fake response; must never be retained


def choice(finish="stop", **message):
    return {"choices": [{"finish_reason": finish, "message": message}], "model": "fake-model"}


class FakeTransport:
    """Returns a fixed response, or builds one from the request via ``respond``."""

    def __init__(self, status=200, body=None, error=None, respond=None):
        self.status, self.body, self.error, self.respond = status, body, error, respond

    def post(self, body, timeout):
        if self.respond is not None:
            return ChatResponse(200, self.respond(body), {}, 0.01)
        return ChatResponse(self.status, self.body, {}, 0.01, self.error)


def request():
    return ModelRequest(
        role="SYNTHESIZER",
        system="system",
        context_json="{}",
        output_schema={"type": "object"},
        max_output_tokens=10,
        timeout_seconds=5,
    )


ENVELOPES = [
    ("length", 200, choice("length", content=SECRET), None, "FINISH_REASON_LENGTH"),
    ("content-filter", 200, choice("content_filter", content=SECRET), None, "FINISH_REASON_OTHER"),
    ("refusal", 200, choice(content=SECRET, refusal=SECRET), None, "REFUSAL"),
    ("tool-call", 200, choice(content=SECRET, tool_calls=[{"id": SECRET}]), None, "TOOL_CALL"),
    ("function-call", 200, choice(content=SECRET, function_call={"n": SECRET}), None,
     "FUNCTION_CALL"),
    ("no-choices", 200, {"choices": [], "text": SECRET}, None, "MALFORMED_RESPONSE"),
    ("two-choices", 200, {"choices": [{}, {}]}, None, "MALFORMED_RESPONSE"),
    ("not-object", 200, SECRET, None, "MALFORMED_RESPONSE"),
    ("message-not-object", 200, {"choices": [{"finish_reason": "stop", "message": SECRET}]}, None,
     "MALFORMED_RESPONSE"),
    ("empty", 200, choice(content="  "), None, "EMPTY_CONTENT"),
    ("reasoning-only", 200, choice(content=[{"type": "reasoning", "text": SECRET}]), None,
     "EMPTY_CONTENT"),
    ("http-400", 400, {"message": SECRET}, None, "REQUEST_REJECTED"),
    ("http-503", 503, {"message": SECRET}, None, "HTTP_ERROR"),
    ("timeout", 0, None, "ReadTimeout", "NO_RESPONSE"),
]  # fmt: skip


def retained(obj, seen=None):
    """Every string reachable from the wrapper's own state (excluding the test's fake)."""
    seen = seen if seen is not None else set()
    if id(obj) in seen or isinstance(obj, FakeTransport):
        return ""
    seen.add(id(obj))
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return " ".join(retained(k, seen) + retained(v, seen) for k, v in obj.items())
    if isinstance(obj, list | tuple | set):
        return " ".join(retained(v, seen) for v in obj)
    if hasattr(obj, "__dict__"):
        return retained(vars(obj), seen)
    return str(obj)


@pytest.mark.parametrize(
    ("status", "body", "error", "reason"),
    [case[1:] for case in ENVELOPES],
    ids=[case[0] for case in ENVELOPES],
)
def test_reason_captured_and_authoritative_failure_unchanged(status, body, error, reason):
    with pytest.raises(NodeError) as pinned:
        DatabricksModelAdapter("ep", transport=FakeTransport(status, body, error)).invoke(request())
    adapter = DiagnosedModelAdapter("ep", transport=FakeTransport(status, body, error))
    with pytest.raises(NodeError) as diagnosed:
        adapter.invoke(request())
    assert diagnosed.value.category == pinned.value.category  # authoritative, unchanged
    assert adapter.last_reason == reason == classify_envelope(status, body)
    assert SECRET not in retained(adapter)


def test_success_is_unchanged_and_has_no_reason():
    body = choice(content='{"ok": true}')
    pinned = DatabricksModelAdapter("ep", transport=FakeTransport(body=body)).invoke(request())
    adapter = DiagnosedModelAdapter("ep", transport=FakeTransport(body=body))
    assert adapter.invoke(request()) == pinned
    assert adapter.last_reason is None and classify_envelope(200, body) is None


def test_reason_resets_between_calls():
    adapter = DiagnosedModelAdapter("ep", transport=FakeTransport(body=choice("length")))
    with pytest.raises(NodeError):
        adapter.invoke(request())
    adapter._transport.inner = FakeTransport(body=choice(content="{}"))
    adapter.invoke(request())
    assert adapter.last_reason is None


def synthesis_transport():
    """A valid Synthesizer endpoint stand-in built on the runtime test's fake."""
    fake = Synthesizer()

    def respond(body):
        text = fake.invoke(SimpleNamespace(context_json=body["messages"][1]["content"])).text
        return choice(content=text)

    return FakeTransport(respond=respond)


def traced_copilot(monkeypatch, synthesizer):
    spans = []

    class Span:
        def __init__(self, name):
            self.name, self.attributes, self.trace_id = name, {}, "trace"
            spans.append(self)

        def set_attributes(self, attributes):
            self.attributes.update(attributes)

    @contextmanager
    def start_span(name):
        yield Span(name)

    monkeypatch.setitem(__import__("sys").modules, "mlflow", SimpleNamespace(start_span=start_span))
    app = copilot(synthesizer, Critic())
    app.mlflow_enabled = True
    return app, spans


@pytest.mark.parametrize(
    ("transport", "failure", "reason"),
    [
        (FakeTransport(body=choice("length", content=SECRET)), "MODEL_OUTPUT_INVALID",
         "FINISH_REASON_LENGTH"),
        (FakeTransport(body=choice(content=[{"type": "reasoning", "text": SECRET}])),
         "MODEL_OUTPUT_INVALID", "EMPTY_CONTENT"),
        (FakeTransport(body=choice(content=SECRET + " not JSON")), "MODEL_OUTPUT_INVALID",
         "JSON_PARSE_FAILED"),
        (FakeTransport(body=choice(content=json.dumps({"x": SECRET}))),
         "SCHEMA_VALIDATION_FAILED", "SCHEMA_VALIDATION_FAILED"),
    ],
    ids=["length", "reasoning-only", "json", "schema"],
)  # fmt: skip
def test_runtime_records_reason_and_still_fails_closed(monkeypatch, transport, failure, reason):
    app, spans = traced_copilot(monkeypatch, DiagnosedModelAdapter("ep", transport=transport))
    result = app.investigate(INVESTIGATION, PROJECT)
    assert result.status == ResultStatus.FAIL_CLOSED and not result.claims
    assert result.validation.failures == (failure,)
    call = result.model_calls[-1]
    assert call.role == "SYNTHESIZER"
    assert (call.outcome, call.diagnostic_reason) == (failure, reason)
    # Investigator rounds, then exactly one synthesis attempt: no retry, no critic.
    assert [c.role for c in result.model_calls] == ["INVESTIGATOR", "INVESTIGATOR", "SYNTHESIZER"]
    synthesis = next(s for s in spans if s.name == "synthesis")
    assert synthesis.attributes["diagnostic_reason"] == reason
    assert spans[0].attributes["model_diagnostic_reasons"] == reason
    assert SECRET not in result.model_dump_json()
    assert SECRET not in json.dumps([s.attributes for s in spans])


def test_runtime_success_through_diagnosed_adapter_is_unchanged(monkeypatch):
    app, spans = traced_copilot(
        monkeypatch, DiagnosedModelAdapter("ep", transport=synthesis_transport())
    )
    result = app.investigate(INVESTIGATION, PROJECT)
    assert result.status == ResultStatus.ANSWER
    assert result.model_calls and all(c.diagnostic_reason is None for c in result.model_calls)
    assert spans[0].attributes["model_diagnostic_reasons"] == ""


def test_timeout_raised_by_transport_is_no_response():
    class Raising:
        def invoke(self, request):
            raise TimeoutError

    result = copilot(Raising(), Critic()).investigate(INVESTIGATION, PROJECT)
    call = result.model_calls[-1]
    assert call.role == "SYNTHESIZER"
    assert (call.outcome, call.diagnostic_reason) == (Failure.MODEL_TIMEOUT.value, "NO_RESPONSE")


def test_critic_length_limit_degrades_to_answer_and_stays_diagnosable(monkeypatch):
    # The live D3B failure: Critic envelope finish_reason "length".
    spans = []

    class Span:
        def __init__(self, name):
            self.name, self.attributes, self.trace_id = name, {}, "trace-1"
            spans.append(self)

        def set_attributes(self, attributes):
            self.attributes.update(attributes)

    @contextmanager
    def start_span(name):
        yield Span(name)

    monkeypatch.setitem(__import__("sys").modules, "mlflow", SimpleNamespace(start_span=start_span))
    critic = DiagnosedModelAdapter(
        "critic-endpoint", transport=FakeTransport(body=choice("length", content=SECRET))
    )
    app = copilot(Synthesizer(), critic)
    app.mlflow_enabled = True
    result = app.investigate(INVESTIGATION, PROJECT)
    assert result.status == ResultStatus.ANSWER and result.claims
    assert {c.support for c in result.claims} == {"NOT_ASSESSED"}
    assert result.validation.critic_status == "FAILED"
    call = result.model_calls[-1]
    assert (call.role, call.outcome, call.diagnostic_reason) == (
        "CRITIC",
        "MODEL_OUTPUT_INVALID",
        "FINISH_REASON_LENGTH",
    )
    critic_span = next(s for s in spans if s.name == "critic")
    assert critic_span.attributes["diagnostic_reason"] == "FINISH_REASON_LENGTH"
    root = spans[0].attributes
    assert root["critic_status"] == "FAILED" and root["status"] == "ANSWER"
    assert "FINISH_REASON_LENGTH" in root["model_diagnostic_reasons"]
    assert SECRET not in json.dumps([s.attributes for s in spans])


def test_synthesizer_length_limit_still_fails_closed_with_usage_counts():
    body = {**choice("length", content=SECRET), "usage": {
        "prompt_tokens": 4100, "completion_tokens": 5000,
        "completion_tokens_details": {"reasoning_tokens": 4700}, "secret": SECRET,
    }}  # fmt: skip
    critic = Critic()
    synthesizer = DiagnosedModelAdapter("synth-endpoint", transport=FakeTransport(body=body))
    result = copilot(synthesizer, critic).investigate(INVESTIGATION, PROJECT)
    assert result.status == ResultStatus.FAIL_CLOSED and not result.claims
    assert not critic.requests  # no candidate claims exist to publish or review
    call = result.model_calls[-1]
    assert (call.role, call.outcome, call.diagnostic_reason) == (
        "SYNTHESIZER",
        "MODEL_OUTPUT_INVALID",
        "FINISH_REASON_LENGTH",
    )
    assert (call.input_tokens, call.output_tokens, call.reasoning_tokens) == (4100, 5000, 4700)
    assert SECRET not in result.model_dump_json()


def test_usage_counts_are_integers_only():
    from worldbank_copilot.copilot.model_diagnostics import usage_counts

    assert usage_counts({"usage": {"prompt_tokens": 3, "completion_tokens": "9"}}) == {
        "input_tokens": 3
    }
    assert usage_counts({"usage": None}) == {} and usage_counts(None) == {}
