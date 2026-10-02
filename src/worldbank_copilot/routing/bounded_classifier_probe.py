"""Capability probe for C_DATABRICKS_BOUNDED_CLASSIFIER (synthetic requests only).

Pre-registered sequence, one attempt per call, no retries:

1. plain chat call (endpoint callable with notebook identity);
2. single calls recording which optional parameters the endpoint accepts;
3. strict-schema classification of every synthetic case (structured-output behaviour);
4. unconstrained JSON mode on selected cases (what the fail-closed parser must reject);
5. malformed / out-of-enum fixtures through the parser (no network);
6. warm latency repeats; 7. a small concurrent burst (rate-limit behaviour).

Gates: endpoint callable, required configuration accepted, strict structured output
works, every reply validates to one allowed label, fixtures fail closed, no tools in
any request, decisions carry only label / route-from-requirements, operational failure
rate of the required non-burst calls. Diagnostic only: parameter probes
(SUPPORTED / UNSUPPORTED / ERROR), latency, synthetic label agreement and the deliberate
burst (its 429s never enter the failure rate).

Nothing is retried, nothing about a World Bank project, routing case or probe case is sent,
and no reasoning text is kept: records hold status, parsed label, failure reason, usage,
latency and selected headers. Transport is injected so the logic is testable offline.
"""

from __future__ import annotations

import hashlib
import re
import statistics
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from worldbank_copilot.routing.bounded_classifier import (
    BoundedClassifierConfig,
    ClassifierOutput,
    InvalidClassifierOutput,
    build_request,
    contract_sha256,
    malformed_fixtures,
    message_text,
    numeric_usage,
    parse_response,
    to_decision,
)
from worldbank_copilot.routing.models import Intent

PLAIN_PROMPT = "Reply with the single word READY."
KEPT_HEADER_MARKERS = ("ratelimit", "retry-after", "request-id")


@dataclass(frozen=True)
class ChatResponse:
    status: int  # 0 = no HTTP response (timeout / connection error)
    body: Any
    headers: dict[str, str]
    latency_s: float
    transport_error: str | None = None


class ChatTransport(Protocol):
    def post(self, body: dict[str, Any], timeout: float) -> ChatResponse: ...


class DatabricksChatTransport:
    """POST /serving-endpoints/<endpoint>/invocations with the notebook/job identity
    (Databricks SDK configuration). No token is read, stored or logged here; no retries."""

    def __init__(self, endpoint: str, config: Any | None = None, session: Any | None = None):
        self.endpoint = endpoint
        self._config = config
        self._session = session

    def _workspace_config(self) -> Any:
        if self._config is None:
            from databricks.sdk import WorkspaceClient

            self._config = WorkspaceClient().config
        return self._config

    def post(self, body: dict[str, Any], timeout: float) -> ChatResponse:
        import requests

        config = self._workspace_config()
        url = f"{config.host.rstrip('/')}/serving-endpoints/{self.endpoint}/invocations"
        started = time.perf_counter()
        try:
            reply = (self._session or requests).post(
                url, json=body, headers=config.authenticate(), timeout=timeout
            )
        except requests.RequestException as exc:
            return ChatResponse(0, None, {}, time.perf_counter() - started, type(exc).__name__)
        latency = time.perf_counter() - started
        try:
            payload = reply.json()
        except ValueError:
            payload = {"text": reply.text[:300]}
        return ChatResponse(reply.status_code, payload, dict(reply.headers), latency)


# -- endpoint discovery (read-only listing) ---------------------------------------------------


def chat_endpoints(raw: Sequence[dict[str, Any]], markers: Sequence[str]) -> list[dict[str, Any]]:
    """Summaries of chat serving endpoints from `serving_endpoints.list()` (as_dict rows)."""
    out = []
    for e in raw:
        if e.get("task") != "llm/v1/chat":
            continue
        entities = (e.get("config") or {}).get("served_entities") or []
        foundation = [
            (s.get("foundation_model") or {}).get("name")
            for s in entities
            if s.get("foundation_model")
        ]
        name = e.get("name", "")
        out.append(
            {
                "name": name,
                "ready": (e.get("state") or {}).get("ready"),
                "endpoint_type": e.get("endpoint_type"),
                "foundation_models": [f for f in foundation if f],
                "small_model_candidate": any(m in name.lower() for m in markers),
            }
        )
    return sorted(out, key=lambda r: r["name"])


# -- call records -----------------------------------------------------------------------------


@dataclass
class CallRecord:
    step: str
    case_id: str
    status: int
    latency_s: float
    ok: bool
    label: str | None = None
    fail_reason: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    model: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    logprobs_present: bool | None = None
    # Model text is never stored: only structure, size and a digest of the final-answer text.
    content_part_types: list[str] = field(default_factory=list)
    answer_text_chars: int | None = None
    answer_text_sha256: str | None = None


def _content_shape(choice: dict[str, Any]) -> tuple[list[str], int | None, str | None]:
    content = (choice.get("message") or {}).get("content")
    if isinstance(content, list):
        types = [str(p.get("type")) if isinstance(p, dict) else type(p).__name__ for p in content]
    else:
        types = [type(content).__name__] if content is not None else []
    text = message_text(content)
    if not text:
        return types, None, None
    return types, len(text), hashlib.sha256(text.encode("utf-8")).hexdigest()


def artifact_name(endpoint: str) -> str:
    """Endpoint-specific capability artifact, so earlier endpoint results are never overwritten."""
    return f"capability_result_{re.sub(r'[^A-Za-z0-9]+', '_', endpoint).strip('_')}.json"


def _kept_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in headers.items() if any(m in k.lower() for m in KEPT_HEADER_MARKERS)}


def _error(body: Any) -> tuple[str | None, str | None]:
    if not isinstance(body, dict):
        return None, None
    err = body.get("error")
    if isinstance(err, dict):
        return err.get("code") or err.get("type"), str(err.get("message", ""))[:300]
    return body.get("error_code"), str(body.get("message", body.get("text", "")))[:300] or None


def _choice(body: Any) -> dict[str, Any]:
    choices = body.get("choices") if isinstance(body, dict) else None
    return (
        choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    )


def classify_record(
    step: str, case_id: str, resp: ChatResponse, config: BoundedClassifierConfig
) -> CallRecord:
    code, message = _error(resp.body)
    choice = _choice(resp.body)
    record = CallRecord(
        step=step,
        case_id=case_id,
        status=resp.status,
        latency_s=round(resp.latency_s, 4),
        ok=False,
        error_code=code if resp.status != 200 else None,
        error_message=(message if resp.status != 200 else None) or resp.transport_error,
        model=resp.body.get("model") if isinstance(resp.body, dict) else None,
        finish_reason=choice.get("finish_reason"),
        usage=numeric_usage(resp.body.get("usage")) if isinstance(resp.body, dict) else {},
        headers=_kept_headers(resp.headers),
        logprobs_present=choice.get("logprobs") is not None if choice else None,
    )
    record.content_part_types, record.answer_text_chars, record.answer_text_sha256 = _content_shape(
        choice
    )
    try:
        output = parse_response(resp.status, resp.body, config)
    except InvalidClassifierOutput as exc:
        record.fail_reason = exc.reason.value
        return record
    record.ok, record.label = True, output.label
    return record


def plain_record(resp: ChatResponse) -> CallRecord:
    code, message = _error(resp.body)
    choice = _choice(resp.body)
    types, chars, digest = _content_shape(choice)
    return CallRecord(
        step="plain",
        case_id="plain",
        status=resp.status,
        latency_s=round(resp.latency_s, 4),
        ok=resp.status == 200 and bool(chars),
        error_code=code if resp.status != 200 else None,
        error_message=(message if resp.status != 200 else None) or resp.transport_error,
        model=resp.body.get("model") if isinstance(resp.body, dict) else None,
        finish_reason=choice.get("finish_reason"),
        usage=numeric_usage(resp.body.get("usage")) if isinstance(resp.body, dict) else {},
        headers=_kept_headers(resp.headers),
        content_part_types=types,
        answer_text_chars=chars,
        answer_text_sha256=digest,
    )


# -- probe ------------------------------------------------------------------------------------

TOOL_KEYS = frozenset({"tools", "tool_choice", "functions", "function_call"})
# What a classifier decision must never carry: scope, authorization or execution material.
FORBIDDEN_DECISION_FIELDS = frozenset(
    {"project_id", "project", "authorized_projects", "user", "scope", "tool", "tools", "sql"}
    | {"arguments", "document_ids", "query", "retrieval_query", "answer"}
)
SCHEMA_FAILURES = frozenset({"NOT_JSON", "WRONG_SHAPE", "TRUNCATED", "EMPTY", "NO_CHOICE"})


class _RecordingTransport:
    """Keeps every request body sent (for the no-execution check); thread-safe."""

    def __init__(self, inner: ChatTransport):
        self.inner, self.bodies, self._lock = inner, [], threading.Lock()

    def post(self, body: dict[str, Any], timeout: float) -> ChatResponse:
        with self._lock:
            self.bodies.append(body)
        return self.inner.post(body, timeout)


def _p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[int(0.95 * (len(ordered) - 1))], 4)


def _latency(values: Sequence[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "p50_s": round(statistics.median(values), 4) if values else None,
        "p95_s": _p95(values),
        "max_s": round(max(values), 4) if values else None,
    }


def parameter_status(record: CallRecord) -> str:
    """SUPPORTED (HTTP 200) / UNSUPPORTED (request rejected: 400, 422) / ERROR (other)."""
    if record.status == 200:
        return "SUPPORTED"
    if record.status in (400, 422):
        return "UNSUPPORTED"
    return "ERROR"


def run_capability_probe(
    config: BoundedClassifierConfig,
    transport: ChatTransport,
    route_of: dict[Intent, str],
    *,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    cap = config.capability
    project, timeout = cap["synthetic_project"], config.request.timeout_seconds
    cases = {c["id"]: c for c in cap["synthetic_cases"]}
    sender = _RecordingTransport(transport)

    def request(case_id: str, **kwargs: Any) -> dict[str, Any]:
        return build_request(config, cases[case_id]["request"], project, "NONE", **kwargs)

    records: list[CallRecord] = []

    log("1/7 plain call")
    plain_body = {
        "messages": [{"role": "user", "content": PLAIN_PROMPT}],
        "max_tokens": config.request.max_tokens,
    }
    records.append(plain_record(sender.post(plain_body, timeout)))

    log("2/7 optional-parameter diagnostics")
    first_case = cap["synthetic_cases"][0]["id"]
    for probe in cap["parameter_probes"]:
        resp = sender.post(request(first_case, extra_params=probe["params"]), timeout)
        records.append(classify_record(f"param:{probe['id']}", first_case, resp, config))

    log("3/7 strict-schema classification")
    for case_id in cases:
        resp = sender.post(request(case_id), timeout)
        records.append(classify_record("structured", case_id, resp, config))

    log("4/7 unconstrained JSON mode")
    for case_id in cap["json_object_case_ids"]:
        resp = sender.post(request(case_id, structured=False), timeout)
        records.append(classify_record("json_object", case_id, resp, config))

    log("5/7 malformed fixtures (offline)")
    fixtures = []
    for name, status, body in malformed_fixtures():
        try:
            parse_response(status, body, config)
            fixtures.append({"fixture": name, "rejected": False, "reason": None})
        except InvalidClassifierOutput as exc:
            fixtures.append({"fixture": name, "rejected": True, "reason": exc.reason.value})

    log("6/7 warm latency")
    warm_id = cap["warm_latency_case_id"]
    for i in range(cap["warm_latency_repeats"]):
        resp = sender.post(request(warm_id), timeout)
        records.append(classify_record("warm", f"{warm_id}#{i + 1}", resp, config))

    log("7/7 concurrent burst (diagnostic)")
    burst_id, burst_body = cap["burst_case_id"], request(cap["burst_case_id"])
    with ThreadPoolExecutor(max_workers=cap["burst_concurrency"]) as pool:
        responses = list(
            pool.map(lambda _: sender.post(burst_body, timeout), range(cap["burst_concurrency"]))
        )
    records += [
        classify_record("burst", f"{burst_id}#{i + 1}", r, config) for i, r in enumerate(responses)
    ]
    return summarise(config, records, fixtures, sender.bodies, route_of)


def decision_bounded(
    label: str, route_of: dict[Intent, str], config: BoundedClassifierConfig
) -> bool:
    """The harness decision for a valid label carries the label only; route from requirements."""
    decision = to_decision(ClassifierOutput(label, None, {}), route_of, version="capability")
    if FORBIDDEN_DECISION_FIELDS & set(decision.model_dump()):
        return False
    if label == config.abstain_label:
        return decision.abstain and decision.intent is None and decision.route is None
    intent = Intent(label)
    return not decision.abstain and decision.intent == intent and decision.route == route_of[intent]


def summarise(
    config: BoundedClassifierConfig,
    records: list[CallRecord],
    fixtures: list[dict[str, Any]],
    bodies: Sequence[dict[str, Any]],
    route_of: dict[Intent, str],
) -> dict[str, Any]:
    cap, gates = config.capability, config.capability["gates"]
    expect = {c["id"]: c["expect"] for c in cap["synthetic_cases"]}
    by_step: dict[str, list[CallRecord]] = {}
    for r in records:
        by_step.setdefault(r.step.split(":")[0], []).append(r)
    plain = by_step["plain"][0]
    structured, warm = by_step.get("structured", []), by_step.get("warm", [])
    burst = by_step.get("burst", [])

    required = [*structured, *warm]  # every one uses the selected (required) configuration
    rejected = [r for r in required if r.status in (400, 422)]
    answered = [r for r in required if r.status == 200]
    schema_broken = [r for r in answered if r.fail_reason in SCHEMA_FAILURES]
    operational = [plain, *required]  # burst excluded by design
    labels = [r.label for r in answered if r.ok]
    metrics = {
        "required_calls": len(required),
        "answered": len(answered),
        "enum_valid_rate": round(sum(r.ok for r in answered) / len(answered), 4)
        if answered
        else 0.0,
        "malformed_fixtures_rejected": round(
            sum(f["rejected"] for f in fixtures) / len(fixtures), 4
        ),
        "operational_failure_rate": round(
            sum(r.status != 200 for r in operational) / len(operational), 4
        ),
        "tool_keys_in_requests": sorted({k for b in bodies for k in b if k in TOOL_KEYS}),
    }
    checks = {
        "preferred_endpoint_callable": plain.ok,
        "required_request_configuration_accepted": bool(answered) and not rejected,
        "strict_structured_output_works": bool(answered) and not schema_broken,
        "replies_validate_to_one_allowed_label": metrics["enum_valid_rate"]
        >= gates["enum_valid_rate"],
        "malformed_fixtures_fail_closed": metrics["malformed_fixtures_rejected"]
        >= gates["malformed_fixtures_rejected"],
        "no_execution_during_classification": not metrics["tool_keys_in_requests"],
        "scope_unchangeable_by_model_output": all(
            decision_bounded(label, route_of, config) for label in [*labels, *config.labels]
        ),
        "operational_failure_rate": metrics["operational_failure_rate"]
        <= gates["max_operational_failure_rate"],
    }
    probe_params = {p["id"]: p["params"] for p in cap["parameter_probes"]}
    parameters = {
        r.step.split(":", 1)[1]: {
            "status": parameter_status(r),  # SUPPORTED / UNSUPPORTED / ERROR
            "http_status": r.status,
            "params": probe_params[r.step.split(":", 1)[1]],
            "required": probe_params[r.step.split(":", 1)[1]].items()
            <= config.request.required_parameters.items(),
            **{k: v for k, v in asdict(r).items() if k not in ("step", "case_id", "status")},
        }
        for r in by_step.get("param", [])
    }
    sanity = [
        {"case_id": r.case_id, "expected": expect[r.case_id], "label": r.label, "ok": r.ok}
        for r in structured
    ]
    return {
        "candidate": config.candidate,
        "endpoint": config.endpoint.preferred,
        "contract_sha256": contract_sha256(config),
        "required_parameters": config.request.required_parameters,
        "models_reported": sorted({r.model for r in records if r.model}),
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "diagnostics": {
            "parameters": parameters,
            "latency": {
                "warm": _latency([r.latency_s for r in warm if r.status == 200]),
                "structured": _latency([r.latency_s for r in structured if r.status == 200]),
                "plain_s": plain.latency_s,
                "concurrent_burst": _latency([r.latency_s for r in burst if r.status == 200]),
                "note": "observational at this checkpoint; judged in the DEV trade-off",
            },
            "synthetic_sanity": {
                "agreement": sum(s["label"] == s["expected"] for s in sanity),
                "n": len(sanity),
                "cases": sanity,
                "note": "diagnostic only; never a gate",
            },
            "json_object_mode": [asdict(r) for r in by_step.get("json_object", [])],
            "burst": {
                "concurrency": len(burst),
                "statuses": dict(Counter(r.status for r in burst)),
                "rate_limited": sum(r.status == 429 for r in burst),
                "retry_after": [r.headers for r in burst if r.status == 429],
                "contract_ok": sum(r.ok for r in burst),
                "note": "deliberate burst; excluded from the operational failure rate",
            },
        },
        "malformed_fixtures": fixtures,
        "fail_reasons": dict(Counter(r.fail_reason for r in records if r.fail_reason)),
        "records": [asdict(r) for r in records],
    }
