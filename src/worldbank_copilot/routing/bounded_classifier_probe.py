"""Capability probe for C_DATABRICKS_BOUNDED_CLASSIFIER (synthetic requests only).

Pre-registered, FIXED schedule (configs/routing/bounded_classifier.yaml capability.schedule);
one attempt per call, no retries, no backoff, no schedule change in response to 429:

1. initial quiet period;
2. GATED ordinary calls: plain chat call, strict-schema classification of every synthetic
   case, warm repeats;
3. DIAGNOSTIC ordinary calls: optional-parameter probes, unconstrained JSON mode;
   (malformed / out-of-enum fixtures go through the parser offline - no network);
4. pre-burst quiet period; 5. a small concurrent DIAGNOSTIC burst (rate-limit behaviour).

Every ordinary call starts no earlier than `ordinary_call_gap_seconds` after the previous
ordinary call ENDED (monotonic clock). The pacer checks this before sending; if it cannot be
honoured the run stops before the call and is marked INVALID - model capability is then not
evaluated. Each call records sequence, start/end offsets, the actual gap and, separately,
the raw Retry-After header and the raw retry_after body field.

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

from worldbank_copilot.retrieval.embeddings import parse_retry_after
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
    # Schedule instrumentation (monotonic seconds relative to run start).
    sequence: int | None = None
    started_s: float | None = None
    ended_s: float | None = None
    gap_before_s: float | None = None  # end of previous ordinary call -> this start
    # Raw, separately: never combined, never acted on (no retry, no backoff).
    retry_after_header: str | None = None
    retry_after_body: Any = None


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


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def artifact_name(endpoint: str, run_id: str | None = None) -> str:
    """Endpoint- and run-specific capability artifact; earlier results are never overwritten.

    Without run_id this is the run-1 name (kept for the immutable run-1 artifact)."""
    suffix = f"__{_sanitize(run_id)}" if run_id else ""
    return f"capability_result_{_sanitize(endpoint)}{suffix}.json"


def retry_after_header(headers: dict[str, str]) -> str | None:
    return next((v for k, v in headers.items() if k.lower() == "retry-after"), None)


def retry_after_body(body: Any) -> Any:
    if not isinstance(body, dict):
        return None
    if "retry_after" in body:
        return body["retry_after"]
    err = body.get("error")
    return err.get("retry_after") if isinstance(err, dict) else None


def _seconds(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return parse_retry_after({"Retry-After": value})
    return None


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


GATED_STEPS = ("plain", "structured", "warm")
DIAGNOSTIC_STEPS = ("param", "json_object")
PACER_MAX_WAITS = 1000  # a pacer that still cannot reach the target is a bug -> INVALID


class ScheduleViolation(RuntimeError):
    """The pacing implementation could not honour the pre-registered schedule."""


@dataclass
class _Pacer:
    gap_s: float
    tolerance_s: float
    clock: Callable[[], float]
    sleep: Callable[[float], None]
    t0: float = 0.0
    previous_end: float | None = None

    def wait_until(self, target: float) -> float:
        for _ in range(PACER_MAX_WAITS):
            now = self.clock()
            if now >= target:
                return now
            self.sleep(target - now)
        return self.clock()

    def before_ordinary_call(self, sequence: int) -> tuple[float, float | None]:
        """Waits for the fixed end->start gap; returns (start, gap). Raises before sending
        if the gap cannot be honoured."""
        if self.previous_end is None:
            return self.clock(), None
        start = self.wait_until(self.previous_end + self.gap_s)
        gap = start - self.previous_end
        if gap < self.gap_s - self.tolerance_s:
            raise ScheduleViolation(f"call {sequence}: gap {gap:.6f}s < {self.gap_s}s")
        return start, gap


def run_capability_probe(
    config: BoundedClassifierConfig,
    transport: ChatTransport,
    route_of: dict[Intent, str],
    *,
    log: Callable[[str], None] = print,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    cap, schedule = config.capability, config.capability["schedule"]
    project, timeout = cap["synthetic_project"], config.request.timeout_seconds
    cases = {c["id"]: c for c in cap["synthetic_cases"]}
    sender = _RecordingTransport(transport)
    pacer = _Pacer(
        schedule["ordinary_call_gap_seconds"], schedule["gap_tolerance_seconds"], clock, sleep
    )

    def request(case_id: str, **kwargs: Any) -> dict[str, Any]:
        return build_request(config, cases[case_id]["request"], project, "NONE", **kwargs)

    plain_body = {
        "messages": [{"role": "user", "content": PLAIN_PROMPT}],
        "max_tokens": config.request.max_tokens,
    }
    first_case, warm_id = cap["synthetic_cases"][0]["id"], cap["warm_latency_case_id"]
    # (step, case_id, body, record builder) in the exact pre-registered order
    ordinary: list[tuple[str, str, dict[str, Any]]] = [("plain", "plain", plain_body)]
    ordinary += [("structured", c, request(c)) for c in cases]
    ordinary += [
        ("warm", f"{warm_id}#{i + 1}", request(warm_id)) for i in range(cap["warm_latency_repeats"])
    ]
    ordinary += [
        (f"param:{p['id']}", first_case, request(first_case, extra_params=p["params"]))
        for p in cap["parameter_probes"]
    ]
    ordinary += [
        ("json_object", c, request(c, structured=False)) for c in cap["json_object_case_ids"]
    ]

    records: list[CallRecord] = []
    pacer.t0 = clock()

    def rel(value: float) -> float:
        return round(value - pacer.t0, 6)

    def instrument(record: CallRecord, resp: ChatResponse, seq: int, start: float, end: float):
        record.sequence, record.started_s, record.ended_s = seq, rel(start), rel(end)
        record.retry_after_header = retry_after_header(resp.headers)
        record.retry_after_body = retry_after_body(resp.body)
        return record

    violation = None
    try:
        log(f"quiet {schedule['initial_quiet_seconds']}s before the first call")
        quiet = pacer.wait_until(pacer.t0 + schedule["initial_quiet_seconds"]) - pacer.t0
        if quiet < schedule["initial_quiet_seconds"] - pacer.tolerance_s:
            raise ScheduleViolation(f"initial quiet {quiet:.6f}s not honoured")
        for seq, (step, case_id, body) in enumerate(ordinary, start=1):
            if seq in (1, 1 + len(cases) + 1 + cap["warm_latency_repeats"]):
                log("gated calls" if seq == 1 else "diagnostic calls")
            start, gap = pacer.before_ordinary_call(seq)
            resp = sender.post(body, timeout)
            end = clock()
            pacer.previous_end = end
            record = (
                plain_record(resp)
                if step == "plain"
                else classify_record(step, case_id, resp, config)
            )
            instrument(record, resp, seq, start, end)
            record.gap_before_s = None if gap is None else round(gap, 6)
            records.append(record)
    except ScheduleViolation as exc:
        violation = str(exc)

    log("malformed fixtures (offline)")
    fixtures = []
    for name, status, body in malformed_fixtures():
        try:
            parse_response(status, body, config)
            fixtures.append({"fixture": name, "rejected": False, "reason": None})
        except InvalidClassifierOutput as exc:
            fixtures.append({"fixture": name, "rejected": True, "reason": exc.reason.value})

    burst_started = None
    if violation is None:
        log(f"quiet {schedule['pre_burst_quiet_seconds']}s, then diagnostic burst")
        burst_started = pacer.wait_until(pacer.previous_end + schedule["pre_burst_quiet_seconds"])
        if burst_started - pacer.previous_end < (
            schedule["pre_burst_quiet_seconds"] - pacer.tolerance_s
        ):
            violation = "pre-burst quiet period not honoured; burst not sent"
    if violation is None:
        burst_id, burst_body = cap["burst_case_id"], request(cap["burst_case_id"])

        def burst_call(_: int) -> tuple[ChatResponse, float, float]:
            started = clock()
            resp = sender.post(burst_body, timeout)
            return resp, started, clock()

        with ThreadPoolExecutor(max_workers=cap["burst_concurrency"]) as pool:
            responses = list(pool.map(burst_call, range(cap["burst_concurrency"])))
        for i, (resp, started, ended) in enumerate(responses):
            record = classify_record("burst", f"{burst_id}#{i + 1}", resp, config)
            records.append(instrument(record, resp, len(ordinary) + i + 1, started, ended))

    validation = validate_schedule(config, records, ordinary, violation, burst_started, pacer)
    if not validation["valid"]:
        return {
            "candidate": config.candidate,
            "endpoint": config.endpoint.preferred,
            "run_id": cap["run_id"],
            "contract_sha256": contract_sha256(config),
            "experiment_status": "INVALID",
            "passed": None,  # model capability NOT evaluated on an invalid schedule
            "schedule": validation,
            "records": [asdict(r) for r in records],
        }
    report = summarise(config, records, fixtures, sender.bodies, route_of)
    return {
        "run_id": cap["run_id"],
        "experiment_status": "VALID",
        "schedule": validation,
        **report,
    }


def validate_schedule(
    config: BoundedClassifierConfig,
    records: Sequence[CallRecord],
    ordinary: Sequence[tuple[str, str, dict[str, Any]]],
    violation: str | None,
    burst_started: float | None,
    pacer: _Pacer,
) -> dict[str, Any]:
    """Verifies the observed schedule against the pre-registered one (no weakening)."""
    schedule = config.capability["schedule"]
    gap_s, tol = schedule["ordinary_call_gap_seconds"], schedule["gap_tolerance_seconds"]
    calls = [r for r in records if r.step != "burst"]
    gaps = [r.gap_before_s for r in calls if r.gap_before_s is not None]
    expected_order = [(step, case) for step, case, _ in ordinary]
    observed_order = [(r.step, r.case_id) for r in calls]
    gated = [r.sequence for r in calls if r.step.split(":")[0] in GATED_STEPS]
    diagnostic = [r.sequence for r in calls if r.step.split(":")[0] in DIAGNOSTIC_STEPS]
    first_start = calls[0].started_s if calls else None
    pre_burst = (
        round(burst_started - pacer.previous_end, 6)
        if burst_started is not None and pacer.previous_end is not None
        else None
    )
    failures = [violation] if violation else []
    if observed_order != expected_order:
        failures.append("ordinary calls missing or out of the pre-registered order")
    if gaps and min(gaps) < gap_s - tol:
        failures.append(f"observed gap {min(gaps)}s < {gap_s}s")
    if len(gaps) != max(len(calls) - 1, 0):
        failures.append("gap not recorded for every ordinary call after the first")
    if first_start is None or first_start < schedule["initial_quiet_seconds"] - tol:
        failures.append("initial quiet period not honoured")
    if gated and diagnostic and max(gated) > min(diagnostic):
        failures.append("a diagnostic call preceded a gated call")
    if pre_burst is None or pre_burst < schedule["pre_burst_quiet_seconds"] - tol:
        failures.append("pre-burst quiet period not honoured")
    return {
        "valid": not failures,
        "failures": failures,
        "configured": dict(schedule),
        "observed": {
            "ordinary_calls": len(calls),
            "min_gap_s": min(gaps) if gaps else None,
            "max_gap_s": max(gaps) if gaps else None,
            "initial_quiet_s": first_start,
            "pre_burst_quiet_s": pre_burst,
            "gated_sequences": gated,
            "diagnostic_sequences": diagnostic,
        },
        "retry_after_over_gap": {  # recorded only; never acted on
            "header": [
                r.sequence for r in records if (_seconds(r.retry_after_header) or 0) > gap_s
            ],
            "body": [r.sequence for r in records if (_seconds(r.retry_after_body) or 0) > gap_s],
        },
    }


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
                "retry_after_header": [r.retry_after_header for r in burst if r.status == 429],
                "retry_after_body": [r.retry_after_body for r in burst if r.status == 429],
                "contract_ok": sum(r.ok for r in burst),
                "note": "deliberate burst; excluded from the operational failure rate",
            },
        },
        "malformed_fixtures": fixtures,
        "fail_reasons": dict(Counter(r.fail_reason for r in records if r.fail_reason)),
        "records": [asdict(r) for r in records],
    }
