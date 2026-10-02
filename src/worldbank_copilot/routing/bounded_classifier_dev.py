"""Candidate C DEV evaluation (Phase 9D): C1 / C2 / hybrid / repeatability, DEV split ONLY.

Approved design (2026-10-02). TEST is never evaluated: every entry point filters the frozen
dataset to `split == "dev"` behind `assert_split_allowed("dev")`, and no TEST case id can
enter a request, a prediction artifact or a report.

* **C1** (production boundary, authoritative): DEV cases for which the frozen 9B.2
  `RoutingService` returns SEMANTIC_CLASSIFICATION_REQUIRED. Derived mechanically with the
  real service and a context factory that raises at the first data access - which in
  `RoutingService.handle` happens only AFTER the semantic boundary - so no data is needed.
  Drift-checked against the recorded 9B.2 baseline; must equal the locked expectation.
* **C2** (shadow, DIAGNOSTIC ONLY): DEV cases with a RESOLVED project in the frozen 9B.2
  run, a non-refusal reviewed intent and an executable expected route.
* **Repeats**: C1 x `c1_repeats` (hard gate) and four mechanically chosen C2 cases x
  `c2_repeats` (diagnostic). Main-pass predictions alone are scored.
* **Hybrid**: recorded main-pass predictions replayed through the real `RoutingService`
  (`RecordedClassifier`); Candidate A = the frozen 9B.2 router alone.

Outcomes are mutually exclusive, decided by precedence:
INVALID > REJECTED_SAFETY > INCONCLUSIVE_OPERATIONAL > REJECTED > ACCEPTED_FOR_NEXT_STAGE.
A failure has one kind (see bounded_semantic): OPERATIONAL failures make a case NOT
EVALUABLE for quality gates rather than also counting as quality failures.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from worldbank_copilot.routing.bounded_classifier import (
    CANDIDATE,
    BoundedClassifierConfig,
    ClassifierOutput,
    contract_sha256,
    to_decision,
)
from worldbank_copilot.routing.bounded_classifier_probe import Pacer, ScheduleViolation
from worldbank_copilot.routing.entities import resolve_project, validate_input
from worldbank_copilot.routing.models import Intent, SemanticDecision
from worldbank_copilot.routing.semantic import SEMANTIC_INTENTS, SemanticQueryContext
from worldbank_copilot.routing.semantic_eval import assert_split_allowed
from worldbank_copilot.routing.service import RoutingService
from worldbank_copilot.routing.temporal import parse_temporal

SEMANTIC_REQUIRED = "SEMANTIC_CLASSIFICATION_REQUIRED"
EXECUTABLE_ROUTES = ("STRUCTURED", "DOCUMENT", "INVESTIGATION")
OUTCOMES = (
    "INVALID",
    "REJECTED_SAFETY",
    "INCONCLUSIVE_OPERATIONAL",
    "REJECTED",
    "ACCEPTED_FOR_NEXT_STAGE",
)
ARTIFACT_PREFIX = "candidate_c_dev_predictions"
# keys that would mean model text / reasoning was persisted into an artifact
FORBIDDEN_ARTIFACT_KEYS = frozenset(
    {"content", "message", "messages", "reasoning", "reasoning_content", "summary", "question"}
)

ACCEPTANCE_RULE: dict[str, Any] = {
    "outcome_precedence": list(OUTCOMES),
    "INVALID": [
        "frozen dataset / freeze-manifest / 9B.2-baseline hash mismatch",
        "protocol-lock mismatch (recomputed lock or artifact lock hash)",
        "C1 population drift (mechanical derivation != locked C1)",
        "Candidate A drift (recomputed 9B.2 DEV outputs != recorded)",
        "schedule integrity violation (quiet period, 5.0 s END->START gap, order, completeness)",
        "artifact/protocol integrity failure (call plan, contract hash, non-DEV case id, "
        "test_evaluated not false)",
    ],
    "REJECTED_SAFETY": [
        "a requested case is outside the locked C2 population or its project was not RESOLVED",
        "classifier invoked outside the production semantic boundary (invoked set != C1)",
        "project / authorisation / temporal context differs from Candidate A or from the "
        "request context",
        "tool/execution fields in any classifier request",
        "an output that neither parsed to an allowed label nor failed closed",
        "reasoning or model text persisted in the artifact",
    ],
    "INCONCLUSIVE_OPERATIONAL": [
        "DEV operational failure rate > max_operational_failure_rate over all scheduled calls",
        "any C1 call (main or repeat) with an OPERATIONAL failure (transport, timeout, 429, "
        "5xx, other non-200)",
        "successful-call inference p95 > max_inference_p95_seconds (service performance)",
    ],
    "REJECTED": [
        "any HTTP 200 reply that violates the frozen classifier output contract/schema",
        "C1: wrong intent, ABSTAIN, wrong final hybrid route, or any wrong executable route",
        "C1 repeatability: not c1_repeats/c1_repeats identical and equal to the main prediction",
        "hybrid: route-correct != Candidate A + |C1|, a change that is not a correction, any "
        "regression, or any non-C1 final output different from Candidate A",
    ],
    "ACCEPTED_FOR_NEXT_STAGE": (
        "every applicable hard gate passes. Means ONLY eligibility for the one frozen ambiguity "
        "probe run, after human review of the C2 diagnostics - not production approval, not a "
        "TEST pass, not Phase 9 complete"
    ),
    "not_evaluable": (
        "a C1 case with an OPERATIONAL failure is NOT EVALUABLE for C1/hybrid/repeat quality "
        "gates (it is covered by INCONCLUSIVE_OPERATIONAL, never also REJECTED)"
    ),
    "c2": "DIAGNOSTIC ONLY - no numeric acceptance threshold; human review before the probe",
    "scoring": "main-pass predictions only; repeats never replace a main prediction",
}


class PopulationDrift(RuntimeError):
    """A mechanically derived population differs from the locked expectation (INVALID)."""


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def acceptance_rule_sha256() -> str:
    return _sha(ACCEPTANCE_RULE)


def artifact_name(run_id: str) -> str:
    return f"{ARTIFACT_PREFIX}__{run_id}.json"


# -- frozen DEV inputs ----------------------------------------------------------------------


def dev_cases(dataset: Any) -> list[Any]:
    """The frozen DEV cases only (the single place the dataset is filtered)."""
    assert_split_allowed("dev")
    return [c for c in dataset.cases if c.split == "dev"]


def dev_baseline(path: Path, dev_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Recorded 9B.2 rows for DEV ids only; other rows are dropped on load, never returned."""
    wanted = set(dev_ids)
    rows = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["results"]
    return {r["case_id"]: r for r in rows if r["case_id"] in wanted}


def dev_guards(dataset: Any, freeze_manifest: dict[str, Any], routing: Any) -> list[str]:
    """Frozen-ground-truth preconditions (DEV ids only); returns failures (empty = pass)."""
    dev = sorted(c.case_id for c in dev_cases(dataset))
    failures = []
    if dev != freeze_manifest["dev_case_ids"] or len(dev) != 29:
        failures.append("DEV ids differ from the frozen 9C manifest (29)")
    if not (dataset.review_status == "REVIEWED" and dataset.split_status == "FROZEN"):
        failures.append("dataset is not REVIEWED / split FROZEN")
    if {c.label_status for c in dev_cases(dataset)} != {"REVIEWED"}:
        failures.append("a DEV label is not REVIEWED")
    if routing.settings.router_version != freeze_manifest["router_version"]:
        failures.append("router version differs from the frozen 9B.2")
    return failures


# -- C1: the production semantic boundary ---------------------------------------------------


class PastSemanticBoundary(Exception):
    """Raised by the boundary context factory: the request passed the semantic boundary."""


def _boundary_context(request_id: str) -> Any:
    raise PastSemanticBoundary(request_id)


def boundary_service(routing: Any, index: Any, semantic: Any = None) -> RoutingService:
    """The frozen production service, unable to touch data: its context factory raises."""
    return RoutingService(routing, index, None, _boundary_context, semantic=semantic)


@dataclass(frozen=True)
class BoundaryRow:
    case_id: str
    past_boundary: bool  # True = rules resolved the intent; data access would follow
    route: str | None
    reason_code: str | None


def semantic_boundary(
    service: RoutingService, cases: Sequence[Any], all_projects: tuple[str, ...]
) -> dict[str, BoundaryRow]:
    rows = {}
    for case in cases:
        try:
            result = service.handle(case.question, case.access(all_projects), case.case_id)
        except PastSemanticBoundary:
            rows[case.case_id] = BoundaryRow(case.case_id, True, None, None)
        else:
            d = result.decision
            rows[case.case_id] = BoundaryRow(case.case_id, False, d.route.value, d.reason_code)
    return rows


def derive_c1(
    boundary: dict[str, BoundaryRow], baseline: dict[str, dict[str, Any]]
) -> tuple[list[str], list[str]]:
    """(C1 ids, drift messages) - C1 = SEMANTIC_CLASSIFICATION_REQUIRED at the boundary."""
    c1, drift = [], []
    for case_id, row in sorted(boundary.items()):
        recorded = baseline[case_id]
        if row.past_boundary:
            if recorded["route"] == SEMANTIC_REQUIRED or recorded["project_status"] != "RESOLVED":
                drift.append(f"{case_id}: passed the boundary but recorded {recorded['route']}")
        elif (row.route, row.reason_code) != (recorded["route"], recorded["reason_code"]):
            drift.append(f"{case_id}: {row.route}/{row.reason_code} != recorded")
        if row.route == SEMANTIC_REQUIRED:
            c1.append(case_id)
    return c1, drift


# -- C2 and the repeat set --------------------------------------------------------------------

SEMANTIC_LABELS = frozenset(i.value for i in SEMANTIC_INTENTS)


def _expected_intent(case: Any) -> str | None:
    return case.expected.intents[0].value if case.expected.intents else None


def derive_c2(cases: Sequence[Any], baseline: dict[str, dict[str, Any]]) -> list[str]:
    """Shadow population: RESOLVED in 9B.2 + non-refusal reviewed intent + executable route."""
    return sorted(
        c.case_id
        for c in cases
        if baseline[c.case_id]["project_status"] == "RESOLVED"
        and c.expected.intents
        and all(i.value in SEMANTIC_LABELS for i in c.expected.intents)
        and c.expected.route in EXECUTABLE_ROUTES
    )


def derive_c2_repeat(
    cases: Sequence[Any], baseline: dict[str, dict[str, Any]], c1: Sequence[str], c2: Sequence[str]
) -> list[str]:
    """Lowest case id per slot (STRUCTURED, DOCUMENT, INVESTIGATION, distinct intent) among C2
    cases that are not C1, not already chosen, and were resolved by the 9B.2 rules."""
    by_id = {c.case_id: c for c in cases}
    pool = [
        cid for cid in sorted(c2) if cid not in c1 and baseline[cid]["route"] != SEMANTIC_REQUIRED
    ]
    chosen: list[str] = []
    for route in EXECUTABLE_ROUTES:
        pick = next(cid for cid in pool if cid not in chosen and by_id[cid].expected.route == route)
        chosen.append(pick)
    used = {_expected_intent(by_id[cid]) for cid in chosen}
    chosen.append(
        next(cid for cid in pool if cid not in chosen and _expected_intent(by_id[cid]) not in used)
    )
    return chosen


def call_plan(
    c1: Sequence[str],
    c2: Sequence[str],
    c2_repeat: Sequence[str],
    c1_repeats: int,
    c2_repeats: int,
) -> list[dict[str, Any]]:
    """Main pass (C1 first, then the rest of C2 by id), then interleaved repeat rounds."""
    entries = [(cid, "main", 0) for cid in [*c1, *[c for c in sorted(c2) if c not in c1]]]
    for round_no in range(1, max(c1_repeats, c2_repeats) + 1):
        ids = [
            *(c1 if round_no <= c1_repeats else []),
            *(c2_repeat if round_no <= c2_repeats else []),
        ]
        entries += [(cid, "repeat", round_no) for cid in ids]
    c1_set = set(c1)
    return [
        {
            "sequence": i,
            "case_id": cid,
            "kind": kind,
            "round": round_no,
            "population": "C1" if cid in c1_set else "C2",
        }
        for i, (cid, kind, round_no) in enumerate(entries, start=1)
    ]


def derive_populations(
    config: BoundedClassifierConfig,
    cases: Sequence[Any],
    baseline: dict[str, dict[str, Any]],
    boundary: dict[str, BoundaryRow],
) -> dict[str, Any]:
    """All populations, mechanically; raises PopulationDrift if they differ from the lock
    expectations in the configuration (the derivation is authoritative)."""
    dev = config.dev_evaluation
    c1, drift = derive_c1(boundary, baseline)
    if drift:
        raise PopulationDrift(f"9B.2 drift: {drift}")
    if c1 != dev["expected_c1"]:
        raise PopulationDrift(f"C1 {c1} != locked expectation {dev['expected_c1']}")
    c2 = derive_c2(cases, baseline)
    if len(c2) != dev["expected_c2_size"] or not set(c1) <= set(c2):
        raise PopulationDrift(f"C2 size {len(c2)} / C1 not inside C2")
    repeat = derive_c2_repeat(cases, baseline, c1, c2)
    if repeat != dev["expected_c2_repeat"]:
        raise PopulationDrift(f"C2 repeat set {repeat} != {dev['expected_c2_repeat']}")
    return {
        "c1": c1,
        "c2": c2,
        "c2_repeat": repeat,
        "c1_repeats": dev["c1_repeats"],
        "c2_repeats": dev["c2_repeats"],
    }


# -- protocol lock ----------------------------------------------------------------------------


def build_lock(
    config: BoundedClassifierConfig,
    populations: dict[str, Any],
    *,
    dataset_sha256_lf: str,
    freeze_manifest_sha256_lf: str,
    baseline_sha256_lf: str,
    router_version: str,
) -> dict[str, Any]:
    dev = config.dev_evaluation
    plan = call_plan(
        populations["c1"],
        populations["c2"],
        populations["c2_repeat"],
        populations["c1_repeats"],
        populations["c2_repeats"],
    )
    return {
        "candidate": CANDIDATE,
        "run_id": dev["run_id"],
        "endpoint": config.endpoint.preferred,
        "required_parameters": config.request.required_parameters,
        "contract_sha256": contract_sha256(config),
        "routing_dataset_sha256_lf": dataset_sha256_lf,
        "freeze_manifest_sha256_lf": freeze_manifest_sha256_lf,
        "baseline_9b2_sha256_lf": baseline_sha256_lf,
        "router_version": router_version,
        "ambiguity_probe_sha256_lf": dev["ambiguity_probe_sha256_lf"],  # reference only
        "populations": populations,
        "call_plan": plan,
        "call_plan_sha256": _sha(plan),
        "dev_scheduled_calls": len(plan),
        "c1_calls": sum(e["population"] == "C1" for e in plan),
        "schedule": dev["schedule"],
        "gates": dev["gates"],
        "candidate_a_dev_route_correct": dev["candidate_a_dev_route_correct"],
        "required_hybrid_route_correct": dev["candidate_a_dev_route_correct"]
        + len(populations["c1"]),
        "acceptance_rule": ACCEPTANCE_RULE,
        "acceptance_rule_sha256": acceptance_rule_sha256(),
        "split_evaluated": "dev",
        "test_evaluated": False,
    }


def lock_sha256(lock: dict[str, Any]) -> str:
    return _sha(lock)


class ProtocolIntegrityError(RuntimeError):
    """A frozen input (dataset, manifest, baseline, split) is not as frozen (INVALID)."""


@dataclass(frozen=True)
class DevProtocol:
    lock: dict[str, Any]
    cases: list[Any]  # DEV only
    baseline: dict[str, dict[str, Any]]  # recorded 9B.2 rows, DEV only


def derive_lock(
    config: BoundedClassifierConfig,
    routing: Any,
    index: Any,
    dataset: Any,
    repo_root: Path,
    all_projects: tuple[str, ...],
) -> DevProtocol:
    """Everything the lock pins, derived mechanically from frozen inputs (no model, no data,
    no TEST rows, no ambiguity-probe access). Raises on any integrity failure or drift."""
    from worldbank_copilot.routing.semantic_eval import canonical_sha256

    evaluation = Path(repo_root) / "evaluation"
    freeze = json.loads((evaluation / "routing_freeze_9c.json").read_text(encoding="utf-8"))
    dataset_sha = canonical_sha256(evaluation / "routing_cases.yaml")
    if dataset_sha != freeze["dataset_sha256_lf"]:
        raise ProtocolIntegrityError("routing dataset hash differs from the frozen 9C hash")
    if failures := dev_guards(dataset, freeze, routing):
        raise ProtocolIntegrityError(f"frozen-ground-truth guards failed: {failures}")
    cases = dev_cases(dataset)
    baseline_path = evaluation / "routing_baseline_9B.2.yaml"
    baseline = dev_baseline(baseline_path, [c.case_id for c in cases])
    boundary = semantic_boundary(boundary_service(routing, index), cases, all_projects)
    populations = derive_populations(config, cases, baseline, boundary)
    lock = build_lock(
        config,
        populations,
        dataset_sha256_lf=dataset_sha,
        freeze_manifest_sha256_lf=canonical_sha256(evaluation / "routing_freeze_9c.json"),
        baseline_sha256_lf=canonical_sha256(baseline_path),
        router_version=routing.settings.router_version,
    )
    return DevProtocol(lock, cases, baseline)


# -- request contexts -------------------------------------------------------------------------


class CapturingClassifier:
    """Semantic hook that records the exact production context and abstains (no data use)."""

    name = "CAPTURE"

    def __init__(self) -> None:
        self.contexts: dict[str, SemanticQueryContext] = {}

    def classify(self, context: SemanticQueryContext) -> SemanticDecision:
        self.contexts[context.request_id] = context
        return SemanticDecision(classifier=self.name, version="0", abstain=True, reason="capture")


def shadow_context(
    case: Any, routing: Any, index: Any, all_projects: tuple[str, ...]
) -> SemanticQueryContext:
    """The context production WOULD send, built with the service's own stage functions."""
    inp = validate_input(case.question, routing)
    project = resolve_project(inp.normalized, case.access(all_projects), index)
    temporal = parse_temporal(inp.normalized)
    return SemanticQueryContext(
        case.case_id, inp.normalized, project.project_id or "", temporal.kind.value, ()
    )


def request_contexts(
    routing: Any,
    index: Any,
    cases: Sequence[Any],
    populations: dict[str, Any],
    all_projects: tuple[str, ...],
) -> dict[str, SemanticQueryContext]:
    """C1: the context captured from the real service; other C2 cases: shadow contexts.
    Raises if the shadow builder disagrees with production on any C1 case."""
    by_id = {c.case_id: c for c in cases}
    capture = CapturingClassifier()
    service = boundary_service(routing, index, semantic=capture)
    for cid in populations["c1"]:
        service.handle(by_id[cid].question, by_id[cid].access(all_projects), cid)
    contexts = {}
    for cid in populations["c2"]:
        shadow = shadow_context(by_id[cid], routing, index, all_projects)
        if cid in populations["c1"]:
            real = capture.contexts[cid]
            if _context_key(real) != _context_key(shadow):
                raise PopulationDrift(f"{cid}: shadow context differs from production")
            contexts[cid] = real
        else:
            contexts[cid] = shadow
    return contexts


def _context_key(context: SemanticQueryContext) -> dict[str, str]:
    return {
        "question_sha256": hashlib.sha256(context.question.encode("utf-8")).hexdigest(),
        "project_id": context.project_id,
        "temporal_kind": context.temporal_kind,
    }


# -- DEV prediction run (the only network stage) ----------------------------------------------


def run_dev_predictions(
    config: BoundedClassifierConfig,
    lock: dict[str, Any],
    contexts: dict[str, SemanticQueryContext],
    classifier: Any,  # bounded_semantic.BoundedSemanticClassifier (call(context))
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Executes the locked call plan sequentially with the fixed schedule. No retries, no
    backoff, no concurrency; a schedule the pacer cannot honour stops BEFORE the call."""
    assert_split_allowed("dev")
    schedule, plan = lock["schedule"], lock["call_plan"]
    pacer = Pacer(
        schedule["ordinary_call_gap_seconds"], schedule["gap_tolerance_seconds"], clock, sleep
    )
    pacer.t0 = clock()
    calls: list[dict[str, Any]] = []
    violation = None
    try:
        log(f"quiet {schedule['initial_quiet_seconds']}s")
        quiet = pacer.wait_until(pacer.t0 + schedule["initial_quiet_seconds"]) - pacer.t0
        if quiet < schedule["initial_quiet_seconds"] - pacer.tolerance_s:
            raise ScheduleViolation(f"initial quiet {quiet:.6f}s not honoured")
        for entry in plan:
            start, gap = pacer.before_ordinary_call(entry["sequence"])
            _, call = classifier.call(contexts[entry["case_id"]])
            end = clock()
            pacer.previous_end = end
            calls.append(
                {
                    **entry,
                    "started_s": round(start - pacer.t0, 6),
                    "ended_s": round(end - pacer.t0, 6),
                    "gap_before_s": None if gap is None else round(gap, 6),
                    **asdict(call),
                }
            )
            log(f"{entry['sequence']}/{len(plan)} {entry['case_id']} {entry['kind']}")
    except ScheduleViolation as exc:
        violation = str(exc)
    return {
        "candidate": CANDIDATE,
        "run_id": lock["run_id"],
        "lock_sha256": lock_sha256(lock),
        "contract_sha256": contract_sha256(config),
        "split_evaluated": "dev",
        "test_evaluated": False,
        "dev_scheduled_calls": len(plan),
        "c1_calls": lock["c1_calls"],
        "contexts": {cid: _context_key(ctx) for cid, ctx in sorted(contexts.items())},
        "schedule": validate_dev_schedule(lock, calls, violation),
        "calls": calls,
    }


def validate_dev_schedule(
    lock: dict[str, Any], calls: Sequence[dict[str, Any]], violation: str | None
) -> dict[str, Any]:
    schedule, plan = lock["schedule"], lock["call_plan"]
    gap_s, tol = schedule["ordinary_call_gap_seconds"], schedule["gap_tolerance_seconds"]
    gaps = [c["gap_before_s"] for c in calls if c["gap_before_s"] is not None]
    failures = [violation] if violation else []
    if len(calls) != len(plan):
        failures.append(f"{len(calls)} of {len(plan)} planned calls made")
    if [(c["sequence"], c["case_id"]) for c in calls] != [
        (e["sequence"], e["case_id"]) for e in plan[: len(calls)]
    ]:
        failures.append("calls out of the locked order")
    if calls and calls[0]["started_s"] < schedule["initial_quiet_seconds"] - tol:
        failures.append("initial quiet period not honoured")
    if gaps and min(gaps) < gap_s - tol:
        failures.append(f"observed gap {min(gaps)}s < {gap_s}s")
    if len(gaps) != max(len(calls) - 1, 0):
        failures.append("gap not recorded for every call after the first")
    return {
        "valid": not failures,
        "failures": failures,
        "configured": dict(schedule),
        "observed": {
            "calls": len(calls),
            "initial_quiet_s": calls[0]["started_s"] if calls else None,
            "min_gap_s": min(gaps) if gaps else None,
            "max_gap_s": max(gaps) if gaps else None,
        },
    }


# -- hybrid replay (local, through the real RoutingService) -----------------------------------


class RecordedClassifier:
    """Semantic hook replaying the recorded MAIN-PASS prediction for the case (by request id).
    Records every invocation, so the replay can prove it ran only at the boundary."""

    name = CANDIDATE

    def __init__(self, artifact: dict[str, Any], route_of: dict[Intent, str], version: str):
        self.main = {c["case_id"]: c for c in artifact["calls"] if c["kind"] == "main"}
        self.route_of, self.version = route_of, version
        self.invoked: list[tuple[str, dict[str, str]]] = []

    def classify(self, context: SemanticQueryContext) -> SemanticDecision:
        self.invoked.append((context.request_id, _context_key(context)))
        call = self.main.get(context.request_id)
        if call is None or call["label"] is None:
            reason = call["fail_reason"] if call else "NOT_PREDICTED"
            return SemanticDecision(
                classifier=CANDIDATE, version=self.version, abstain=True, reason=reason
            )
        output = ClassifierOutput(call["label"], call.get("model"), {})
        return to_decision(output, self.route_of, version=self.version)


def replay(
    service: RoutingService,
    cases: Sequence[Any],
    all_projects: tuple[str, ...],
    baseline_of: Callable[[Any, Any], Any],
) -> dict[str, dict[str, Any]]:
    """Runs DEV cases through `service` (with or without a semantic hook) -> baseline rows."""
    assert_split_allowed("dev")
    return {
        c.case_id: baseline_of(
            c, service.handle(c.question, c.access(all_projects), c.case_id)
        ).model_dump(mode="json")
        for c in cases
        if c.split == "dev"
    }


# -- evaluation ---------------------------------------------------------------------------------

PROJECT_FIELDS = (
    "project_status",
    "project_id",
    "project_basis",
    "temporal_kind",
    "temporal_explicit",
)


def _final(row: dict[str, Any]) -> str:
    return "CLARIFY" if row["route"] == SEMANTIC_REQUIRED else row["route"]


def _p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[int(0.95 * (len(ordered) - 1))], 4)


def _forbidden_keys(obj: Any, path: str = "") -> list[str]:
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in FORBIDDEN_ARTIFACT_KEYS:
                found.append(f"{path}/{key}")
            found += _forbidden_keys(value, f"{path}/{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            found += _forbidden_keys(value, f"{path}[{i}]")
    return found


def evaluate(
    config: BoundedClassifierConfig,
    lock: dict[str, Any],
    artifact: dict[str, Any],
    cases: Sequence[Any],
    route_of: dict[Intent, str],
    *,
    recorded_a: dict[str, dict[str, Any]],
    recomputed_a: dict[str, dict[str, Any]],
    hybrid: dict[str, dict[str, Any]],
    invoked: Sequence[tuple[str, dict[str, str]]],
    recomputed_lock_sha256: str,
    recomputed_c1: Sequence[str],
) -> dict[str, Any]:
    """The verdict (one outcome, by precedence) plus every metric and diagnostic."""
    assert_split_allowed("dev")
    dev = {c.case_id: c for c in cases if c.split == "dev"}
    pops, gates, plan = lock["populations"], lock["gates"], lock["call_plan"]
    c1, c2 = list(pops["c1"]), list(pops["c2"])
    calls = artifact.get("calls", [])
    main = {c["case_id"]: c for c in calls if c["kind"] == "main"}
    labels = set(SEMANTIC_LABELS) | {config.abstain_label}
    findings: dict[str, list[str]] = {o: [] for o in OUTCOMES[:-1]}

    # 1 INVALID ------------------------------------------------------------------------------
    inv = findings["INVALID"]
    if recomputed_lock_sha256 != lock_sha256(lock):
        inv.append("recomputed protocol lock differs from the committed lock")
    if artifact.get("lock_sha256") != lock_sha256(lock):
        inv.append("artifact was produced under a different protocol lock")
    if artifact.get("contract_sha256") != lock["contract_sha256"]:
        inv.append("artifact contract hash differs from the lock")
    if list(recomputed_c1) != c1:
        inv.append(f"C1 drift: {list(recomputed_c1)} != {c1}")
    if recomputed_a != recorded_a:
        inv.append("Candidate A drift: recomputed 9B.2 DEV outputs != recorded baseline")
    if not artifact.get("schedule", {}).get("valid"):
        inv.append(f"schedule INVALID: {artifact.get('schedule', {}).get('failures')}")
    if [(c["sequence"], c["case_id"], c["kind"], c["round"]) for c in calls] != [
        (e["sequence"], e["case_id"], e["kind"], e["round"]) for e in plan
    ]:
        inv.append("artifact calls do not match the locked call plan")
    if artifact.get("test_evaluated") is not False or artifact.get("split_evaluated") != "dev":
        inv.append("artifact does not declare DEV-only / test_evaluated=false")
    if {c["case_id"] for c in calls} - set(dev) or set(artifact.get("contexts", {})) - set(dev):
        inv.append("a non-DEV case id is present in the artifact")
    if (
        sum(_final(recorded_a[cid]) == dev[cid].expected.route for cid in dev)
        != lock["candidate_a_dev_route_correct"]
    ):
        inv.append("Candidate A DEV accuracy differs from the locked value")

    # 2 REJECTED_SAFETY ----------------------------------------------------------------------
    saf = findings["REJECTED_SAFETY"]
    if keys := sorted({k for c in calls for k in c.get("tool_keys_in_request", [])}):
        saf.append(f"tool/execution fields in classifier requests: {keys}")
    outside = sorted({c["case_id"] for c in calls} - set(c2))
    if outside:
        saf.append(f"requests outside the C2 population: {outside}")
    if any(
        recorded_a[c["case_id"]]["project_status"] != "RESOLVED"
        for c in calls
        if c["case_id"] in recorded_a
    ):
        saf.append("a request was made for a case without a RESOLVED project")
    invoked_ids = [cid for cid, _ in invoked]
    if sorted(invoked_ids) != sorted(c1):
        saf.append(f"classifier invoked for {sorted(invoked_ids)}, boundary is {sorted(c1)}")
    for cid, key in invoked:
        if artifact.get("contexts", {}).get(cid) != key:
            saf.append(f"{cid}: replay context differs from the request context")
    for cid in dev:
        if any(hybrid[cid][f] != recorded_a[cid][f] for f in PROJECT_FIELDS):
            saf.append(f"{cid}: project/authorisation/temporal context changed")
    for c in calls:
        bounded = (c["failure_kind"] == "NONE" and c["label"] in labels) or (
            c["failure_kind"] in ("OPERATIONAL", "CONTRACT") and c["label"] is None
        )
        if not bounded:
            saf.append(f"call {c['sequence']}: output neither bounded nor failed closed")
    if leaked := _forbidden_keys(artifact):
        saf.append(f"model/reasoning text persisted at {leaked[:5]}")

    # 3 INCONCLUSIVE_OPERATIONAL ---------------------------------------------------------------
    ops = findings["INCONCLUSIVE_OPERATIONAL"]
    operational = [c for c in calls if c["failure_kind"] == "OPERATIONAL"]
    rate = round(len(operational) / len(plan), 4) if plan else 0.0
    if rate > gates["max_operational_failure_rate"]:
        ops.append(f"operational failure rate {rate} > {gates['max_operational_failure_rate']}")
    c1_op = sorted({c["case_id"] for c in operational if c["case_id"] in c1})
    if c1_op:
        ops.append(f"C1 operational failures: {c1_op}")
    ok_latency = [c["latency_s"] for c in calls if c["status"] == 200]
    p95 = _p95(ok_latency)
    if p95 is not None and p95 > gates["max_inference_p95_seconds"]:
        ops.append(f"inference p95 {p95}s > {gates['max_inference_p95_seconds']}s")

    # 4 REJECTED (quality) - C1 cases with operational failures are NOT EVALUABLE here --------
    rej = findings["REJECTED"]
    contract = [c for c in calls if c["failure_kind"] == "CONTRACT"]
    if contract:
        rej.append(f"contract/schema violations in calls {[c['sequence'] for c in contract]}")
    not_evaluable = set(c1_op)
    c1_rows = []
    for cid in c1:
        expected_intent, expected_route = _expected_intent(dev[cid]), dev[cid].expected.route
        call, final = main.get(cid, {}), hybrid[cid]["route"]
        reps = [c for c in calls if c["case_id"] == cid and c["kind"] == "repeat"]
        stable = (
            all(c["label"] == call.get("label") for c in reps) and len(reps) == pops["c1_repeats"]
        )
        row = {
            "case_id": cid,
            "expected_intent": expected_intent,
            "expected_route": expected_route,
            "router_reason": recorded_a[cid]["reason_code"],
            "project_id": recorded_a[cid]["project_id"],
            "temporal_kind": recorded_a[cid]["temporal_kind"],
            "predicted": call.get("label"),
            "final_hybrid_route": final,
            "abstain": call.get("label") == config.abstain_label,
            "intent_correct": call.get("label") == expected_intent,
            "route_correct": final == expected_route,
            "wrong_executable_route": final in EXECUTABLE_ROUTES and final != expected_route,
            "latency_s": call.get("latency_s"),
            "status": call.get("status"),
            "failure_kind": call.get("failure_kind"),
            "repeat_labels": [c["label"] for c in reps],
            "repeat_stable": stable,
            "evaluable": cid not in not_evaluable,
        }
        c1_rows.append(row)
        if cid in not_evaluable:
            continue
        if row["abstain"]:
            rej.append(f"{cid}: C1 ABSTAIN")
        elif not row["intent_correct"]:
            rej.append(f"{cid}: C1 intent {row['predicted']} != {expected_intent}")
        if not row["route_correct"]:
            rej.append(f"{cid}: C1 final route {final} != {expected_route}")
        if row["wrong_executable_route"]:
            rej.append(f"{cid}: wrong executable route {final}")
        if not stable:
            rej.append(f"{cid}: C1 repeats {row['repeat_labels']} vs main {row['predicted']}")

    hybrid_rows, changes = [], []
    for cid in dev:
        a_final, h_final = _final(recorded_a[cid]), hybrid[cid]["route"]
        expected = dev[cid].expected.route
        a_ok, h_ok = a_final == expected, h_final == expected
        effect = (
            "unchanged"
            if a_final == h_final
            else "correction"
            if h_ok and not a_ok
            else "regression"
            if a_ok and not h_ok
            else "changed_still_wrong"
        )
        hybrid_rows.append(
            {
                "case_id": cid,
                "expected_route": expected,
                "candidate_a": a_final,
                "hybrid": h_final,
                "hybrid_intent": hybrid[cid]["intent"],
                "effect": effect,
            }
        )
        if effect != "unchanged":
            changes.append((cid, effect))
        if cid not in c1 and hybrid[cid] != {**recorded_a[cid]}:
            rej.append(f"{cid}: non-C1 final output differs from Candidate A")
    hybrid_correct = sum(r["hybrid"] == r["expected_route"] for r in hybrid_rows)
    if [cid for cid, effect in changes if effect == "regression"]:
        rej.append(f"hybrid regressions: {[c for c, e in changes if e == 'regression']}")
    if not not_evaluable:
        if hybrid_correct != lock["required_hybrid_route_correct"]:
            rej.append(
                f"hybrid {hybrid_correct}/{len(dev)} != required "
                f"{lock['required_hybrid_route_correct']}/{len(dev)}"
            )
        if [e for _, e in changes if e != "correction"]:
            rej.append(f"hybrid changes that are not corrections: {changes}")

    outcome = next((o for o in OUTCOMES[:-1] if findings[o]), "ACCEPTED_FOR_NEXT_STAGE")
    return {
        "candidate": CANDIDATE,
        "run_id": lock["run_id"],
        "outcome": outcome,
        "findings": findings,
        "lock_sha256": lock_sha256(lock),
        "acceptance_rule_sha256": lock["acceptance_rule_sha256"],
        "contract_sha256": lock["contract_sha256"],
        "split_evaluated": "dev",
        "test_evaluated": False,
        "counts": {
            "dev_scheduled_calls": len(plan),
            "c1_calls": lock["c1_calls"],
            "calls_made": len(calls),
            "operational_failures": len(operational),
            "contract_failures": len(contract),
            "operational_failure_rate": rate,
        },
        "latency": {
            "inference_p50_s": round(statistics.median(ok_latency), 4) if ok_latency else None,
            "inference_p95_s": p95,
            "note": "inference only; the 5 s gap is request pacing, not inference time",
        },
        "c1": c1_rows,
        "hybrid": {
            "candidate_a_route_correct": lock["candidate_a_dev_route_correct"],
            "hybrid_route_correct": hybrid_correct,
            "hybrid_intent_correct": sum(
                hybrid[cid]["intent"] == _expected_intent(dev[cid]) for cid in dev
            ),
            "n": len(dev),
            "delta": hybrid_correct - lock["candidate_a_dev_route_correct"],
            "changes": [{"case_id": c, "effect": e} for c, e in changes],
            "rows": hybrid_rows,
        },
        "c2": c2_diagnostics(c2, dev, main, recorded_a, route_of, config.abstain_label),
        "repeatability": repeat_diagnostics(calls, main),
        "retry_after": [
            {
                "sequence": c["sequence"],
                "header": c["retry_after_header"],
                "body": c["retry_after_body"],
            }
            for c in calls
            if c["retry_after_header"] is not None or c["retry_after_body"] is not None
        ],
    }


def c2_diagnostics(
    c2: Sequence[str],
    dev: dict[str, Any],
    main: dict[str, dict[str, Any]],
    recorded_a: dict[str, dict[str, Any]],
    route_of: dict[Intent, str],
    abstain_label: str,
) -> dict[str, Any]:
    """Shadow DIAGNOSTICS only - never an acceptance gate; reviewed by a human."""
    rows = []
    for cid in c2:
        expected_intent, expected_route = _expected_intent(dev[cid]), dev[cid].expected.route
        label = main.get(cid, {}).get("label")
        gpt_route = route_of[Intent(label)] if label and label != abstain_label else "CLARIFY"
        det_intent, det_route = recorded_a[cid]["intent"], _final(recorded_a[cid])
        det_ok, gpt_ok = det_route == expected_route, gpt_route == expected_route
        rows.append(
            {
                "case_id": cid,
                "expected_intent": expected_intent,
                "expected_route": expected_route,
                "deterministic_intent": det_intent,
                "deterministic_route": det_route,
                "gpt_intent": label,
                "gpt_route": gpt_route,
                "gpt_intent_correct": label == expected_intent,
                "gpt_route_correct": gpt_ok,
                "agrees_with_deterministic_intent": label == det_intent,
                "deterministic_correct": det_ok,
                "effect_if_replaced": "preserve"
                if det_ok and gpt_ok
                else "regress"
                if det_ok
                else "improve"
                if gpt_ok
                else "both_wrong",
                "abstain": label == abstain_label,
                "latency_s": main.get(cid, {}).get("latency_s"),
                "status": main.get(cid, {}).get("status"),
                "failure_kind": main.get(cid, {}).get("failure_kind"),
            }
        )
    n = len(rows)
    answered = [r for r in rows if r["gpt_intent"] and not r["abstain"]]
    effects = Counter(r["effect_if_replaced"] for r in rows)
    confusions = Counter(
        (r["expected_intent"], r["gpt_intent"]) for r in rows if not r["gpt_intent_correct"]
    )
    return {
        "role": "SHADOW DIAGNOSTIC ONLY - not the production distribution, never a gate",
        "n": n,
        "intent_accuracy": round(sum(r["gpt_intent_correct"] for r in rows) / n, 4) if n else None,
        "route_accuracy": round(sum(r["gpt_route_correct"] for r in rows) / n, 4) if n else None,
        "abstain_rate": round(sum(r["abstain"] for r in rows) / n, 4) if n else None,
        "non_abstain_precision": round(
            sum(r["gpt_intent_correct"] for r in answered) / len(answered), 4
        )
        if answered
        else None,
        "coverage": round(len(answered) / n, 4) if n else None,
        "matrix_route": {
            "both_correct": effects["preserve"],
            "deterministic_only": effects["regress"],
            "gpt_only": effects["improve"],
            "both_wrong": effects["both_wrong"],
        },
        "deterministic_errors_corrected": effects["improve"],
        "deterministic_correct_broken": effects["regress"],
        "net_route_change_if_replaced": effects["improve"] - effects["regress"],
        "confusions": [
            {"expected": e, "predicted": p, "count": k} for (e, p), k in confusions.most_common()
        ],
        "patterns_for_review": [
            {"expected": e, "predicted": p, "count": k}
            for (e, p), k in confusions.most_common()
            if k >= 2
        ],
        "human_review_required": True,
        "rows": rows,
    }


def repeat_diagnostics(
    calls: Sequence[dict[str, Any]], main: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    out = []
    for cid in sorted({c["case_id"] for c in calls if c["kind"] == "repeat"}):
        labels = [c["label"] for c in calls if c["case_id"] == cid and c["kind"] == "repeat"]
        modal, modal_n = Counter(labels).most_common(1)[0]
        out.append(
            {
                "case_id": cid,
                "population": next(c["population"] for c in calls if c["case_id"] == cid),
                "main": main.get(cid, {}).get("label"),
                "repeats": labels,
                "modal_label": modal,
                "modal_share": round(modal_n / len(labels), 4),
                "label_changes": sum(a != b for a, b in zip(labels, labels[1:], strict=False)),
                "all_equal_main": all(lab == main.get(cid, {}).get("label") for lab in labels),
            }
        )
    return out


def render_markdown(report: dict[str, Any]) -> str:
    """Human-readable DEV report (case ids and labels only - no question text)."""
    h = report["hybrid"]
    lines = [
        f"# Candidate C DEV evaluation - {report['run_id']}",
        "",
        f"**Outcome: {report['outcome']}** (precedence: {' > '.join(OUTCOMES)})",
        "",
        "TEST was not evaluated. C2 is shadow diagnostic evidence only.",
        "",
        "## Findings",
        *[f"- {o}: {', '.join(f) if f else 'none'}" for o, f in report["findings"].items()],
        "",
        "## C1 (production boundary)",
        "| case | expected | predicted | final route | intent ok | route ok | repeats stable |",
        "|---|---|---|---|---|---|---|",
        *[
            f"| {r['case_id']} | {r['expected_intent']}/{r['expected_route']} | {r['predicted']} "
            f"| {r['final_hybrid_route']} | {r['intent_correct']} | {r['route_correct']} "
            f"| {r['repeat_stable']} |"
            for r in report["c1"]
        ],
        "",
        "## Hybrid vs Candidate A",
        f"Candidate A {h['candidate_a_route_correct']}/{h['n']}; hybrid "
        f"{h['hybrid_route_correct']}/{h['n']} (delta {h['delta']:+d}); changes: "
        f"{h['changes'] or 'none'}",
        "",
        "## C2 shadow diagnostics (human review required)",
        f"intent accuracy {report['c2']['intent_accuracy']}, route accuracy "
        f"{report['c2']['route_accuracy']}, abstain rate {report['c2']['abstain_rate']}, "
        f"matrix {report['c2']['matrix_route']}, patterns {report['c2']['patterns_for_review']}",
        "",
        f"Latency: {report['latency']}. Counts: {report['counts']}.",
    ]
    return "\n".join(lines) + "\n"
