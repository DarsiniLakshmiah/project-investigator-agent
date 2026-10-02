"""Phase 9F-C bounded acceptance harness; no live work on import.

Adapters only count calls and bridge RoutingService to DocumentRetrieval without a
second document search. No routing/retrieval decisions, tuning, agents or retries.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from worldbank_copilot.common.dependency_health import check_environment
from worldbank_copilot.common.exceptions import ConfigurationError
from worldbank_copilot.ingestion.documents import load_document_manifest
from worldbank_copilot.intelligence.rules import load_rules, load_scales
from worldbank_copilot.retrieval import contract as rc
from worldbank_copilot.retrieval import pipeline as rp
from worldbank_copilot.retrieval.adaptive_eval import (
    build_lock,
    load_retrieval_production,
    lock_sha256,
    runtime_index_failures,
)
from worldbank_copilot.retrieval.adaptive_rerank import load_adaptive_config
from worldbank_copilot.retrieval.config import load_retrieval_settings
from worldbank_copilot.retrieval.embeddings import embedding_provider
from worldbank_copilot.retrieval.rerank import CrossEncoderReranker
from worldbank_copilot.retrieval.retriever import ChunkStore, Retriever
from worldbank_copilot.routing import execution as ex
from worldbank_copilot.routing.config import load_routing_config
from worldbank_copilot.routing.entities import EntityIndex
from worldbank_copilot.routing.models import (
    PROVENANCE_REQUIREMENTS,
    AccessContext,
    InvestigationPlan,
    ProjectStatus,
    Route,
)
from worldbank_copilot.routing.requirements import validate_call
from worldbank_copilot.routing.semantic_eval import canonical_sha256
from worldbank_copilot.routing.service import RoutingService
from worldbank_copilot.tools.base import ToolContext
from worldbank_copilot.tools.evidence import provenance_violations
from worldbank_copilot.tools.executor import ToolExecutor
from worldbank_copilot.tools.models import (
    ProvenanceClass,
    ToolResult,
    ToolStatus,
    iter_provenance_classes,
)
from worldbank_copilot.tools.reader import SparkTableReader
from worldbank_copilot.tools.registry import TOOL_SPECS

CASE_FILE = "evaluation/phase9_contract_cases.yaml"
LOCK_FILE = "evaluation/phase9_contract_lock.json"
RUN_ID = "9f1"
ARTIFACT_DIR = "phase9_contract"
ARTIFACT_NAME = "phase9_contract_validation__9f1.json"
EXPECTED_IDS = tuple(f"C{i:02}" for i in range(1, 11))
LOCK_9E = "480d0a1ec02e6e54aeb7c4a86d1122c002e1b0607d76ad74a9571a7d231b591e"
INDEX_NAME = "worldbank_copilot.silver.document_chunk_index_qwen3_v1"
REQUIREMENTS = [
    "requirements-databricks.txt",
    "requirements-retrieval.txt",
    "requirements-reranker.txt",
]
COUNT_NAMES = (
    "structured_tool_calls",
    "structured_reads",
    "structured_pin_calls",
    "document_search_calls",
    "first_stage_calls",
    "lexical_search_calls",
    "dense_search_calls",
    "ai_search_calls",
    "embedding_calls",
    "rerank_calls",
    "semantic_model_calls",
    "agent_calls",
)


class PreflightError(ConfigurationError):
    def __init__(self, failures: list[str], integrity: dict[str, Any] | None = None):
        self.failures = failures
        self.integrity = integrity or {}
        super().__init__("STOP before C01: " + "; ".join(failures))


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: Literal["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10"]
    query: str = Field(min_length=1, max_length=1000)
    active_project_id: str | None
    source_question_id: str | None = None
    expected_project_status: ProjectStatus
    expected_project_id: str | None
    expected_router_route: Route
    expected_execution_mode: ex.ExecutionMode
    expected_router_reason: str
    expected_execution_reason: str
    translated: bool = False
    structured_execution_allowed: bool
    retrieval_execution_allowed: bool
    investigation_plan_expected: bool
    citations_required: bool
    expected_tools: tuple[str, ...]
    allowed_provenance: tuple[ProvenanceClass, ...]

    @model_validator(mode="after")
    def expectations(self) -> Case:
        mode = ex.ROUTE_MODES[self.expected_router_route]
        if mode != self.expected_execution_mode:
            raise ValueError("router route/execution mode disagree")
        if (
            self.structured_execution_allowed != (mode == ex.ExecutionMode.EXECUTE_STRUCTURED)
            or self.retrieval_execution_allowed != (mode == ex.ExecutionMode.EXECUTE_DOCUMENT)
            or self.investigation_plan_expected != (mode == ex.ExecutionMode.PLAN_INVESTIGATION)
            or self.citations_required != self.retrieval_execution_allowed
            or self.translated
            != (self.expected_router_route == Route.SEMANTIC_CLASSIFICATION_REQUIRED)
        ):
            raise ValueError("malformed execution expectation")
        if self.translated and self.expected_execution_reason != ex.INTENT_NOT_RESOLVED:
            raise ValueError("Candidate A reason must be INTENT_NOT_RESOLVED")
        if self.expected_project_status == ProjectStatus.RESOLVED and not self.expected_project_id:
            raise ValueError("resolved project requires an id")
        if ProvenanceClass.AI_INTERPRETATION in self.allowed_provenance:
            raise ValueError("AI_INTERPRETATION is not deterministic evidence")
        if not set(self.expected_tools) <= {s.name for s in TOOL_SPECS}:
            raise ValueError("tool outside allowlist")
        return self


class CaseSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal[1]
    status: Literal["FROZEN"]
    run_id: Literal["9f1"]
    cases: tuple[Case, ...]

    @model_validator(mode="after")
    def exact_set(self) -> CaseSet:
        if tuple(c.case_id for c in self.cases) != EXPECTED_IDS:
            raise ValueError(
                "exactly 10 ordered cases C01-C10 required; no duplicates or omissions"
            )
        return self


def load_cases(path: Path) -> CaseSet:
    return CaseSet.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass
class Protocol:
    cases: CaseSet
    manifest: dict[str, Any]
    profile: rc.RetrievalProfile
    settings: Any
    integrity: dict[str, Any]


def prepare_protocol(repo: Path, *, dependency_ok: bool) -> Protocol:
    """Local frozen-file checks before opening any live reader or index."""
    failures = []

    def check(ok: bool, label: str) -> None:
        if not ok:
            failures.append(label)

    cases = load_cases(repo / CASE_FILE)
    lock = json.loads((repo / LOCK_FILE).read_text(encoding="utf-8"))
    check(dependency_ok, "DEPENDENCY_HEALTH")
    check(lock["status"] == "FROZEN" and lock["run_id"] == RUN_ID, "VALIDATION_LOCK")
    case_hash = canonical_sha256(repo / CASE_FILE)
    manifest_hash = canonical_sha256(repo / "configs/phase9_closure.yaml")
    check(case_hash == lock["case_set_sha256_lf"], "CASE_SET_HASH")
    check(manifest_hash == lock["closure_manifest_sha256_lf"], "CLOSURE_MANIFEST_HASH")
    for path, sha in lock["frozen_files_sha256_lf"].items():
        check(canonical_sha256(repo / path) == sha, "FROZEN_FILE:" + path)
    manifest = yaml.safe_load((repo / "configs/phase9_closure.yaml").read_text(encoding="utf-8"))
    rs = load_retrieval_settings(repo / "configs")
    try:
        profile = rc.load_quality_baseline(repo / "configs", rs)
    except ConfigurationError as exc:
        raise PreflightError(failures + ["BASELINE_CONFIGURATION"]) from exc
    check(profile.profile_id == "phase8_quality_baseline@1", "BASELINE_ID")
    routing = load_routing_config(repo / "configs")
    r = manifest["routing"]
    check(r["router_version"] == routing.versions["router"], "ROUTER_VERSION")
    check(r["execution_policy"] == ex.EXECUTION_POLICY, "EXECUTION_POLICY")
    check(r["semantic_fallback"] == "disabled", "SEMANTIC_FALLBACK")
    check(
        r["semantic_required_execution"]
        == {"mode": "CLARIFY", "reason_code": ex.INTENT_NOT_RESOLVED},
        "CANDIDATE_A",
    )
    check(
        set(r["clarification_reason_codes"]) == ex.CLARIFICATION_REASON_CODES, "CLARIFICATION_CODES"
    )
    freeze = json.loads((repo / r["evaluation_9c"]["freeze_manifest"]).read_text(encoding="utf-8"))
    check(
        r["evaluation_9c"]["dataset_sha256_lf"]
        == freeze["dataset_sha256_lf"]
        == canonical_sha256(repo / r["evaluation_9c"]["dataset"]),
        "9C_DATASET",
    )
    check(r["evaluation_9c"]["case_count"] == freeze["case_count"] == 80, "9C_COUNT")
    d = r["decision_9d"]
    decision = yaml.safe_load((repo / d["frozen_config"]).read_text(encoding="utf-8"))
    check(d["frozen_config_sha256_lf"] == canonical_sha256(repo / d["frozen_config"]), "9D_HASH")
    check(
        decision["status"] == d["status"] == "FROZEN"
        and decision["config"] is None
        and decision["test_evaluated"] is False,
        "9D_FROZEN",
    )
    check(
        decision["selected_strategy"] == d["selected_strategy"]
        and d["semantic_fallback_promoted"] is False,
        "9D_SELECTION",
    )
    a = manifest["adaptive_reranking"]
    check(a["promoted"] is False and a["status"] == "DIAGNOSTIC_ONLY", "ADAPTIVE_UNPROMOTED")
    committed_lock = json.loads((repo / a["protocol_lock"]).read_text(encoding="utf-8"))
    cfg = load_adaptive_config(repo / "configs")
    rebuilt = build_lock(cfg, repo, load_retrieval_production(repo / "configs"))
    check(
        committed_lock == rebuilt and lock_sha256(rebuilt) == a["protocol_lock_sha256"] == LOCK_9E,
        "9E_LOCK",
    )
    check(committed_lock["production_null"] is True, "PRODUCTION_NULL")
    check(
        committed_lock["policy_definitions_sha256"] == a["policy_definitions_sha256"],
        "POLICY_DEFINITIONS",
    )
    for path, sha in a["committed_results_sha256_lf"].items():
        check(canonical_sha256(repo / path) == sha, "9E_RESULT:" + path)
    results = json.loads((repo / "evaluation/adaptive_rerank_9e.json").read_text(encoding="utf-8"))
    selection = json.loads(
        (repo / "evaluation/adaptive_rerank_9e_live_selection.json").read_text(encoding="utf-8")
    )
    check(
        results["collection_artifact_sha256"]
        == selection["collection_artifact_sha256"]
        == a["collection_artifact_sha256"],
        "9E_COLLECTION_ID",
    )
    check(selection["point"] == a["latency_validation_point"], "9E_LATENCY_POINT")
    for key, value in manifest["retrieval"]["validated_metrics"].items():
        if key in results["points"]["P1"]:
            check(abs(value - results["points"]["P1"][key]) <= 5e-5, "RECORDED_METRIC:" + key)
    check(manifest["tools"] == {s.name: s.version for s in TOOL_SPECS}, "TOOL_VERSIONS")
    check(
        tuple(manifest["phase10_contract"]["provenance_requirements"]) == PROVENANCE_REQUIREMENTS,
        "PROVENANCE_REQUIREMENTS",
    )
    check(
        set(manifest["phase10_contract"]["provenance_origins"])
        == {p.value for p in ProvenanceClass},
        "PROVENANCE_CLASSES",
    )
    check(
        [p.name for p in (repo / "src/worldbank_copilot/agents").glob("*.py")] == ["__init__.py"],
        "NO_AGENTS",
    )
    questions = {
        q["id"]: q
        for q in yaml.safe_load(
            (repo / "evaluation/retrieval_questions.yaml").read_text(encoding="utf-8")
        )["questions"]
    }
    prior = {q["question_id"]: q for q in results["counterfactual"]["per_question"]}
    for case in cases.cases:
        if case.retrieval_execution_allowed:
            qid = case.source_question_id
            check(
                qid in questions and qid not in {"q28", "q29", "q30", "q36"},
                case.case_id + ":SOURCE",
            )
            if qid in questions:
                check(
                    case.query == questions[qid]["question"]
                    and case.expected_project_id == questions[qid]["project_id"],
                    case.case_id + ":SOURCE_QUESTION",
                )
                check(
                    prior[qid]["in_candidate_set"]
                    and 0 < (prior[qid]["first_relevant_rank"]["p1"] or 999) <= 5,
                    case.case_id + ":KNOWN_RETRIEVABLE",
                )
    integrity = {
        "closure_manifest_sha256_lf": manifest_hash,
        "case_set_sha256_lf": case_hash,
        "phase9e_lock_sha256": lock_sha256(rebuilt),
        "retrieval_profile": profile.profile_id,
        "embedding_endpoint": profile.embedding_endpoint,
        "embedding_dimension": profile.embedding_dimension,
    }
    if failures:
        raise PreflightError(failures, integrity)
    return Protocol(cases, manifest, profile, rs, integrity)


@dataclass
class Counters:
    values: Counter = field(default_factory=Counter)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.values.clear()
        self.tool_calls.clear()

    def snapshot(self) -> dict[str, int]:
        return {name: self.values[name] for name in COUNT_NAMES}


class Counted:
    """Transparent validation proxy; metadata inspection never invokes live methods."""

    def __init__(self, target: Any, methods: dict[str, tuple[str, ...]], counters: Counters):
        self.target, self.methods, self.counters = target, methods, counters

    def __getattr__(self, name: str) -> Any:
        value = getattr(self.target, name)
        if name not in self.methods:
            return value

        def call(*args: Any, **kwargs: Any) -> Any:
            # Empty table pin requests (the document tool) perform no structured I/O.
            if name != "pin" or args[0]:
                for counter in self.methods[name]:
                    self.counters.values[counter] += 1
            return value(*args, **kwargs)

        return call


class RecordingExecutor(ToolExecutor):
    def __init__(self, counters: Counters):
        super().__init__(TOOL_SPECS)
        self.counters = counters
        self.last: ToolResult | None = None

    def run(self, name, arguments, ctx, *, scope_project_id, authorized_projects=None):
        key = "document_search_calls" if name == rc.DOCUMENT_TOOL else "structured_tool_calls"
        self.counters.values[key] += 1
        self.counters.tool_calls.append(
            {
                "tool": name,
                "project_id": arguments.get("project_id"),
                "scope_project_id": scope_project_id,
            }
        )
        self.last = super().run(
            name,
            arguments,
            ctx,
            scope_project_id=scope_project_id,
            authorized_projects=authorized_projects,
        )
        return self.last


class ContractExecutor:
    """One document search through the approved interface; return its original ToolResult."""

    def __init__(self, retrieval: rc.DocumentRetrieval, backend: RecordingExecutor):
        self.retrieval, self.backend = retrieval, backend
        self.document_result: rc.RetrievalResult | None = None

    def run(self, name, arguments, ctx, *, scope_project_id, authorized_projects=None):
        if name != rc.DOCUMENT_TOOL:
            return self.backend.run(
                name,
                arguments,
                ctx,
                scope_project_id=scope_project_id,
                authorized_projects=authorized_projects,
            )
        if arguments["project_id"] != scope_project_id:
            raise ConfigurationError("document request differs from routing scope")
        self.backend.last = None
        self.document_result = self.retrieval.retrieve(
            rc.RetrievalRequest(**arguments), ctx, authorized_projects=authorized_projects or ()
        )
        if self.backend.last is not None:
            return self.backend.last
        return ToolResult(
            tool=name,
            tool_version="1",
            request_id=ctx.request_id,
            project_id=scope_project_id,
            status=ToolStatus.ERROR,
            error="retrieval integrity failed before execution",
        )


@dataclass
class Runtime:
    service: RoutingService
    retrieval: rc.DocumentRetrieval
    executor: ContractExecutor
    counters: Counters
    identity: dict[str, Any]
    authorized_projects: tuple[str, ...]


def wire_runtime(
    retriever: Retriever,
    encoder: Any,
    registry: Any,
    routing: Any,
    index: Any,
    context_factory: Callable,
    protocol: Protocol,
    identity: dict[str, Any],
) -> Runtime:
    """Validation-only instrumentation; no production module is edited."""
    counts = Counters()
    retriever.store = Counted(retriever.store, {"lexical": ("lexical_search_calls",)}, counts)
    retriever.dense = Counted(
        retriever.dense, {"search": ("dense_search_calls", "ai_search_calls")}, counts
    )
    retriever.embeddings = Counted(retriever.embeddings, {"embed": ("embedding_calls",)}, counts)
    observed = Counted(retriever, {"first_stage": ("first_stage_calls",)}, counts)
    encoder = Counted(encoder, {"score": ("rerank_calls",)}, counts)
    search = rc.build_document_search(protocol.profile, observed, encoder)
    backend = RecordingExecutor(counts)
    retrieval = rc.DocumentRetrieval(protocol.profile, search, backend)
    executor = ContractExecutor(retrieval, backend)

    def context(request_id):
        ctx = context_factory(request_id)
        ctx.reader = Counted(
            ctx.reader, {"read": ("structured_reads",), "pin": ("structured_pin_calls",)}, counts
        )
        return ctx

    service = RoutingService(routing, index, executor, context, documents=search, semantic=None)
    return Runtime(service, retrieval, executor, counts, identity, tuple(registry.project_ids))


def runtime_preflight(protocol: Protocol, runtime: Runtime) -> list[str]:
    failures = rc._search_mismatches(runtime.retrieval.search, protocol.profile)
    if runtime.service.semantic is not None:
        failures.append("SEMANTIC_FALLBACK_CONFIGURED")
    expected = protocol.manifest["retrieval"]["index"]
    identity = runtime.identity
    for key, want in (
        ("endpoint", expected["endpoint"]),
        ("index_name", INDEX_NAME),
        ("index_rows", expected["index_rows"]),
        ("corpus_rows", expected["corpus_rows"]),
        ("ready", True),
    ):
        if identity.get(key) != want:
            failures.append("RUNTIME_IDENTITY:" + key)
    failures += runtime_index_failures(identity.get("index_name"), INDEX_NAME)
    if (
        runtime.service.documents is not runtime.retrieval.search
        or runtime.service.executor is not runtime.executor
    ):
        failures.append("SERVICE_WIRING")
    if set(runtime.authorized_projects) != {"P130544", "P179039", "P506272"}:
        failures.append("PROJECT_SCOPE")
    return sorted(set(failures))


def validate_case(
    case: Case,
    result: Any,
    decision: Any,
    document: rc.RetrievalResult | None,
    counts: dict[str, int],
    tool_calls: list[dict[str, Any]],
    latency_ms: float,
) -> dict[str, Any]:
    """Acceptance only: no exact values, chunks, rankings, scores or latency gates."""
    failures, observed = [], set()
    citation_validation = "NOT_APPLICABLE"

    def check(ok, reason):
        if not ok:
            failures.append(reason)

    p = result.understanding.project
    check(result.decision.route == case.expected_router_route, "ROUTER_ROUTE")
    check(result.decision.reason_code == case.expected_router_reason, "ROUTER_REASON")
    check(decision is not None and decision.mode == case.expected_execution_mode, "EXECUTION_MODE")
    check(
        decision is not None and decision.reason_code == case.expected_execution_reason,
        "EXECUTION_REASON",
    )
    check(
        p is not None
        and p.project_id == case.expected_project_id
        and p.status == case.expected_project_status,
        "PROJECT_RESOLUTION",
    )
    check(
        decision is not None
        and decision.router_route == result.decision.route
        and decision.router_reason_code == result.decision.reason_code
        and decision.translated == case.translated,
        "ROUTER_OUTPUT_PRESERVED",
    )
    check(
        decision is not None
        and decision.clarification_options == result.decision.clarification_options,
        "CLARIFICATION_OPTIONS_PRESERVED",
    )
    check(result.semantic is None and counts["semantic_model_calls"] == 0, "SEMANTIC_MODEL_CALLED")
    check(counts["agent_calls"] == 0, "AGENT_CALLED")
    check(not result.anchor_results, "UNEXPECTED_ANCHOR_EXECUTION")
    if not case.structured_execution_allowed:
        for key in ("structured_tool_calls", "structured_reads", "structured_pin_calls"):
            check(counts[key] == 0, "FORBIDDEN:" + key)
    if not case.retrieval_execution_allowed:
        for key in (
            "document_search_calls",
            "first_stage_calls",
            "lexical_search_calls",
            "dense_search_calls",
            "ai_search_calls",
            "embedding_calls",
            "rerank_calls",
        ):
            check(counts[key] == 0, "FORBIDDEN:" + key)
        check(not result.retrieval_executed and document is None, "FORBIDDEN_RETRIEVAL_RESULT")
    check(tuple(result.executed_tools) == case.expected_tools, "EXECUTED_TOOLS")
    check(tuple(c["tool"] for c in tool_calls) == case.expected_tools, "OBSERVED_TOOLS")
    for call in tool_calls:
        check(
            call["project_id"] == call["scope_project_id"] == case.expected_project_id,
            "TOOL_CALL_SCOPE",
        )
    specs = {s.name: s for s in TOOL_SPECS}
    for envelope in result.tool_results:
        check(isinstance(envelope, ToolResult), "TYPED_TOOL_RESULT")
        check(envelope.tool in specs and envelope.tool in case.expected_tools, "TOOL_ALLOWLIST")
        check(envelope.project_id == case.expected_project_id, "TOOL_RESULT_SCOPE")
        check(
            envelope.tool in specs and envelope.tool_version == specs[envelope.tool].version,
            "TOOL_VERSION",
        )
        for item in envelope.items:
            if hasattr(item, "project_id"):
                check(item.project_id == case.expected_project_id, "TOOL_ITEM_SCOPE")
        check(not provenance_violations(envelope), "INVALID_PROVENANCE")
        observed.update(c.value for c in iter_provenance_classes(envelope.items))
        try:
            items = [
                specs[envelope.tool].item_model.model_validate(i.model_dump())
                for i in envelope.items
            ]
            ToolResult.model_validate(envelope.model_dump(exclude={"items"}) | {"items": items})
        except (ValueError, KeyError, TypeError, AttributeError):
            failures.append("INVALID_TYPED_EVIDENCE")
    check(observed <= {c.value for c in case.allowed_provenance}, "PROVENANCE_CLASS")
    if case.structured_execution_allowed:
        check(
            counts["structured_tool_calls"] == len(case.expected_tools)
            and counts["structured_reads"] > 0,
            "STRUCTURED_EXECUTION",
        )
        check(
            len(result.tool_results) == len(case.expected_tools)
            and all(e.status == ToolStatus.OK for e in result.tool_results),
            "STRUCTURED_STATUS",
        )
    if case.retrieval_execution_allowed:
        check(document is not None, "DOCUMENT_RESULT_MISSING")
        check(
            counts["document_search_calls"]
            == counts["first_stage_calls"]
            == counts["lexical_search_calls"]
            == counts["dense_search_calls"]
            == counts["ai_search_calls"]
            == 1,
            "DOCUMENT_EXECUTION",
        )
        check(result.retrieval_executed, "DOCUMENT_ROUTING_EXECUTION")
        if document is not None:
            check(document.status in set(rc.RetrievalStatus), "RETRIEVAL_STATUS_ENUM")
            check(
                document.status in (rc.RetrievalStatus.OK, rc.RetrievalStatus.NO_EVIDENCE),
                "RETRIEVAL_EXECUTION_ERROR",
            )
            check(
                document.profile == "phase8_quality_baseline@1"
                and document.top_k == 5
                and document.project_id == case.expected_project_id,
                "RETRIEVAL_PROFILE_SCOPE",
            )
            if document.status == rc.RetrievalStatus.OK:
                check(
                    document.rerank_decision is not None
                    and document.rerank_decision.policy == "always"
                    and document.rerank_decision.rerank
                    and counts["rerank_calls"] == 1,
                    "ALWAYS_RERANK",
                )
                check(
                    bool(document.evidence) and len(document.evidence) <= 5,
                    "DOCUMENT_EVIDENCE_COUNT",
                )
                citation_validation = "PASS"
                for item in document.evidence:
                    e = item.evidence
                    check(
                        e.project_id == e.citation.project_id == case.expected_project_id,
                        "CROSS_PROJECT_EVIDENCE",
                    )
                    check(
                        item.provenance_class == ProvenanceClass.DOCUMENTED_FINDING
                        and item.relevance_verified is False,
                        "DOCUMENT_PROVENANCE",
                    )
                    if not (
                        e.document_id
                        and e.citation.document_id == e.document_id
                        and e.citation.pages
                        and e.citation.source_file
                    ):
                        citation_validation = "FAIL"
                        failures.append("MISSING_CITATION")
            else:
                citation_validation = "NO_EVIDENCE"
            try:
                rc.RetrievalResult.model_validate(document.model_dump())
            except ValueError:
                failures.append("DOCUMENT_RESULT_CONTRACT")
            for envelope in result.tool_results:
                if envelope.status == ToolStatus.OK:
                    for item in envelope.items:
                        check(
                            item.candidate_k == 50
                            and item.final_k == 5
                            and item.reranker == "cross_encoder",
                            "DOCUMENT_SEARCH_PROFILE",
                        )
    check(
        (result.investigation_plan is not None) == case.investigation_plan_expected,
        "INVESTIGATION_PLAN_PRESENCE",
    )
    if case.investigation_plan_expected and result.investigation_plan is not None:
        plan = result.investigation_plan
        check(plan.executed is False, "INVESTIGATION_EXECUTED")
        check(
            plan.project_id == case.expected_project_id
            and plan.temporal is not None
            and plan.clarification_state == "NONE",
            "INVESTIGATION_SCOPE",
        )
        check(
            bool(plan.structured_calls) and bool(plan.document_retrievals),
            "INVESTIGATION_REQUIREMENTS",
        )
        check(
            plan.provenance_requirements == PROVENANCE_REQUIREMENTS
            and plan.retrieval_profile == "phase8_quality_baseline@1",
            "INVESTIGATION_HANDOFF",
        )
        check(
            all(
                c.validated
                and c.arguments.get("project_id") == case.expected_project_id
                and validate_call(c.tool, c.arguments) is None
                for c in plan.structured_calls
            ),
            "INVESTIGATION_TOOL_VALIDATION",
        )
        try:
            InvestigationPlan.model_validate(plan.model_dump())
        except ValueError:
            failures.append("INVESTIGATION_PLAN_CONTRACT")
    if not case.expected_tools:
        check(not result.tool_results, "FORBIDDEN_TOOL_RESULT")
    return {
        "case_id": case.case_id,
        "expected_router_route": case.expected_router_route.value,
        "actual_router_route": result.decision.route.value,
        "expected_execution_mode": case.expected_execution_mode.value,
        "actual_execution_mode": decision.mode.value if decision else None,
        "expected_router_reason": case.expected_router_reason,
        "actual_router_reason": result.decision.reason_code,
        "expected_execution_reason": case.expected_execution_reason,
        "actual_execution_reason": decision.reason_code if decision else None,
        "expected_project": case.expected_project_id,
        "resolved_project": p.project_id if p else None,
        "project_status": p.status.value if p else None,
        "translated": decision.translated if decision else None,
        "result_status": document.status.value if document else result.outcome.value,
        "tool_statuses": [{"tool": e.tool, "status": e.status.value} for e in result.tool_results],
        "retrieval_profile": document.profile if document else None,
        "provenance_classes": sorted(observed),
        "citation_validation": citation_validation,
        "execution_counters": counts,
        "latency_ms": round(latency_ms, 3),
        "status": "FAIL" if failures else "PASS",
        "failure_reasons": sorted(set(failures)),
    }


def execute_case(case: Case, runtime: Runtime, protocol: Protocol) -> dict[str, Any]:
    runtime.counters.reset()
    runtime.executor.document_result = None
    start = time.perf_counter()
    failures = runtime_preflight(protocol, runtime)
    if failures:
        raise PreflightError(failures)
    access = AccessContext(
        user_ref="phase9-contract-validation",
        authorized_projects=runtime.authorized_projects,
        active_project_id=case.active_project_id,
    )
    result = runtime.service.handle(case.query, access, request_id=RUN_ID + ":" + case.case_id)
    try:
        decision = ex.execution_decision(result)
    except ex.ExecutionContractError:
        decision = None
    return validate_case(
        case,
        result,
        decision,
        runtime.executor.document_result,
        runtime.counters.snapshot(),
        runtime.counters.tool_calls,
        (time.perf_counter() - start) * 1000,
    )


def summary(rows: list[dict[str, Any]], preflight_status: str) -> dict[str, Any]:
    passed = sum(r["status"] == "PASS" and not r["failure_reasons"] for r in rows)
    failed = len(rows) - passed
    complete = tuple(r["case_id"] for r in rows) == EXPECTED_IDS
    return {
        "total": 10,
        "passed": passed,
        "failed": failed,
        "not_run": 10 - len(rows),
        "preflight_status": preflight_status,
        "overall_status": "PASS"
        if preflight_status == "PASS" and complete and passed == 10
        else "FAIL",
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    rows = artifact["case_results"]
    if len(rows) > 10 or tuple(r["case_id"] for r in rows) != EXPECTED_IDS[: len(rows)]:
        raise ValueError("artifact cases are duplicated, reordered or outside the frozen set")
    if artifact["summary"] != summary(rows, artifact["preflight"]["status"]):
        raise ValueError("artifact summary does not match preflight and the ten cases")
    if any((r["status"] == "PASS") != (not r["failure_reasons"]) for r in rows):
        raise ValueError("case PASS/FAIL disagrees with failure reasons")


def run_first(
    *,
    output: Path,
    commit_sha: str,
    environment: dict[str, Any],
    prepare_local: Callable[[], Protocol],
    prepare_runtime: Callable[[Protocol], Runtime],
) -> dict[str, Any]:
    """Reserve first run exclusively; checkpoint each case; never retry or overwrite.

    Failed preflight and interrupted suites also consume run 9f1. Existing output stops
    before either callback. A crash leaves the reservation: diagnose, never rerun.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
        raise ConfigurationError("a full Git commit SHA is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x+", encoding="utf-8") as handle:
        artifact = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "timestamp": datetime.now(UTC).isoformat(),
            "commit_sha": commit_sha,
            "environment": environment,
            "integrity": {
                name: None
                for name in (
                    "closure_manifest_sha256_lf",
                    "case_set_sha256_lf",
                    "phase9e_lock_sha256",
                    "endpoint",
                    "index_name",
                    "index_rows",
                    "corpus_rows",
                    "embedding_endpoint",
                    "embedding_dimension",
                    "retrieval_profile",
                )
            },
            "preflight": {"status": "FAIL", "failures": ["RUN_NOT_COMPLETED"]},
            "case_results": [],
        }

        def checkpoint():
            artifact["summary"] = summary(artifact["case_results"], artifact["preflight"]["status"])
            validate_artifact(artifact)
            handle.seek(0)
            handle.write(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
            handle.truncate()
            handle.flush()

        checkpoint()
        try:
            protocol = prepare_local()
            artifact["integrity"].update(protocol.integrity)
            runtime = prepare_runtime(protocol)
            artifact["integrity"].update(runtime.identity)
            failures = runtime_preflight(protocol, runtime)
            if failures:
                raise PreflightError(failures)
        except Exception as exc:
            if isinstance(exc, PreflightError):
                artifact["integrity"].update(exc.integrity)
            artifact["preflight"] = {
                "status": "FAIL",
                "failures": exc.failures
                if isinstance(exc, PreflightError)
                else ["PREFLIGHT_EXCEPTION:" + type(exc).__name__],
            }
            checkpoint()
            return artifact
        artifact["preflight"] = {"status": "PASS", "failures": []}
        checkpoint()
        for case in protocol.cases.cases:
            start = time.perf_counter()
            try:
                row = execute_case(case, runtime, protocol)
            except Exception as exc:
                row = {
                    "case_id": case.case_id,
                    "expected_router_route": case.expected_router_route.value,
                    "actual_router_route": None,
                    "expected_execution_mode": case.expected_execution_mode.value,
                    "actual_execution_mode": None,
                    "expected_router_reason": case.expected_router_reason,
                    "actual_router_reason": None,
                    "expected_execution_reason": case.expected_execution_reason,
                    "actual_execution_reason": None,
                    "expected_project": case.expected_project_id,
                    "resolved_project": None,
                    "project_status": None,
                    "translated": None,
                    "result_status": "ERROR",
                    "provenance_classes": [],
                    "citation_validation": "NOT_ASSESSED",
                    "execution_counters": runtime.counters.snapshot(),
                    "latency_ms": round((time.perf_counter() - start) * 1000, 3),
                    "status": "FAIL",
                    "failure_reasons": ["CASE_EXCEPTION:" + type(exc).__name__],
                }
            artifact["case_results"].append(row)
            checkpoint()
        # Also catch drift introduced during the final case.
        try:
            failures = runtime_preflight(protocol, runtime)
        except Exception as exc:
            failures = ["FINAL_INTEGRITY_EXCEPTION:" + type(exc).__name__]
        if failures:
            artifact["preflight"] = {"status": "FAIL", "failures": failures}
        checkpoint()
        return artifact


def run_databricks_validation(spark: Any, settings: Any, *, commit_sha: str) -> dict[str, Any]:
    """Explicit LIVE entry point. Never call during local implementation/validation."""
    if settings.environment.value != "databricks":
        raise ConfigurationError("07e is a Databricks-only live entry point")
    repo = settings.repo_root
    output = Path(settings.artifact_volume_path) / ARTIFACT_DIR / ARTIFACT_NAME
    environment = {
        "kind": "databricks",
        "python": platform.python_version(),
        "commit_sha_source": "user_declared_git_folder_revision",
    }
    # Databricks Git folders may not expose Git metadata. Verify where available;
    # otherwise clearly record that the supplied revision is declared, not verified.
    try:
        actual = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL, timeout=10
            )
            .decode()
            .strip()
        )
    except (OSError, subprocess.SubprocessError):
        actual = None
    if actual is not None and actual != commit_sha:
        raise ConfigurationError("declared commit SHA differs from accessible Git HEAD")
    if actual:
        environment["commit_sha_source"] = "verified_git_head"

    def local():
        health = check_environment(repo, REQUIREMENTS, settings.config_dir)
        environment["package_versions"] = health.versions
        return prepare_protocol(repo, dependency_ok=health.ok)

    def live(protocol):
        from worldbank_copilot.common import load_project_registry

        registry = load_project_registry(settings.config_dir)
        rs = protocol.settings
        store = ChunkStore.from_table(spark, rp.names(settings, rs)["chunks"])
        dense = rp.vector_index(settings, rs)
        description = dense.describe()  # identity only, never similarity_search
        identity = {
            "endpoint": description.get("endpoint_name"),
            "index_name": description.get("name"),
            "index_rows": (description.get("status") or {}).get("indexed_row_count"),
            "corpus_rows": len(store.rows),
            "ready": (description.get("status") or {}).get("ready"),
        }
        routing = load_routing_config(settings.config_dir)
        entities = EntityIndex.build(
            registry,
            load_document_manifest(settings.config_dir / "document_manifest.yaml"),
            routing,
        )
        rules, scales = load_rules(settings.config_dir), load_scales(settings.config_dir)

        def ctx(request_id):
            return ToolContext(
                SparkTableReader.for_settings(spark, settings), registry, scales, rules, request_id
            )

        retriever = Retriever(
            store, rs, registry.project_ids, embedding_provider(rs.embeddings), dense
        )
        return wire_runtime(
            retriever,
            CrossEncoderReranker(rs.retrieval.reranker),
            registry,
            routing,
            entities,
            ctx,
            protocol,
            identity,
        )

    return run_first(
        output=output,
        commit_sha=commit_sha,
        environment=environment,
        prepare_local=local,
        prepare_runtime=live,
    )
