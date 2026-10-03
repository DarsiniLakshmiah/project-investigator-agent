"""Bounded evidence capability harness; imports perform no live work.

Four immutable fixtures, no models, retries, tuning or semantic quality judgments.
Method counters observe adapter calls, not hidden transport attempts or billing.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shlex
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from worldbank_copilot.common.exceptions import ConfigurationError
from worldbank_copilot.investigation.assembly import (
    assemble,
    merge_references,
    package_identity,
    references,
)
from worldbank_copilot.investigation.evidence import EvidenceExecutor
from worldbank_copilot.investigation.evidence_models import (
    EvidenceExecutionReport,
    EvidenceLimits,
    OperationType,
)
from worldbank_copilot.investigation.gate import AdmissionContext, AdmissionOutcome, admit
from worldbank_copilot.investigation.models import (
    BASELINE_ID,
    AssessmentStatus,
    EvidenceRequirement,
    InvestigationStatus,
    PurposeCode,
    fingerprint,
    requirement_id,
)
from worldbank_copilot.investigation.policy import AttemptStatus, InvestigationPolicy
from worldbank_copilot.retrieval import contract as rc
from worldbank_copilot.routing.models import (
    AccessContext,
    RetrievalSpec,
    TemporalScope,
    ToolCallSpec,
)
from worldbank_copilot.routing.semantic_eval import canonical_sha256
from worldbank_copilot.routing.service import RoutingService
from worldbank_copilot.validation import phase9_contract as prior

CASE_FILE = "evaluation/phase10c_evidence_cases.yaml"
LOCK_FILE = "evaluation/phase10c_evidence_lock.json"
IDS = tuple(f"D10C-{i:02}" for i in range(1, 5))
ARTIFACT_DIR = "phase10c_evidence"
SOURCE_QUESTION = "What changed and why?"
NOTEBOOK_FILE = "notebooks/09_phase10c_evidence_validation.py"


def notebook_identity(source: str) -> dict:
    """Semantic identity of the thin Python/source notebook, without executing it.

    Ignore markdown, source headers, comments and cell splits between complete
    statements. Decode Databricks MAGIC Python cells and preserve ordered Python
    AST nodes and non-Python directives. Arbitrary languages/magics fail closed.
    This describes representation equivalence, not general Python equivalence.
    """
    program = []
    cells = re.split(r"(?m)^# COMMAND -+[ \t]*$", source.replace("\r\n", "\n"))
    for cell in cells:
        lines = [re.sub(r"^# MAGIC ?", "", line) for line in cell.splitlines()]
        code = "\n".join(lines)
        significant = [
            line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")
        ]
        if not significant:
            continue
        first = significant[0]
        if first == "%md" or first.startswith("%md "):
            continue  # Markdown cells do not execute Python.
        if first == "%python":
            lines.remove(next(line for line in lines if line.strip() == "%python"))
            code = "\n".join(lines)
        elif first.startswith("%"):
            if first.split()[0] not in ("%pip", "%run"):
                raise ConfigurationError("NOTEBOOK_UNAPPROVED_MAGIC")
            try:
                program.append({"directive": shlex.split(code, comments=True)})
            except ValueError as exc:
                raise ConfigurationError("NOTEBOOK_INVALID_DIRECTIVE") from exc
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise ConfigurationError("NOTEBOOK_INVALID_PYTHON") from exc
        program.extend(
            {"python_ast": ast.dump(node, include_attributes=False)} for node in tree.body
        )
    payload = json.dumps(program, sort_keys=True, separators=(",", ":")).encode()
    return {"scheme": "databricks_wrapper_ast@1", "sha256": hashlib.sha256(payload).hexdigest()}


def verify_notebook(repo, expected):
    actual = notebook_identity((repo / NOTEBOOK_FILE).read_text(encoding="utf-8"))
    if actual != expected:
        raise ConfigurationError("NOTEBOOK_SEMANTIC_MISMATCH:" + NOTEBOOK_FILE)
    return actual


class CostAccounting(BaseModel):
    """Cost is not an acceptance criterion; infrastructure cost is not zero."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    applicable: Literal[False] = False
    phase10_model_calls: Literal[0] = 0
    actual_billed_cost: Literal["UNAVAILABLE"] = "UNAVAILABLE"
    pricing_version: Literal[None] = None
    cost_ceiling: Literal[None] = None
    reason: str = (
        "Phase 10C is a deterministic evidence-execution capability validation. "
        "The frozen retrieval interfaces do not expose trustworthy actual billed cost, "
        "and no Phase 10 model calls occur. Underlying infrastructure may incur monetary cost."
    )


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str
    project_id: str
    source_question: str
    objective: str
    structured_call: ToolCallSpec
    document_retrieval: RetrievalSpec | None
    source_question_id: str | None
    temporal_scope: TemporalScope
    expected_operation_types: tuple[OperationType, ...]
    expected_statuses: tuple[str, ...]
    expected_profile: str
    purpose_code: PurposeCode
    rationale: str


def load_cases(repo):
    data = yaml.safe_load((repo / CASE_FILE).read_text(encoding="utf-8"))
    if data["status"] != "FROZEN" or data["version"] != 1:
        raise ConfigurationError("CASE_SET_NOT_FROZEN")
    cases = tuple(Case.model_validate(c) for c in data["cases"])
    if tuple(c.case_id for c in cases) != IDS:
        raise ConfigurationError("EXACT_FOUR_ORDERED_CASES_REQUIRED")
    return cases


def prepare(repo, *, dependency_ok=True):
    if (
        Path(__file__).resolve()
        != repo.resolve() / "src/worldbank_copilot/validation/phase10c_evidence.py"
    ):
        raise ConfigurationError("EXECUTING_HARNESS_NOT_REVIEWED_SNAPSHOT")
    protocol = prior.prepare_protocol(repo, dependency_ok=dependency_ok)
    lock = json.loads((repo / LOCK_FILE).read_text(encoding="utf-8"))
    if lock["status"] != "FROZEN" or lock["case_set_sha256_lf"] != canonical_sha256(
        repo / CASE_FILE
    ):
        raise ConfigurationError("CASE_LOCK_MISMATCH")
    for name, sha in lock["files_sha256_lf"].items():
        if canonical_sha256(repo / name) != sha:
            raise ConfigurationError("CONTENT_MISMATCH:" + name)
    verify_notebook(repo, lock["notebook_semantic_identity"])
    cases = load_cases(repo)
    old = {c.source_question_id: c for c in protocol.cases.cases if c.retrieval_execution_allowed}
    for case in cases:
        if case.source_question != SOURCE_QUESTION or case.expected_profile != BASELINE_ID:
            raise ConfigurationError("FIXTURE_SOURCE_OR_PROFILE:" + case.case_id)
        requirement(case)  # validate scope/project/fingerprints before live wiring
        if case.document_retrieval is not None:
            source = old.get(case.source_question_id)
            if (
                source is None
                or source.query != case.document_retrieval.query
                or source.expected_project_id != case.project_id
            ):
                raise ConfigurationError("REVIEWED_RETRIEVAL_FIXTURE:" + case.case_id)
        types = (OperationType.STRUCTURED,) + (
            (OperationType.DOCUMENT,) if case.document_retrieval else ()
        )
        if case.expected_operation_types != types or case.expected_statuses != ("OK",) * len(types):
            raise ConfigurationError("CASE_EXPECTATIONS:" + case.case_id)
    return protocol, cases, lock


def requirement(case):
    return EvidenceRequirement(
        requirement_id=requirement_id(
            case.project_id,
            case.temporal_scope,
            case.structured_call,
            case.document_retrieval,
            case.purpose_code,
            None,
        ),
        objective=case.objective,
        project_id=case.project_id,
        temporal_scope=case.temporal_scope,
        structured_call=case.structured_call,
        document_retrieval=case.document_retrieval,
        purpose_code=case.purpose_code,
        rationale=case.rationale,
    )


class RecordingTools(prior.RecordingExecutor):
    def __init__(self, counts):
        super().__init__(counts)
        self.results = []

    def run(self, *args, **kwargs):
        result = super().run(*args, **kwargs)
        self.results.append(result.model_dump(mode="json"))
        return result


class RecordingRetrieval(rc.DocumentRetrieval):
    def __init__(self, profile, search, backend, counts):
        super().__init__(profile, search, backend)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "results", [])

    def retrieve(self, *args, **kwargs):
        self.counts.values["document_retrieval_calls"] += 1
        result = super().retrieve(*args, **kwargs)
        self.results.append(result.model_dump(mode="json"))
        return result


@dataclass
class Runtime:
    service: RoutingService
    tools: RecordingTools
    documents: RecordingRetrieval
    counts: prior.Counters
    config_dir: Path
    profile: rc.RetrievalProfile
    identity: dict


def wire(retriever, encoder, routing, entities, context_factory, protocol, identity, config_dir):
    counts = prior.Counters()
    retriever.store = prior.Counted(retriever.store, {"lexical": ("lexical_search_calls",)}, counts)
    retriever.dense = prior.Counted(
        retriever.dense, {"search": ("dense_search_calls", "ai_search_calls")}, counts
    )
    retriever.embeddings = prior.Counted(
        retriever.embeddings, {"embed": ("embedding_calls",)}, counts
    )
    observed = prior.Counted(retriever, {"first_stage": ("first_stage_calls",)}, counts)
    encoder = prior.Counted(encoder, {"score": ("rerank_calls",)}, counts)
    search = rc.build_document_search(protocol.profile, observed, encoder)
    tools = RecordingTools(counts)
    documents = RecordingRetrieval(protocol.profile, search, tools, counts)

    def context(request_id):
        ctx = context_factory(request_id)
        ctx.reader = prior.Counted(
            ctx.reader, {"read": ("structured_reads",), "pin": ("structured_pin_calls",)}, counts
        )
        return ctx

    service = RoutingService(routing, entities, tools, context, documents=search, semantic=None)
    return Runtime(service, tools, documents, counts, config_dir, protocol.profile, identity)


def counters(runtime):
    result = runtime.counts.snapshot()
    result["document_retrieval_calls"] = runtime.counts.values["document_retrieval_calls"]
    result["forbidden_execution"] = {
        "investigator": 0,
        "llm_model_serving": 0,
        "synthesis": 0,
        "critic": 0,
        "agent": 0,
        "basis": (
            "No such adapters are wired; no planning model entry point is invoked; "
            "ledger model_calls is independently checked."
        ),
    }
    result["unavailable"] = [
        "physical_transport_attempts",
        "server_internal_model_activity",
        "actual_billed_cost",
    ]
    return result


def validate_report(case, state, report, runtime):
    failures = []
    checks = {}

    def check(label, ok):
        checks[label] = bool(ok)
        if not ok:
            failures.append(label)

    package = report.package
    req = requirement(case)
    check(
        "PROJECT_ISOLATION",
        state.project_id == case.project_id == req.project_id == package.project_id
        and all(o.project_id == case.project_id for o in package.operations)
        and all(r.project_id == case.project_id for r in package.evidence_index),
    )
    check(
        "SOURCE_PLAN_UNEXECUTED",
        state.source_plan.executed is False and report.investigation.source_plan.executed is False,
    )
    check("SOURCE_STATE_PRESERVED", report.investigation.model_dump() == state.model_dump())
    check(
        "OPERATION_FINGERPRINTS",
        tuple(o.operation_id for o in package.operations) == req.operation_ids,
    )
    check(
        "EXPECTED_OPERATION_TYPES",
        tuple(o.operation_type for o in package.operations) == case.expected_operation_types,
    )
    check(
        "EXPECTED_STATUSES",
        tuple(o.status.value for o in package.operations) == case.expected_statuses
        and report.terminal_failure is None,
    )
    check(
        "NO_SEMANTIC_SATISFIED",
        package.semantic_sufficiency == "NOT_ASSESSED"
        and all(
            s.assessment.status != AssessmentStatus.SATISFIED for s in package.requirement_summaries
        ),
    )
    limits = EvidenceLimits()
    rebuilt = ()
    for op in package.operations:
        rebuilt = merge_references(rebuilt, references(op)[0], limits.max_references)
    check("SOURCE_PROVENANCE_CITATIONS_IDENTITIES_PRESERVED", rebuilt == package.evidence_index)
    check(
        "PACKAGE_FINGERPRINT_DETERMINISTIC",
        assemble(state, package.operations, rebuilt, runtime.profile, limits) == package,
    )
    check(
        "EVIDENCE_IDS_UNIQUE",
        len({e.evidence_id for e in package.evidence_index}) == len(package.evidence_index),
    )
    check(
        "MIXED_JOIN",
        {e.source_type for e in package.evidence_index} == set(case.expected_operation_types),
    )
    check(
        "REQUIREMENT_LINKS",
        all(
            o.requirement_ids == (req.requirement_id,) and o.evidence_links
            for o in package.operations
        )
        and package.requirement_summaries[0].evidence_ids
        == tuple(sorted(e.evidence_id for e in package.evidence_index)),
    )
    structured = [
        o.structured_result for o in package.operations if o.structured_result is not None
    ]
    docs = [o.retrieval_result for o in package.operations if o.retrieval_result is not None]
    check(
        "TOOL_EXECUTOR_ENVELOPE_PRESERVED",
        [r.model_dump(mode="json") for r in structured] == runtime.tools.results[:1]
        and len(structured) == 1,
    )
    check(
        "STRUCTURED_SOURCE_REFS",
        any(
            e.source is not None
            for e in package.evidence_index
            if e.source_type == OperationType.STRUCTURED
        ),
    )
    snapshots = {k: v for r in structured for k, v in r.data_snapshot.items()}
    check("SNAPSHOTS_PRESERVED", package.snapshots.table_versions == snapshots and bool(snapshots))
    check(
        "ATOMICITY_NOT_ESTABLISHED",
        package.snapshots.globally_atomic is False
        and package.snapshots.index_version is None
        and package.snapshots.corpus_identity is None,
    )
    check(
        "DOCUMENT_RETRIEVAL_ENVELOPE_PRESERVED",
        [r.model_dump(mode="json") for r in docs] == runtime.documents.results,
    )
    check(
        "FROZEN_PROFILE",
        not rc._search_mismatches(runtime.documents.search, runtime.profile)
        and all(r.profile == BASELINE_ID and r.top_k == 5 for r in docs),
    )
    check(
        "DOCUMENT_CITATIONS_CHUNK_HASH",
        all(
            e.citation is not None
            and e.citation.project_id == case.project_id
            and e.citation.pages
            and e.citation.document_label
            and e.citation.source_file
            and e.identity.get("chunk_id")
            and e.identity.get("source_hash")
            for e in package.evidence_index
            if e.source_type == OperationType.DOCUMENT
        ),
    )
    check(
        "HINTS_RECORDED_NOT_FILTER_CLAIMS",
        all(
            r.recorded_hints.applied is False
            and r.recorded_hints.temporal_scope == req.temporal_scope
            and r.recorded_hints.document_type_hints == req.document_retrieval.document_type_hints
            for r in docs
        ),
    )
    check(
        "LATEST_FILTER_APPLIED",
        case.temporal_scope.kind.value != "LATEST"
        or all(r.filters_applied and r.filters_applied.get("isr_sequences") for r in docs),
    )
    reserved = report.budget.reservations
    count = len(case.expected_operation_types)
    check(
        "BUDGET_LOGICAL_OPERATIONS",
        sum(r.operations for r in reserved) == count
        and sum(r.initial_operations for r in reserved) == count
        and sum(r.retrieval_operations for r in reserved) == len(docs),
    )
    check(
        "BUDGET_CONSUMPTIONS",
        len(report.budget.consumptions) == count
        and all(c.status == AttemptStatus.SUCCEEDED for c in report.budget.consumptions),
    )
    check(
        "ZERO_MODELS_REPAIRS",
        sum(r.model_calls for r in reserved) == 0
        and sum(r.repair_cycles for r in reserved) == 0
        and state.retry_count == 0,
    )
    observed = counters(runtime)
    check(
        "CALL_PATH_COUNTS",
        observed["structured_tool_calls"] == 1
        and observed["document_retrieval_calls"] == len(docs)
        and observed["document_search_calls"] == len(docs)
        and observed["ai_search_calls"] == len(docs)
        and observed["embedding_calls"] == len(docs)
        and observed["rerank_calls"] == len(docs)
        and observed["lexical_search_calls"] == len(docs),
    )
    return checks, sorted(failures)


def execute_case(case, runtime, policy, request_id, *, offline=False):
    runtime.counts.reset()
    runtime.tools.results.clear()
    runtime.documents.results.clear()
    access = AccessContext(
        user_ref="10c-capability-validation",
        authorized_projects=(case.project_id,),
        active_project_id=case.project_id,
    )
    source = runtime.service.handle(case.source_question, access, request_id)
    if runtime.counts.tool_calls:
        raise ConfigurationError("SOURCE_ROUTING_EXECUTED_EVIDENCE")
    context = AdmissionContext(
        request_id,
        case.source_question,
        access,
        runtime.service.config,
        runtime.service.index,
        runtime.config_dir,
        policy,
        True,
    )
    admitted = admit(source, context)
    if admitted.outcome != AdmissionOutcome.ADMITTED:
        raise ConfigurationError("FIXTURE_ADMISSION:" + admitted.reason.value)
    state = admitted.state.transition(
        append=(requirement(case),), status=InvestigationStatus.PLANNED
    )
    # This trusted harness alone excludes cost from capability acceptance. The
    # existing flag skips only the executor's pricing gate; adapters still execute
    # normally. run_attempt separately records the actual validation environment.
    executor = EvidenceExecutor(
        context, runtime.tools, runtime.documents, runtime.service.context_factory, offline=True
    )
    report = executor.execute(state)
    checks, failures = validate_report(case, state, report, runtime)
    return {
        "case_id": case.case_id,
        "project_id": case.project_id,
        "status": "FAIL" if failures else "PASS",
        "failure_reasons": failures,
        "checks": checks,
        "counters": counters(runtime),
        "report": report.model_dump(mode="json"),
        "cross_source_atomicity": "NOT_ESTABLISHED",
    }


def summary(rows, preflight):
    passed = sum(row["status"] == "PASS" for row in rows)
    return {
        "cases_executed": len(rows),
        "cases_passed": passed,
        "overall_status": "PASS" if preflight == "PASS" and len(rows) == passed == 4 else "FAIL",
    }


def validate_artifact(artifact):
    CostAccounting.model_validate(artifact["cost_accounting"])
    rows = artifact["case_results"]
    if tuple(r["case_id"] for r in rows) != IDS[: len(rows)] or len(rows) > 4:
        raise ValueError("ARTIFACT_CASE_SEQUENCE")
    if artifact["summary"] != summary(rows, artifact["preflight"]["status"]):
        raise ValueError("ARTIFACT_SUMMARY")
    if any((r["status"] == "PASS") != (not r["failure_reasons"]) for r in rows):
        raise ValueError("ARTIFACT_CASE_STATUS")
    for row in rows:
        if row["status"] != "PASS":
            continue
        if not row.get("checks") or not all(row["checks"].values()):
            raise ValueError("PASS_REQUIRES_ALL_ASSERTIONS")
        raw = deepcopy(row["report"])
        # Policy intentionally refuses caller-supplied float/string money. Rehydrate
        # only its persisted exact Decimal fields at this artifact-reading boundary.
        for ledger in (raw["budget"], raw["investigation"]["budget"]):
            policy = ledger["policy"]
            if policy["cost_ceiling"] is not None:
                policy["cost_ceiling"] = Decimal(policy["cost_ceiling"])
            for record in (*ledger["reservations"], *ledger["consumptions"]):
                for key in ("cost", "actual_cost"):
                    if record.get(key) is not None:
                        record[key] = Decimal(record[key])
        policy = raw["investigation"]["policy"]
        if policy["cost_ceiling"] is not None:
            policy["cost_ceiling"] = Decimal(policy["cost_ceiling"])
        report = EvidenceExecutionReport.model_validate(raw)
        package = report.package
        if (
            package.package_fingerprint
            != fingerprint("package", package_identity(package.model_dump(mode="json")))
            or package.project_id != row["project_id"]
            or report.investigation.source_plan.executed is not False
            or package.snapshots.globally_atomic is not False
            or any(r.model_calls or r.repair_cycles for r in report.budget.reservations)
        ):
            raise ValueError("PASS_REPORT_INTEGRITY")


def read_completed(path, *, require_final=True):
    payload = path.read_bytes()
    receipt = json.loads(prior.completion_receipt(path).read_bytes())
    if receipt != {
        "artifact": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }:
        raise ValueError("COMPLETION_RECEIPT_MISMATCH")
    artifact = json.loads(payload)
    validate_artifact(artifact)
    if (
        artifact["persistence"]["completion_receipt"] != prior.completion_receipt(path).name
        or require_final
        and not artifact["persistence"]["finalized"]
    ):
        raise ValueError("NOT_FINAL_ARTIFACT")
    return artifact


class ArtifactWriter:
    """Exclusive attempt reservation; sequential immutable sidecars and one final JSON."""

    def __init__(self, output):
        self.output, self.sequence = output, 0
        output.parent.mkdir(parents=True, exist_ok=True)
        if any(output.parent.glob(output.stem + "*")):
            raise FileExistsError("ATTEMPT_ALREADY_EXISTS; preserve history and use a new run ID")
        prior._write_new_json(
            output.with_name(output.stem + ".attempt.json"),
            {"status": "RESERVED", "artifact": output.name},
        )

    def write(self, artifact, *, final=False):
        path = (
            self.output
            if final
            else self.output.with_name(f"{self.output.stem}.checkpoint-{self.sequence:02}.json")
        )
        artifact["persistence"] = {
            "finalized": final,
            "completion_receipt": prior.completion_receipt(path).name,
            "sequence": self.sequence,
        }
        validate_artifact(artifact)
        payload = prior._write_new_json(path, artifact)
        if path.read_bytes() != payload:
            raise OSError("ARTIFACT_READBACK_MISMATCH")
        prior._write_new_json(
            prior.completion_receipt(path),
            {
                "artifact": path.name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
        )
        read_completed(path, require_final=final)
        self.sequence += 1


def run_attempt(
    repo, output, *, commit_sha, policy, prepare_runtime, dependency_ok=True, offline=False
):
    revision = prior.revision_identity(repo, commit_sha)
    writer = ArtifactWriter(output)
    artifact = {
        "schema": "phase10c_evidence_validation@1",
        "cost_accounting": CostAccounting().model_dump(mode="json"),
        "validation_environment": "OFFLINE_STAND_INS" if offline else "DATABRICKS",
        "revision": revision.model_dump(mode="json"),
        "preflight": {"status": "PENDING"},
        "case_results": [],
        "summary": summary([], "PENDING"),
        "cross_source_atomicity": "NOT_ESTABLISHED",
        "policy": policy.model_dump(mode="json"),
    }
    writer.write(artifact)
    try:
        if revision.runtime_git_status == "MISMATCH":
            raise ConfigurationError("TRUSTWORTHY_SOURCE_REVISION_MISMATCH")
        protocol, cases, lock = prepare(repo, dependency_ok=dependency_ok)
        if revision.runtime_git_status == "VERIFIED":
            # HEAD alone does not establish that the reviewed uncommitted work
            # was subsequently committed. Ordinary frozen inputs must be clean;
            # notebook representation differences are checked semantically above.
            dirty = subprocess.check_output(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                    "--",
                    *lock["files_sha256_lf"],
                    LOCK_FILE,
                ],
                cwd=repo,
                stderr=subprocess.DEVNULL,
                timeout=20,
            )
            if dirty.strip():
                raise ConfigurationError("TRUSTWORTHY_SOURCE_NOT_CLEAN_COMMITTED_SNAPSHOT")
        artifact["content_identity"] = lock
        artifact["profile"] = protocol.profile.model_dump(mode="json")
        runtime = prepare_runtime(protocol)
        if rc._search_mismatches(runtime.documents.search, protocol.profile):
            raise ConfigurationError("RUNTIME_PROFILE_MISMATCH")
        artifact["runtime_identity"] = runtime.identity
        artifact["preflight"] = {"status": "PASS"}
        artifact["summary"] = summary([], "PASS")
        writer.write(artifact)
        for case in cases:
            try:
                row = execute_case(
                    case, runtime, policy, output.stem + "__" + case.case_id, offline=offline
                )
            except Exception as exc:
                row = {
                    "case_id": case.case_id,
                    "project_id": case.project_id,
                    "status": "FAIL",
                    "failure_reasons": ["CASE_EXECUTION_EXCEPTION"],
                    "checks": {"CASE_EXECUTION_COMPLETED": False},
                    "error_class": type(exc).__name__,
                    "counters": counters(runtime),
                    "cross_source_atomicity": "NOT_ESTABLISHED",
                }
                if getattr(exc, "budget", None) is not None:
                    row["failed_budget"] = exc.budget.model_dump(mode="json")
                    row["failed_events"] = [e.model_dump(mode="json") for e in exc.events]
            artifact["case_results"].append(row)
            artifact["summary"] = summary(artifact["case_results"], "PASS")
            writer.write(artifact)
            if row["status"] != "PASS":
                break
    except Exception as exc:
        artifact["failure"] = {"error_class": type(exc).__name__, "reason": "STOP_PRESERVE_ATTEMPT"}
        if isinstance(exc, prior.PreflightError):
            artifact["failure"]["invariant_failures"] = exc.failures
        elif isinstance(exc, ConfigurationError) and re.fullmatch(r"[A-Z0-9_:./a-z-]+", str(exc)):
            artifact["failure"]["invariant_failures"] = [str(exc)]
        if artifact["preflight"]["status"] == "PENDING":
            artifact["preflight"] = {"status": "FAIL"}
        artifact["summary"] = summary(artifact["case_results"], artifact["preflight"]["status"])
    if "failure" in artifact and artifact["summary"]["overall_status"] == "PASS":
        artifact["preflight"] = {"status": "FAIL"}
        artifact["summary"] = summary(artifact["case_results"], "FAIL")
    writer.write(artifact, final=True)
    return read_completed(output)


def run_databricks_validation(spark, settings, *, commit_sha, run_id):
    """User-run LIVE boundary after immutable reservation and local checks."""
    if settings.environment.value != "databricks" or not re.fullmatch(r"10c[1-9][0-9]*", run_id):
        raise ConfigurationError("DATABRICKS_AND_NEW_10C_RUN_ID_REQUIRED")
    policy = InvestigationPolicy()

    def live(protocol):
        from worldbank_copilot.common import load_project_registry
        from worldbank_copilot.ingestion.documents import load_document_manifest
        from worldbank_copilot.intelligence.rules import load_rules, load_scales
        from worldbank_copilot.retrieval import pipeline as rp
        from worldbank_copilot.retrieval.embeddings import embedding_provider
        from worldbank_copilot.retrieval.rerank import CrossEncoderReranker
        from worldbank_copilot.retrieval.retriever import ChunkStore, Retriever
        from worldbank_copilot.routing.config import load_routing_config
        from worldbank_copilot.routing.entities import EntityIndex
        from worldbank_copilot.tools.base import ToolContext
        from worldbank_copilot.tools.reader import SparkTableReader

        rs = protocol.settings
        registry = load_project_registry(settings.config_dir)
        routing = load_routing_config(settings.config_dir)
        entities = EntityIndex.build(
            registry,
            load_document_manifest(settings.config_dir / "document_manifest.yaml"),
            routing,
        )
        store = ChunkStore.from_table(spark, rp.names(settings, rs)["chunks"])
        dense = rp.vector_index(settings, rs)
        description = dense.describe()
        identity = {
            "endpoint": description.get("endpoint_name"),
            "index_name": description.get("name"),
            "index_rows": (description.get("status") or {}).get("indexed_row_count"),
            "corpus_rows": len(store.rows),
            "ready": (description.get("status") or {}).get("ready"),
            "cross_source_atomicity": "NOT_ESTABLISHED",
        }
        expected = protocol.manifest["retrieval"]["index"]
        if (
            identity["endpoint"] != protocol.profile.vector_search_endpoint
            or identity["index_name"] != prior.INDEX_NAME
            or identity["index_rows"] != expected["index_rows"]
            or identity["corpus_rows"] != expected["corpus_rows"]
            or identity["ready"] is not True
        ):
            raise ConfigurationError("LIVE_INDEX_IDENTITY_MISMATCH")
        rules, scales = load_rules(settings.config_dir), load_scales(settings.config_dir)

        def context(request_id):
            return ToolContext(
                SparkTableReader.for_settings(spark, settings), registry, scales, rules, request_id
            )

        retriever = Retriever(
            store, rs, registry.project_ids, embedding_provider(rs.embeddings), dense
        )
        return wire(
            retriever,
            CrossEncoderReranker(rs.retrieval.reranker),
            routing,
            entities,
            context,
            protocol,
            identity,
            settings.config_dir,
        )

    from worldbank_copilot.common.dependency_health import check_environment

    health = check_environment(settings.repo_root, prior.REQUIREMENTS, settings.config_dir)
    output = (
        Path(settings.artifact_volume_path)
        / ARTIFACT_DIR
        / f"phase10c_evidence_validation__{run_id}.json"
    )
    return run_attempt(
        settings.repo_root,
        output,
        commit_sha=commit_sha,
        policy=policy,
        prepare_runtime=live,
        dependency_ok=health.ok,
    )
