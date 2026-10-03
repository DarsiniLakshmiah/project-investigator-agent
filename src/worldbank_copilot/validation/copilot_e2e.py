"""User-run immutable comparison/E2E harness. No live work on import."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from decimal import Decimal
from pathlib import Path

from worldbank_copilot.application.service import AnswerRequest
from worldbank_copilot.investigation.bounded import Decision
from worldbank_copilot.investigation.claims import Failure
from worldbank_copilot.investigation.policy import BudgetLedger, InvestigationPolicy, Reservation
from worldbank_copilot.investigation.synthesis import source_identity, validate_claims
from worldbank_copilot.validation import phase10d_models as prior
from worldbank_copilot.validation.phase10d_fixtures import draft, fixture
from worldbank_copilot.validation.phase10d_notebook_v3 import notebook_identity

CASES = "evaluation/copilot_e2e_cases.json"
LOCK = "evaluation/copilot_e2e_lock.json"
FILES = (
    "src/worldbank_copilot/investigation/bounded.py",
    "src/worldbank_copilot/application/service.py",
    "src/worldbank_copilot/application/__init__.py",
    "src/worldbank_copilot/application/guardrails.py",
    "src/worldbank_copilot/application/projection.py",
    "src/worldbank_copilot/application/runtime.py",
    "src/worldbank_copilot/application/observability.py",
    "src/worldbank_copilot/application/job_client.py",
    "src/worldbank_copilot/validation/copilot_e2e.py",
    "tests/unit/test_copilot_application.py",
    "frontend/app.py",
    "app.yaml",
    "requirements.txt",
    "requirements-app.txt",
    "requirements-copilot-runtime.txt",
    CASES,
)
NOTEBOOKS = (
    "notebooks/11_copilot_request.py",
    "notebooks/12_copilot_validation.py",
    "notebooks/13_investigator_capability.py",
)


def build_lock(root):
    root = Path(root)
    return {
        "schema": "copilot_e2e_protocol@1",
        "status": "PREREGISTERED",
        "files_sha256_lf": {name: prior.canonical_sha256(root / name) for name in FILES},
        "notebooks": {
            name: notebook_identity((root / name).read_text("utf-8")) for name in NOTEBOOKS
        },
        "phase10d_lock_sha256_lf": prior.canonical_sha256(root / prior.LOCK_FILE),
        "modes": ["A", "B", "C"],
        "hard_invariant_violations_allowed": 0,
        "architecture_selection": "HUMAN_REVIEW_REQUIRED",
        "model_quality_acceptance": "NOT_ESTABLISHED_BY_MECHANICAL_CHECKS",
    }


def prepare(root):
    root = Path(root)
    if (
        Path(__file__).resolve()
        != root.resolve() / "src/worldbank_copilot/validation/copilot_e2e.py"
    ):
        raise ValueError("EXECUTING_E2E_HARNESS_NOT_REVIEWED_SNAPSHOT")
    prior.prepare(root)
    lock = json.loads((root / LOCK).read_text("utf-8"))
    if lock != build_lock(root):
        raise ValueError("COPILOT_SOURCE_OR_PROTOCOL_MISMATCH")
    return json.loads((root / CASES).read_text("utf-8")), lock


def validate_artifact(value):
    if value.get("schema") not in ("copilot_comparison_e2e@1", "investigator_capability@1"):
        raise ValueError("UNKNOWN_ARTIFACT_SCHEMA")
    if value.get("acceptance") != "NOT_ACCEPTED":
        raise ValueError("MECHANICAL_RUN_CANNOT_CLAIM_ACCEPTANCE")
    rows = value["results"]
    identities = [(r.get("mode"), r["case_id"]) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("DUPLICATE_RESULT")
    if any(r["status"] == "PASS" and "checks" in r and not all(r["checks"].values()) for r in rows):
        raise ValueError("PASS_WITH_FAILED_CHECK")
    if value.get("contract_status") == "PASS":
        if (
            value["preflight"] != "PASS"
            or not rows
            or "failure" in value
            or any(r["status"] != "PASS" for r in rows)
        ):
            raise ValueError("INVALID_CONTRACT_PASS")
        if len(rows) != value.get("expected_results"):
            raise ValueError("INCOMPLETE_RESULT_SCHEDULE")
        if value["schema"] == "copilot_comparison_e2e@1":
            if identities != [tuple(item) for item in value["schedule"]] or not all(
                value["deterministic_checks"].values()
            ):
                raise ValueError("INVALID_SCHEDULE_OR_DETERMINISTIC_CHECKS")


def read_completed(path, *, require_final=True):
    path = Path(path)
    payload = path.read_bytes()
    receipt = json.loads(prior.prior.completion_receipt(path).read_bytes())
    if receipt != {
        "artifact": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }:
        raise ValueError("RECEIPT_MISMATCH")
    value = json.loads(payload)
    validate_artifact(value)
    if require_final and not value["persistence"]["finalized"]:
        raise ValueError("CHECKPOINT_IS_NOT_FINAL")
    return value


class Writer:
    def __init__(self, path):
        self.path, self.sequence = Path(path), 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if list(self.path.parent.glob(self.path.stem + ".*")):
            raise FileExistsError("PRESERVE_ATTEMPT_USE_NEW_ID")
        prior.prior._write_new_json(
            self.path.with_name(self.path.stem + ".attempt.json"), {"status": "RESERVED"}
        )

    def write(self, value, final=False):
        path = (
            self.path
            if final
            else self.path.with_name(f"{self.path.stem}.checkpoint-{self.sequence:03}.json")
        )
        value["persistence"] = {"finalized": final, "sequence": self.sequence}
        validate_artifact(value)
        payload = prior.prior._write_new_json(path, value)
        if path.read_bytes() != payload:
            raise OSError("READBACK_MISMATCH")
        prior.prior._write_new_json(
            prior.prior.completion_receipt(path),
            {
                "artifact": path.name,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
        )
        read_completed(path, require_final=final)
        self.sequence += 1


def deterministic_checks():
    """Actual pure contract checks, distinct from model-dependent quality results."""
    result = {}
    context = fixture()
    result["invented_citation_rejected"] = Failure.EVIDENCE_REFERENCE_INVALID in validate_claims(
        draft(context, evidence_ids=["ev_fabricated"]), context
    )
    result["prediction_rejected"] = bool(
        validate_claims(draft(context, claim_text="The project will fail."), context)
    )
    for key, decision in (
        ("unknown_action_rejected", {"action": "SQL", "justification": "invalid"}),
        (
            "project_mutation_rejected",
            {"action": "STOP", "justification": "invalid", "project_id": "P999999"},
        ),
        (
            "temporal_mutation_rejected",
            {"action": "STOP", "justification": "invalid", "date_from": "2099-01-01"},
        ),
    ):
        try:
            Decision.model_validate(decision)
            result[key] = False
        except ValueError:
            result[key] = True
    try:
        BudgetLedger(policy=InvestigationPolicy(max_model_calls=0)).reserve(
            Reservation(reservation_id="forbidden", model_calls=1)
        )
        result["model_budget_exhaustion_stops"] = False
    except ValueError:
        result["model_budget_exhaustion_stops"] = True
    return result


def checks(answer, case, *, require_mlflow):
    from worldbank_copilot.investigation.evidence import _owned

    scoped = True
    try:
        _owned(answer.model_dump(mode="json"), case["project_id"])
    except ValueError:
        scoped = False
    citations = True
    if answer.final:
        refs = {e.get("evidence_id"): e for e in answer.evidence}
        from worldbank_copilot.investigation.evidence_models import EvidenceReference

        for claim in answer.final.published_claims:
            for citation in claim.citations:
                ref = refs.get(citation.evidence_id)
                if ref is None or citation.source_identity != source_identity(
                    EvidenceReference.model_validate(ref)
                ):
                    citations = False
    trace = answer.trace
    return {
        "route": answer.route == case["expected_route"],
        "project_isolation": scoped,
        "citation_resolution": citations,
        "request_completed_safely": answer.status != "FAIL",
        "trace_generated": trace is not None
        and (not require_mlflow or bool(trace.mlflow_trace_id)),
        "one_additional_action": trace is not None and trace.investigator_actions <= 1,
        "model_budget": trace is not None
        and trace.model_calls <= InvestigationPolicy().max_model_calls,
    }


def run_validation(root, output, *, commit_sha, application_factory, require_mlflow=True):
    """Factory opens Databricks services only after reservation and integrity checks."""
    root = Path(root)
    revision = prior.prior.revision_identity(root, commit_sha)
    writer = Writer(output)
    value = {
        "schema": "copilot_comparison_e2e@1",
        "revision": revision.model_dump(mode="json"),
        "preflight": "PENDING",
        "results": [],
        "acceptance": "NOT_ACCEPTED",
        "quality_review": "REQUIRED",
        "architecture_selection": "UNDECIDED",
    }
    writer.write(value)
    try:
        if revision.runtime_git_status == "MISMATCH":
            raise ValueError("REVIEWED_REVISION_MISMATCH")
        value["stage"] = "SOURCE_PREFLIGHT"
        cases, lock = prepare(root)
        value["schedule"] = [[mode, c["case_id"]] for mode in lock["modes"] for c in cases]
        if revision.runtime_git_status == "VERIFIED":
            dirty = subprocess.check_output(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                    "--",
                    *lock["files_sha256_lf"],
                    LOCK,
                ],
                cwd=root,
                timeout=20,
            )
            if dirty.strip():
                raise ValueError("REVIEWED_SOURCE_NOT_CLEAN")
        value["preflight"] = "PASS"
        value["content_identity"] = lock
        value["deterministic_checks"] = deterministic_checks()
        writer.write(value)
        # Fresh wiring per comparison avoids cross-mode request/ledger reuse.
        for mode in lock["modes"]:
            value["stage"] = "RUNTIME_INITIALIZATION"
            app = application_factory()
            for case in cases:
                request = AnswerRequest(question=case["question"], project_id=case["project_id"])
                value["stage"] = "CASE_EXECUTION"
                answer = app.answer_question(request, mode=mode)
                tests = checks(answer, case, require_mlflow=require_mlflow)
                value["results"].append(
                    {
                        "mode": mode,
                        "case_id": case["case_id"],
                        "checks": tests,
                        "status": "PASS" if all(tests.values()) else "FAIL",
                        "response_status": answer.status,
                        "trace": answer.trace.model_dump(mode="json") if answer.trace else None,
                        "final": answer.final.model_dump(mode="json") if answer.final else None,
                    }
                )
                writer.write(value)
        value["comparison_metrics"] = comparison_metrics(value["results"])
        value["expected_results"] = len(cases) * len(lock["modes"])
        value["contract_status"] = (
            "PASS"
            if (
                len(value["results"]) == value["expected_results"]
                and all(r["status"] == "PASS" for r in value["results"])
                and all(value["deterministic_checks"].values())
            )
            else "FAIL"
        )
    except Exception as exc:
        value["failure"] = {
            "error_class": type(exc).__name__,
            "category": "PREFLIGHT_OR_RUNTIME_FAILURE",
            "stage": value.get("stage", "INITIALIZATION"),
        }
        value["contract_status"] = "FAIL"
        if value["preflight"] == "PENDING":
            value["preflight"] = "FAIL"
    writer.write(value, final=True)
    return read_completed(output)


def run_databricks(spark, settings, *, commit_sha, run_id):
    if settings.environment.value != "databricks" or not re.fullmatch(
        r"(?:10f|e2e)[1-9][0-9]*", run_id
    ):
        raise ValueError("Databricks and a new comparison/E2E run ID required")
    from worldbank_copilot.application.runtime import build_databricks_application

    return run_validation(
        settings.repo_root,
        Path(settings.artifact_volume_path) / "copilot_e2e" / (run_id + ".json"),
        commit_sha=commit_sha,
        application_factory=lambda: build_databricks_application(
            spark, settings, models_enabled=True
        ),
    )


def run_investigator_probe(spark, settings, *, commit_sha, run_id):
    """Three real proposal calls on explicitly synthetic empty-evidence contexts.

    No proposed tool is executed. Validity is mechanical, not usefulness/quality.
    """
    if settings.environment.value != "databricks" or not re.fullmatch(r"10e[1-9][0-9]*", run_id):
        raise ValueError("Databricks and a NEW 10eN run required")
    import os

    from worldbank_copilot.application.runtime import build_databricks_application
    from worldbank_copilot.investigation.assembly import assemble
    from worldbank_copilot.investigation.bounded import (
        InvestigatorAdapter,
        InvestigatorRequest,
        parse_output,
        validate_decision,
    )
    from worldbank_copilot.investigation.evidence_models import (
        EvidenceExecutionReport,
        EvidenceLimits,
    )
    from worldbank_copilot.investigation.gate import AdmissionContext, admit
    from worldbank_copilot.investigation.planning import template_plan
    from worldbank_copilot.investigation.policy import AttemptStatus, Consumption
    from worldbank_copilot.routing.models import AccessContext

    writer = Writer(Path(settings.artifact_volume_path) / "copilot_e2e" / (run_id + ".json"))
    value = {
        "schema": "investigator_capability@1",
        "preflight": "PENDING",
        "results": [],
        "acceptance": "NOT_ACCEPTED",
        "quality_review": "REQUIRED",
    }
    writer.write(value)
    try:
        revision = prior.prior.revision_identity(Path(settings.repo_root), commit_sha)
        value["revision"] = revision.model_dump(mode="json")
        if revision.runtime_git_status == "MISMATCH":
            raise ValueError("reviewed revision mismatch")
        _, lock = prepare(settings.repo_root)
        value["content_identity"] = lock
        app = build_databricks_application(spark, settings, models_enabled=True)
        endpoint = os.environ["WBC_APP_INVESTIGATOR_ENDPOINT"]
        model = InvestigatorAdapter(endpoint)
        value["preflight"] = "PASS"
        ledger = BudgetLedger(policy=app.policy)
        probe_started = time.monotonic()
        for sequence, project in enumerate(app.allowed_projects[:3]):
            question = "What changed and why about the closing date?"
            rid = f"probe-{run_id}-{sequence}"
            access = AccessContext(authorized_projects=(project,), active_project_id=project)
            route = app.router.handle(question, access, request_id=rid)
            admission = AdmissionContext(
                rid,
                question,
                access,
                app.router.config,
                app.router.index,
                app.config_dir,
                app.policy,
                True,
            )
            state = template_plan(admit(route, admission).state, admission).state
            package = assemble(state, (), (), app.documents.profile, EvidenceLimits())
            report = EvidenceExecutionReport(
                investigation=state, budget=state.budget, package=package, events=()
            )
            payload = {
                "fixture": "SYNTHETIC_EMPTY_EVIDENCE_NOT_A_REAL_PROJECT_FINDING",
                "question": question,
                "project_id": project,
                "temporal_scope": state.temporal_scope.model_dump(mode="json"),
                "evidence": [],
                "unresolved_requirements": [
                    {"requirement_id": r.requirement_id, "objective": r.objective}
                    for r in state.requirements
                ],
                "allowed_actions": ["STOP", "SEARCH_DOCUMENTS"],
                "remaining": {"operations": 1, "model_calls": 1, "repair_cycles": 1},
            }
            request = InvestigatorRequest(
                context_json=json.dumps(payload, sort_keys=True),
                max_output_tokens=app.policy.max_planning_output_tokens,
                timeout_seconds=min(
                    60, app.policy.overall_deadline_seconds - (time.monotonic() - probe_started)
                ),
            )
            bound = (
                len(request.context_json.encode())
                + len(request.system.encode())
                + len(json.dumps(request.output_schema).encode())
            )
            if bound > app.policy.max_investigator_context_tokens:
                raise ValueError("context budget exhausted")
            reservation_id = f"probe-{sequence}"
            ledger = ledger.reserve(
                Reservation(
                    reservation_id=reservation_id,
                    model_calls=1,
                    tokens=bound + request.max_output_tokens,
                    cost=app.worst_case_model_cost,
                )
            )
            row = {
                "case_id": str(sequence),
                "project_id": project,
                "status": "FAIL",
                "tool_calls": 0,
            }
            try:
                reply = model.invoke(request)
                decision = parse_output(reply.text, Decision)
                if decision.action not in payload["allowed_actions"]:
                    raise ValueError("unapproved fixture action")
                validate_decision(decision, report, admission)
                row.update(
                    status="PASS",
                    action=decision.action,
                    model_identity=reply.model_identity,
                    input_tokens=reply.input_tokens,
                    output_tokens=reply.output_tokens,
                )
            except Exception as exc:
                row["error_class"] = type(exc).__name__
            ledger = ledger.consume(
                Consumption(
                    reservation_id=reservation_id,
                    status=AttemptStatus.SUCCEEDED
                    if row["status"] == "PASS"
                    else AttemptStatus.FAILED,
                )
            )
            value["results"].append(row)
            ledger = ledger.advance_elapsed(Decimal(str(time.monotonic() - probe_started)))
            writer.write(value)
        value["expected_results"] = min(3, len(app.allowed_projects))
        value["budget"] = ledger.model_dump(mode="json")
        value["contract_status"] = (
            "PASS"
            if len(value["results"]) == min(3, len(app.allowed_projects))
            and value["results"]
            and all(r["status"] == "PASS" for r in value["results"])
            else "FAIL"
        )
    except Exception as exc:
        value["failure"] = {"error_class": type(exc).__name__}
        value["contract_status"] = "FAIL"
    writer.write(value, final=True)
    return read_completed(writer.path)


def comparison_metrics(rows):
    result = {}
    for mode in ("A", "B", "C"):
        traces = [r["trace"] for r in rows if r["mode"] == mode and r.get("trace")]
        total = sum(t.get("required_total", 0) for t in traces)
        result[mode] = {
            "requests": len(traces),
            "latency_ms_total": sum(t["latency_ms"] for t in traces),
            "model_calls": sum(t["model_calls"] for t in traces),
            "retrieval_calls": sum(t["retrieval_calls"] for t in traces),
            "mechanical_requirement_coverage": sum(
                t.get("required_with_evidence", 0) for t in traces
            )
            / total
            if total
            else None,
            "claim_count": sum(t["claim_count"] for t in traces),
            "semantic_groundedness": "HUMAN_REVIEW_REQUIRED",
            "billed_cost": "UNAVAILABLE",
        }
    return result
