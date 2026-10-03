"""Preregistered synthetic synthesis/critic capability validation, user-run only.

No model is selected/promoted here. No SQL, retrieval, agent or repair execution.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

from worldbank_copilot.common.exceptions import ConfigurationError
from worldbank_copilot.investigation.claims import (
    PROMPT_VERSION,
    CriticOutput,
    Failure,
    ModelRequest,
    NodeError,
    SynthesisOutput,
)
from worldbank_copilot.investigation.model_adapter import DatabricksModelAdapter
from worldbank_copilot.investigation.synthesis import (
    CRITIC_INSTRUCTIONS,
    INSTRUCTIONS,
    parse_output,
    validate_claims,
    validate_critic,
)
from worldbank_copilot.routing.semantic_eval import canonical_sha256
from worldbank_copilot.validation import phase9_contract as prior
from worldbank_copilot.validation import phase10c_evidence as accepted
from worldbank_copilot.validation.phase10d_fixtures import draft, fixture
from worldbank_copilot.validation.phase10d_notebook_v3 import notebook_identity

CASE_FILE = "evaluation/phase10d_model_cases.json"
LOCK_FILE = "evaluation/phase10d_model_lock.json"
PREVIOUS_LOCK_FILE = "evaluation/phase10d_model_lock_v3.json"
NOTEBOOK_FILE = "notebooks/09_phase10d_model_validation.py"
SOURCE_FILES = (
    "src/worldbank_copilot/investigation/claims.py",
    "src/worldbank_copilot/investigation/synthesis.py",
    "src/worldbank_copilot/investigation/model_adapter.py",
    "src/worldbank_copilot/validation/phase10d_fixtures.py",
    "src/worldbank_copilot/validation/phase10d_models.py",
    "tests/unit/test_phase10d_models.py",
    "src/worldbank_copilot/validation/phase10d_notebook_v3.py",
    "tests/unit/test_phase10d_notebook_v3.py",
    CASE_FILE,
    "tests/unit/test_phase10d_transport_compatibility.py",
    "tests/unit/test_phase10d_unknown_provenance.py",
    "notebooks/_bootstrap.py",
    "requirements-phase10d.txt",
    "requirements-databricks.txt",
    "constraints-databricks.txt",
)


def build_lock(root):
    files = {name: canonical_sha256(root / name) for name in SOURCE_FILES}
    return {
        "schema": "phase10d_capability_lock@4",
        "supersedes_lock_sha256_lf": canonical_sha256(root / PREVIOUS_LOCK_FILE),
        "status": "PREREGISTERED",
        "files_sha256_lf": files,
        "case_set_sha256_lf": files[CASE_FILE],
        "notebook_semantic_identity": notebook_identity((root / NOTEBOOK_FILE).read_text("utf-8")),
        "phase10c_lock_sha256_lf": canonical_sha256(root / accepted.LOCK_FILE),
        "max_endpoints": 2,
        "max_calls_per_endpoint": 19,
        "max_output_tokens": 2000,
        "timeout_seconds": 60,
        "temperature": 0,
        "retries": 0,
        "required_repeatability": "ALL_REPEATS_CONTRACT_STABLE",
    }


def prepare(root):
    root = Path(root)
    if (
        Path(__file__).resolve()
        != root.resolve() / "src/worldbank_copilot/validation/phase10d_models.py"
    ):
        raise ConfigurationError("EXECUTING_HARNESS_NOT_REVIEWED_SNAPSHOT")
    accepted.prepare(root)
    lock = json.loads((root / LOCK_FILE).read_text("utf-8"))
    if lock != build_lock(root):
        raise ConfigurationError("PHASE10D_CONTENT_OR_PROTOCOL_MISMATCH")
    cases = json.loads((root / CASE_FILE).read_text("utf-8-sig"))
    if sum(c["repeats"] for c in cases) != lock["max_calls_per_endpoint"]:
        raise ConfigurationError("PHASE10D_CALL_SCHEDULE_MISMATCH")
    return cases, lock


def case_input(case):
    context = fixture(case["kind"])
    output = None
    if case["role"] == "SYNTHESIZER":
        data = context.model_dump(mode="json")
        data["scenario"] = case["scenario"]
    else:
        changes = {
            "NONE": {},
            "UNSUPPORTED": {"claim_text": "The project budget doubled."},
            "FABRICATED": {"evidence_ids": ["ev_fabricated"]},
            "PROJECT": {"project_id": "P999999"},
            "PROVENANCE": {"provenance_label": "FACT"},
            "OVERCLAIM": {"claim_text": "The project will fail."},
            "TEMPORAL": {"temporal_scope": {**context.temporal_scope, "date_to": "2099-01-01"}},
        }[case["mutation"]]
        output = draft(context, **changes)
        data = {
            **context.model_dump(mode="json"),
            "candidate_output": output.model_dump(mode="json"),
        }
    return context, output, data


def evaluate(case, parsed, context, candidate):
    if case["role"] == "SYNTHESIZER":
        errors = validate_claims(parsed, context)
        shape = (
            parsed.insufficient_evidence
            if case["expectation"] == "INSUFFICIENT"
            else not parsed.insufficient_evidence and bool(parsed.candidate_claims)
        )
        expected_provenance = case["kind"]
        provenance = all(c.provenance_label == expected_provenance for c in parsed.candidate_claims)
        if case["expectation"] != "INSUFFICIENT":
            provenance = provenance and bool(parsed.candidate_claims)
        # Controlled literal fixture: entailment oracle only for these synthetic cases.
        semantic = case["expectation"] == "INSUFFICIENT" or all(
            "delay" in c.claim_text.lower()
            and not any(
                w in c.claim_text.lower() for w in ("budget", "doubled", "million", "failure")
            )
            for c in parsed.candidate_claims
        )
        checks = {
            "schema": True,
            "mechanical": not errors,
            "expected_disposition": shape,
            "provenance": provenance,
            "synthetic_semantic_oracle": semantic,
        }
        stable = {
            "insufficient": parsed.insufficient_evidence,
            "claims": sorted(
                (
                    c.claim_type.value,
                    c.provenance_label.value,
                    sorted(c.evidence_ids),
                    sorted(c.requirement_ids),
                )
                for c in parsed.candidate_claims
            ),
            "checks": dict(checks),
        }
    else:
        errors = validate_critic(parsed, candidate, context)
        checks = {
            "schema": True,
            "mechanical": not errors,
            "critic_detection": len(parsed.findings) == 1
            and parsed.findings[0].code == case["expectation"],
        }
        stable = {
            "findings": sorted(
                (f.claim_id, f.code.value, sorted(f.evidence_ids)) for f in parsed.findings
            ),
            "checks": dict(checks),
        }
    return checks, stable


def summary(rows, preflight, expected):
    success = len(rows) == expected and all(r["status"] == "PASS" for r in rows)
    return {
        "calls_executed": len(rows),
        "calls_passed": sum(r["status"] == "PASS" for r in rows),
        "overall_status": "PASS" if preflight == "PASS" and success else "FAIL",
    }


def validate_artifact(value):
    rows = value["case_results"]
    if value["summary"] != summary(rows, value["preflight"]["status"], value["expected_calls"]):
        raise ValueError("summary mismatch")
    if value["cost_accounting"] != {
        "actual_billed_cost": "UNAVAILABLE",
        "pricing_version": None,
        "reason": (
            "No trustworthy billing exposed by this capability adapter; unavailable is not zero."
        ),
    }:
        raise ValueError("cost accounting mismatch")
    if any(
        r["status"] == "PASS" and (not r["checks"] or not all(r["checks"].values())) for r in rows
    ):
        raise ValueError("passing row with failed checks")
    identities = [(r["endpoint"], r["case_id"], r["repeat"]) for r in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate call identity")
    if value["preflight"]["status"] == "PASS" and value["summary"]["overall_status"] == "PASS":
        schedule = value["schedule"]
        if identities != [(r["endpoint"], r["case_id"], r["repeat"]) for r in schedule]:
            raise ValueError("incomplete or reordered schedule")
        if value["expected_calls"] != len(schedule) or not schedule:
            raise ValueError("invalid expected calls")
        if any(
            value[k] != 0 for k in ("agent_calls", "tool_calls", "retrieval_calls", "repair_cycles")
        ):
            raise ValueError("forbidden execution")
        groups = {}
        for row in rows:
            groups.setdefault((row["endpoint"], row["case_id"]), []).append(row["stability"])
        if any(any(s != group[0] for s in group) for group in groups.values()):
            raise ValueError("repeatability mismatch")


def read_completed(path, *, require_final=True):
    path = Path(path)
    payload = path.read_bytes()
    receipt = json.loads(prior.completion_receipt(path).read_bytes())
    if receipt != {
        "artifact": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }:
        raise ValueError("completion receipt mismatch")
    value = json.loads(payload)
    validate_artifact(value)
    if value["persistence"]["completion_receipt"] != prior.completion_receipt(path).name:
        raise ValueError("receipt identity mismatch")
    if require_final and not value["persistence"]["finalized"]:
        raise ValueError("not a final artifact")
    return value


class Writer:
    def __init__(self, path):
        self.path, self.sequence = Path(path), 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if any(self.path.parent.glob(self.path.stem + "*")):
            raise FileExistsError("ATTEMPT_ALREADY_EXISTS")
        prior._write_new_json(
            self.path.with_name(self.path.stem + ".attempt.json"),
            {"status": "RESERVED", "artifact": self.path.name},
        )

    def write(self, value, *, final=False):
        path = (
            self.path
            if final
            else self.path.with_name(f"{self.path.stem}.checkpoint-{self.sequence:03}.json")
        )
        value["persistence"] = {
            "finalized": final,
            "sequence": self.sequence,
            "completion_receipt": prior.completion_receipt(path).name,
        }
        validate_artifact(value)
        payload = prior._write_new_json(path, value)
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
    root,
    output,
    *,
    commit_sha,
    endpoints,
    adapter_factory=DatabricksModelAdapter,
    offline=False,
    clock=time.monotonic,
):
    if offline and adapter_factory is DatabricksModelAdapter:
        raise ValueError("offline execution requires an injected fake adapter")
    root = Path(root)
    if not 1 <= len(endpoints) <= 2 or len(set(endpoints)) != len(endpoints):
        raise ValueError("one or two distinct user-approved endpoints required")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", e) for e in endpoints):
        raise ValueError("invalid endpoint")
    revision = prior.revision_identity(root, commit_sha)
    writer = Writer(output)
    artifact = {
        "schema": "phase10d_model_capability@1",
        "validation_environment": "OFFLINE_STAND_INS" if offline else "DATABRICKS",
        "revision": revision.model_dump(mode="json"),
        "preflight": {"status": "PENDING"},
        "expected_calls": 19 * len(endpoints),
        "case_results": [],
        "summary": summary([], "PENDING", 19 * len(endpoints)),
        "cost_accounting": {
            "actual_billed_cost": "UNAVAILABLE",
            "pricing_version": None,
            "reason": (
                "No trustworthy billing exposed by this capability adapter; "
                "unavailable is not zero."
            ),
        },
        "promotion": "NOT_DECIDED",
        "agent_calls": 0,
        "tool_calls": 0,
        "retrieval_calls": 0,
        "repair_cycles": 0,
        "prompt_version": PROMPT_VERSION,
        "schema_version": "candidate_claims@1",
    }
    writer.write(artifact)
    try:
        if revision.runtime_git_status == "MISMATCH":
            raise ConfigurationError("TRUSTWORTHY_SOURCE_REVISION_MISMATCH")
        cases, lock = prepare(root)
        if revision.runtime_git_status == "VERIFIED":
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
                cwd=root,
                timeout=20,
            )
            if dirty.strip():
                raise ConfigurationError("TRUSTWORTHY_SOURCE_NOT_CLEAN_COMMITTED_SNAPSHOT")
        artifact["schedule"] = [
            {"endpoint": endpoint, "case_id": case["case_id"], "repeat": repeat}
            for endpoint in endpoints
            for case in cases
            for repeat in range(1, case["repeats"] + 1)
        ]
        artifact["content_identity"] = lock
        artifact["preflight"] = {"status": "PASS"}
        artifact["summary"] = summary([], "PASS", artifact["expected_calls"])
        writer.write(artifact)
        for endpoint in endpoints:
            adapter = adapter_factory(endpoint)
            for case in cases:
                baselines = []
                for repeat in range(1, case["repeats"] + 1):
                    context, candidate, data = case_input(case)
                    schema = SynthesisOutput if case["role"] == "SYNTHESIZER" else CriticOutput
                    request = ModelRequest(
                        role=case["role"],
                        system=INSTRUCTIONS if schema == SynthesisOutput else CRITIC_INSTRUCTIONS,
                        context_json=json.dumps(data, sort_keys=True),
                        output_schema=schema.model_json_schema(),
                        max_output_tokens=lock["max_output_tokens"],
                        timeout_seconds=lock["timeout_seconds"],
                    )
                    row = {
                        "endpoint": endpoint,
                        "case_id": case["case_id"],
                        "repeat": repeat,
                        "role": case["role"],
                        "status": "FAIL",
                        "checks": {},
                        "temperature": 0,
                        "attempt_count": 1,
                    }
                    began = clock()
                    try:
                        reply = adapter.invoke(request)
                        parsed = parse_output(reply.text, schema)
                        checks, stability = evaluate(case, parsed, context, candidate)
                        baselines.append(stability)
                        checks["repeatability"] = all(s == baselines[0] for s in baselines)
                        row.update(
                            checks=checks,
                            status="PASS" if all(checks.values()) else "FAIL",
                            model_identity=reply.model_identity,
                            input_tokens=reply.input_tokens,
                            output_tokens=reply.output_tokens,
                            stability=stability,
                            structured_output=parsed.model_dump(mode="json"),
                        )
                    except NodeError as exc:
                        row["error_category"] = exc.category.value
                    except TimeoutError:
                        row["error_category"] = Failure.MODEL_TIMEOUT.value
                    except Exception:
                        row["error_category"] = Failure.INTERNAL_VALIDATION_ERROR.value
                    row["latency_ms"] = (clock() - began) * 1000
                    artifact["case_results"].append(row)
                    artifact["summary"] = summary(
                        artifact["case_results"], "PASS", artifact["expected_calls"]
                    )
                    writer.write(artifact)
    except Exception as exc:
        artifact["failure"] = {
            "error_class": type(exc).__name__,
            "category": "PREFLIGHT_OR_HARNESS_FAILURE",
        }
        if isinstance(exc, ConfigurationError):
            artifact["failure"]["invariant_failures"] = [str(exc)]
        if artifact["preflight"]["status"] == "PENDING":
            artifact["preflight"] = {"status": "FAIL"}
        artifact["summary"] = summary(artifact["case_results"], "FAIL", artifact["expected_calls"])
        # A harness-level exception cannot become PASS even if every call completed.
        artifact["preflight"] = {"status": "FAIL"}
    writer.write(artifact, final=True)
    return read_completed(output)


def run_databricks_validation(settings, *, commit_sha, run_id, endpoints):
    if settings.environment.value != "databricks" or not re.fullmatch(r"10d[1-9][0-9]*", run_id):
        raise ConfigurationError("DATABRICKS_ENVIRONMENT_AND_NEW_10D_RUN_ID_REQUIRED")
    root = Path(settings.repo_root)
    output = Path(settings.artifact_volume_path) / "phase10d_models" / f"{run_id}.json"
    return run_attempt(
        root,
        output,
        commit_sha=commit_sha,
        endpoints=tuple(e.strip() for e in endpoints.split(",") if e.strip()),
    )
