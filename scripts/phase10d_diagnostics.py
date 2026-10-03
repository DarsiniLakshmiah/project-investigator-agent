"""Read-only Phase 10D diagnostics. Never reserves a run or invokes an endpoint.

Load in a separate Databricks notebook after the reviewed bootstrap. This file is
outside the locked harness; it neither changes nor substitutes acceptance checks.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

REVIEWED_SHA = "0228b4dfce5f10c9fefe0c7a363905cba7d66a1a"
REVIEWED_LOCK_SHA256_LF = "ec1b9bfc634592a327e6908f4492635b0e35d6bce2e410908733f3eb2dca08f5"


def diagnose(settings, *, declared_sha, run_id="10d1", endpoints="databricks-gpt-oss-20b"):
    """Return observations, not a validation result. No writes or external calls.

    Exception messages, source text, environment values, SDK configuration and
    model output are deliberately omitted. Shared Phase 9 checks are observed
    through a temporary thread-local trace, restored even on failure. If tracing
    is already active, leave it alone and report the shared gate as one unit.
    """
    if not re.fullmatch(r"10d[1-9][0-9]*", run_id):
        raise ValueError("invalid diagnostic run ID")
    root = Path(settings.repo_root)
    output = Path(settings.artifact_volume_path) / "phase10d_models" / f"{run_id}.json"
    report = {
        "diagnostic_only": True,
        "acceptance_performed": False,
        "model_calls": 0,
        "writes": 0,
        "expected_reviewed_sha": REVIEWED_SHA,
        "declared_reviewed_sha": declared_sha,
        "repo_root": str(root),
        "artifact_path": str(output),
        "gates": [],
        "observations": {},
    }
    gates = report["gates"]

    def gate(name, operation, expected=None, *, blocking=True):
        row = {"check": name, "blocking": blocking, "expected": expected}
        gates.append(row)
        try:
            actual = operation()
            row["actual"] = actual
            row["status"] = "PASS" if expected is None or actual == expected else "FAIL"
            return actual
        except Exception as exc:
            row.update(status="ERROR", error_class=type(exc).__name__)
            # Only invariant labels from the reviewed shared harness, never raw exceptions.
            labels = getattr(exc, "failures", None)
            if isinstance(labels, list):
                row["invariant_failures"] = [
                    s
                    for s in labels
                    if isinstance(s, str) and re.fullmatch(r"[A-Za-z0-9_:./-]+", s)
                ]
            return None

    gate("environment", lambda: settings.environment.value, "databricks")
    gate("run_id_format", lambda: bool(re.fullmatch(r"10d[1-9][0-9]*", run_id)), True)
    gate("declared_sha_format", lambda: bool(re.fullmatch(r"[0-9a-f]{40}", declared_sha)), True)
    gate("declared_sha_matches_reviewed", lambda: declared_sha, REVIEWED_SHA)
    parsed = tuple(e.strip() for e in endpoints.split(",") if e.strip())
    # Endpoint strings are only reported after validating the safe name grammar.
    valid_names = all(re.fullmatch(r"[A-Za-z0-9_-]{1,200}", e) for e in parsed)
    gate("endpoint_names", lambda: bool(valid_names), True)
    gate(
        "endpoint_count_distinct",
        lambda: 1 <= len(parsed) <= 2 and len(set(parsed)) == len(parsed),
        True,
    )
    report["observations"]["endpoints"] = list(parsed) if valid_names else "INVALID_NAMES_REDACTED"
    report["observations"]["endpoint_readiness_and_schema_support"] = (
        "NOT_CHECKED: production has no metadata preflight; no external calls in diagnostic"
    )

    def load_harness():
        from worldbank_copilot.validation import phase10d_models

        return phase10d_models

    h = None

    def import_probe():
        nonlocal h
        h = load_harness()
        return True

    gate("harness_import", import_probe, True)
    if h is not None:
        c, p = h.accepted, h.prior
        revision = gate(
            "runtime_revision_identity",
            lambda: p.revision_identity(root, declared_sha).model_dump(mode="json"),
        )
        if revision:
            gate(
                "trustworthy_revision_not_mismatched",
                lambda: revision["runtime_git_status"] != "MISMATCH",
                True,
            )
        for module, name in ((h, "phase10d_models"), (c, "phase10c_evidence")):
            gate(
                "executing_module:" + name,
                lambda m=module, n=name: str(Path(m.__file__).resolve()),
                str(root.resolve() / f"src/worldbank_copilot/validation/{name}.py"),
            )

        locks = {}
        for owner in (p, c, h):
            lock = gate(
                "read_lock:" + owner.LOCK_FILE,
                lambda o=owner: json.loads((root / o.LOCK_FILE).read_text("utf-8")),
            )
            if isinstance(lock, dict):
                locks[owner.LOCK_FILE] = lock
                for name, wanted in lock.get(
                    "files_sha256_lf", lock.get("frozen_files_sha256_lf", {})
                ).items():
                    gate(
                        owner.LOCK_FILE + ":file:" + name,
                        lambda n=name: h.canonical_sha256(root / n),
                        wanted,
                    )
                gate(
                    owner.LOCK_FILE + ":case_hash",
                    lambda o=owner: h.canonical_sha256(root / o.CASE_FILE),
                    lock.get("case_set_sha256_lf"),
                )
        # Do not emit full lock content; protocol keys/hashes only are sufficient.
        for row in gates:
            if row["check"].startswith("read_lock:") and isinstance(row.get("actual"), dict):
                value = row["actual"]
                row["actual"] = {k: value[k] for k in ("schema", "status", "run_id") if k in value}
        dlock = locks.get(h.LOCK_FILE, {})
        gate(
            "reviewed_10d_lock_identity",
            lambda: h.canonical_sha256(root / h.LOCK_FILE),
            REVIEWED_LOCK_SHA256_LF,
        )
        gate(
            "referenced_10c_lock_identity",
            lambda: h.canonical_sha256(root / c.LOCK_FILE),
            dlock.get("phase10c_lock_sha256_lf"),
        )
        for owner in (c, h):
            lock = locks.get(owner.LOCK_FILE, {})
            gate(
                "notebook_semantic_identity:" + owner.NOTEBOOK_FILE,
                lambda o=owner: o.notebook_identity((root / o.NOTEBOOK_FILE).read_text("utf-8")),
                lock.get("notebook_semantic_identity"),
            )
        report["observations"]["semantic_scheme"] = {
            "phase10c": "databricks_wrapper_ast@2",
            "phase10d": "databricks_wrapper_ast@3",
        }
        report["observations"]["10d_wrapper_import_widget_commutation"] = (
            "@3: exact import before/after the adjacent ordered commit_sha/run_id/endpoints block"
        )

        # Observe each shared check without replacing its implementation or result.
        previous_trace = sys.gettrace()

        def trace(frame, event, arg):
            if (
                event == "return"
                and frame.f_code.co_name == "check"
                and frame.f_code.co_filename == p.__file__
                and "label" in frame.f_locals
                and "ok" in frame.f_locals
            ):
                ok = bool(frame.f_locals["ok"])
                gates.append(
                    {
                        "check": "shared_protocol:" + frame.f_locals["label"],
                        "expected": True,
                        "actual": ok,
                        "status": "PASS" if ok else "FAIL",
                        "blocking": True,
                    }
                )
            return trace

        try:
            if previous_trace is None:
                sys.settrace(trace)
            gate(
                "shared_protocol_prepare",
                lambda: bool(p.prepare_protocol(root, dependency_ok=True)),
                True,
            )
        finally:
            sys.settrace(previous_trace)
        gate("phase10c_prepare_including_fixtures", lambda: bool(c.prepare(root)), True)
        gate("phase10d_build_lock_equal", lambda: h.build_lock(root) == dlock, True)
        gate("phase10d_production_prepare", lambda: bool(h.prepare(root)), True)
        cases = gate(
            "phase10d_case_parse", lambda: json.loads((root / h.CASE_FILE).read_text("utf-8-sig"))
        )
        if isinstance(cases, list):
            # Avoid printing prompt/case content.
            next(r for r in gates if r["check"] == "phase10d_case_parse")["actual"] = {
                "count": len(cases)
            }
            gate(
                "call_schedule",
                lambda: sum(case["repeats"] for case in cases),
                dlock.get("max_calls_per_endpoint"),
            )
            for case in cases:
                gate(
                    "fixture_and_request:" + case["case_id"],
                    lambda case=case: _request_constructs(h, case),
                    True,
                )
        if revision and revision["runtime_git_status"] == "VERIFIED":
            gate(
                "runtime_git_clean_locked_10d_files",
                lambda: (
                    not subprocess.check_output(
                        [
                            "git",
                            "status",
                            "--porcelain",
                            "--untracked-files=all",
                            "--",
                            *dlock.get("files_sha256_lf", {}),
                            h.LOCK_FILE,
                        ],
                        cwd=root,
                        stderr=subprocess.DEVNULL,
                        timeout=20,
                    ).strip()
                ),
                True,
            )
        else:
            report["observations"]["runtime_git_clean_check"] = (
                "SKIPPED_AS_IN_PRODUCTION: runtime Git is not VERIFIED"
            )
        if valid_names:
            for endpoint in parsed:
                # Reviewed constructor is lazy; does not create WorkspaceClient or call post.
                gate(
                    "adapter_constructor:" + endpoint,
                    lambda e=endpoint: bool(h.DatabricksModelAdapter(e)),
                    True,
                )

    def pins():
        result = {}
        for name in ("requirements-databricks.txt", "requirements-phase10d.txt"):
            for line in (root / name).read_text("utf-8").splitlines():
                match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9.+!-]+)", line.strip())
                if match:
                    package, expected = match.groups()
                    try:
                        actual = metadata.version(package)
                    except metadata.PackageNotFoundError:
                        actual = None
                    result[package] = {
                        "expected": expected,
                        "actual": actual,
                        "match": actual == expected,
                    }
        return result

    report["observations"]["requirement_pins"] = gate(
        "installed_pins_observation", pins, blocking=False
    )
    report["observations"]["dependency_health_gate"] = (
        "10D calls accepted.prepare with dependency_ok=True; no pip-check/pin/s"
        "hadowing gate is executed"
    )

    def persistence():
        parent = output.parent
        files = (
            sorted(path.name for path in parent.glob(output.stem + "*")) if parent.exists() else []
        )
        existing_parent = parent
        while not existing_parent.exists() and existing_parent != existing_parent.parent:
            existing_parent = existing_parent.parent
        return {
            "parent_exists": parent.exists(),
            "nearest_existing_parent": str(existing_parent),
            "existing_attempt_files": files,
            "reservation_exists": output.with_name(output.stem + ".attempt.json").exists(),
            "final_exists": output.exists(),
            "receipt_exists": output.with_name(output.name + ".complete.json").exists(),
            "volume_path_shape": str(output).startswith("/Volumes/"),
            "write_readback_exclusive_create_support": "NOT_TESTED_READ_ONLY",
            "reuse_allowed": not files,
        }

    gate("persistence_observation", persistence, blocking=False)
    if h is not None and output.is_file():

        def artifact_probe():
            artifact = h.read_completed(output)
            failure = artifact.get("failure", {})
            return {
                "summary": artifact["summary"],
                "preflight": artifact["preflight"],
                "revision": artifact["revision"],
                "persistence": artifact["persistence"],
                "schedule_present": "schedule" in artifact,
                "content_identity_present": "content_identity" in artifact,
                "failure": {
                    k: failure[k]
                    for k in ("error_class", "category", "invariant_failures")
                    if k in failure
                },
            }

        gate("preserved_attempt_receipt_and_state", artifact_probe, blocking=False)
    failed = [row["check"] for row in gates if row["blocking"] and row["status"] != "PASS"]
    report["observed_preflight_status"] = "FAIL" if failed else "NO_FAILURE_OBSERVED_NOT_ACCEPTANCE"
    report["first_failing_diagnostic_gate"] = failed[0] if failed else None
    report["failed_gates"] = failed
    report["observations"]["gate_order"] = (
        "Diagnostic inventory order; preserved artifact identifies the historic"
        "al failure. Later gates are probed independently even after earlier fa"
        "ilures."
    )
    return report


def _request_constructs(h, case):
    _, _, data = h.case_input(case)
    schema = h.SynthesisOutput if case["role"] == "SYNTHESIZER" else h.CriticOutput
    h.ModelRequest(
        role=case["role"],
        system=h.INSTRUCTIONS if schema == h.SynthesisOutput else h.CRITIC_INSTRUCTIONS,
        context_json=json.dumps(data, sort_keys=True),
        output_schema=schema.model_json_schema(),
        max_output_tokens=2000,
        timeout_seconds=60,
    )
    return True
