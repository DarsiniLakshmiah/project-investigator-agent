# Phase 10D UNKNOWN-evidence provenance revision (@4)

User-approved deterministic validator hardening. Not evaluation tuning: it can only make
`validate_claims` stricter, and no case, expectation, prompt or budget changes.

## Historical record

10d6 (`phase10d_capability_lock@3`, commit `0a2e5eee735e4227dd3e0716e02e85cee2eca607`,
`databricks-gpt-oss-20b`) executed all 19 calls: 14 PASS, overall FAIL. All 19 calls generated
schema-valid output; none was `MODEL_REQUEST_INVALID`. The five failures were model-capability
results: S-unknown x3 (UNKNOWN evidence relabeled AI_INTERPRETATION/DOCUMENTED_FINDING with
`insufficient_evidence=false`), C-overclaim (UNSUPPORTED instead of OVERCLAIMED) and
C-temporal (SUPPORTED despite a changed `date_to`). 10d6 remains a valid historical FAIL under
@3. It is not rewritten, rerun or rescored. Preserve 10d1-10d6 and all receipts.

## Root cause of the latent gap

Forensic review of 10d6 S-unknown repeats 1-2 showed `mechanical=true` for an
AI_INTERPRETATION claim citing UNKNOWN evidence. `validate_claims` required non-interpretation
labels to match cited evidence provenance, but AI_INTERPRETATION only required some evidence,
and `finalize` routes to INSUFFICIENT_EVIDENCE only for claims *labeled* UNKNOWN. A model could
therefore relabel unknown evidence and leave publication dependent on the probabilistic critic.
The harness's provenance check caught it in 10d6; the production deterministic layer did not.

## Invariant

If a candidate claim cites evidence whose authoritative provenance (the approved context entry
built from the EvidencePackage `EvidenceReference.provenance`, never the claim's label) is
`UNKNOWN`, the claim must be labeled `UNKNOWN` (and therefore `UNCERTAINTY`, as before).
Otherwise `validate_claims` returns `PROVENANCE_VIOLATION`. This applies to every label and to
mixed citations. All other validator behavior is unchanged.

## Lock lineage

- `evaluation/phase10d_model_lock_v3.json`: byte-for-byte copy of the @3 active lock
  (SHA256_LF `3bf743732a9f8fca8e3d4ce22a7d89720b6c3d2b66b2630d9b7cd9c06423323b`).
- `evaluation/phase10d_model_lock.json`: `phase10d_capability_lock@4`, superseding that digest.
  Only `schema`, `supersedes_lock_sha256_lf` and `files_sha256_lf` differ. Changed file digests:
  `synthesis.py`, `phase10d_models.py`, `test_phase10d_transport_compatibility.py` (its @3
  lineage test now reads the archived v2/v3 files explicitly, assertions retained) and the new
  locked `tests/unit/test_phase10d_unknown_provenance.py`.
- Unchanged: case bytes, 11 cases, 19 calls per endpoint, max 2 endpoints, temperature 0,
  retries 0, 2000 output tokens, 60 s timeout, repeatability criterion, notebook identity
  `databricks_wrapper_ast@3`, Phase 10C lock, prompts and `bounded_synthesis_critic@2`,
  transport compatibility.
- `evaluation/copilot_e2e_lock.json` pins the active 10D lock digest, so only its
  `phase10d_lock_sha256_lf` changes; prior bytes are archived as
  `evaluation/copilot_e2e_lock_before_10d_v4.json`.
- `tests/unit/test_phase10d_diagnostics.py` now names the v2 archive explicitly, because the
  standalone diagnostic remains pinned to that older reviewed lock.

Old capability artifacts, including 10d6, intentionally do not satisfy the @4 acceptance gate.

## Clean-room zero-model-call preflight

After review, commit, push and Git-folder synchronization: install the existing notebook's
requirements/constraints, restart Python, `%run ./_bootstrap`, then run this cell in a
separate notebook. It creates no reservation or artifact, constructs no adapter and invokes no
endpoint. Do not start any new capability attempt until it passes and the user approves.

```python
import json
import subprocess
from pathlib import Path

from worldbank_copilot.common.dependency_health import check_environment
from worldbank_copilot.investigation.claims import Failure
from worldbank_copilot.investigation.synthesis import validate_claims
from worldbank_copilot.validation import phase10d_models as h
from worldbank_copilot.validation.phase10d_fixtures import draft, fixture

REVIEWED_SHA = "<replace with full lowercase 40-character committed SHA>"
V3_SHA = "3bf743732a9f8fca8e3d4ce22a7d89720b6c3d2b66b2630d9b7cd9c06423323b"
root = Path(settings.repo_root)
revision = h.prior.revision_identity(root, REVIEWED_SHA)
assert revision.runtime_git_status != "MISMATCH", "REVIEWED_REVISION_MISMATCH"
requirements = [*h.prior.REQUIREMENTS, "requirements-phase10d.txt"]
health = check_environment(root, requirements, settings.config_dir)
assert health.ok, "DEPENDENCY_HEALTH_FAILED"
h.prior.prepare_protocol(root, dependency_ok=health.ok)
h.accepted.prepare(root, dependency_ok=health.ok)
cases, lock = h.prepare(root)
assert lock == h.build_lock(root)
previous = json.loads((root / h.PREVIOUS_LOCK_FILE).read_text("utf-8"))
assert lock["schema"] == "phase10d_capability_lock@4"
assert previous["schema"] == "phase10d_capability_lock@3"
assert h.canonical_sha256(root / h.PREVIOUS_LOCK_FILE) == V3_SHA
assert lock["supersedes_lock_sha256_lf"] == V3_SHA
assert len(cases) == 11 and sum(c["repeats"] for c in cases) == 19
for field in (
    "case_set_sha256_lf",
    "max_endpoints",
    "max_calls_per_endpoint",
    "max_output_tokens",
    "timeout_seconds",
    "temperature",
    "retries",
    "required_repeatability",
    "notebook_semantic_identity",
    "phase10c_lock_sha256_lf",
):
    assert lock[field] == previous[field], field
# Pure deterministic check of the @4 invariant; no model call.
context = fixture("UNKNOWN")
relabeled = draft(context, claim_type="INTERPRETATION", provenance_label="AI_INTERPRETATION")
assert validate_claims(relabeled, context) == (Failure.PROVENANCE_VIOLATION,)
# 10d6 is preserved, readable and bound to @3; the next ID is unused.
attempts = Path(settings.artifact_volume_path) / "phase10d_models"
artifact = h.read_completed(attempts / "10d6.json")
assert artifact["summary"] == {"calls_executed": 19, "calls_passed": 14, "overall_status": "FAIL"}
assert artifact["content_identity"] == previous
assert not list(attempts.glob("10d7*")), "10d7 ALREADY EXISTS"
if revision.runtime_git_status == "VERIFIED":
    dirty = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *lock["files_sha256_lf"],
            h.LOCK_FILE,
            h.PREVIOUS_LOCK_FILE,
        ],
        cwd=root,
        timeout=20,
    )
    assert not dirty.strip(), "REVIEWED_SOURCE_NOT_CLEAN"
print(
    json.dumps(
        {
            "preflight": "PASS",
            "model_calls": 0,
            "reservation_created": False,
            "reviewed_sha": REVIEWED_SHA,
            "revision": revision.model_dump(mode="json"),
            "active_lock_sha256_lf": h.canonical_sha256(root / h.LOCK_FILE),
            "supersedes_lock_sha256_lf": lock["supersedes_lock_sha256_lf"],
            "notebook": lock["notebook_semantic_identity"],
            "case_count": len(cases),
            "calls_per_endpoint": sum(c["repeats"] for c in cases),
            "10d6_preserved_under_lock_v3": True,
        },
        indent=2,
    )
)
```
