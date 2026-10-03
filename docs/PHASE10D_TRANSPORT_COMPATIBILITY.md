# Phase 10D transport compatibility revision

User-reported real diagnosis: 10d1 failed preflight under the old canonicalizer; 10d2 was interrupted after reservation; 10d3 had a preflight identity mismatch; 10d4 reported the same preflight problem. Preserve all attempts.

10d5 preflight passed and its complete 19-call schedule executed, with 0/19 passing; the then-current adapter normalized all failures to MODEL_UNAVAILABLE. Subsequent direct endpoint diagnostics established HTTP 400 BAD_REQUEST for the exact Phase 10D request: Invalid JSON schema - the "pattern" keyword is not supported. These requests failed before model generation, so 10d5 did not evaluate GPT-OSS synthesis/critic semantic quality. Historical artifacts must not be rewritten. The later successful targeted diagnostics do not substitute for the frozen 19-call suite.

Databricks GPT-OSS structured output rejected the Pydantic-generated JSON Schema 'pattern' keyword with HTTP 400. The transport schema was therefore normalized to remove only the unsupported keyword while retaining the authoritative Pydantic constraint and adding equivalent bounded model guidance. This is a platform compatibility correction, not evaluation tuning.

The existing helper still recursively removes default and applies strict-object required/additionalProperties normalization; it additionally removes pattern on a copied transport schema. All other schema keywords remain. CandidateClaim claim_id/project_id and CriticFinding claim_id retain their authoritative patterns. Output still passes parse_output, Pydantic and unchanged deterministic evidence/citation/project/time/provenance validators. Synthesis receives bounded C1/C2/C15/C123 guidance and exact project copying; the critic copies candidate IDs unchanged. Prompt version advances to bounded_synthesis_critic@2; authoritative output schema version remains candidate_claims@1.

HTTP 400 becomes the single new bounded MODEL_REQUEST_INVALID category, without response bodies or headers. Timeout remains MODEL_TIMEOUT. Network/no-response and other non-200 statuses, including auth, permission, throttling and server errors, remain MODEL_UNAVAILABLE: no broad error taxonomy was added. Malformed successful envelopes/content remain MODEL_OUTPUT_INVALID; invalid generated schema remains SCHEMA_VALIDATION_FAILED. Reasoning content continues to be ignored.

## Lock lineage

phase10d_model_lock_v1.json remains byte-for-byte unchanged. The immediately previous active lock is copied byte-for-byte to phase10d_model_lock_v2.json; the new active lock is phase10d_capability_lock@3 and supersedes that archive's canonical SHA. Notebook serialization remains databricks_wrapper_ast@3 with its unchanged semantic digest. Case bytes, all expectations/repeats, 11 cases, 19 calls per endpoint, endpoint limit, output bound, timeout, temperature, retries and repeatability criterion are unchanged. Phase 9/10C sources and locks are unchanged.

The historical canonicalizer test now explicitly verifies its archived v1-to-v2 transition, while a new regression verifies v2-to-v3. Its assertions were retained. The older standalone diagnostic remains pinned to the previous reviewed lock; its safety tests now require the specific reviewed-lock mismatch against the superseding active lock. Use the new clean-room cell for this revision. The dependent application lock changes only its Phase 10D lock digest; its prior bytes are archived as copilot_e2e_lock_before_10d_v3.json. No Phase 10E/10F/App architecture changed. Old completed capability artifacts intentionally do not satisfy the new active protocol's live acceptance gate.

## Clean-room zero-model-call preflight

After user review/commit/push/synchronization, install the existing notebook's requirements/constraints, restart Python and execute its shared bootstrap. In a separate diagnostic notebook run this cell. It creates no reservation or artifact, constructs no adapter, invokes no endpoint and retains the accepted USER_DECLARED_REVIEWED_REVISION fallback for ambiguous workspace Git. Do not run the capability suite before this preflight passes. 10d6 has NOT been run; it remains the next unused attempt, to be started separately by the user only after successful preflight and review.

```python
# Run ONLY in a restarted Python session after installing the existing notebook
# requirements/constraints and executing %run ./_bootstrap. No validation runner.
import json, subprocess
from pathlib import Path
from worldbank_copilot.common.dependency_health import check_environment
from worldbank_copilot.validation import phase10d_models as h

REVIEWED_SHA = "<replace with full lowercase 40-character committed SHA>"
root = Path(settings.repo_root)
revision = h.prior.revision_identity(root, REVIEWED_SHA)
assert revision.runtime_git_status != "MISMATCH", "REVIEWED_REVISION_MISMATCH"
health = check_environment(root, [*h.prior.REQUIREMENTS, "requirements-phase10d.txt"], settings.config_dir)
assert health.ok, "DEPENDENCY_HEALTH_FAILED"
h.prior.prepare_protocol(root, dependency_ok=health.ok)
h.accepted.prepare(root, dependency_ok=health.ok)
cases, lock = h.prepare(root)
assert lock == h.build_lock(root)
previous = json.loads((root / h.PREVIOUS_LOCK_FILE).read_text("utf-8"))
assert lock["schema"] == "phase10d_capability_lock@3"
assert lock["supersedes_lock_sha256_lf"] == h.canonical_sha256(root / h.PREVIOUS_LOCK_FILE)
assert lock["supersedes_lock_sha256_lf"] == "ec1b9bfc634592a327e6908f4492635b0e35d6bce2e410908733f3eb2dca08f5"
assert lock["case_set_sha256_lf"] == previous["case_set_sha256_lf"]
assert len(cases) == 11 and sum(c["repeats"] for c in cases) == 19
assert lock["max_calls_per_endpoint"] == 19
for field in ("max_endpoints", "max_output_tokens", "timeout_seconds", "temperature", "retries", "required_repeatability", "notebook_semantic_identity", "phase10c_lock_sha256_lf"):
    assert lock[field] == previous[field], field
if revision.runtime_git_status == "VERIFIED":
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all", "--", *lock["files_sha256_lf"], h.LOCK_FILE, h.PREVIOUS_LOCK_FILE], cwd=root, timeout=20)
    assert not dirty.strip(), "REVIEWED_SOURCE_NOT_CLEAN"
print(json.dumps({"preflight": "PASS", "model_calls": 0, "reservation_created": False,
    "reviewed_sha": REVIEWED_SHA, "revision": revision.model_dump(mode="json"),
    "active_lock_sha256_lf": h.canonical_sha256(root / h.LOCK_FILE),
    "supersedes_lock_sha256_lf": lock["supersedes_lock_sha256_lf"],
    "notebook": lock["notebook_semantic_identity"], "case_count": len(cases),
    "calls_per_endpoint": sum(c["repeats"] for c in cases)}, indent=2))
```

## Prepared file identities

SHA256_LF uses the repository canonical line-ending normalization; raw SHA256 records actual bytes.

| File | SHA256_LF | Raw SHA256 |
|---|---|---|
| `evaluation/phase10d_model_lock_v2.json` | `ec1b9bfc634592a327e6908f4492635b0e35d6bce2e410908733f3eb2dca08f5` | `eb0c4510299a8ab0bd32bfa10a0fd3e471e9af7a939d46667b8ff5cd0669d84a` |
| `evaluation/phase10d_model_lock.json` | `3bf743732a9f8fca8e3d4ce22a7d89720b6c3d2b66b2630d9b7cd9c06423323b` | `28c839fad6ba92fcbf904f8f7f012fd949539da3b4c247eebbfaa188cf00e982` |
| `src/worldbank_copilot/investigation/model_adapter.py` | `729399130f5e4e85e4833ee13cedcc4626f6662c552e1e1afc30d03bead4546d` | `31198a59486572f9673de4c473a0fcbd54ec9cfa32386eb6d62ee29ccf1ec21d` |
| `src/worldbank_copilot/investigation/synthesis.py` | `f58b6114a347b4b6255f734739716c02674e78b3a9c87f19cbf4962df9d009c8` | `12ad37a643a7d5746bb23fd49abafb21de04bf1cb838930805fe0ded77df4a88` |
| `src/worldbank_copilot/investigation/claims.py` | `2432baf1a8a085fbc7d5c295a418115482bbc4ef65e3e7a738ba030566fcc176` | `d4e465ece5ca597ecb0536f344ef8933a6222f0a6e750b587e70750ba6de0f0c` |
| `src/worldbank_copilot/validation/phase10d_models.py` | `6fe750f61a50e9f76d96239b2d23a5b7bb07d7621a6ba5557161ab9568120e76` | `c6ccf258e9a24288d4b8301a8ac6c425bb059f8ea4c1ca3c4ebaa674f48d3f19` |
| `tests/unit/test_phase10d_notebook_v3.py` | `fe765641abf7d62d9f2f138a7d87bd8ff5a5322040d16edd02676a7a67f88544` | `bb86b47d45752679e57a5ff54a7fa644f6d8d633e884d38d41242c25e632bfc8` |
| `tests/unit/test_phase10d_transport_compatibility.py` | `be418fd534f9e508681c5fe386454b3d59f788a48ac6d25fdcf23b18f861a24f` | `1f43c597ea80795c47d39b645411b0ee670cae69b413c76a4eb5e708f6c90efb` |
