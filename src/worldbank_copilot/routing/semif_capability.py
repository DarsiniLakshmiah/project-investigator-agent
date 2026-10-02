"""P0 (workspace capability) and P1 (pinned model capability) for C_SEMIF_OPENJEV.

Pure logic used by notebooks/08c_semif_capability.py; the notebook only gathers raw
facts (nvidia-smi text, HTTP status codes, file I/O results, subprocess output).

P0 passes only if: a CUDA GPU is visible, its compute capability supports the pinned
bfloat16 dtype (>= 8.0), GitHub / Hugging Face / PyPI are reachable for the pinned
artefacts, local disk can hold the model, and the artifact Volume is writable.
P1 passes only if every synthetic decision returns a result that satisfies the pinned
contract (validate_result), the SemIf process exits cleanly, and infrastructure
provenance holds (installed SemIf version = pin; a resolved HF snapshot, when exposed,
equals the pinned revision). Neither stage changes the model, revision, backend, dtype
or grid when something fails - the notebook STOPs. Compute capability < 8.0 means the
preregistered bfloat16 configuration is incompatible with the GPU: STOP, no fp16.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from worldbank_copilot.routing.semif_contract import (
    InvalidSemifResult,
    SemifConfig,
    lock_sha256,
    validate_result,
)

MIN_COMPUTE_CAPABILITY_BF16 = 8.0
MIN_FREE_DISK_GB = 25.0  # model (~9 GB bf16) + torch/CUDA wheels + headroom


def reachability_targets(config: SemifConfig) -> dict[str, str]:
    commit, model = config.implementation["commit"], config.model
    return {
        "github_semif_commit": f"https://github.com/TheoLeeCJ/SemIf-OpenJev/commit/{commit}",
        "huggingface_model_revision": (
            f"https://huggingface.co/api/models/{model['id']}/revision/{model['revision']}"
        ),
        "pypi_torch": "https://pypi.org/simple/torch/",
    }


def parse_nvidia_smi(text: str) -> list[dict[str, Any]]:
    """Rows of `nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap
    --format=csv,noheader,nounits`."""
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        name, memory, driver, cap = parts
        gpus.append(
            {
                "name": name,
                "memory_total_mib": int(float(memory)),
                "driver_version": driver,
                "compute_capability": float(cap),
            }
        )
    return gpus


def p0_verdict(
    gpus: Sequence[dict],
    reachability: dict[str, int | None],
    free_disk_gb: float,
    volume_write_ok: bool,
    git_available: bool = False,
) -> dict[str, Any]:
    checks = {
        "gpu_visible": bool(gpus),
        "bf16_capable_gpu": bool(gpus)
        and max(g["compute_capability"] for g in gpus) >= MIN_COMPUTE_CAPABILITY_BF16,
        **{f"reachable_{k}": v == 200 for k, v in reachability.items()},
        "local_disk_sufficient": free_disk_gb >= MIN_FREE_DISK_GB,
        "artifact_volume_writable": volume_write_ok,
        "git_available_for_pinned_install": git_available,
    }
    return {"passed": all(checks.values()), "checks": checks}


def synthetic_rows(config: SemifConfig) -> list[dict[str, Any]]:
    """The exact pre-registered P1 decisions (synthetic text only), plus warm repeats."""
    rows = []
    for spec in config.p1_synthetic:
        row = {
            "id": spec["id"],
            "state": spec["state"],
            "question": config.criterion["V1"]
            if spec["question"] == "USE_CRITERION_V1"
            else spec["question"],
            "options": [dict(o) for o in config.options]
            if spec["options"] == "USE_FIXED_OPTIONS"
            else [dict(o) for o in spec["options"]],
        }
        rows.append(row)
    twelve = next(r for r in rows if r["id"] == "syn-12opt")
    for i in range(1, config.p1_repeats_for_warm_latency + 1):
        rows.append({**twelve, "id": f"syn-12opt-r{i}"})
    return rows


def hf_snapshot_provenance(hf_home: Path, model_id: str, pinned_revision: str) -> dict[str, Any]:
    """What the local Hugging Face cache exposes about the loaded model revision.

    Layout: <HF_HOME>/hub/models--<org>--<name>/{snapshots/<commit>/, refs/<ref>}. A
    dedicated, initially empty HF_HOME is used for P1, so the snapshots present are the
    ones the loader resolved. MATCHES_PIN only if the sole snapshot is the pinned
    revision; MISMATCH if any other snapshot exists; NOT_EXPOSED if there is none (the
    pinned revision supplied to the loader is still recorded). The pin is never altered.
    """
    repo = Path(hf_home) / "hub" / ("models--" + model_id.replace("/", "--"))
    snap_dir, ref_dir = repo / "snapshots", repo / "refs"
    snapshots = (
        sorted(p.name for p in snap_dir.iterdir() if p.is_dir()) if snap_dir.is_dir() else []
    )
    refs = (
        {p.name: p.read_text(encoding="utf-8").strip() for p in ref_dir.iterdir() if p.is_file()}
        if ref_dir.is_dir()
        else {}
    )
    if not snapshots:
        status, resolved = "NOT_EXPOSED", None
    elif snapshots == [pinned_revision]:
        status, resolved = "MATCHES_PIN", pinned_revision
    else:
        status, resolved = "MISMATCH", snapshots[0] if len(snapshots) == 1 else snapshots
    return {
        "requested_model_id": model_id,
        "requested_revision": pinned_revision,
        "cache_repo_dir": str(repo),
        "local_snapshots": snapshots,
        "local_refs": refs,
        "resolved_snapshot_commit": resolved,
        "status": status,
    }


def p1_provenance(
    config: SemifConfig,
    lock: dict,
    snapshot: dict[str, Any],
    environment: dict[str, Any],
    gpus: Sequence[dict],
) -> dict[str, Any]:
    """Infrastructure provenance persisted with P1 (and later with every DEV run).

    `environment` is reported by the isolated SemIf venv itself: semif_phase1, torch,
    transformers versions, torch CUDA version, device name/capability.
    """
    return {
        "requested_model_id": config.model["id"],
        "requested_revision": config.model["revision"],
        "hf_snapshot": snapshot,
        "semif_git_commit": config.implementation["commit"],
        "semif_package_pin": config.implementation["package"],
        "installed": environment,
        "gpus": list(gpus),
        "option_set_sha256": lock["option_set_sha256"],
        "criterion_sha256": lock["criterion_sha256"],
        "protocol_lock_sha256": lock_sha256(lock),
        "probabilities": "UNCALIBRATED model scores; thresholds are harness decision thresholds",
    }


def provenance_failures(config: SemifConfig, provenance: dict[str, Any]) -> list[str]:
    failures = []
    snapshot = provenance["hf_snapshot"]
    if snapshot["status"] == "MISMATCH":
        failures.append(
            f"HF snapshot {snapshot['resolved_snapshot_commit']} != pinned "
            f"{config.model['revision']}"
        )
    pinned_version = config.implementation["package"].split("==")[1]
    installed = provenance["installed"].get("semif_phase1")
    if installed != pinned_version:
        failures.append(f"installed semif-phase1 {installed!r} != pinned {pinned_version}")
    for key in ("torch", "transformers"):
        if not provenance["installed"].get(key):
            failures.append(f"{key} version not reported by the SemIf environment")
    return failures


def p1_verdict(
    rows: Sequence[dict],
    results: Sequence[dict],
    config: SemifConfig,
    returncode: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    by_id = {r.get("id"): r for r in results}
    records, failures = [], provenance_failures(config, provenance)
    for row in rows:
        result = by_id.get(row["id"])
        if result is None:
            failures.append(f"{row['id']}: no result")
            continue
        try:
            scores = validate_result(result, row, config)
        except InvalidSemifResult as exc:
            failures.append(f"{row['id']}: {exc}")
            continue
        top = max(scores.probabilities, key=scores.probabilities.get)
        records.append(
            {
                "id": row["id"],
                "options": len(row["options"]),
                "argmax": top,
                "top_probability": round(scores.probabilities[top], 4),
                "probability_sum": round(sum(scores.probabilities.values()), 6),
                "prompt_sha256": scores.prompt_sha256,
                "forward_seconds": scores.forward_seconds,
                "total_seconds": scores.total_seconds,
            }
        )
    warm = [r["total_seconds"] for r in records[1:] if isinstance(r["total_seconds"], int | float)]
    latency = None
    if warm:
        ordered = sorted(warm)
        latency = {
            "n": len(ordered),
            "p50_s": round(statistics.median(ordered), 4),
            "p95_s": round(ordered[int(0.95 * (len(ordered) - 1))], 4),
        }
    twelve_ok = any(r["options"] == 12 for r in records)
    passed = returncode == 0 and not failures and len(records) == len(rows) and twelve_ok
    return {
        "passed": passed,
        "provenance": provenance,
        "returncode": returncode,
        "failures": failures,
        "records": records,
        "warm_latency": latency,
        "twelve_option_contract_ok": twelve_ok,
    }
