# Databricks notebook source
# MAGIC %md
# MAGIC ## 08c_semif_capability: C_SEMIF_OPENJEV capability probe — P0 then P1 (Phase 9D, B-nb)
# MAGIC Thin entry point. Logic: `worldbank_copilot.routing.semif_capability` / `semif_contract`.
# MAGIC
# MAGIC **Compute:** attach this notebook to **Serverless GPU (AI Runtime)**. No Model Serving
# MAGIC endpoint is created. SemIf-OpenJev is an independent open implementation; it is NOT
# MAGIC TypeSafe Jev.
# MAGIC
# MAGIC * **P0** workspace capability only: GPU visible + bf16-capable, GitHub / Hugging Face /
# MAGIC   PyPI reachable for the PINNED artefacts, local disk, artifact Volume writable.
# MAGIC   Any failure STOPs. Nothing is downloaded or installed in P0.
# MAGIC * **P1** (only if P0 passed in this session): verify the protocol lock, install the
# MAGIC   pinned SemIf commit into an ISOLATED virtual environment (its pins - protobuf 7,
# MAGIC   torch 2.10, transformers 5.17 - never touch this notebook's environment), run
# MAGIC   exactly the pre-registered SYNTHETIC decisions with the pinned model revision, and
# MAGIC   validate the bounded option contract. No World Bank, dataset, probe or TEST text.
# MAGIC * Failures are never worked around (no other model size, revision, backend, dtype,
# MAGIC   grid or architecture). After P1: STOP and report.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-databricks.txt -c ../constraints-databricks.txt

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821 (Databricks built-in)

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

# P0: workspace capability (no download, no install, no deployment).
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import requests  # noqa: E402

from worldbank_copilot.routing.semif_capability import (  # noqa: E402
    p0_verdict,
    parse_nvidia_smi,
    reachability_targets,
)
from worldbank_copilot.routing.semif_contract import load_semif_config  # noqa: E402

semif = load_semif_config(settings.config_dir)  # noqa: F821
out_dir = Path(settings.artifact_volume_path) / "semif_capability"  # noqa: F821
started_p0 = datetime.now(UTC).isoformat()

smi = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
     "--format=csv,noheader,nounits"],
    capture_output=True, text=True, check=False,
)
gpus = parse_nvidia_smi(smi.stdout) if smi.returncode == 0 else []

reachability, reach_errors = {}, {}
for name, url in reachability_targets(semif).items():
    try:
        reachability[name] = requests.get(url, timeout=20, allow_redirects=True).status_code
    except requests.RequestException as exc:
        reachability[name], reach_errors[name] = None, type(exc).__name__

git_available = shutil.which("git") is not None
local_root = Path("/local_disk0") if Path("/local_disk0").exists() else Path("/tmp")
free_disk_gb = round(shutil.disk_usage(local_root).free / 1e9, 1)

volume_write_ok, volume_error = False, None
try:
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_file = out_dir / ".p0_write_check"
    probe_file.write_text("ok", encoding="utf-8")
    volume_write_ok = probe_file.read_text(encoding="utf-8") == "ok"
    probe_file.unlink()
except OSError as exc:
    volume_error = type(exc).__name__

endpoints_visible = None
try:  # read-only listing, for production documentation only (never a B-nb blocker)
    from databricks.sdk import WorkspaceClient

    endpoints_visible = len(list(WorkspaceClient().serving_endpoints.list()))
except Exception as exc:  # recorded, not fatal
    endpoints_visible = f"unavailable: {type(exc).__name__}"

p0 = {
    "stage": "P0", "candidate": semif.candidate, "started_at": started_p0,
    "gpus": gpus, "nvidia_smi_returncode": smi.returncode,
    "reachability_status": reachability, "reachability_errors": reach_errors,
    "local_disk": str(local_root), "free_disk_gb": free_disk_gb,
    "artifact_volume_writable": volume_write_ok, "volume_error": volume_error,
    "serving_endpoints_listable": endpoints_visible,
    "gpu_model_serving_note": "Free Edition docs: no GPU serving endpoints / no custom models "
    "on GPU - not a B-nb blocker; recorded for production documentation only",
    **p0_verdict(gpus, reachability, free_disk_gb, volume_write_ok, git_available),
}
if volume_write_ok:
    (out_dir / "p0_result.json").write_text(json.dumps(p0, indent=1), encoding="utf-8")
print(json.dumps(p0, indent=1))
if not p0["passed"]:
    failed = [k for k, v in p0["checks"].items() if not v]
    if failed == ["bf16_capable_gpu"]:
        raise RuntimeError(
            "STOP: the preregistered bfloat16 configuration is incompatible with the available "
            f"GPU (compute capability < 8.0: {gpus}). No fp16, other GPU, model size or revision."
        )
    raise RuntimeError(f"STOP: P0 failed - {failed}")

# COMMAND ----------

# P1: pinned SemIf + model on SYNTHETIC decisions only (requires P0 passed in this session).
import os  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

from worldbank_copilot.routing.semantic_eval import canonical_sha256  # noqa: E402
from worldbank_copilot.routing.semif_capability import (  # noqa: E402
    hf_snapshot_provenance,
    p1_provenance,
    p1_verdict,
    synthetic_rows,
)
from worldbank_copilot.routing.semif_contract import protocol_lock  # noqa: E402

require_state("p0", step="P0")  # noqa: F821
if not p0["passed"]:
    raise RuntimeError("STOP: P1 requires a passing P0 in this session")
repo = settings.repo_root  # noqa: F821
lock_now = protocol_lock(
    semif,
    canonical_sha256(repo / "evaluation" / "routing_cases.yaml"),
    canonical_sha256(repo / semif.probe_set["file"]),
)
lock_committed = json.loads((repo / "evaluation" / "semif_protocol_lock.json").read_text("utf-8"))
if lock_now != lock_committed:
    raise RuntimeError("STOP: protocol drift - configuration differs from the committed lock")

work = local_root / "semif_p1"
venv = work / "venv"
work.mkdir(parents=True, exist_ok=True)
env = {**os.environ, "HF_HOME": str(work / "hf_cache"), "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
steps = []


def step(name, cmd):
    started = time.perf_counter()
    done = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    steps.append({"step": name, "returncode": done.returncode,
                  "seconds": round(time.perf_counter() - started, 1),
                  "stderr_tail": done.stderr[-1500:]})
    if done.returncode != 0:
        raise RuntimeError(f"STOP: P1 step {name!r} failed:\n{done.stderr[-1500:]}")
    return done


commit = semif.implementation["commit"]
step("create_isolated_venv", [sys.executable, "-m", "venv", str(venv)])
pip = str(venv / "bin" / "pip")
step("install_pinned_semif",
     [pip, "install", "-q", f"semif-phase1 @ git+https://github.com/TheoLeeCJ/SemIf-OpenJev@{commit}"])
freeze = step("venv_freeze", [pip, "freeze"]).stdout.splitlines()

rows = synthetic_rows(semif)
in_file, out_file = work / "p1_input.jsonl", work / "p1_output.jsonl"
in_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
cmd = [str(venv / "bin" / "semif-score"), "--mode", semif.model["mode"],
       "--model", semif.model["id"], "--revision", semif.model["revision"],
       "--backend", semif.model["backend"], "--device", semif.model["device"],
       "--dtype", semif.model["dtype"], "--input", str(in_file), "--output", str(out_file)]

peak_mib, stop_polling = [0], threading.Event()


def poll_gpu_memory():
    while not stop_polling.is_set():
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True)
        if q.returncode == 0 and q.stdout.strip():
            peak_mib[0] = max(peak_mib[0], max(int(float(x)) for x in q.stdout.split()))
        time.sleep(1)


poller = threading.Thread(target=poll_gpu_memory, daemon=True)
poller.start()
started = time.perf_counter()
run = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
process_seconds = round(time.perf_counter() - started, 1)
stop_polling.set()
poller.join(timeout=5)

results = []
if out_file.exists():
    results = [json.loads(line) for line in out_file.read_text("utf-8").splitlines() if line]

# Infrastructure provenance, reported by the isolated SemIf environment itself.
ENV_REPORT = (
    "import json, importlib.metadata as m, torch, transformers\n"
    "cuda = torch.cuda.is_available()\n"
    "print(json.dumps({'semif_phase1': m.version('semif-phase1'),"
    " 'torch': torch.__version__, 'transformers': transformers.__version__,"
    " 'huggingface_hub': m.version('huggingface-hub'), 'protobuf': m.version('protobuf'),"
    " 'torch_cuda': torch.version.cuda, 'cuda_available': cuda,"
    " 'device_name': torch.cuda.get_device_name(0) if cuda else None,"
    " 'device_capability': list(torch.cuda.get_device_capability(0)) if cuda else None,"
    " 'bf16_supported': torch.cuda.is_bf16_supported() if cuda else False}))"
)
environment = json.loads(
    step("environment_report", [str(venv / "bin" / "python"), "-c", ENV_REPORT]).stdout
)
snapshot = hf_snapshot_provenance(work / "hf_cache", semif.model["id"], semif.model["revision"])
provenance = p1_provenance(semif, lock_committed, snapshot, environment, p0["gpus"])
p1 = {
    "stage": "P1", "candidate": semif.candidate, "protocol_lock": lock_committed,
    "venv_versions": [p for p in freeze if p.split("==")[0].split(" @ ")[0].lower() in
                      {"semif-phase1", "torch", "transformers", "protobuf", "accelerate",
                       "huggingface-hub", "safetensors", "tokenizers"}],
    "install_steps": steps, "process_seconds_incl_load": process_seconds,
    "semif_stderr_tail": run.stderr[-1500:], "peak_gpu_memory_mib": peak_mib[0],
    "model_metadata_sample": results[0].get("model") if results else None,
    **p1_verdict(rows, results, semif, run.returncode, provenance),
}
(out_dir / "p1_result.json").write_text(json.dumps(p1, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in p1.items() if k != "protocol_lock"}, indent=1, default=str))
if not p1["passed"]:
    raise RuntimeError(f"STOP: P1 failed - {p1['failures'] or 'process/contract failure'}")
print("P1 passed. STOP - report before any DEV, C1, C2 or probe prediction.")
