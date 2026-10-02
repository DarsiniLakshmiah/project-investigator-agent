"""Phase 9E: offline frontier from the REAL collection artifact (local, no network).

Verifies the protocol lock and the artifact, then - fail-closed - either writes an INVALID
report (lock / artifact / hybrid-equivalence / nondeterminism / drift problems; NO frontier)
or the full descriptive frontier: every preregistered point, counterfactual classes,
trigger diagnostics, Random(r), Oracle (USES GROUND TRUTH — NOT DEPLOYABLE), Pareto sets
and per-project breakdowns. No winner is chosen. For a VALID report it also writes the
live latency-validation selection (rerank-rate rule only; NOT a production winner).

    .venv\\Scripts\\python scripts/adaptive_rerank_9e.py <collection__9e1.json>

Outputs (never overwritten): evaluation/adaptive_rerank_9e.json and .md, and
evaluation/adaptive_rerank_9e_live_selection.json (VALID only).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldbank_copilot.retrieval import evaluation as ev  # noqa: E402
from worldbank_copilot.retrieval.adaptive_eval import (  # noqa: E402
    analyse,
    build_lock,
    load_retrieval_production,
    lock_sha256,
    render_markdown,
)
from worldbank_copilot.retrieval.adaptive_rerank import load_adaptive_config  # noqa: E402

LOCK = ROOT / "evaluation" / "adaptive_rerank_9e_lock.json"
REPORT_JSON = ROOT / "evaluation" / "adaptive_rerank_9e.json"
REPORT_MD = ROOT / "evaluation" / "adaptive_rerank_9e.md"
SELECTION = ROOT / "evaluation" / "adaptive_rerank_9e_live_selection.json"


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    if any(p.exists() for p in (REPORT_JSON, REPORT_MD, SELECTION)):
        print("STOP: a 9E report or live selection already exists - never overwritten")
        return 1
    artifact_path = Path(argv[0])
    raw = artifact_path.read_bytes()
    config = load_adaptive_config(ROOT / "configs")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    recomputed = build_lock(config, ROOT, load_retrieval_production(ROOT / "configs"))
    questions = ev.load_questions(ROOT / config.questions_file)
    report = analyse(
        config,
        lock,
        json.loads(raw),
        questions,
        lock_sha256=lock_sha256(lock),
        recomputed_lock_sha256=lock_sha256(recomputed),
    ) | {"collection_artifact_sha256": hashlib.sha256(raw).hexdigest()}
    REPORT_JSON.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", "utf-8")
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(f"STATUS: {report['status']}")
    if report["status"] != "VALID":
        print(json.dumps(report.get("failures"), indent=1))
        return 1
    selection = report["live_validation_point"] | {
        "lock_sha256": report["lock_sha256"],
        "collection_artifact_sha256": report["collection_artifact_sha256"],
    }
    SELECTION.write_text(json.dumps(selection, indent=1, ensure_ascii=False) + "\n", "utf-8")
    print("live latency validation point (NOT a production winner):", selection["point"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
