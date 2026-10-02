"""Phase 9E: composed vs live latency from the REAL collection and live artifacts (local).

Checks both artifacts belong to the committed protocol lock and that the live run used the
committed live selection, then reports composed and live p50/p95, their differences, and
decision / top-5 agreement for P0@50 and the latency-validation point (NOT a winner).

    .venv\\Scripts\\python scripts/adaptive_rerank_9e_live.py <collection.json> <live.json>

Outputs (never overwritten): evaluation/adaptive_rerank_9e_live.json and .md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldbank_copilot.retrieval import evaluation as ev  # noqa: E402
from worldbank_copilot.retrieval.adaptive_eval import (  # noqa: E402
    compare_live,
    file_sha256,
    lock_sha256,
    render_live_markdown,
)
from worldbank_copilot.retrieval.adaptive_rerank import load_adaptive_config  # noqa: E402

LOCK = ROOT / "evaluation" / "adaptive_rerank_9e_lock.json"
SELECTION = ROOT / "evaluation" / "adaptive_rerank_9e_live_selection.json"
OUT_JSON = ROOT / "evaluation" / "adaptive_rerank_9e_live.json"
OUT_MD = ROOT / "evaluation" / "adaptive_rerank_9e_live.md"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    if OUT_JSON.exists() or OUT_MD.exists():
        print("STOP: the live report already exists - never overwritten")
        return 1
    config = load_adaptive_config(ROOT / "configs")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    sha = lock_sha256(lock)
    collection = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    live = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    if {collection.get("lock_sha256"), live.get("lock_sha256"), selection["lock_sha256"]} != {
        sha
    } or live["selection"] != selection:
        print("STOP: artifacts or selection belong to a different lock / selection")
        return 1
    if live.get("policy_definitions_sha256") != lock["policy_definitions_sha256"]:
        print("STOP: live run used different policy definitions")
        return 1
    expected = selection["collection_artifact_sha256"]
    if file_sha256(Path(argv[0])) != expected or live.get("collection_artifact_sha256") != expected:
        print("STOP: collection artifact is not the one the live selection was derived from")
        return 1
    report = compare_live(config, collection, live, ev.load_questions(ROOT / config.questions_file))
    OUT_JSON.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", "utf-8")
    OUT_MD.write_text(render_live_markdown(report), encoding="utf-8")
    print(json.dumps(report["points"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
