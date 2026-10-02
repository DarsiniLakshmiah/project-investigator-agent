"""Phase 9E: write (or --check) the adaptive-rerank protocol lock BEFORE any collection.

The lock pins the question set, retrieval configuration hashes, the production-null
assertion, index / corpus identity, CrossEncoder identity, retrieval parameters, the 13
preregistered policy points and their definitions, the classification rules, the diagnostic
formulas, Random / Oracle / Pareto specifications, the drift references and tolerance, the
pass counts, the live-selection rule and the run ids. It never depends on 9E results.

    .venv\\Scripts\\python scripts/adaptive_rerank_lock.py           # write
    .venv\\Scripts\\python scripts/adaptive_rerank_lock.py --check   # verify, exit 1 on drift
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldbank_copilot.retrieval.adaptive_eval import (  # noqa: E402
    build_lock,
    load_retrieval_production,
    lock_sha256,
)
from worldbank_copilot.retrieval.adaptive_rerank import load_adaptive_config  # noqa: E402

LOCK = ROOT / "evaluation" / "adaptive_rerank_9e_lock.json"


def main(check: bool) -> int:
    config_dir = ROOT / "configs"
    production = load_retrieval_production(config_dir)
    lock = build_lock(load_adaptive_config(config_dir), ROOT, production)
    if not lock["production_null"]:
        print("STOP: configs/retrieval/retrieval.yaml production is not null")
        return 1
    if check:
        committed = json.loads(LOCK.read_text(encoding="utf-8"))
        ok = committed == lock
        print(f"lock {'matches' if ok else 'DIFFERS'}: {lock_sha256(lock)}")
        return 0 if ok else 1
    LOCK.write_text(json.dumps(lock, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {LOCK.relative_to(ROOT)} sha256={lock_sha256(lock)}")
    print(f"policy definitions sha256={lock['policy_definitions_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--check" in sys.argv))
