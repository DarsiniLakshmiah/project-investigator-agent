"""Phase 9D Candidate C DEV: write (or --check) the protocol lock BEFORE any DEV prediction.

Derives C1 / C2 / repeat populations mechanically from the frozen DEV dataset and the frozen
9B.2 RoutingService (no model, no data reads, no TEST rows), verifies the ambiguity-probe
file against its frozen hash (integrity reference only - its cases are never parsed), and
writes evaluation/candidate_c_dev_lock.json.

    .venv\\Scripts\\python scripts/candidate_c_dev_lock.py           # write
    .venv\\Scripts\\python scripts/candidate_c_dev_lock.py --check   # verify, exit 1 on drift
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldbank_copilot.common import load_project_registry, load_settings  # noqa: E402
from worldbank_copilot.ingestion.documents import load_document_manifest  # noqa: E402
from worldbank_copilot.routing.bounded_classifier import (  # noqa: E402
    load_bounded_classifier_config,
)
from worldbank_copilot.routing.bounded_classifier_dev import derive_lock, lock_sha256  # noqa: E402
from worldbank_copilot.routing.config import load_routing_config  # noqa: E402
from worldbank_copilot.routing.entities import EntityIndex  # noqa: E402
from worldbank_copilot.routing.evaluation import load_dataset  # noqa: E402
from worldbank_copilot.routing.semantic_eval import canonical_sha256  # noqa: E402

LOCK = ROOT / "evaluation" / "candidate_c_dev_lock.json"


def main(check: bool) -> int:
    settings = load_settings("local", env={})
    registry = load_project_registry(settings.config_dir)
    routing = load_routing_config(settings.config_dir)
    config = load_bounded_classifier_config(settings.config_dir)
    manifest = load_document_manifest(settings.config_dir / "document_manifest.yaml")
    index = EntityIndex.build(registry, manifest, routing)
    probe = canonical_sha256(ROOT / "evaluation" / "semantic_ambiguity_probe.yaml")  # hash only
    if probe != config.dev_evaluation["ambiguity_probe_sha256_lf"]:
        print("STOP: ambiguity probe differs from its frozen hash")
        return 1
    dataset = load_dataset(ROOT / "evaluation" / "routing_cases.yaml")
    lock = derive_lock(config, routing, index, dataset, ROOT, tuple(registry.project_ids)).lock
    if check:
        committed = json.loads(LOCK.read_text(encoding="utf-8"))
        ok = committed == lock
        print(f"lock {'matches' if ok else 'DIFFERS'}: {lock_sha256(lock)}")
        return 0 if ok else 1
    LOCK.write_text(json.dumps(lock, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {LOCK.relative_to(ROOT)} sha256={lock_sha256(lock)}")
    print("populations:", json.dumps(lock["populations"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--check" in sys.argv))
