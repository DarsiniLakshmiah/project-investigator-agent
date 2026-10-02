"""Write evaluation/semif_protocol_lock.json - the C_SEMIF_OPENJEV experiment protection.

Run BEFORE P1 or any DEV prediction. The lock pins the SemIf commit, model id and revision,
option set, criterion, the pre-registered threshold grids, the frozen ambiguity probe hash
and the frozen routing dataset hash. tests/unit/test_routing_semif_contract.py and the
notebook guards refuse to proceed if any of them drifts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldbank_copilot.routing.semantic_eval import canonical_sha256  # noqa: E402
from worldbank_copilot.routing.semif_contract import (  # noqa: E402
    load_semif_config,
    protocol_lock,
)


def main() -> None:
    config = load_semif_config(ROOT / "configs")
    lock = protocol_lock(
        config,
        canonical_sha256(ROOT / "evaluation" / "routing_cases.yaml"),
        canonical_sha256(ROOT / config.probe_set["file"]),
    )
    out = ROOT / "evaluation" / "semif_protocol_lock.json"
    out.write_text(json.dumps(lock, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(lock, indent=1))


if __name__ == "__main__":
    main()
