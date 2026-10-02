"""Phase 9D Candidate C DEV: hybrid replay + outcome, LOCALLY, from the real prediction artifact.

Builds the same in-memory Gold harness that produced evaluation/routing_baseline_9B.2.yaml
(scripts/routing_review_9c.governed_rows), then over the DEV split ONLY:

1. recomputes Candidate A (frozen 9B.2 router alone) and checks it equals the recorded rows;
2. replays the recorded MAIN-PASS predictions through the real RoutingService
   (RecordedClassifier), which invokes them only where the rules return
   SEMANTIC_CLASSIFICATION_REQUIRED;
3. evaluates the preregistered acceptance rule (one outcome by precedence) and writes
   evaluation/candidate_c_dev_9d.json and .md (never overwritten; no question text).

    .venv-spark\\Scripts\\python scripts/candidate_c_dev_9d.py <predictions artifact .json>
    .venv-spark\\Scripts\\python scripts/candidate_c_dev_9d.py --check-harness

`--check-harness` runs step 1 only (no predictions, no model, nothing written).
TEST is never evaluated; the ambiguity probe is never opened.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from routing_review_9c import governed_rows  # noqa: E402

from worldbank_copilot.common import load_project_registry, load_settings  # noqa: E402
from worldbank_copilot.ingestion.documents import load_document_manifest  # noqa: E402
from worldbank_copilot.intelligence.rules import load_rules, load_scales  # noqa: E402
from worldbank_copilot.routing.bounded_classifier import (  # noqa: E402
    contract_sha256,
    load_bounded_classifier_config,
)
from worldbank_copilot.routing.bounded_classifier_dev import (  # noqa: E402
    RecordedClassifier,
    derive_lock,
    evaluate,
    lock_sha256,
    render_markdown,
    replay,
)
from worldbank_copilot.routing.config import load_routing_config  # noqa: E402
from worldbank_copilot.routing.entities import EntityIndex  # noqa: E402
from worldbank_copilot.routing.evaluation import baseline_of, load_dataset  # noqa: E402
from worldbank_copilot.routing.semantic import route_map  # noqa: E402
from worldbank_copilot.routing.service import RoutingService  # noqa: E402
from worldbank_copilot.tools.base import ToolContext  # noqa: E402
from worldbank_copilot.tools.reader import InMemoryReader  # noqa: E402
from worldbank_copilot.tools.registry import default_executor  # noqa: E402

LOCK = ROOT / "evaluation" / "candidate_c_dev_lock.json"
REPORT_JSON = ROOT / "evaluation" / "candidate_c_dev_9d.json"
REPORT_MD = ROOT / "evaluation" / "candidate_c_dev_9d.md"


def main(argv: list[str]) -> int:
    check_only = "--check-harness" in argv
    paths = [a for a in argv if not a.startswith("--")]
    if not check_only and len(paths) != 1:
        print(__doc__)
        return 2
    if not check_only and (REPORT_JSON.exists() or REPORT_MD.exists()):
        print("STOP: the DEV report already exists - never overwritten")
        return 1

    settings = load_settings("local", env={})
    registry = load_project_registry(settings.config_dir)
    routing = load_routing_config(settings.config_dir)
    config = load_bounded_classifier_config(settings.config_dir)
    manifest = load_document_manifest(settings.config_dir / "document_manifest.yaml")
    index = EntityIndex.build(registry, manifest, routing)
    all_projects = tuple(registry.project_ids)
    protocol = derive_lock(
        config,
        routing,
        index,
        load_dataset(ROOT / "evaluation" / "routing_cases.yaml"),
        ROOT,
        all_projects,
    )
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    rules, scales = load_rules(settings.config_dir), load_scales(settings.config_dir)
    rows = governed_rows(settings)

    def service(semantic: object | None) -> RoutingService:
        return RoutingService(
            routing,
            index,
            default_executor(),
            lambda rid: ToolContext(InMemoryReader(rows), registry, scales, rules, request_id=rid),
            documents=None,
            semantic=semantic,
        )

    recomputed_a = replay(service(None), protocol.cases, all_projects, baseline_of)
    drift = sorted(cid for cid in protocol.baseline if recomputed_a[cid] != protocol.baseline[cid])
    print(f"Candidate A recompute vs recorded 9B.2 (DEV {len(recomputed_a)}): drift={drift}")
    if check_only:
        return 0 if not drift and protocol.lock == lock else 1

    artifact = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
    route_of = route_map(routing.requirements)
    recorder = RecordedClassifier(artifact, route_of, contract_sha256(config)[:12])
    hybrid = replay(service(recorder), protocol.cases, all_projects, baseline_of)
    report = evaluate(
        config,
        lock,
        artifact,
        protocol.cases,
        route_of,
        recorded_a=protocol.baseline,
        recomputed_a=recomputed_a,
        hybrid=hybrid,
        invoked=recorder.invoked,
        recomputed_lock_sha256=lock_sha256(protocol.lock),
        recomputed_c1=protocol.lock["populations"]["c1"],
    )
    REPORT_JSON.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(f"OUTCOME: {report['outcome']}")
    print(json.dumps(report["findings"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
