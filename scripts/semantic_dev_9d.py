"""Phase 9D: development-only experiment for the semantic fallback (Candidate B).

Runs the pre-registered grid with leave-one-family-out on the frozen DEVELOPMENT split,
records every experiment, applies the selection rule and writes:

  evaluation/semantic_dev_experiments_9d.json - every configuration and its metrics
  evaluation/semantic_config_9d.yaml            - the selected configuration (PROVISIONAL)
  evaluation/semantic_dev_report_9d.md          - the development report

Embedder: ``--embedder lexical`` (default; deterministic, runs locally). The Qwen
(Databricks Model Serving) embedder runs in notebooks/08_semantic_routing_dev.py; a
local stand-in never establishes a learned model's quality. The frozen TEST split is
never touched here (assert_split_allowed refuses it).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldbank_copilot.common import load_settings  # noqa: E402
from worldbank_copilot.routing.config import load_routing_config  # noqa: E402
from worldbank_copilot.routing.evaluation import BaselineResult, load_dataset  # noqa: E402
from worldbank_copilot.routing.semantic import LexicalEmbedder, route_map  # noqa: E402
from worldbank_copilot.routing.semantic_eval import (  # noqa: E402
    assert_split_allowed,
    canonical_sha256,
    load_protocol,
    run_development,
)
from worldbank_copilot.routing.semantic_report import render as render_report  # noqa: E402


def main() -> None:
    assert_split_allowed("dev")
    settings = load_settings("local", env={})
    routing = load_routing_config(settings.config_dir)
    protocol = load_protocol(settings.config_dir)
    dataset_path = ROOT / "evaluation" / "routing_cases.yaml"
    dataset = load_dataset(dataset_path)
    baseline = {
        r["case_id"]: r
        for r in yaml.safe_load(
            (ROOT / "evaluation" / "routing_baseline_9B.2.yaml").read_text("utf-8")
        )["results"]
    }
    started = datetime.now(UTC).isoformat()
    lexical = run_development(
        dataset, route_map(routing.requirements), protocol, LexicalEmbedder(), baseline
    )
    log = {
        "checkpoint": "9D development (first stop)",
        "started": started,
        "split_evaluated": "dev",
        "test_evaluated": False,
        "dataset_sha256": canonical_sha256(dataset_path),
        "dataset_hash_note": "SHA-256 with CRLF normalised to LF (see routing_freeze_9c.json)",
        "dev_case_ids": sorted(c.case_id for c in dataset.cases if c.split == "dev"),
        "protocol": protocol.model_dump(),
        "runs": {"B-lexical": lexical},
    }
    (ROOT / "evaluation" / "semantic_dev_experiments_9d.json").write_text(
        json.dumps(log, indent=1, default=str), encoding="utf-8"
    )
    chosen = lexical["selected"]
    decision = {
        "status": "PROVISIONAL",
        "selected_candidate": "B-lexical"
        if chosen
        else "A (rules-only, semantic fallback disabled)",
        "config": chosen["config"] if chosen else None,
        "reason": (
            "selected by the pre-registered development rule"
            if chosen
            else "no B-lexical configuration met the pre-registered development rule "
            f"(route precision >= {protocol.selection['min_route_precision']} at coverage >= "
            f"{protocol.selection['min_coverage']}); abstention -> CLARIFY is preferred to a "
            "wrong route"
        ),
        "candidates": {
            "A": "rules-only 9B.2 (baseline, always available)",
            "B-lexical": "evaluated on DEVELOPMENT - " + ("selected" if chosen else "REJECTED"),
            "B-qwen": "PENDING - run notebooks/08_semantic_routing_dev.py (DEVELOPMENT only) "
            "in Databricks against the validated Qwen endpoint",
            "C": "PROPOSED, not implemented - awaiting approval and an endpoint probe",
        },
        "freeze_rule": "becomes FROZEN only after the B-qwen development run is reviewed; "
        "TEST stays blocked until then (assert_split_allowed)",
        "selected_on": "dev (leave-one-family-out)",
        "test_evaluated": False,
        "dataset_sha256": log["dataset_sha256"],
        "recorded": started,
    }
    (ROOT / "evaluation" / "semantic_config_9d.yaml").write_text(
        yaml.safe_dump(decision, sort_keys=False), encoding="utf-8"
    )
    base_results = [
        BaselineResult(**r)
        for r in yaml.safe_load(
            (ROOT / "evaluation" / "routing_baseline_9B.2.yaml").read_text("utf-8")
        )["results"]
    ]
    (ROOT / "evaluation" / "semantic_dev_report_9d.md").write_text(
        render_report(log, decision, dataset, base_results), encoding="utf-8"
    )
    summary = {k: lexical[k] for k in ("majority_route_baseline", "classifier_latency_ms")}
    summary["selected"] = chosen and chosen["name"]
    summary["rules_only_dev"] = {k: v for k, v in lexical["rules_only_dev"].items() if k != "rows"}
    summary["hypothetical_fallback_dev"] = {
        k: v for k, v in lexical["hypothetical_fallback_dev"].items() if k != "rows"
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
