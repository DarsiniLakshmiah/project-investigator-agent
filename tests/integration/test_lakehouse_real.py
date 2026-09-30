"""Phase 6 over the real local sources (LOCAL validation only; no Databricks).

Builds every Delta-ready dataset from the real sources exactly as the Databricks
notebook does, without persisting anything. Run with: pytest -m integration
"""

from collections import Counter

import pytest

from worldbank_copilot.common import load_project_registry, load_settings
from worldbank_copilot.lakehouse.contracts import validate_rows
from worldbank_copilot.lakehouse.pipeline import LOCAL_DRY_RUN, run_platform

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def platform():
    settings = load_settings("local", env={})
    if not (settings.local_output_root / "parsed").is_dir():
        pytest.skip("no parsed outputs; run scripts/parse_documents.py first")
    registry = load_project_registry(settings.config_dir)
    from worldbank_copilot.lakehouse.pipeline import build_platform_datasets

    try:
        build = build_platform_datasets(settings, registry)
    except FileNotFoundError as exc:
        pytest.skip(f"source files not available: {exc}")
    return build, run_platform(settings, registry, build=build)


def test_sources_verified_and_local_run_is_a_dry_run(platform):
    build, report = platform
    assert report.mode == LOCAL_DRY_RUN and report.writes == []
    assert report.integrity == {"MATCH": 56, "MISMATCH": 0, "MISSING": 0}
    kinds = Counter(f.kind for f in build.snapshot.files)
    assert kinds == {"document": 53, "structured": 3}


def test_current_snapshot_reconciles_with_committed_expectations(platform):
    build, report = platform
    assert report.expected_diffs == {}
    for dataset in build.datasets.values():
        assert validate_rows(dataset.contract, dataset.rows) == []


def test_document_derived_counts_for_current_snapshot(platform):
    build, _ = platform
    rows = {name: ds.rows for name, ds in build.datasets.items()}
    isr = rows["silver.isr_snapshots"]
    assert Counter(r["project_id"] for r in isr) == {"P130544": 24, "P179039": 8, "P506272": 3}
    assert len(rows["silver.project_results"]) == 837
    assert len({r["indicator_key"] for r in rows["silver.project_results"]}) == 237
    assert len(rows["silver.appraisal_risks"]) == 89
    assert len(rows["silver.project_events"]) == 27
    assert all(r["pdo_rating"] and r["overall_risk_rating"] for r in isr)


def test_semantics_survive_platformization(platform):
    build, _ = platform
    rows = {name: ds.rows for name, ds in build.datasets.items()}
    papers = [
        e
        for e in rows["silver.project_events"]
        if e["event_type"] == "RESTRUCTURING" and e["event_date"] is None
    ]
    assert len(papers) == 4
    assert all(
        e["candidate_event_date"] and e["candidate_date_status"] == "DERIVED_FROM_EXPLICIT_SOURCE"
        for e in papers
    )
    seq5 = next(
        r
        for r in rows["silver.isr_snapshots"]
        if r["project_id"] == "P179039" and r["isr_sequence"] == 5
    )
    assert (str(seq5["header_date"]), str(seq5["archive_date"])) == ("2025-05-31", "2024-09-11")
    quality = rows["silver.data_quality_observations"]
    assert any(
        q["check_code"] == "SEQUENCE_DATE_ANOMALY" and q["project_id"] == "P179039" for q in quality
    )
    candidates = rows["silver.indicator_match_candidates"]
    assert candidates and {c["review_status"] for c in candidates} == {"PENDING_REVIEW"}
    keys = {r["indicator_key"] for r in rows["silver.project_results"]}
    assert all(c["indicator_key_a"] in keys and c["indicator_key_b"] in keys for c in candidates)
    assert not any(name.startswith("gold.") for name in rows)
