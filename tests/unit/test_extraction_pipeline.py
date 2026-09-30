"""Phase 5 pipeline pieces: cache keys, per-document extraction, observation mapping."""

from tests.support.extraction_builders import RATINGS_ROWS, Tbl, isr_doc

from worldbank_copilot.common.quality import CheckCode, Severity
from worldbank_copilot.extraction.pipeline import (
    DocumentExtraction,
    _load_cached,
    cache_key,
    extract_document,
    to_observation,
)
from worldbank_copilot.extraction.provenance import ExtractionMethod, evidence, issue


def test_cache_key_depends_on_source_hash_and_parser_config():
    doc = isr_doc([])
    assert cache_key(doc) == cache_key(doc.model_copy())
    assert cache_key(doc) != cache_key(doc.model_copy(update={"source_hash": "1" * 64}))
    parser = doc.parser.model_copy(update={"config_hash": "other"})
    assert cache_key(doc) != cache_key(doc.model_copy(update={"parser": parser}))


def test_extract_document_without_text_layer_and_cache_roundtrip(tmp_path):
    doc = isr_doc([Tbl(1, RATINGS_ROWS, after="Ratings follow.")])
    extraction = extract_document(doc, None)  # no PDF available: fallback returns nothing
    assert extraction.snapshot.pdo_rating.normalized_rating == "Moderately Satisfactory"
    assert extraction.risks == [] and extraction.events == []
    path = tmp_path / "x.json"
    path.write_text(extraction.model_dump_json(), encoding="utf-8")
    assert isinstance(_load_cached(path, extraction.cache_key), DocumentExtraction)
    assert _load_cached(path, "stale-key") is None


def test_issue_to_observation_keeps_evidence_and_project():
    doc = isr_doc([])
    ref = evidence(doc, 3, ExtractionMethod.PDF_TEXT_FALLBACK, text="line")
    obs = to_observation(issue(CheckCode.RESULTS_TABLE_INVALID, "WARNING", "m", ref, pages=[3]))
    assert (obs.check, obs.severity, obs.project_id) == (
        CheckCode.RESULTS_TABLE_INVALID,
        Severity.WARNING,
        "P130544",
    )
    assert obs.details["evidence"]["page_number"] == 3 and obs.details["pages"] == [3]
    bare = to_observation(issue(CheckCode.EXTRACTION_LIMITATION, "INFO", "m", project_id="P1"))
    assert bare.project_id == "P1" and bare.source == "silver_documents"


def test_quality_report_keeps_distinct_findings_with_identical_text():
    from worldbank_copilot.extraction.pipeline import ExtractionRun, build_quality_report

    doc = isr_doc([])
    ref = evidence(doc, 3, ExtractionMethod.DOCLING_TABLE)
    same = "possible same indicator, kept separate (reason)"
    run = ExtractionRun(generated_at="t", documents={}, extractions=[], issues=[
        issue(CheckCode.POSSIBLE_INDICATOR_MATCH, "INFO", same, ref, indicator_keys=["a", "b"]),
        issue(CheckCode.POSSIBLE_INDICATOR_MATCH, "INFO", same, ref, indicator_keys=["a", "c"]),
        issue(CheckCode.POSSIBLE_INDICATOR_MATCH, "INFO", same, ref, indicator_keys=["a", "c"]),
    ])  # fmt: skip
    report = build_quality_report(run)
    assert len(report.by_check(CheckCode.POSSIBLE_INDICATOR_MATCH)) == 2
