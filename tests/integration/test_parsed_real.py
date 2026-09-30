"""Checks over the real parsed outputs (report-only: nothing is reparsed).

Run after `python scripts/parse_documents.py`, with: pytest -m integration
Skipped when the parsed outputs are not present.
"""

from datetime import date

import pytest

from worldbank_copilot.common import load_project_registry, load_settings
from worldbank_copilot.common.quality import CheckCode, Severity
from worldbank_copilot.parsing.models import ParseStatus, evidence_location
from worldbank_copilot.parsing.pipeline import run_parsing
from worldbank_copilot.parsing.report import build_parsing_report

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def parsed():
    settings = load_settings("local", env={})
    if not (settings.local_output_root / "parsed").is_dir():
        pytest.skip("no parsed outputs; run scripts/parse_documents.py first")
    registry = load_project_registry(settings.config_dir)
    try:
        run = run_parsing(settings, registry, None)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"source documents not available: {exc}")
    if any(o.action == "missing" for o in run.outcomes):
        pytest.skip("parsing has not completed for every document")
    return registry, run, build_parsing_report(run, registry)


def test_all_53_documents_parsed_and_hashes_unchanged(parsed):
    _, run, report = parsed
    assert len(run.expected) == 53
    assert len(run.documents) == 53
    assert all(d.parse_status is ParseStatus.SUCCESS for d in run.documents)
    assert run.hashes_before == run.hashes_after
    assert not report.by_check(CheckCode.SOURCE_HASH_CHANGED)
    assert not report.by_check(CheckCode.PARSED_DOCUMENT_MISSING)


def test_isr_inventory_complete(parsed):
    _, run, report = parsed
    sequences = {}
    for doc in run.documents:
        if doc.document_type == "ISR":
            sequences.setdefault(doc.project_id, []).append(doc.isr_sequence)
    assert {pid: sorted(s) for pid, s in sequences.items()} == {
        "P130544": list(range(1, 25)),
        "P179039": list(range(1, 9)),
        "P506272": list(range(1, 4)),
    }
    assert not report.by_check(CheckCode.ISR_INVENTORY_INCOMPLETE)


def test_isr_sequence_8_keeps_both_dates(parsed):
    _, run, _ = parsed
    seq8 = next(d for d in run.documents if d.project_id == "P130544" and d.isr_sequence == 8)
    assert seq8.report_date == date(2019, 6, 14)
    assert seq8.archive_date == date(2019, 2, 21)
    assert (seq8.document_date, seq8.document_date_basis) == (
        date(2019, 2, 21),
        "isr_archived_date",
    )


def test_every_element_has_valid_page_provenance(parsed):
    _, run, report = parsed
    for doc in run.documents:
        assert doc.page_count > 0
        for element in [*doc.blocks, *doc.tables]:
            assert element.page_numbers and all(
                1 <= p <= doc.page_count for p in element.page_numbers
            )
    assert not report.by_check(CheckCode.PAGE_PROVENANCE_INVALID)
    doc = next(d for d in run.documents if d.project_id == "P130544" and d.isr_sequence == 18)
    block = next(b for b in doc.blocks if not b.is_furniture and b.page_number > 1)
    assert evidence_location(doc, block).label.startswith("Source: ISR Sequence 18, page ")


def test_project_ids_and_types_resolved(parsed):
    _, run, report = parsed
    assert not report.by_check(CheckCode.DOCUMENT_PROJECT_MISMATCH)
    assert not report.by_check(CheckCode.DOCUMENT_TYPE_UNRESOLVED)
    assert all(d.document_type not in (None, "OTHER") for d in run.documents)


def test_no_parsing_errors(parsed):
    _, _, report = parsed
    errors = [o.message for o in report.observations if o.severity is Severity.ERROR]
    assert errors == []
