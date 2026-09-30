"""Parsing pipeline with a fake parser: cache, failures, reconciliation, provenance, report."""

import json
from datetime import date

import pytest

from support.parsing_builders import FakeParser, T, content_from_pages, isr_pages
from worldbank_copilot.common import load_project_registry, load_settings
from worldbank_copilot.common.quality import CheckCode, Severity
from worldbank_copilot.parsing.document_builder import build_document
from worldbank_copilot.parsing.models import (
    ParsedDocument,
    ParseStatus,
    ValidationStatus,
    evidence_location,
)
from worldbank_copilot.parsing.parser import DocumentParser
from worldbank_copilot.parsing.pipeline import load_parsed, run_parsing
from worldbank_copilot.parsing.report import aggregates, build_parsing_report

SEQ1 = "Disclosable-Version-of-the-ISR-IN-X-P130544-Sequence-No-01.pdf"
DATED = "ISR-Disclosable-P130544-04-10-2017-1491820303083.pdf"
LOAN = "RAD696381150.pdf"
P179 = "P179039-3c28639e.pdf"


def _contents(**overrides):
    contents = {
        SEQ1: content_from_pages(isr_pages(seq=1, archived="28-Jun-2016",
                                           header_date="6/28/2016")),
        DATED: content_from_pages(isr_pages(seq=2, archived="10-Apr-2017",
                                            header_date="4/10/2017")),
        LOAN: content_from_pages([[(T, "LOAN NUMBER 8601-IN"), (T, "Loan Agreement between INDIA "
                                   "and INTERNATIONAL BANK FOR RECONSTRUCTION AND DEVELOPMENT")]]),
        P179: content_from_pages(isr_pages(pid="P179039", seq=1, archived="16-Jun-2023",
                                           header_date="6/16/2023")),
    }  # fmt: skip
    contents.update(overrides)
    return contents


@pytest.fixture
def env(synthetic_config):
    return load_settings(config_dir=synthetic_config, env={}), load_project_registry(
        synthetic_config
    )


def _by_name(run):
    return {d.filename: d for d in run.documents}


def test_fake_parser_satisfies_interface():
    assert isinstance(FakeParser({}), DocumentParser)


def test_full_run_parses_reconciles_and_reports(env):
    settings, registry = env
    run = run_parsing(settings, registry, FakeParser(_contents()))
    assert {o.action for o in run.outcomes} == {"parsed"}
    docs = _by_name(run)
    assert set(docs) == {SEQ1, DATED, LOAN, P179}

    dated = docs[DATED]
    statuses = {v.field: v.status for v in dated.metadata_validation}
    assert statuses["isr_sequence"] is ValidationStatus.CONFIRMED  # manifest 2 == document 2
    assert statuses["document_date"] is ValidationStatus.CONFIRMED
    assert (dated.report_date, dated.archive_date) == (date(2017, 4, 10), date(2017, 4, 10))

    seq1 = docs[SEQ1]
    seq1_status = {v.field: v.status for v in seq1.metadata_validation}
    assert seq1_status["document_date"] is ValidationStatus.DOCUMENT_ONLY
    assert (
        seq1.document_date == date(2016, 6, 28) and seq1.document_date_basis == "isr_archived_date"
    )

    loan = docs[LOAN]
    assert loan.document_type == "LOAN_AGREEMENT"
    assert loan.loan_number == "8601-IN" and loan.normalized_loan_number == "8601-IN"
    assert {v.field: v.status for v in loan.metadata_validation}["loan_number"] is (
        ValidationStatus.CONFIRMED
    )

    report = build_parsing_report(run, registry)
    assert not report.has_errors, [o.message for o in report.observations]
    assert run.hashes_before == run.hashes_after
    agg = aggregates(run)
    assert agg["isr"]["P130544"]["sequences"] == [1, 2]
    assert agg["by_status"] == {"SUCCESS": 4}


def test_outputs_written_per_project_and_round_trip(env):
    settings, registry = env
    run = run_parsing(settings, registry, FakeParser(_contents()))
    root = settings.local_output_root / "parsed"
    for doc in run.documents:
        path = root / doc.project_id / f"{doc.document_id}.json"
        assert load_parsed(path) == doc
        json.loads(path.read_text(encoding="utf-8"))
    assert (root / "_parse_index.json").is_file()


def test_unchanged_documents_are_not_reparsed(env):
    settings, registry = env
    run_parsing(settings, registry, FakeParser(_contents()))
    second = FakeParser(_contents())
    run = run_parsing(settings, registry, second)
    assert second.calls == []
    assert {o.action for o in run.outcomes} == {"cached"}


def test_force_and_parser_config_change_reparse(env):
    settings, registry = env
    run_parsing(settings, registry, FakeParser(_contents()))
    forced = FakeParser(_contents())
    run_parsing(settings, registry, forced, force=True)
    assert len(forced.calls) == 4
    upgraded = FakeParser(_contents(), version="2")
    run_parsing(settings, registry, upgraded)
    assert len(upgraded.calls) == 4  # parser version is part of the cache identity


def test_one_failure_does_not_stop_the_run_and_can_be_retried(env):
    settings, registry = env
    run = run_parsing(settings, registry, FakeParser(_contents(), fail={LOAN}))
    docs = _by_name(run)
    assert docs[LOAN].parse_status is ParseStatus.FAILED
    assert "simulated parser failure" in docs[LOAN].error
    assert all(d.parse_status is ParseStatus.SUCCESS for n, d in docs.items() if n != LOAN)
    report = build_parsing_report(run, registry)
    assert (CheckCode.PARSE_FAILED, Severity.ERROR) in {
        (o.check, o.severity) for o in report.observations
    }

    retry = FakeParser(_contents())
    run = run_parsing(settings, registry, retry, failed_only=True)
    assert retry.calls == [LOAN]
    assert _by_name(run)[LOAN].parse_status is ParseStatus.SUCCESS


def test_single_document_selection(env):
    settings, registry = env
    parser = FakeParser(_contents())
    run = run_parsing(settings, registry, parser, document=LOAN)
    assert parser.calls == [LOAN] and not run.full_projects
    with pytest.raises(ValueError):
        run_parsing(settings, registry, parser, document="nope.pdf")


def test_report_only_mode(env):
    settings, registry = env
    run = run_parsing(settings, registry, None)
    assert {o.action for o in run.outcomes} == {"missing"}
    report = build_parsing_report(run, registry)
    assert CheckCode.PARSED_DOCUMENT_MISSING in {o.check for o in report.observations}
    run_parsing(settings, registry, FakeParser(_contents()))
    run = run_parsing(settings, registry, None)
    assert {o.action for o in run.outcomes} == {"loaded"}


def test_changed_source_gets_new_identity(env):
    settings, registry = env
    first = run_parsing(settings, registry, FakeParser(_contents()))
    old_id = _by_name(first)[SEQ1].document_id
    (settings.documents_root / "P130544" / SEQ1).write_bytes(b"new content")
    parser = FakeParser(_contents())
    second = run_parsing(settings, registry, parser)
    assert parser.calls == [SEQ1]
    assert _by_name(second)[SEQ1].document_id != old_id


def test_document_correction_and_isr_duplicate_are_reported(env):
    settings, registry = env
    wrong = content_from_pages(isr_pages(seq=1, archived="10-Apr-2017", header_date="4/10/2017"))
    run = run_parsing(settings, registry, FakeParser(_contents(**{DATED: wrong})))
    dated = _by_name(run)[DATED]
    seq = next(v for v in dated.metadata_validation if v.field == "isr_sequence")
    assert (seq.status, seq.manifest_value, seq.document_value, seq.resolved_value) == (
        ValidationStatus.CORRECTED_FROM_DOCUMENT,
        2,
        1,
        1,
    )
    report = build_parsing_report(run, registry)
    checks = {(o.check, o.project_id) for o in report.observations}
    assert (CheckCode.METADATA_CORRECTED_FROM_DOCUMENT, "P130544") in checks
    assert (CheckCode.ISR_INVENTORY_INCOMPLETE, "P130544") in checks  # sequences 1,1


def test_document_from_another_project_is_an_error(env):
    settings, registry = env
    misfiled = content_from_pages(isr_pages(pid="P130544", seq=1, archived="16-Jun-2023"))
    run = run_parsing(settings, registry, FakeParser(_contents(**{P179: misfiled})))
    report = build_parsing_report(run, registry)
    assert (CheckCode.DOCUMENT_PROJECT_MISMATCH, Severity.ERROR) in {
        (o.check, o.severity) for o in report.observations if o.project_id == "P179039"
    }


def test_page_provenance_and_evidence_label(env):
    settings, registry = env
    run = run_parsing(settings, registry, FakeParser(_contents()))
    doc: ParsedDocument = _by_name(run)[DATED]
    block = next(b for b in doc.blocks if b.text_clean.startswith("The project is progressing"))
    location = evidence_location(doc, block)
    assert location.label == "Source: ISR Sequence 2, page 2"
    assert location.section_title == "Implementation Status and Key Decisions"
    assert (location.project_id, location.filename) == ("P130544", DATED)
    assert all(b.page_number >= 1 for b in doc.blocks)


def test_pdf_page_count_mismatch_and_empty_pages():
    content = content_from_pages([[(T, "LOAN NUMBER 9835-IN Loan Agreement " * 20)], []])
    record = {"project_id": "P506272", "document_id": "P506272-x", "filename": "x.pdf",
              "relative_path": "P506272/x.pdf", "sha256": "0" * 64, "file_size_bytes": 1,
              "document_type": "LOAN_AGREEMENT", "classification_method": "MANIFEST"}  # fmt: skip
    parser = FakeParser({}).info
    doc = build_document(record, content, parser, "t", 1.0, pdf_page_count=3)
    codes = {(i.code, i.severity) for i in doc.quality_issues}
    assert (CheckCode.PDF_PAGE_COUNT_MISMATCH, "ERROR") in codes
    assert (CheckCode.EMPTY_PAGES, "WARNING") in codes  # 1 of 2 pages empty


def test_text_coverage_detects_lost_content():
    content = content_from_pages(
        [
            [(T, "Original Closing Date 30-Nov-2022 " * 30)],
            [(T, "The project is progressing well overall " * 30)],
        ]
    )
    record = {"project_id": "P130544", "document_id": "d", "filename": "f.pdf",
              "relative_path": "P130544/f.pdf", "sha256": "0" * 64, "file_size_bytes": 1,
              "document_type": "LOAN_AGREEMENT", "classification_method": "MANIFEST"}  # fmt: skip
    pdf_text = [
        "Original Closing Date 30-Nov-2022",
        "The project is progressing well overall. Disbursement Baseline Target Actual",
    ]
    doc = build_document(record, content, FakeParser({}).info, "t", 1.0, pdf_page_texts=pdf_text)
    assert doc.pages[0].pdf_text_coverage == 1.0
    # 9 PDF tokens (3+ chars) on page 2; 4 (disbursement, baseline, target, actual) are missing.
    assert doc.pages[1].pdf_text_coverage == 0.5556
    assert doc.metadata["pdf_page_count"] == 2
    issue = next(i for i in doc.quality_issues if i.code == CheckCode.LOW_TEXT_COVERAGE)
    assert "p2=0.56" in issue.message


def test_running_headers_are_not_counted_as_lost_content():
    # The parser omitted the running header on every page: not a content loss.
    bodies = [
        "procurement delays reported",
        "utility reform progressing",
        "disbursement accelerated sharply",
        "safeguards compliance reviewed",
    ]
    pages = [[(T, f"{body} " * 40)] for body in bodies]
    content = content_from_pages(pages)
    pdf_text = [f"The World Bank Karnataka Project\n{body}" for body in bodies]
    record = {"project_id": "P130544", "document_id": "d", "filename": "f.pdf",
              "relative_path": "P130544/f.pdf", "sha256": "0" * 64, "file_size_bytes": 1,
              "document_type": "LOAN_AGREEMENT", "classification_method": "MANIFEST"}  # fmt: skip
    doc = build_document(record, content, FakeParser({}).info, "t", 1.0, pdf_page_texts=pdf_text)
    assert doc.metadata["pdf_text_coverage"] == 1.0
    assert all(p.pdf_text_coverage == 1.0 for p in doc.pages)


def test_full_coverage_raises_no_issue():
    content = content_from_pages([[(T, "Loan Agreement between India and the Bank " * 30)]])
    record = {"project_id": "P130544", "document_id": "d", "filename": "f.pdf",
              "relative_path": "P130544/f.pdf", "sha256": "0" * 64, "file_size_bytes": 1,
              "document_type": "LOAN_AGREEMENT", "classification_method": "MANIFEST"}  # fmt: skip
    doc = build_document(
        record,
        content,
        FakeParser({}).info,
        "t",
        1.0,
        pdf_page_texts=["Loan Agreement between India and the Bank"],
    )
    assert doc.metadata["pdf_text_coverage"] == 1.0
    assert not [i for i in doc.quality_issues if i.code == CheckCode.LOW_TEXT_COVERAGE]


def test_table_cell_drop_warnings_surface():
    content = content_from_pages(isr_pages()).model_copy(update={"parser_warnings": [
        "MatchingPostProcessor: 8 of 34 pdf cells matched neither a row nor a column band of "
        "the 3x6 grid and were dropped from the table"]})  # fmt: skip
    record = {
        "project_id": "P130544",
        "document_id": "d",
        "filename": "f.pdf",
        "relative_path": "P130544/f.pdf",
        "sha256": "0" * 64,
        "file_size_bytes": 1,
        "document_type": "ISR",
        "isr_sequence": 5,
        "classification_method": "MANIFEST",
    }
    doc = build_document(record, content, FakeParser({}).info, "t", 1.0)
    issue = next(i for i in doc.quality_issues if i.code == CheckCode.TABLE_CELLS_DROPPED)
    assert "8 PDF cells" in issue.message
