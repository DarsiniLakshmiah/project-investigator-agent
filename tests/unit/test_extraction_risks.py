"""Appraisal risks: formal SORT ratings vs assessment findings, deterministic sources only."""

from datetime import date

from tests.support.extraction_builders import S, T, Tbl, make_doc, text_source

from worldbank_copilot.common.quality import CheckCode
from worldbank_copilot.extraction.provenance import ExtractionMethod, ExtractionStatus
from worldbank_copilot.extraction.risks import (
    FINDING,
    FORMAL,
    extract_appraisal_risks,
)

SORT_HEADER = ["Risk Category", "Rating"]


def _pad(tables, pages=None, **kw):
    pages = pages or [
        [(S, "Datasheet"), (T, "Synthetic datasheet.")],
        [(S, "Systematic Operations Risk-Rating Tool (SORT)"), (T, "Ratings below.")],
        [(S, "Technical risks"), (T, "Table below.")],
    ]
    defaults = {
        "project_id": "P179039",
        "document_type": "APPRAISAL_DOCUMENT",
        "document_date": date(2023, 3, 3),
        "filename": "pad.pdf",
    }
    defaults.update(kw)
    return make_doc(pages, tables, **defaults)


def test_sort_rows_are_formal_ratings_with_provenance():
    sort = [
        SORT_HEADER,
        ["1. Political and Governance", "⚫ Moderate"],
        ["2. Fiduciary", "⚫ Substantial"],
        ["9. Other", ""],
        ["10. Overall", "⚫ Substantial"],
    ]
    doc = _pad([Tbl(2, sort, after="Ratings below.")])
    risks, _ = extract_appraisal_risks(doc, text_source(doc))
    formal = [r for r in risks if r.framing == FORMAL]
    assert [r.risk_category for r in formal] == [
        "Political and Governance",
        "Fiduciary",
        "Other",
        "Overall",
    ]
    fiduciary = formal[1]
    assert fiduciary.risk_rating.raw_rating == "Substantial"  # glyph removed by clean_cell
    assert fiduciary.risk_rating.normalized_rating == "Substantial"
    assert fiduciary.status in (ExtractionStatus.EXACT, ExtractionStatus.NORMALIZED)
    assert formal[2].risk_rating is None  # blank rating is not invented
    assert (fiduciary.source_page, fiduciary.identified_date) == (2, date(2023, 3, 3))
    assert fiduciary.source_refs[0].table_id == "t0001"


def test_sort_duplicate_copy_with_different_rating_is_conflict():
    datasheet = [SORT_HEADER, ["Fiduciary", "Substantial"]]
    annex = [["Risk Category", "Rating"], ["Fiduciary", "Moderate"]]
    doc = _pad([Tbl(1, datasheet), Tbl(2, annex, after="Ratings below.")])
    risks, _ = extract_appraisal_risks(doc, text_source(doc))
    fiduciary = next(r for r in risks if r.risk_category == "Fiduciary")
    assert fiduciary.status is ExtractionStatus.CONFLICT
    assert len(fiduciary.source_refs) == 2
    assert CheckCode.EXTRACTION_CONFLICT in {i.code for i in fiduciary.quality_issues}


def test_risk_table_with_parenthetical_rating_is_assessment_finding():
    table = [
        ["Risk", "Mitigation Action"],
        ["Synthetic budget risk (Moderate)", "Synthetic mitigation one."],
        ["Synthetic staffing risk (Substantial)", "Synthetic mitigation two."],
    ]
    doc = _pad([Tbl(3, table, after="Table below.")])
    risks, _ = extract_appraisal_risks(doc, text_source(doc))
    findings = [r for r in risks if r.framing == FINDING]
    assert [r.risk_rating.normalized_rating for r in findings] == ["Moderate", "Substantial"]
    assert findings[0].mitigation_text == "Synthetic mitigation one."
    assert findings[0].risk_category == "Technical"


def test_long_body_row_mentioning_risk_is_not_a_header():
    long_cell = "Develop screening procedures so that environmental risk is " + "managed " * 10
    table = [
        ["Action Description", "DLI", "Completion Measurement"],
        [long_cell, "DLI 2", "Synthetic mitigation measure"],
    ]
    doc = _pad([Tbl(3, table, after="Table below.")], document_type="ESSA")
    risks, _ = extract_appraisal_risks(doc, text_source(doc))
    assert risks == []


def test_fiduciary_table_with_merged_sn_risk_continuation():
    first = [["SN", "Risk", "Mitigation Measure"], ["1", "Synthetic risk A.", "Mitigation A."]]
    # Real layout: the continuation repeats the header with 'SN' and 'Risk' merged.
    continuation = [["SN Risk", "Mitigation Measure"], ["2 Synthetic risk B.", "Mitigation B."]]
    pages = [[(S, "Procurement risk assessment"), (T, "Risks.")], [(T, "Continued.")]]
    doc = _pad(
        [Tbl(1, first, after="Risks."), Tbl(2, continuation)],
        pages=pages,
        document_type="FIDUCIARY_ASSESSMENT",
    )
    risks, _ = extract_appraisal_risks(doc, text_source(doc))
    assert [r.risk_description for r in risks] == ["Synthetic risk A.", "Synthetic risk B."]
    assert [r.mitigation_text for r in risks] == ["Mitigation A.", "Mitigation B."]
    assert risks[0].risk_category == "Fiduciary - Procurement"


def test_essa_page_split_fragment_is_merged_and_aggregate_kept():
    header = [
        "Proposed Investments/ Activity",
        "Benefit",
        "Impact",
        "Justification for rating",
        "Risk Rating",
    ]
    page1 = [header, ["Activity A", "Benefit A", "Impact A part one", "Why A", "Moderate"]]
    page2 = [
        header,
        ["", "", "impact A part two", "more why", ""],
        ["Aggregate Risk Rating"] * 4 + ["Moderate"],
    ]
    pages = [[(S, "Environmental Risk Analysis"), (T, "One.")], [(T, "Two.")]]
    doc = _pad([Tbl(1, page1, after="One."), Tbl(2, page2)], pages=pages, document_type="ESSA")
    risks, _ = extract_appraisal_risks(doc, text_source(doc))
    first, aggregate = risks
    assert first.risk_description == "Impact A part one impact A part two"
    assert first.rating_justification == "Why A more why"
    assert first.activity == "Activity A" and len(first.source_refs) == 2
    assert first.risk_category == "Environmental"
    assert (aggregate.risk_description, aggregate.risk_rating.normalized_rating) == (
        "Aggregate Risk Rating",
        "Moderate",
    )


def test_row_starting_mid_sentence_is_flagged_ambiguous():
    table = [
        ["Program Activities", "Risks", "Mitigation/Risk Management"],
        ["Activity A", "Risk A.", "Mitigation A."],
        ["Activity B", "continues from a split row. Risk B.", "Mitigation B."],
    ]
    doc = _pad([Tbl(3, table, after="Table below.")], document_type="ESSA")
    risks, _ = extract_appraisal_risks(doc, text_source(doc))
    assert risks[1].status is ExtractionStatus.AMBIGUOUS
    assert CheckCode.EXTRACTION_AMBIGUOUS in {i.code for i in risks[1].quality_issues}


def test_technical_assessment_blocks_table_and_text_fallback():
    table = [
        ["Risk -1 Synthetic O&M budget risk", ""],
        ["Risk Rating", "Moderate"],
        ["Mitigation Actions", "Synthetic mitigation."],
    ]
    pages = [
        [(S, "Risks and mitigation"), (T, "Intro.")],
        [(S, "Risk -2"), (T, "Misordered block text.")],
    ]
    doc = _pad([Tbl(1, table, after="Intro.")], pages=pages, document_type="TECHNICAL_ASSESSMENT")
    text = {
        2: [
            "Risk -2 Synthetic depletion risk",
            "Risk Rating Moderate",
            "Mitigation",
            "Actions",
            "Synthetic mitigation text line.",
        ]
    }
    risks, issues = extract_appraisal_risks(doc, text_source(doc, text))
    one, two = risks
    assert (one.risk_description, one.risk_rating.normalized_rating) == (
        "Synthetic O&M budget risk",
        "Moderate",
    )
    assert one.extraction_method is ExtractionMethod.DOCLING_TABLE
    assert two.risk_description == "Synthetic depletion risk"
    assert two.extraction_method is ExtractionMethod.PDF_TEXT_FALLBACK
    assert two.mitigation_text == "Synthetic mitigation text line."
    assert CheckCode.PDF_TEXT_FALLBACK_USED in {i.code for i in issues}


def test_non_appraisal_documents_are_ignored():
    doc = _pad([], document_type="ISR")
    assert extract_appraisal_risks(doc, text_source(doc)) == ([], [])
