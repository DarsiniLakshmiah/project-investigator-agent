"""Results-framework extraction: WIDE/BLOCK tables, text fallback, identity rules."""

from datetime import date

from tests.support.extraction_builders import WIDE_HEADER, Tbl, isr_doc, text_source

from worldbank_copilot.common.quality import CheckCode
from worldbank_copilot.extraction.identity import (
    base_name,
    indicator_key,
    load_aliases,
    resolve_identities,
)
from worldbank_copilot.extraction.provenance import ExtractionMethod, ExtractionStatus
from worldbank_copilot.extraction.results import (
    extract_results,
    normalize_indicator_name,
    parse_block_text,
    split_name_suffixes,
)
from worldbank_copilot.extraction.text_source import FallbackLog

NAME = "►Direct project beneficiaries (Number, Corporate)"
BLOCK = [
    [NAME] * 5,
    ["IN00000001", "Baseline", "Actual (Previous)", "Actual (Current)", "End Target"],
    ["Value", "0.00", "10.00", "20.00", "100.00"],
    ["Date", "01-Apr-2016", "01-Jan-2019", "01-Jan-2020", "30-Nov-2022"],
]
TEXT_LINES = [
    "IN00000001",
    NAME,
    "Baseline Actual (Previous) Actual (Current) End Target",
    "Value 0.00 10.00 20.00 100.00",
    "Date 01-Apr-2016 01-Jan-2019 01-Jan-2020 30-Nov-2022",
    "Comments:",
    "Short synthetic comment.",
]


def _results(tables, lines=None, log=None, **kw):
    doc = isr_doc(tables, **kw)
    return extract_results(doc, text_source(doc, lines, log), date(2020, 1, 15))


def test_block_table_values_dates_and_provenance():
    observations, issues, fallback = _results([Tbl(3, BLOCK, after="Indicators.")])
    assert fallback == []
    obs = observations[0]
    assert obs.indicator_name_raw == "Direct project beneficiaries (Number, Corporate)"
    assert (obs.unit, obs.indicator_type, obs.layout) == ("Number", "PDO", "BLOCK")
    assert (obs.baseline_value, obs.current_value, obs.target_value) == ("0.00", "20.00", "100.00")
    assert obs.target_date == "30-Nov-2022"
    assert obs.indicator_id_source == "IN00000001"
    assert (obs.source_page, obs.source_table, obs.status) == (3, "t0001", ExtractionStatus.EXACT)
    assert obs.source_ref.extraction_method is ExtractionMethod.DOCLING_TABLE
    assert obs.observation_date == date(2020, 1, 15) and obs.isr_sequence == 5


def test_incomplete_block_table_falls_back_to_pdf_text():
    broken = [r[:] for r in BLOCK]
    broken[2] = ["Value", "0.00", "10.00", "20.00"]  # Docling dropped a cell
    log = FallbackLog()
    observations, issues, fallback = _results(
        [Tbl(3, broken, after="Indicators.")], {3: TEXT_LINES}, log
    )
    assert fallback == [3]
    obs = observations[0]
    assert obs.extraction_method is ExtractionMethod.PDF_TEXT_FALLBACK
    assert (obs.previous_value, obs.target_value) == ("10.00", "100.00")
    assert obs.comments == "Short synthetic comment."
    assert any(inv.element == "results" for inv in log.invocations)
    assert CheckCode.PDF_TEXT_FALLBACK_USED in {i.code for i in issues}


def test_incomplete_table_without_text_layer_is_not_invented():
    broken = [r[:] for r in BLOCK]
    broken[2] = ["Value", "0.00", "10.00", "20.00"]
    observations, _, fallback = _results([Tbl(3, broken, after="Indicators.")], {})
    assert fallback == [3]
    assert observations == []  # nothing is fabricated when neither source validates


def test_block_text_grammar_sets_area_and_type():
    items = parse_block_text(
        {5: ["Intermediate Results Indicators", "RA 1: Synthetic area", *TEXT_LINES]}, None
    )
    assert len(items) == 1 and items[0].valid
    assert (items[0].indicator_type, items[0].result_area) == (
        "INTERMEDIATE",
        "RA 1: Synthetic area",
    )


def test_wide_table_with_text_values_and_tags():
    rows = WIDE_HEADER + [
        [
            "Tap connections installed (Number) DLI",
            "0",
            "Jun/2023",
            "10",
            "15-Sep-2024",
            "20",
            "01-Mar-2025",
            "100",
            "Jun/2028",
        ],
        [
            "Policy adopted (Text)",
            "Not started",
            "Jun/2023",
            "Not started",
            "15-Sep-2024",
            "Drafted",
            "01-Mar-2025",
            "Adopted",
            "Jun/2028",
        ],
    ]
    observations, issues, _ = _results([Tbl(3, rows, after="Indicators.")])
    taps, policy = observations
    assert taps.indicator_name_raw == "Tap connections installed (Number)"
    assert taps.indicator_tags == ["DLI"] and taps.layout == "WIDE"
    assert (taps.baseline_date, taps.current_value, taps.target_date) == (
        "Jun/2023",
        "20",
        "Jun/2028",
    )
    assert (policy.baseline_value, policy.current_value) == ("Not started", "Drafted")


def test_wide_row_that_does_not_match_header_is_left_null():
    rows = WIDE_HEADER + [
        [
            "Merged objective text Direct beneficiaries (Number)",
            "piped 254,700",
            "water supply in Apr/2016",
            "",
            "",
            "",
            "",
            "",
            "",
        ]
    ]
    observations, issues, _ = _results([Tbl(3, rows, after="Indicators.")])
    obs = observations[0]
    assert obs.status is ExtractionStatus.AMBIGUOUS
    assert obs.baseline_value is None and obs.target_value is None
    assert CheckCode.RESULTS_TABLE_INVALID in {i.code for i in issues}


def test_loan_table_is_never_a_results_continuation():
    loans = [
        [
            "Loan/Credit/TF",
            "Status",
            "Original",
            "Revised",
            "Cancelled",
            "Disbursed",
            "Undisbursed",
            "% Disbursed",
            "x",
        ],
        [
            "IBRD-86010",
            "Effective",
            "100.00",
            "100.00",
            "0.00",
            "71.67",
            "28.33",
            "71.67%",
            "100.00",
        ],
    ]
    rows = WIDE_HEADER + [
        [
            "Indicator A (Number)",
            "0",
            "Jun/2023",
            "1",
            "15-Sep-2024",
            "2",
            "01-Mar-2025",
            "3",
            "Jun/2028",
        ]
    ]
    observations, _, _ = _results([Tbl(3, rows, after="Indicators."), Tbl(4, loans)])
    assert [o.indicator_name_raw for o in observations] == ["Indicator A (Number)"]


def test_dli_status_and_tags_split_from_name():
    assert split_name_suffixes("1:Tap connections installed (Number) Partially achieved") == (
        "1:Tap connections installed (Number)",
        [],
        "Partially achieved",
    )
    assert split_name_suffixes("X (Number) CRI DLI") == ("X (Number)", ["CRI", "DLI"], None)
    assert split_name_suffixes("DLI 1 name without unit") == ("DLI 1 name without unit", [], None)


def _observations(names_by_isr):
    out = []
    for sequence, names in names_by_isr.items():
        rows = WIDE_HEADER + [
            [n, "0", "Jun/2023", "1", "15-Sep-2024", "2", "01-Mar-2025", "3", "Jun/2028"]
            for n in names
        ]
        doc = isr_doc(
            [Tbl(3, rows, after="Indicators.")],
            isr_sequence=sequence,
            document_id=f"isr-{sequence}",
        )
        out += extract_results(doc, text_source(doc), None)[0]
    return out


def test_identity_exact_name_links_across_isrs_and_ids_do_not():
    observations = _observations(
        {1: ["Female plumbers educated (Number)"], 2: ["female  plumbers educated (Number)"]}
    )
    observations[0].indicator_id_source, observations[1].indicator_id_source = "IN1", "IN2"
    resolve_identities(observations)
    assert observations[0].indicator_key == observations[1].indicator_key
    assert {o.identity_basis for o in observations} == {"EXACT_NAME"}
    assert observations[0].indicator_key == indicator_key(
        "P130544", normalize_indicator_name("Female plumbers educated (Number)")
    )


def test_identity_possible_match_is_reported_not_merged():
    observations = _observations(
        {1: ["Female plumbers educated (Number, Custom)"], 2: ["Female plumbers educated (Number)"]}
    )
    issues = resolve_identities(observations)
    assert observations[0].indicator_key != observations[1].indicator_key
    matches = [i for i in issues if i.code == CheckCode.POSSIBLE_INDICATOR_MATCH]
    assert len(matches) == 1 and matches[0].details["isr_sequences"] == [[1], [2]]
    assert base_name("female plumbers educated (number, custom)") == "female plumbers educated"


def test_identity_documented_alias(tmp_path):
    path = tmp_path / "aliases.yaml"
    path.write_text(
        "projects:\n  P130544:\n    - canonical: 'New name (Number)'\n"
        "      aliases: ['Old name (Number)']\n",
        encoding="utf-8",
    )
    observations = _observations({1: ["Old name (Number)"], 2: ["New name (Number)"]})
    resolve_identities(observations, load_aliases(path))
    assert observations[0].indicator_key == observations[1].indicator_key
    assert observations[0].identity_basis == "ALIAS"
    assert load_aliases(tmp_path / "missing.yaml") == {}
