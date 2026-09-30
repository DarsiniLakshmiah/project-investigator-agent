"""Cleaning, furniture marking, section/page assembly and table rendering."""

from support.parsing_builders import F, H, S, T, content_from_pages
from worldbank_copilot.parsing.assembly import (
    FRONT_MATTER,
    PageSize,
    RawBlock,
    assemble_content,
    render_table_text,
)
from worldbank_copilot.parsing.cleaning import (
    clean_block_text,
    mark_furniture,
    normalize_unicode,
    repair_hyphenation,
)
from worldbank_copilot.parsing.models import BlockLabel, BoundingBox, ParsedTable, TextBlock


def test_unicode_normalisation():
    assert normalize_unicode("é") == "é"  # NFC
    assert normalize_unicode("ﬁnancial ﬂow") == "financial flow"
    assert normalize_unicode("imple­mentation​") == "implementation"


def test_hyphenation_repair_is_conservative():
    assert repair_hyphenation("imple-\nmentation") == "implementation"
    assert repair_hyphenation("Hubballi-\nDharwad") == "Hubballi-\nDharwad"  # capital: kept
    assert repair_hyphenation("co-financing") == "co-financing"


def test_whitespace_normalisation_keeps_words():
    assert clean_block_text("  The  Project\n  has   overcome challenges ") == (
        "The Project has overcome challenges"
    )


def _block(i, text, page, label=BlockLabel.TEXT, top=400.0, source=None):
    return TextBlock(block_id=f"b{i}", page_number=page, page_numbers=[page], label=label,
                     source_label=source, text_raw=text, text_clean=clean_block_text(text),
                     bbox=BoundingBox(left=50, top=top, right=500, bottom=top - 10),
                     reading_order=i)  # fmt: skip


def test_furniture_rules():
    heights = {p: 792.0 for p in range(1, 5)}
    blocks = [
        _block(1, "The World Bank", 1, BlockLabel.PAGE_HEADER),
        _block(2, "Header text", 1, source="text@furniture"),
        _block(3, "Page 3 of 10", 1),
        _block(4, "Public Disclosure Authorized Public Disclosure Authorized", 1),
        _block(5, "Public Disclosure Authorized WATER", 1),  # mixed content: kept
    ]
    # Same short line on every page in the top margin -> furniture; mid-page -> kept.
    blocks += [_block(10 + p, f"Confidential draft {p}", p, top=780.0) for p in range(1, 5)]
    blocks += [_block(20 + p, "Key Dates", p, top=400.0) for p in range(1, 5)]
    marked = {b.block_id: b.furniture_reason for b in mark_furniture(blocks, 4, heights)}
    assert marked["b1"] == "parser_label:page_header"
    assert marked["b2"] == "parser_content_layer:furniture"
    assert marked["b3"] == "page_number"
    assert marked["b4"] == "disclosure_stamp"
    assert marked["b5"] is None
    assert all(marked[f"b{10 + p}"] == "repeated_margin_text" for p in range(1, 5))
    assert all(marked[f"b{20 + p}"] is None for p in range(1, 5))  # repetition alone: kept


def test_pages_sections_and_provenance():
    content = content_from_pages([
        [(H, "The World Bank"), (T, "Cover text"), (S, "I. BASIC DATA"), (T, "Basic data.")],
        [(F, "Page 2 of 3"), (S, "II. SUMMARY OF PROJECT STATUS"), (T, "Status text.")],
        [],
    ])  # fmt: skip
    assert content.page_count == 3
    assert [p.page_number for p in content.pages] == [1, 2, 3]
    front, basic, summary = content.sections
    assert front.title == FRONT_MATTER and front.heading_block_id is None
    assert (basic.title, basic.numbering, basic.start_page) == ("I. BASIC DATA", "I", 1)
    assert (summary.numbering, summary.start_page, summary.end_page) == ("II", 2, 2)
    page1 = content.pages[0]
    assert "The World Bank" in page1.text_raw and "The World Bank" not in page1.text_clean
    assert content.pages[2].is_empty and content.pages[2].text_raw == ""
    for block in content.blocks:
        assert block.page_numbers == [block.page_number]
        if not block.is_furniture:
            assert block.section_id is not None


def test_multi_page_block_keeps_all_pages():
    raw = [RawBlock([2, 3], BlockLabel.TEXT, "spans pages", 10)]
    content = assemble_content(raw, [], {2: PageSize(), 3: PageSize()}, 3)
    (block,) = content.blocks
    assert block.page_number == 2 and block.page_numbers == [2, 3]


def test_tables_keep_page_and_section():
    table = ParsedTable(
        table_id="t0001",
        page_number=2,
        page_numbers=[2],
        n_rows=2,
        n_cols=2,
        header_rows=1,
        headers=["Ln", "Closing"],
        rows=[["Ln", "Closing"], ["IBRD-86010", "30-Nov-2022"]],
        text="Ln: IBRD-86010 | Closing: 30-Nov-2022",
        reading_order=25,
    )
    content = content_from_pages([[(T, "a")], [(S, "LOAN CLOSING DATE(S)")]], tables=[table])
    (t,) = content.tables
    assert t.page_number == 2
    assert content.section(t.section_id).title == "LOAN CLOSING DATE(S)"
    assert "IBRD-86010" in content.pages[1].text_clean  # table text is part of page text
    assert content.pages[1].table_ids == ["t0001"]


def test_table_text_rendering():
    rows = [["", "", "Net", "Net"], ["Ln/Cr/Tf", "Approval", "Closing", "Commitment"],
            ["IBRD-86010", "31-Mar-2016", "30-Nov-2022", "100.00"]]  # fmt: skip
    text = render_table_text(rows, header_rows=2, caption="Financing")
    assert text.splitlines()[0] == "Table: Financing"
    assert "Ln/Cr/Tf: IBRD-86010" in text and "Net / Commitment: 100.00" in text


def test_table_text_collapses_spanned_cells():
    rows = [["PDO", "PDO", "PDO"], ["Indicator", "10", "20"]]
    assert render_table_text(rows, header_rows=0).splitlines()[0] == "PDO"


def test_picture_counts_recorded():
    content = assemble_content([], [], {1: PageSize()}, 1, picture_counts={1: 2})
    assert content.pages[0].picture_count == 2 and content.pages[0].is_empty
