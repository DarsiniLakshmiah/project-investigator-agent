"""Docling adapter: DoclingDocument -> internal model (fast; needs docling-core only).

The end-to-end PDF test runs real Docling models and is marked ``docling``
(deselected by default; run with ``pytest -m docling``).
"""

import pytest

docling_core = pytest.importorskip("docling_core")

from docling_core.types.doc import (  # noqa: E402
    BoundingBox,
    ContentLayer,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    Size,
    TableCell,
    TableData,
)

from support.parsing_builders import minimal_pdf  # noqa: E402
from worldbank_copilot.parsing.docling_parser import (  # noqa: E402
    DoclingDocumentParser,
    convert_docling_document,
)
from worldbank_copilot.parsing.models import BlockLabel  # noqa: E402


def _prov(page, top, text, start=0):
    return ProvenanceItem(
        page_no=page,
        bbox=BoundingBox(l=50, t=top, r=500, b=top - 12, coord_origin=CoordOrigin.BOTTOMLEFT),
        charspan=(start, start + len(text)),
    )


def _document():
    doc = DoclingDocument(name="t")
    for page in (1, 2, 3):
        doc.add_page(page_no=page, size=Size(width=612, height=792))
    doc.add_text(
        label=DocItemLabel.PAGE_HEADER,
        text="The World Bank",
        prov=_prov(1, 780, "The World Bank"),
        content_layer=ContentLayer.FURNITURE,
    )
    doc.add_heading(text="I. BASIC DATA", level=1, prov=_prov(1, 700, "I. BASIC DATA"))
    first, second = "Yes Explanation", "This restructuring is processed."
    merged = doc.add_text(label=DocItemLabel.TEXT, text=f"{first} {second}",
                          prov=_prov(1, 100, first))  # fmt: skip
    merged.prov.append(_prov(2, 700, second, start=len(first) + 1))
    cells = [
        TableCell(text="Ln/Cr/Tf", start_row_offset_idx=0, end_row_offset_idx=1,
                  start_col_offset_idx=0, end_col_offset_idx=1, column_header=True),
        TableCell(text="Original Closing", start_row_offset_idx=0, end_row_offset_idx=1,
                  start_col_offset_idx=1, end_col_offset_idx=2, column_header=True),
        TableCell(text="IBRD-86010", start_row_offset_idx=1, end_row_offset_idx=2,
                  start_col_offset_idx=0, end_col_offset_idx=1),
        TableCell(text="30-Nov-2022", start_row_offset_idx=1, end_row_offset_idx=2,
                  start_col_offset_idx=1, end_col_offset_idx=2),
    ]  # fmt: skip
    doc.add_table(data=TableData(num_rows=2, num_cols=2, table_cells=cells),
                  prov=_prov(2, 600, ""))  # fmt: skip
    return doc


def test_conversion_preserves_pages_and_provenance():
    content = convert_docling_document(_document())
    assert content.page_count == 3
    assert [p.page_number for p in content.pages] == [1, 2, 3]
    assert content.pages[2].is_empty  # page 3 has no items

    header = next(b for b in content.blocks if b.text_raw == "The World Bank")
    assert header.is_furniture and header.label is BlockLabel.PAGE_HEADER

    # A text item spanning pages 1-2 is split per provenance fragment, keeping each page.
    pieces = {b.text_raw: b.page_number for b in content.blocks if b.label is BlockLabel.TEXT}
    assert pieces == {"Yes Explanation": 1, "This restructuring is processed.": 2}

    (section,) = content.sections
    assert (section.title, section.numbering, section.start_page, section.end_page) == (
        "I. BASIC DATA", "I", 1, 2)  # fmt: skip


def test_conversion_preserves_table_structure():
    content = convert_docling_document(_document())
    (table,) = content.tables
    assert table.page_number == 2 and table.page_numbers == [2]
    assert table.header_rows == 1 and table.headers == ["Ln/Cr/Tf", "Original Closing"]
    assert table.rows == [["Ln/Cr/Tf", "Original Closing"], ["IBRD-86010", "30-Nov-2022"]]
    assert "Original Closing: 30-Nov-2022" in table.text
    assert table.markdown and "IBRD-86010" in table.markdown
    assert table.section_id == content.sections[0].section_id


def test_text_inside_picture_regions_is_kept_and_tagged():
    doc = DoclingDocument(name="t")
    doc.add_page(page_no=1, size=Size(width=612, height=792))
    picture = doc.add_picture(prov=_prov(1, 500, ""))
    doc.add_text(
        label=DocItemLabel.TEXT,
        text="Systematic Operations Risk-rating Tool",
        prov=_prov(1, 480, "Systematic Operations Risk-rating Tool"),
        parent=picture,
    )
    content = convert_docling_document(doc)
    (block,) = content.blocks
    assert block.text_raw == "Systematic Operations Risk-rating Tool"
    assert block.source_label.endswith("@picture") and not block.is_furniture
    assert content.pages[0].picture_count == 1


def test_parser_info_is_stable_and_versioned():
    a, b = DoclingDocumentParser(), DoclingDocumentParser(num_threads=2)
    assert a.info.config_hash == b.info.config_hash  # threads do not change output
    assert a.info.config["do_ocr"] is False
    assert DoclingDocumentParser(table_mode="fast").info.config_hash != a.info.config_hash


@pytest.mark.docling
def test_real_docling_parse_of_generated_pdf(tmp_path):
    path = tmp_path / "sample.pdf"
    path.write_bytes(minimal_pdf(["Implementation Status Report page one", None,
                                  "Third page text"]))  # fmt: skip
    content = DoclingDocumentParser(num_threads=2).parse(path)
    assert content.page_count == 3
    texts = {p.page_number: p.text_clean for p in content.pages}
    assert "page one" in texts[1] and "Third page" in texts[3]
    assert content.pages[1].is_empty


@pytest.mark.docling
def test_real_docling_failure_is_a_parser_error(tmp_path):
    from worldbank_copilot.parsing.parser import ParserError

    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4 not really a pdf")
    with pytest.raises(ParserError):
        DoclingDocumentParser(num_threads=2).parse(path)
