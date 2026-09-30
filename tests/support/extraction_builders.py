"""Small structural ParsedDocument fixtures for Phase 5 extraction tests.

Fixtures reproduce document *layouts* (table shapes, headers, label grammar)
with short synthetic strings. They contain no copied document text.
"""

from __future__ import annotations

from datetime import date

from tests.support.parsing_builders import FakeParser, content_from_pages

from worldbank_copilot.extraction.text_source import DictTextSource, FallbackLog, LoggedTextSource
from worldbank_copilot.parsing.document_builder import build_document
from worldbank_copilot.parsing.models import BlockLabel, ParsedDocument, ParsedTable

S = BlockLabel.SECTION_HEADER
T = BlockLabel.TEXT


class Tbl:
    """A table placed on ``page`` right after the block whose text is ``after``."""

    def __init__(self, page: int, rows: list[list[str]], after: str | None = None):
        self.page, self.rows, self.after = page, rows, after


def make_doc(
    pages: list[list[tuple[BlockLabel, str]]],
    tables: list[Tbl] = (),
    *,
    project_id: str = "P130544",
    document_type: str = "ISR",
    filename: str = "doc.pdf",
    document_id: str | None = None,
    isr_sequence: int | None = None,
    report_date: date | None = None,
    archive_date: date | None = None,
    document_date: date | None = None,
    source_hash: str = "0" * 64,
) -> ParsedDocument:
    orders: dict[tuple[int, str], int] = {}
    last: dict[int, int] = {}
    order = 0
    for number, blocks in enumerate(pages, start=1):
        for _, text in blocks:
            order += 10
            orders.setdefault((number, text), order)
            last[number] = order
    parsed_tables = []
    for index, spec in enumerate(tables, start=1):
        anchor = orders.get((spec.page, spec.after)) if spec.after else last.get(spec.page, 0)
        n_cols = max((len(r) for r in spec.rows), default=0)
        parsed_tables.append(
            ParsedTable(
                table_id=f"t{index:04d}",
                page_number=spec.page,
                page_numbers=[spec.page],
                n_rows=len(spec.rows),
                n_cols=n_cols,
                rows=spec.rows,
                text="\n".join(" | ".join(r) for r in spec.rows),
                reading_order=(anchor or 0) + 1 + index % 5,
            )
        )
    content = content_from_pages(pages, parsed_tables)
    doc_id = document_id or f"{project_id}-{filename}"
    record = {
        "project_id": project_id,
        "document_id": doc_id,
        "filename": filename,
        "relative_path": f"{project_id}/{filename}",
        "sha256": source_hash,
        "file_size_bytes": 1,
        "document_type": document_type,
        "classification_method": "MANIFEST",
    }
    doc = build_document(record, content, FakeParser({}).info, "t", 1.0)
    return doc.model_copy(
        update={
            "document_type": document_type,
            "isr_sequence": isr_sequence,
            "report_date": report_date,
            "archive_date": archive_date,
            "document_date": document_date,
            "source_hash": source_hash,
        }
    )


def text_source(
    doc: ParsedDocument, pages: dict[int, list[str]] | None = None, log: FallbackLog | None = None
) -> LoggedTextSource:
    return LoggedTextSource(
        DictTextSource(pages or {}), log or FallbackLog(), doc.document_id, doc.filename
    )


RATINGS_ROWS = [
    ["Name", "Previous Rating", "Current Rating"],
    ["Progress towards achievement of PDO", "Moderately Unsatisfactory", "Moderately Satisfactory"],
    ["Overall Implementation Progress (IP)", "Moderately Satisfactory", "Satisfactory"],
    ["Overall Risk Rating", "Substantial", "Substantial"],
]

WIDE_HEADER = [
    [
        "Indicator Name",
        "Baseline",
        "Baseline",
        "Actual (Previous)",
        "Actual (Previous)",
        "Actual (Current)",
        "Actual (Current)",
        "Closing Period",
        "Closing Period",
    ],
    ["", "Value", "Month/Year", "Value", "Date", "Value", "Date", "Value", "Month/Year"],
]


def isr_doc(
    tables: list[Tbl], extra_pages: list[list[tuple[BlockLabel, str]]] = (), **kw
) -> ParsedDocument:
    pages = [
        [
            (S, "Key Dates"),
            (T, "Synthetic key dates block."),
            (S, "Overall Ratings"),
            (T, "Ratings follow."),
        ],
        [
            (S, "Implementation Status and Key Decisions"),
            (T, "Status narrative A."),
            (S, "Key Issues"),
            (T, "Issue narrative B."),
        ],
        [(S, "Results"), (S, "PDO Indicators by Objectives / Outcomes"), (T, "Indicators.")],
        [
            (S, "Disbursements (by loan)"),
            (T, "Loans."),
            (S, "Key Dates (by loan)"),
            (T, "Dates."),
            (S, "Restructuring History"),
            (T, "Level 2 Approved on 20-May-2021"),
            (T, "Restructuring Level 2 Approved on 23-Jul-2024"),
        ],
        *extra_pages,
    ]
    defaults = {
        "isr_sequence": 5,
        "report_date": date(2020, 1, 15),
        "archive_date": date(2020, 1, 10),
    }
    defaults.update(kw)
    return make_doc(pages, tables, **defaults)
