"""Build ParsedContent / fake parsers for parsing tests (no Docling, no real PDFs)."""

from __future__ import annotations

from pathlib import Path

from worldbank_copilot.parsing.assembly import PageSize, RawBlock, assemble_content
from worldbank_copilot.parsing.models import (
    BlockLabel,
    BoundingBox,
    ParsedContent,
    ParsedTable,
    ParserInfo,
)
from worldbank_copilot.parsing.parser import ParserError, config_hash

H = BlockLabel.PAGE_HEADER
F = BlockLabel.PAGE_FOOTER
S = BlockLabel.SECTION_HEADER
T = BlockLabel.TEXT


def content_from_pages(
    pages: list[list[tuple[BlockLabel, str]]],
    tables: list[ParsedTable] | None = None,
    height: float = 792.0,
) -> ParsedContent:
    """Each page is a list of (label, text); blocks get a mid-page bbox."""
    raw, order = [], 0
    for number, blocks in enumerate(pages, start=1):
        for label, text in blocks:
            order += 10
            raw.append(RawBlock([number], label, text, order, label.value, None,
                                BoundingBox(left=50, top=400, right=500, bottom=380)))  # fmt: skip
    sizes = {n: PageSize(612.0, height) for n in range(1, len(pages) + 1)}
    return assemble_content(raw, tables or [], sizes, len(pages))


def isr_pages(pid="P130544", seq=5, archived="29-Nov-2017", header_date="11/29/2017",
              isr_number="ISR30462") -> list[list[tuple[BlockLabel, str]]]:  # fmt: skip
    header = f"{header_date} Page 1 of 2" if header_date else "Page 1 of 2"
    return [
        [
            (H, "The World Bank Implementation Status & Results Report"),
            (H, f"IN Karnataka Urban Water Supply Modernization Project ({pid}) {header}"),
            (T, "Public Disclosure Authorized"),
            (T, f"IN Karnataka Urban Water Supply Modernization Project ({pid})"),
            (T, f"SOUTH ASIA | India | IBRD/IDA | Investment Project Financing | FY 2016 | "
                f"Seq No: {seq} | ARCHIVED on {archived} | {isr_number} | Implementing Agencies"),
            (S, "Key Dates"),
            (T, "Original Closing Date: 30-Nov-2022 Revised Closing Date: 30-Nov-2022"),
        ],
        [(S, "Implementation Status and Key Decisions"),
         (T, "The project is progressing and contracts are being implemented. " * 25)],
    ]  # fmt: skip


class FakeParser:
    """Deterministic parser for pipeline tests: content keyed by filename."""

    name = "fake"

    def __init__(self, contents: dict[str, ParsedContent], fail: set[str] | None = None,
                 version: str = "1"):  # fmt: skip
        self.contents = contents
        self.fail = fail or set()
        self.calls: list[str] = []
        self.version = version

    @property
    def info(self) -> ParserInfo:
        config = {"mode": "test"}
        return ParserInfo(
            name=self.name,
            version=self.version,
            config=config,
            config_hash=config_hash(self.name, self.version, config),
        )

    def parse(self, path: Path) -> ParsedContent:
        self.calls.append(path.name)
        if path.name in self.fail:
            raise ParserError("simulated parser failure")
        return self.contents[path.name]


def minimal_pdf(pages: list[str | None]) -> bytes:
    """A valid PDF with one text line per page (None = blank page)."""
    objects: list[bytes] = []
    kids = []
    font_id = 3
    for i, text in enumerate(pages):
        page_id = 4 + 2 * i
        content_id = page_id + 1
        kids.append(f"{page_id} 0 R")
        stream = (
            b"" if text is None else (f"BT /F1 14 Tf 72 700 Td ({text}) Tj ET".encode("latin-1"))
        )
        objects.append(
            (
                page_id,
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
                f"/Contents {content_id} 0 R >>".encode(),
            )
        )
        objects.append(
            (content_id, b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        )
    header = [
        (1, b"<< /Type /Catalog /Pages 2 0 R >>"),
        (2, f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>".encode()),
        (3, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
    ]
    all_objects = sorted(header + objects)
    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for number, body in all_objects:
        offsets[number] = len(out)
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(all_objects) + 1}\n0000000000 65535 f \n".encode()
    for number in range(1, len(all_objects) + 1):
        out += f"{offsets[number]:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(all_objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF"
            ).encode()  # fmt: skip
    return bytes(out)
