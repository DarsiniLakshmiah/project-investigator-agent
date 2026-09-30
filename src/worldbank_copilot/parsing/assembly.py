"""Parser-independent assembly: raw elements -> cleaned blocks, sections, pages.

A parser adapter only has to emit ``RawBlock``s and ``ParsedTable``s in reading
order with their page provenance. Cleaning, furniture marking, section building
and page aggregation happen here, identically for every parser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from worldbank_copilot.parsing.cleaning import clean_block_text, mark_furniture
from worldbank_copilot.parsing.models import (
    BlockLabel,
    BoundingBox,
    ParsedContent,
    ParsedPage,
    ParsedTable,
    Section,
    TextBlock,
)

# "1.", "1.1", "I. BASIC DATA", "III.DETAILED CHANGES" (no space after roman numeral), "A."
_NUMBERING = re.compile(
    r"^((?:\d{1,2}(?:\.\d{1,2})*\.?(?=\s))|(?:[IVXLC]+\.)|(?:[A-H]\.(?=\s)))\s*(?=\S)"
)
FRONT_MATTER = "(front matter)"
# Bump when assembly/cleaning output changes, so cached parses are invalidated.
ASSEMBLY_VERSION = "3"


@dataclass
class RawBlock:
    page_numbers: list[int]
    label: BlockLabel
    text: str
    reading_order: int
    source_label: str | None = None
    heading_level: int | None = None
    bbox: BoundingBox | None = None


@dataclass
class PageSize:
    width: float | None = None
    height: float | None = None


def render_table_text(rows: list[list[str]], header_rows: int, caption: str | None = None) -> str:
    """Readable row-wise text: 'Header: value | Header: value' per data row."""
    lines = [f"Table: {caption}"] if caption else []
    headers = []
    if header_rows:
        columns = list(zip(*rows[:header_rows], strict=False))
        headers = [" / ".join(dict.fromkeys(c for c in col if c)) for col in columns]
    for row in rows[header_rows:]:
        parts = []
        previous = None
        for col, value in enumerate(row):
            # A cell spanning several columns is repeated in each grid slot: render it once.
            if value and value != previous:
                header = headers[col] if col < len(headers) else ""
                parts.append(f"{header}: {value}" if header else value)
            previous = value
        if parts:
            lines.append(" | ".join(parts))
    return "\n".join(lines)


@dataclass
class _SectionBuilder:
    section: Section
    pages: set[int] = field(default_factory=set)


def assemble_content(
    raw_blocks: list[RawBlock],
    tables: list[ParsedTable],
    page_sizes: dict[int, PageSize],
    page_count: int,
    parser_warnings: list[str] | None = None,
    picture_counts: dict[int, int] | None = None,
) -> ParsedContent:
    blocks = [
        TextBlock(
            block_id=f"b{i:05d}",
            page_number=raw.page_numbers[0],
            page_numbers=raw.page_numbers,
            label=raw.label,
            source_label=raw.source_label,
            heading_level=raw.heading_level,
            text_raw=raw.text,
            text_clean=clean_block_text(raw.text),
            bbox=raw.bbox,
            reading_order=raw.reading_order,
        )
        for i, raw in enumerate(sorted(raw_blocks, key=lambda b: b.reading_order), start=1)
    ]
    blocks = mark_furniture(blocks, page_count, {p: s.height for p, s in page_sizes.items()})
    blocks, tables, sections = _build_sections(blocks, tables)
    pages = _build_pages(blocks, tables, page_sizes, page_count, picture_counts or {})
    return ParsedContent(
        page_count=page_count,
        pages=pages,
        blocks=blocks,
        tables=tables,
        sections=sections,
        parser_warnings=parser_warnings or [],
    )


def _build_sections(
    blocks: list[TextBlock], tables: list[ParsedTable]
) -> tuple[list[TextBlock], list[ParsedTable], list[Section]]:
    elements = sorted(
        [("block", b.reading_order, b) for b in blocks]
        + [("table", t.reading_order, t) for t in tables],
        key=lambda e: e[1],
    )
    builders: list[_SectionBuilder] = []
    current: _SectionBuilder | None = None
    block_section: dict[str, str] = {}
    table_section: dict[str, str] = {}

    for kind, _, element in elements:
        is_heading = (
            kind == "block"
            and element.label in (BlockLabel.SECTION_HEADER, BlockLabel.TITLE)
            and not element.is_furniture
            and element.text_clean
        )
        if is_heading:
            match = _NUMBERING.match(element.text_clean)
            current = _SectionBuilder(
                Section(
                    section_id=f"s{len(builders) + 1:04d}",
                    title=element.text_clean,
                    level=element.heading_level,
                    numbering=match.group(1).rstrip(".") if match else None,
                    heading_block_id=element.block_id,
                    start_page=element.page_number,
                    end_page=element.page_number,
                )
            )
            builders.append(current)
        elif current is None:
            if kind == "block" and element.is_furniture:
                continue  # furniture before any content does not open a section
            current = _SectionBuilder(
                Section(section_id="s0000", title=FRONT_MATTER, start_page=element.page_number,
                        end_page=element.page_number)
            )  # fmt: skip
            builders.append(current)
        if kind == "block":
            if element.is_furniture:
                continue  # furniture belongs to the page, not to a section
            current.section.block_ids.append(element.block_id)
            block_section[element.block_id] = current.section.section_id
        else:
            current.section.table_ids.append(element.table_id)
            table_section[element.table_id] = current.section.section_id
        current.pages.update(element.page_numbers)

    sections = []
    for builder in builders:
        pages = builder.pages or {builder.section.start_page}
        sections.append(
            builder.section.model_copy(update={"start_page": min(pages), "end_page": max(pages)})
        )
    blocks = [b.model_copy(update={"section_id": block_section.get(b.block_id)}) for b in blocks]
    tables = [t.model_copy(update={"section_id": table_section.get(t.table_id)}) for t in tables]
    return blocks, tables, sections


def _build_pages(
    blocks: list[TextBlock],
    tables: list[ParsedTable],
    page_sizes: dict[int, PageSize],
    page_count: int,
    picture_counts: dict[int, int],
) -> list[ParsedPage]:
    pages = []
    for number in range(1, page_count + 1):
        page_blocks = [b for b in blocks if b.page_number == number]
        page_tables = [t for t in tables if t.page_number == number]
        elements = sorted(
            [(b.reading_order, b.text_raw, b.text_clean if not b.is_furniture else None)
             for b in page_blocks]
            + [(t.reading_order, t.text, t.text) for t in page_tables],
            key=lambda e: e[0],
        )  # fmt: skip
        raw = "\n".join(text for _, text, _ in elements if text)
        clean = "\n".join(text for _, _, text in elements if text)
        size = page_sizes.get(number, PageSize())
        pages.append(
            ParsedPage(
                page_number=number,
                width=size.width,
                height=size.height,
                block_ids=[b.block_id for b in page_blocks],
                table_ids=[t.table_id for t in page_tables],
                heading_block_ids=[
                    b.block_id
                    for b in page_blocks
                    if b.label in (BlockLabel.SECTION_HEADER, BlockLabel.TITLE)
                ],
                text_raw=raw,
                text_clean=clean,
                char_count_clean=len(clean),
                picture_count=picture_counts.get(number, 0),
                is_empty=not clean.strip(),
            )
        )
    return pages
