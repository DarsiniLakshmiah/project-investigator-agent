"""Docling implementation of ``DocumentParser``.

This is the only module that imports Docling. Docling is imported lazily so
the rest of the package (and the unit tests) work without it installed.

Configuration (recorded in ``ParserInfo`` and part of the parse cache key):
* OCR off: every in-scope PDF has a text layer (verified in Phase 1); OCR would
  be slow and could replace good embedded text with recognition errors.
* Table structure on, TableFormer ``accurate`` mode, cell matching on.
* All content layers are read (body and furniture); furniture is marked, not dropped.

Thread count does not change output, so it is not part of the cache key.
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from worldbank_copilot.parsing.assembly import (
    ASSEMBLY_VERSION,
    PageSize,
    RawBlock,
    assemble_content,
    render_table_text,
)
from worldbank_copilot.parsing.cleaning import FURNITURE_LAYER_SUFFIX
from worldbank_copilot.parsing.models import (
    BlockLabel,
    BoundingBox,
    ParsedContent,
    ParsedTable,
    ParserInfo,
)
from worldbank_copilot.parsing.parser import ParserError, config_hash

_LABELS = {
    "title": BlockLabel.TITLE,
    "section_header": BlockLabel.SECTION_HEADER,
    "text": BlockLabel.TEXT,
    "paragraph": BlockLabel.TEXT,
    "reference": BlockLabel.TEXT,
    "list_item": BlockLabel.LIST_ITEM,
    "caption": BlockLabel.CAPTION,
    "footnote": BlockLabel.FOOTNOTE,
    "page_header": BlockLabel.PAGE_HEADER,
    "page_footer": BlockLabel.PAGE_FOOTER,
}


class _WarningCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(f"{record.name}: {record.getMessage()}")


@contextmanager
def _collect_warnings(logger_name: str = "") -> Iterator[list[str]]:
    """Collect WARNING+ records during conversion (root logger: Docling's table
    post-processor does not log under the ``docling`` namespace)."""
    collector = _WarningCollector()
    logger = logging.getLogger(logger_name)
    logger.addHandler(collector)
    try:
        yield collector.messages
    finally:
        logger.removeHandler(collector)


class DoclingDocumentParser:
    name = "docling"

    def __init__(
        self,
        *,
        table_mode: str = "accurate",
        do_cell_matching: bool = True,
        num_threads: int | None = None,
    ) -> None:
        self._config = {
            "do_ocr": False,
            "do_table_structure": True,
            "table_mode": table_mode,
            "do_cell_matching": do_cell_matching,
            "content_layers": "all",
            "traverse_pictures": True,
            "assembly_version": ASSEMBLY_VERSION,
        }
        self._num_threads = num_threads or os.cpu_count() or 4
        self._converter: Any = None

    @property
    def info(self) -> ParserInfo:
        version = importlib.metadata.version("docling")
        return ParserInfo(
            name=self.name,
            version=version,
            config=self._config,
            config_hash=config_hash(self.name, version, self._config),
        )

    def _get_converter(self) -> Any:
        if self._converter is None:
            from docling.datamodel import pipeline_options as po
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import DocumentConverter, PdfFormatOption

            options = po.PdfPipelineOptions(
                do_ocr=False,
                do_table_structure=True,
                accelerator_options=po.AcceleratorOptions(num_threads=self._num_threads),
            )
            options.table_structure_options.mode = po.TableFormerMode(self._config["table_mode"])
            options.table_structure_options.do_cell_matching = self._config["do_cell_matching"]
            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
            )
        return self._converter

    def parse(self, path: Path) -> ParsedContent:
        from docling.datamodel.base_models import ConversionStatus

        converter = self._get_converter()
        with _collect_warnings() as warnings:
            try:
                result = converter.convert(str(path), raises_on_error=False)
            except Exception as exc:  # noqa: BLE001 - surface any parser crash as ParserError
                raise ParserError(f"Docling raised {type(exc).__name__}: {exc}") from exc
        status = result.status
        if status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
            errors = "; ".join(str(e) for e in result.errors) or "no details"
            raise ParserError(f"Docling conversion status {status}: {errors}")
        if status == ConversionStatus.PARTIAL_SUCCESS:
            warnings.append(f"docling: partial success: {[str(e) for e in result.errors]}")
        return convert_docling_document(result.document, warnings)


# ---------------------------------------------------------------------------
# DoclingDocument -> ParsedContent (pure; testable with a hand-built document)
# ---------------------------------------------------------------------------


def _bbox(prov: Any) -> BoundingBox:
    box = prov.bbox
    origin = getattr(box.coord_origin, "value", str(box.coord_origin))
    return BoundingBox(left=box.l, top=box.t, right=box.r, bottom=box.b, coord_origin=str(origin))


def _segments(text: str, provs: list[Any]) -> list[tuple[str, Any]] | None:
    """Split a multi-provenance item into (text, prov) fragments via charspans."""
    pieces = []
    for prov in provs:
        span = getattr(prov, "charspan", None)
        if not span or len(span) != 2:
            return None
        start, end = span
        if not (0 <= start <= end <= len(text)):
            return None
        piece = text[start:end].strip()
        if piece:
            pieces.append((piece, prov))
    return pieces or None


PICTURE_SUFFIX = "@picture"


def _inside_picture(item: Any, doc: Any) -> bool:
    """True if the item is nested (at any depth) under a picture region."""
    from docling_core.types.doc import PictureItem

    parent_ref = getattr(item, "parent", None)
    for _ in range(20):  # bounded walk up the tree
        if parent_ref is None:
            return False
        try:
            parent = parent_ref.resolve(doc)
        except Exception:  # noqa: BLE001 - malformed reference: treat as not nested
            return False
        if isinstance(parent, PictureItem):
            return True
        parent_ref = getattr(parent, "parent", None)
    return False


def _label(item: Any, doc: Any = None) -> tuple[BlockLabel, str]:
    raw = getattr(item.label, "value", str(item.label))
    layer = getattr(getattr(item, "content_layer", None), "value", None)
    source = f"{raw}{FURNITURE_LAYER_SUFFIX}" if layer == "furniture" else raw
    if doc is not None and _inside_picture(item, doc):
        source += PICTURE_SUFFIX
    return _LABELS.get(raw, BlockLabel.OTHER), source


def _table(item: Any, doc: Any, order: int, index: int) -> ParsedTable:
    grid = item.data.grid
    rows = [[(cell.text or "").strip() for cell in row] for row in grid]
    header_rows = 0
    for row in grid:
        filled = [c for c in row if (c.text or "").strip()]
        if filled and all(c.column_header for c in filled):
            header_rows += 1
        else:
            break
    headers = []
    if header_rows:
        for column in zip(*rows[:header_rows], strict=False):
            headers.append(" / ".join(dict.fromkeys(v for v in column if v)))
    caption = (item.caption_text(doc) or "").strip() or None
    pages = sorted({p.page_no for p in item.prov}) or [0]
    return ParsedTable(
        table_id=f"t{index:04d}",
        page_number=pages[0],
        page_numbers=pages,
        caption=caption,
        n_rows=len(rows),
        n_cols=max((len(r) for r in rows), default=0),
        header_rows=header_rows,
        headers=headers,
        rows=rows,
        markdown=item.export_to_markdown(doc),
        text=render_table_text(rows, header_rows, caption),
        bbox=_bbox(item.prov[0]) if item.prov else None,
        reading_order=order,
    )


def convert_docling_document(doc: Any, warnings: list[str] | None = None) -> ParsedContent:
    from docling_core.types.doc import ContentLayer, PictureItem, TableItem

    warnings = list(warnings or [])
    page_sizes = {
        int(no): PageSize(width=page.size.width, height=page.size.height)
        for no, page in doc.pages.items()
    }
    page_count = max(page_sizes, default=0)
    raw_blocks: list[RawBlock] = []
    tables: list[ParsedTable] = []
    pictures: Counter[int] = Counter()
    order = 0

    # traverse_pictures=True: Docling nests text found inside picture regions under the
    # picture item; skipping it lost real content (e.g. an ISR "Risks" heading and the
    # "Systematic Operations Risk-rating Tool" table title). Such blocks are tagged "@picture".
    items = doc.iterate_items(included_content_layers=set(ContentLayer), traverse_pictures=True)
    for item, _depth in items:
        order += 1
        provs = list(getattr(item, "prov", []) or [])
        if isinstance(item, TableItem):
            if not provs:
                warnings.append(f"table without page provenance skipped (order {order})")
                continue
            tables.append(_table(item, doc, order * 100, len(tables) + 1))
            continue
        if isinstance(item, PictureItem):
            for prov in provs:
                pictures[prov.page_no] += 1
            continue
        text = getattr(item, "text", None)
        if not text or not text.strip():
            continue
        if not provs:
            warnings.append(f"text without page provenance skipped: {text[:60]!r}")
            continue
        label, source_label = _label(item, doc)
        level = getattr(item, "level", None) if label is BlockLabel.SECTION_HEADER else None
        fragments = _segments(text, provs) if len(provs) > 1 else [(text, provs[0])]
        if fragments is None:
            pages = sorted({p.page_no for p in provs})
            if len(pages) > 1:
                warnings.append(f"multi-page text without usable charspans kept whole on "
                                f"pages {pages}: {text[:60]!r}")  # fmt: skip
            raw_blocks.append(RawBlock(pages, label, text, order * 100, source_label, level,
                                       _bbox(provs[0])))  # fmt: skip
            continue
        for sub, (piece, prov) in enumerate(fragments):
            raw_blocks.append(RawBlock([prov.page_no], label, piece, order * 100 + sub,
                                       source_label, level, _bbox(prov)))  # fmt: skip

    return assemble_content(raw_blocks, tables, page_sizes, page_count, warnings, dict(pictures))
