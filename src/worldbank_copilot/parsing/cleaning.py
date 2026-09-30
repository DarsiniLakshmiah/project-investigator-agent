"""Conservative text cleaning for parsed documents.

Allowed: Unicode normalisation (NFC, ligatures, invisible characters), whitespace
normalisation, hyphenated line-break repair between lowercase letters, and
marking page furniture (running headers/footers, page numbers) so that it is
excluded from *clean* text. Raw text is always kept. No rewriting, summarising
or LLM involvement; tables are never cleaned.

Furniture is only marked when one of these holds:

1. the parser labels the block as a page header/footer, or places it in its
   furniture content layer;
2. the block is a bare page-number artifact ("7", "Page 7 of 10");
3. the block consists *only* of disclosure stamps ("Public Disclosure
   Authorized", "Public Disclosure Copy", "For Official Use Only", "Official Use
   Only"), which World Bank PDFs stamp in the margins;
4. the same short text (digits masked) recurs on many pages *and* sits in the
   top/bottom margin band. Repetition alone never removes content.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

from worldbank_copilot.parsing.models import BlockLabel, TextBlock

_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}
_INVISIBLE = dict.fromkeys(map(ord, "­​‌‍﻿"))
_SPACES = re.compile(r"[ \t  -   　]+")
_HYPHEN_BREAK = re.compile(r"(?<=[a-z])-\n(?=[a-z])")
_PAGE_NUMBER = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.IGNORECASE)
_STAMP_PHRASE = r"(?:public disclosure (?:authorized|copy)|(?:for )?official use only)"
_STAMP_ONLY = re.compile(rf"^(?:{_STAMP_PHRASE}\s*)+$", re.IGNORECASE)
FURNITURE_LAYER_SUFFIX = "@furniture"

MARGIN_BAND = 0.12  # fraction of page height treated as header/footer band
REPEAT_MIN_PAGES = 3
REPEAT_MIN_SHARE = 0.5
REPEAT_MAX_CHARS = 200


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for ligature, replacement in _LIGATURES.items():
        text = text.replace(ligature, replacement)
    return text.translate(_INVISIBLE)


def repair_hyphenation(text: str) -> str:
    """Join 'imple-\\nmentation' -> 'implementation' (lowercase on both sides only)."""
    return _HYPHEN_BREAK.sub("", text)


def normalize_whitespace(text: str) -> str:
    lines = [_SPACES.sub(" ", line).strip() for line in text.splitlines()]
    return " ".join(line for line in lines if line)


def clean_block_text(raw: str) -> str:
    return normalize_whitespace(repair_hyphenation(normalize_unicode(raw)))


def _signature(text: str) -> str:
    return re.sub(r"\d+", "#", normalize_whitespace(text).lower())


def _in_margin(block: TextBlock, page_heights: dict[int, float | None]) -> bool:
    height = page_heights.get(block.page_number)
    if block.bbox is None or not height:
        return False
    top, bottom = block.bbox.top, block.bbox.bottom
    if block.bbox.coord_origin.upper() == "BOTTOMLEFT":
        top, bottom = height - top, height - bottom  # distances from the page top
    upper, lower = sorted((top, bottom))
    return upper <= height * MARGIN_BAND or lower >= height * (1 - MARGIN_BAND)


def mark_furniture(
    blocks: list[TextBlock], page_count: int, page_heights: dict[int, float | None]
) -> list[TextBlock]:
    """Return blocks with ``is_furniture`` / ``furniture_reason`` set (inputs unchanged)."""
    pages_by_signature: dict[str, set[int]] = defaultdict(set)
    for block in blocks:
        if len(block.text_clean) <= REPEAT_MAX_CHARS:
            pages_by_signature[_signature(block.text_raw)].add(block.page_number)
    threshold = max(REPEAT_MIN_PAGES, page_count * REPEAT_MIN_SHARE)

    result = []
    for block in blocks:
        reason = None
        if block.label in (BlockLabel.PAGE_HEADER, BlockLabel.PAGE_FOOTER):
            reason = f"parser_label:{block.label.value}"
        elif (block.source_label or "").endswith(FURNITURE_LAYER_SUFFIX):
            reason = "parser_content_layer:furniture"
        elif _PAGE_NUMBER.match(block.text_clean):
            reason = "page_number"
        elif _STAMP_ONLY.match(block.text_clean):
            reason = "disclosure_stamp"
        elif (
            len(block.text_clean) <= REPEAT_MAX_CHARS
            and len(pages_by_signature[_signature(block.text_raw)]) >= threshold
            and _in_margin(block, page_heights)
        ):
            reason = "repeated_margin_text"
        result.append(
            block.model_copy(
                update={"is_furniture": reason is not None, "furniture_reason": reason}
            )
        )
    return result
