"""Targeted PDF text-layer access for extraction fallbacks.

Docling output (the cached ParsedDocument) is always the primary source. The PDF
text layer is opened only when an extractor's quality check fails for a specific
page, and every read is logged (document, page, element, reason) so each fallback
invocation can be reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class PageTextSource(Protocol):
    def page_lines(self, page_number: int) -> list[str] | None: ...


@dataclass
class FallbackInvocation:
    document_id: str
    filename: str
    page_number: int
    element: str
    reason: str


@dataclass
class FallbackLog:
    invocations: list[FallbackInvocation] = field(default_factory=list)

    def record(self, document_id: str, filename: str, page: int, element: str, reason: str):
        self.invocations.append(FallbackInvocation(document_id, filename, page, element, reason))


class PdfTextLayer:
    """Lazily opens a PDF with pypdfium2 and returns non-empty lines per page."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._pages: dict[int, list[str]] | None = None

    def _load(self) -> dict[int, list[str]]:
        if self._pages is None:
            import pypdfium2

            pages: dict[int, list[str]] = {}
            pdf = pypdfium2.PdfDocument(str(self.path))
            try:
                for index, page in enumerate(pdf, start=1):
                    text_page = page.get_textpage()
                    text = text_page.get_text_range()
                    text_page.close()
                    page.close()
                    pages[index] = [
                        " ".join(line.split()) for line in text.splitlines() if line.strip()
                    ]
            finally:
                pdf.close()
            self._pages = pages
        return self._pages

    def page_lines(self, page_number: int) -> list[str] | None:
        try:
            return self._load().get(page_number)
        except Exception:  # noqa: BLE001 - unreadable text layer: fallback unavailable
            return None


class DictTextSource:
    """In-memory text source (tests and fixtures)."""

    def __init__(self, pages: dict[int, list[str]]):
        self.pages = pages

    def page_lines(self, page_number: int) -> list[str] | None:
        return self.pages.get(page_number)


class LoggedTextSource:
    """Wraps a source so every page read is recorded in the fallback log."""

    def __init__(
        self, source: PageTextSource | None, log: FallbackLog, document_id: str, filename: str
    ):
        self.source = source
        self.log = log
        self.document_id = document_id
        self.filename = filename

    def page_lines(self, page_number: int, *, element: str, reason: str) -> list[str] | None:
        if self.source is None:
            return None
        self.log.record(self.document_id, self.filename, page_number, element, reason)
        return self.source.page_lines(page_number)
