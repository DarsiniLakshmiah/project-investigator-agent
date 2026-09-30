"""Parser abstraction. Downstream code depends on this interface only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from worldbank_copilot.parsing.models import ParsedContent, ParserInfo


class ParserError(RuntimeError):
    """The parser could not produce content for a document."""


@runtime_checkable
class DocumentParser(Protocol):
    @property
    def info(self) -> ParserInfo:
        """Name, version and output-affecting configuration (part of cache identity)."""
        ...

    def parse(self, path: Path) -> ParsedContent:
        """Parse one PDF. Raise ``ParserError`` if no usable content is produced."""
        ...


def config_hash(name: str, version: str, config: dict[str, Any]) -> str:
    payload = json.dumps({"name": name, "version": version, "config": config}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
