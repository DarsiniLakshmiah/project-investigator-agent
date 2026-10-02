"""Shared tool plumbing: context, argument base, specs and small row helpers (Phase 9)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from worldbank_copilot.common.exceptions import CopilotError
from worldbank_copilot.common.project_registry import ProjectRegistry, validate_project_id_format
from worldbank_copilot.intelligence.rules import RatingScales, RuleSet
from worldbank_copilot.tools.models import ProvenanceClass, SourceRef, ToolOutcome
from worldbank_copilot.tools.reader import ReadRequest, TableReader


class ToolTimeout(CopilotError):
    """The request deadline passed before the tool finished its reads."""


class DataIntegrityError(CopilotError):
    """An expected authoritative record is missing, duplicated or inconsistent."""


class ToolArgs(BaseModel):
    """Every tool argument model: strict, frozen, project-scoped."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str

    @field_validator("project_id")
    @classmethod
    def _pid(cls, value: str) -> str:
        return validate_project_id_format(value)


@dataclass
class ToolContext:
    reader: TableReader
    registry: ProjectRegistry
    scales: RatingScales
    rules: RuleSet
    request_id: str
    deadline: float | None = None  # time.monotonic() value
    documents: Any | None = None  # tools.documents.DocumentSearch (search tool only)

    def check_deadline(self) -> None:
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise ToolTimeout("request deadline passed")

    def read(self, request: ReadRequest) -> list[dict[str, Any]]:
        """Cooperative timeout: checked before every read (a running Spark job is not killed)."""
        self.check_deadline()
        return self.reader.read(request)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    version: str
    description: str
    args_model: type[ToolArgs]
    item_model: type[BaseModel]
    tables: tuple[str, ...]  # governed tables the tool may read (pinned at request start)
    run: Callable[[ToolContext, Any], ToolOutcome]


def gold_ref(table: str, row: dict[str, Any]) -> SourceRef:
    """Row-level provenance of a Gold row (its record plus the evidence behind it)."""
    return SourceRef(
        table=table,
        record_id=row.get("record_id"),
        source_table=row.get("source_table"),
        source_record_id=row.get("source_record_id"),
        supporting_record_ids=tuple(row.get("supporting_record_ids") or ()),
        document_id=row.get("document_id"),
        page_number=row.get("page_number"),
        section=row.get("section"),
        table_id=row.get("table_id"),
        extraction_method=row.get("extraction_method"),
        extraction_status=row.get("extraction_status") or row.get("evidence_status"),
    )


def row_class(row: dict[str, Any]) -> ProvenanceClass:
    return ProvenanceClass(row["provenance_class"])


GOLD_PROVENANCE = (
    "record_id",
    "source_table",
    "source_record_id",
    "document_id",
    "page_number",
    "section",
    "extraction_method",
    "provenance_class",
)
