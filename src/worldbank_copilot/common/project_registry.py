"""The prototype's project scope, loaded from ``configs/projects.yaml``.

The registry is the single allow-list for project IDs: ingestion filters on it,
tools validate arguments against it, and anything outside it is rejected.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from worldbank_copilot.common.exceptions import ConfigurationError, UnknownProjectError

PROJECT_ID_PATTERN = re.compile(r"^P\d{6}$")


class ProcurementCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    covered_by_ipf_contract_awards: bool
    note: str


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str
    name: str
    instrument: str
    lifecycle_stage: Literal["early", "mid", "mature"]
    purpose: str
    analytical_question: str
    documents_dir: str
    procurement_coverage: ProcurementCoverage
    # Number of ISRs expected in the local corpus; None disables the count check.
    expected_isr_count: int | None = Field(default=None, ge=0)
    # Raw loan numbers expected in the loan snapshot; None disables the check.
    expected_loan_numbers: tuple[str, ...] | None = None

    @field_validator("project_id")
    @classmethod
    def _valid_project_id(cls, value: str) -> str:
        return validate_project_id_format(value)


class ProjectRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    projects: tuple[ProjectConfig, ...]

    @model_validator(mode="after")
    def _unique_ids(self) -> ProjectRegistry:
        ids = [p.project_id for p in self.projects]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"Duplicate project IDs: {', '.join(duplicates)}")
        if not ids:
            raise ValueError("At least one project must be configured")
        return self

    @property
    def project_ids(self) -> list[str]:
        return [p.project_id for p in self.projects]

    def __contains__(self, project_id: object) -> bool:
        return isinstance(project_id, str) and project_id.strip().upper() in self.project_ids

    def get(self, project_id: str) -> ProjectConfig:
        """Return the project config, or raise ``UnknownProjectError``."""
        normalized = project_id.strip().upper() if isinstance(project_id, str) else project_id
        for project in self.projects:
            if project.project_id == normalized:
                return project
        raise UnknownProjectError(str(project_id), self.project_ids)

    def validate_project_id(self, project_id: str) -> str:
        """Return the canonical project ID if it is in scope, else raise."""
        return self.get(project_id).project_id


def validate_project_id_format(value: str) -> str:
    """Canonicalise and check the World Bank project ID shape (``P`` + 6 digits)."""
    if not isinstance(value, str):
        raise ValueError(f"Project ID must be a string, got {type(value).__name__}")
    canonical = value.strip().upper()
    if not PROJECT_ID_PATTERN.match(canonical):
        raise ValueError(f"Invalid project ID {value!r}; expected 'P' followed by 6 digits")
    return canonical


def load_project_registry(config_dir: Path | str) -> ProjectRegistry:
    """Load and validate ``<config_dir>/projects.yaml``."""
    path = Path(config_dir) / "projects.yaml"
    if not path.is_file():
        raise ConfigurationError(f"Project configuration not found: {path}")
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    try:
        return ProjectRegistry.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid project configuration in {path}:\n{exc}") from exc
