"""Attention-rule configuration and rating scales (pure Python, no Spark).

``configs/intelligence/attention_rules.yaml`` is the reviewed source of every rule's
id, version, category, severity logic, thresholds and rationale.
``configs/intelligence/rating_scales.yaml`` holds the ordinal rating scales.
Both are validated on load; an enabled rule without an implementation, or an
implementation without a configured rule, fails loudly (see ``signals.py``).
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from worldbank_copilot.common.exceptions import ConfigurationError

RULES_FILE = Path("intelligence") / "attention_rules.yaml"
SCALES_FILE = Path("intelligence") / "rating_scales.yaml"

SEVERITIES = ("INFO", "WATCH", "HIGH")
CATEGORIES = (
    "SCHEDULE",
    "IMPLEMENTATION_RATING",
    "RESULTS",
    "FINANCIAL_EXECUTION",
    "RESTRUCTURING_SCOPE",
    "RISK",
)


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str
    version: int = Field(ge=1)
    enabled: bool
    category: str
    title: str
    description: str
    required_fields: list[str]
    severity: dict[str, str]
    thresholds: dict[str, Decimal] = Field(default_factory=dict)
    applies_to_instruments: list[str] | None = None
    rationale: str

    @field_validator("category")
    @classmethod
    def _category(cls, value: str) -> str:
        if value not in CATEGORIES:
            raise ValueError(f"unknown category {value!r}; expected one of {CATEGORIES}")
        return value

    @field_validator("severity")
    @classmethod
    def _severity(cls, value: dict[str, str]) -> dict[str, str]:
        unknown = set(value) - set(SEVERITIES)
        if unknown or not value:
            raise ValueError(f"severity levels must be a non-empty subset of {SEVERITIES}")
        return value

    def threshold(self, name: str) -> Decimal:
        if name not in self.thresholds:
            raise ConfigurationError(f"rule {self.rule_id} has no threshold {name!r}")
        return self.thresholds[name]

    @property
    def rule_version(self) -> str:
        return f"{self.rule_id}@v{self.version}"


class DeferredRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate: str
    reason: str


class RuleSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rules: list[Rule]
    deferred: list[DeferredRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique(self) -> RuleSet:
        ids = [r.rule_id for r in self.rules]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate rule ids {duplicates}")
        return self

    @property
    def enabled(self) -> dict[str, Rule]:
        return {r.rule_id: r for r in self.rules if r.enabled}

    def get(self, rule_id: str) -> Rule:
        rule = self.enabled.get(rule_id)
        if rule is None:
            raise ConfigurationError(f"rule {rule_id} is not configured or not enabled")
        return rule


class RatingScales(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    performance: dict[str, int]
    performance_below_satisfactory_max_rank: int
    risk: dict[str, int]

    @model_validator(mode="after")
    def _ordinal(self) -> RatingScales:
        for name, scale in (("performance", self.performance), ("risk", self.risk)):
            if len(set(scale.values())) != len(scale):
                raise ValueError(f"{name} scale ranks must be distinct")
        if self.performance_below_satisfactory_max_rank not in self.performance.values():
            raise ValueError("performance_below_satisfactory_max_rank is not a scale rank")
        return self


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigurationError(f"{path} not found")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_rules(config_dir: Path) -> RuleSet:
    try:
        return RuleSet.model_validate(_read(Path(config_dir) / RULES_FILE))
    except ValueError as exc:
        raise ConfigurationError(f"invalid attention rules: {exc}") from exc


def load_scales(config_dir: Path) -> RatingScales:
    try:
        return RatingScales.model_validate(_read(Path(config_dir) / SCALES_FILE))
    except ValueError as exc:
        raise ConfigurationError(f"invalid rating scales: {exc}") from exc
