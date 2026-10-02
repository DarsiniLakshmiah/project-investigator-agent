"""Routing configuration (configs/routing/*.yaml), validated on load (Phase 9B)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from worldbank_copilot.common.exceptions import ConfigurationError
from worldbank_copilot.common.project_registry import ProjectRegistry
from worldbank_copilot.routing.models import Intent, TemporalKind

FOLDER = "routing"
GROUP_PRECEDENCE = {
    "refusal": 1,
    "investigation": 2,
    "document_scope": 3,
    "explanatory": 4,
    "subject": 5,
}
SUBJECTS = (
    "RATINGS",
    "FINANCE",
    "RESULTS",
    "RISKS",
    "ATTENTION",
    "OVERVIEW",
    "TIMELINE",
    "DECISION",
)
RATING_TYPES = ("PDO", "IP", "OVERALL_RISK")
# Phrases that must never become aliases (they fit several projects).
GENERIC_ALIAS_WORDS = {
    "water",
    "project",
    "program",
    "programme",
    "karnataka",
    "the",
    "in",
    "supply",
}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InputConfig(_Model):
    max_chars: int = Field(ge=10, le=10_000)
    injection_patterns: tuple[str, ...] = ()


class RoutingSettings(_Model):
    router_version: str
    input: InputConfig
    # Relative project references ("the other program") that never identify a project by
    # themselves (Phase 9C human-review fix, case r073).
    relative_project_references: tuple[str, ...] = Field(min_length=1)


class IntentRule(_Model):
    id: str
    group: Literal["refusal", "investigation", "document_scope", "explanatory", "subject"]
    patterns: tuple[str, ...] = Field(min_length=1)
    intent: Intent | None = None
    subject: str | None = None
    rating_type: str | None = None
    investigation: str | None = None
    non_isr_only: bool = False

    @model_validator(mode="after")
    def _shape(self) -> IntentRule:
        if self.group in ("refusal", "investigation") and self.intent is None:
            raise ValueError(f"{self.id}: {self.group} rules need an intent")
        if self.group == "subject" and self.subject not in SUBJECTS:
            raise ValueError(f"{self.id}: subject must be one of {SUBJECTS}")
        if self.rating_type is not None and self.rating_type not in RATING_TYPES:
            raise ValueError(f"{self.id}: rating_type must be one of {RATING_TYPES}")
        if self.group == "investigation" and not self.investigation:
            raise ValueError(f"{self.id}: investigation rules need an investigation kind")
        if self.group == "document_scope" and not all("{doc}" in p for p in self.patterns):
            raise ValueError(f"{self.id}: document_scope patterns must contain {{doc}}")
        return self

    @property
    def precedence(self) -> int:
        return GROUP_PRECEDENCE[self.group]


class IntentRules(_Model):
    version: int
    rules: tuple[IntentRule, ...]
    document_types: dict[str, tuple[str, ...]]

    @model_validator(mode="after")
    def _unique_and_compilable(self) -> IntentRules:
        ids = [r.id for r in self.rules]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"duplicate rule ids {dup}")
        for rule in self.rules:
            for pattern in rule.patterns:
                try:
                    re.compile(pattern.replace("{doc}", "(?:x)"))
                except re.error as exc:
                    raise ValueError(f"{rule.id}: invalid pattern {pattern!r}: {exc}") from exc
        return self


class IntentRequirement(_Model):
    route: Literal["STRUCTURED", "DOCUMENT", "INVESTIGATION"]
    tools: tuple[str, ...]
    default_time: TemporalKind
    supported_time: tuple[TemporalKind, ...]


class PlannedCall(_Model):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class RequirementsConfig(_Model):
    version: int
    intents: dict[Intent, IntentRequirement]
    explicit_time_required: tuple[dict[str, str], ...] = ()
    investigation_plans: dict[str, tuple[PlannedCall, ...]]

    @model_validator(mode="after")
    def _complete(self) -> RequirementsConfig:
        from worldbank_copilot.routing.models import REFUSAL_INTENTS

        missing = set(Intent) - REFUSAL_INTENTS - set(self.intents)
        if missing:
            raise ValueError(f"no requirement for intents {sorted(missing)}")
        extra = set(self.intents) & REFUSAL_INTENTS
        if extra:
            raise ValueError(f"refusal intents never get requirements: {sorted(extra)}")
        return self


class AliasConfig(_Model):
    version: int
    reviewed_by: str | None = None
    aliases: dict[str, tuple[str, ...]]


def normalize_phrase(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9&]+", (text or "").lower()))


def validate_aliases(config: AliasConfig, registry: ProjectRegistry) -> dict[str, str]:
    """alias (normalised) -> project. Rejects shared, nested or generic aliases."""
    out: dict[str, str] = {}
    for project_id, aliases in config.aliases.items():
        if project_id not in registry:
            raise ConfigurationError(f"alias project {project_id} is not in the registry")
        for alias in aliases:
            norm = normalize_phrase(alias)
            words = norm.split()
            if len(words) < 2 or set(words) <= GENERIC_ALIAS_WORDS:
                raise ConfigurationError(f"alias {alias!r} is too generic to be unambiguous")
            if norm in out and out[norm] != project_id:
                raise ConfigurationError(f"alias {alias!r} names two projects")
            out[norm] = project_id
    for a, pa in out.items():
        for b, pb in out.items():
            if pa != pb and re.search(rf"\b{re.escape(a)}\b", b):
                raise ConfigurationError(f"alias {a!r} ({pa}) is contained in {b!r} ({pb})")
    return out


class RoutingConfig(_Model):
    settings: RoutingSettings
    rules: IntentRules
    requirements: RequirementsConfig
    aliases: AliasConfig

    @property
    def versions(self) -> dict[str, str]:
        return {
            "router": self.settings.router_version,
            "intent_rules": str(self.rules.version),
            "requirements": str(self.requirements.version),
            "aliases": str(self.aliases.version),
        }


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"{path} not found")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_routing_config(config_dir: Path) -> RoutingConfig:
    folder = Path(config_dir) / FOLDER
    try:
        return RoutingConfig(
            settings=RoutingSettings.model_validate(_read(folder / "routing.yaml")),
            rules=IntentRules.model_validate(_read(folder / "intents.yaml")),
            requirements=RequirementsConfig.model_validate(_read(folder / "requirements.yaml")),
            aliases=AliasConfig.model_validate(_read(folder / "project_aliases.yaml")),
        )
    except ValidationError as exc:
        raise ConfigurationError(f"invalid routing configuration: {exc}") from exc
