"""Runtime configuration for ``Copilot.investigate`` (``configs/copilot.yaml`` + env)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from worldbank_copilot.common.exceptions import ConfigurationError
from worldbank_copilot.investigation.policy import InvestigationPolicy

CONFIG_FILE = "copilot.yaml"
_ENDPOINT = r"^[A-Za-z0-9_-]{1,200}$"
_ENV_OVERRIDES = {
    "WBC_COPILOT_SYNTHESIZER_ENDPOINT": "synthesizer_endpoint",
    "WBC_COPILOT_CRITIC_ENDPOINT": "critic_endpoint",
    "WBC_COPILOT_CRITIC_ENABLED": "critic_enabled",
}


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(_Config):
    synthesizer_endpoint: str = Field(pattern=_ENDPOINT)
    critic_endpoint: str = Field(pattern=_ENDPOINT)
    critic_enabled: bool
    max_output_tokens: int = Field(ge=1, le=8000)
    timeout_seconds: float = Field(gt=0, le=600)
    capability_note: str = Field(min_length=1)


class CopilotConfig(_Config):
    models: ModelConfig
    allowed_projects: tuple[str, ...] = Field(min_length=1)
    overall_deadline_seconds: int = Field(ge=1, le=3600)
    cost_ceiling: Decimal | None = None
    pricing_version: str | None = None
    mlflow_experiment: str | None = None

    @field_validator("allowed_projects")
    @classmethod
    def _project_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for project_id in value:
            if not (len(project_id) == 7 and project_id[0] == "P" and project_id[1:].isdigit()):
                raise ValueError(f"invalid project id {project_id!r}")
        return value

    @property
    def pricing_configured(self) -> bool:
        """True when both budget-pricing fields are set; enables the executor pricing gate.

        Unset (the default) follows the accepted Phase 10C live validation: cost is not an
        acceptance criterion and actual billed cost is UNAVAILABLE, not zero.
        """
        return self.cost_ceiling is not None and self.pricing_version is not None

    def policy(self) -> InvestigationPolicy:
        """The existing Phase 10C policy, with this runtime's deadline and cost settings."""
        return InvestigationPolicy(
            overall_deadline_seconds=self.overall_deadline_seconds,
            cost_ceiling=self.cost_ceiling,
            pricing_version=self.pricing_version,
        )


def load_copilot_config(
    config_dir: Path | str, env: Mapping[str, str] | None = None
) -> CopilotConfig:
    """Read ``configs/copilot.yaml``; selected model settings may be overridden by env."""
    path = Path(config_dir) / CONFIG_FILE
    if not path.is_file():
        raise ConfigurationError(f"Configuration file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    env = os.environ if env is None else env
    models = dict(raw.get("models") or {})
    for variable, key in _ENV_OVERRIDES.items():
        value = env.get(variable, "").strip()
        if value:
            models[key] = value
    policy = raw.get("policy") or {}
    ceiling = policy.get("cost_ceiling")
    return CopilotConfig(
        models=ModelConfig(**models),
        allowed_projects=tuple((raw.get("projects") or {}).get("allowed") or ()),
        overall_deadline_seconds=policy.get("overall_deadline_seconds", 180),
        cost_ceiling=None if ceiling is None else Decimal(str(ceiling)),
        pricing_version=policy.get("pricing_version"),
        mlflow_experiment=(raw.get("mlflow") or {}).get("experiment"),
    )
