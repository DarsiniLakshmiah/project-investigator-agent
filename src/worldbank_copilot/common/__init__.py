"""Cross-cutting utilities: configuration, project scope, logging, exceptions."""

from worldbank_copilot.common.config import Environment, Layer, Settings, load_settings
from worldbank_copilot.common.exceptions import (
    ConfigurationError,
    CopilotError,
    UnknownProjectError,
)
from worldbank_copilot.common.project_registry import (
    ProjectConfig,
    ProjectRegistry,
    load_project_registry,
)

__all__ = [
    "ConfigurationError",
    "CopilotError",
    "Environment",
    "Layer",
    "ProjectConfig",
    "ProjectRegistry",
    "Settings",
    "UnknownProjectError",
    "load_project_registry",
    "load_settings",
]
