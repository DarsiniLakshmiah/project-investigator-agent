"""Fail-closed protection for FROZEN decision artifacts (Phase 9F).

A script that writes a decision artifact must call ``assert_not_frozen`` before it computes
or writes anything: once a decision is FROZEN, an accidental rerun must not overwrite it.
Superseding a frozen decision needs an explicit, versioned new artifact, never an overwrite.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from worldbank_copilot.common.exceptions import CopilotError

FROZEN = "FROZEN"


class FrozenArtifactError(CopilotError):
    """Refused: the target decision artifact is FROZEN."""


def artifact_status(path: Path) -> str | None:
    """The ``status`` of a YAML/JSON decision artifact, or None if absent or unreadable."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise FrozenArtifactError(f"{path} is unreadable; refusing to overwrite it") from exc
    return data.get("status") if isinstance(data, dict) else None


def assert_not_frozen(path: Path) -> None:
    if artifact_status(path) == FROZEN:
        raise FrozenArtifactError(
            f"{Path(path).name} is FROZEN; refusing to recompute or overwrite it. Record any "
            "new decision as a new, versioned artifact."
        )
