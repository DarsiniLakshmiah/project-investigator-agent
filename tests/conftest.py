from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_CONFIG_DIR = REPO_ROOT / "configs"


@pytest.fixture
def repo_config_dir() -> Path:
    return REPO_CONFIG_DIR


@pytest.fixture
def write_config(tmp_path: Path) -> Callable[..., Path]:
    """Create a throwaway configs/ directory from Python dicts."""

    def _write(
        base: dict | None = None,
        local: dict | None = None,
        databricks: dict | None = None,
        projects: dict | None = None,
    ) -> Path:
        config_dir = tmp_path / "configs"
        env_dir = config_dir / "environments"
        env_dir.mkdir(parents=True, exist_ok=True)
        for name, content in (("base", base), ("local", local), ("databricks", databricks)):
            (env_dir / f"{name}.yaml").write_text(yaml.safe_dump(content or {}), encoding="utf-8")
        if projects is not None:
            (config_dir / "projects.yaml").write_text(yaml.safe_dump(projects), encoding="utf-8")
        return config_dir

    return _write


@pytest.fixture
def synthetic_config(tmp_path: Path) -> Path:
    """A throwaway data/ + configs/ tree with small synthetic source files."""
    from support.synthetic import build_source_tree

    return build_source_tree(tmp_path, REPO_CONFIG_DIR)


@pytest.fixture
def synthetic_run(synthetic_config: Path):
    """(settings, registry, result) for a Bronze ingestion over the synthetic tree."""
    from datetime import UTC, datetime

    from worldbank_copilot.common import load_project_registry, load_settings
    from worldbank_copilot.ingestion.pipeline import ingest_bronze

    settings = load_settings(config_dir=synthetic_config, env={})
    registry = load_project_registry(synthetic_config)
    result = ingest_bronze(
        settings, registry, ingested_at=datetime(2026, 9, 26, tzinfo=UTC), run_id="test-run"
    )
    return settings, registry, result
