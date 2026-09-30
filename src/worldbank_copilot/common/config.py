"""Environment-aware configuration.

Resolution order (later wins):

1. ``configs/environments/base.yaml``
2. ``configs/environments/<environment>.yaml``
3. ``WBC_*`` environment variables (see ``ENV_VAR_OVERRIDES``)

The environment is selected by, in order: the explicit ``environment`` argument,
``WBC_ENV``, auto-detection of a Databricks runtime (``DATABRICKS_RUNTIME_VERSION``),
and finally ``local``.

Loading settings never requires Databricks credentials. Values that are not
configured stay ``None``; code that needs one calls ``Settings.require`` and
gets a ``ConfigurationError`` naming the environment variable to set.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from worldbank_copilot.common.exceptions import ConfigurationError

ENV_VAR_ENVIRONMENT = "WBC_ENV"
ENV_VAR_CONFIG_DIR = "WBC_CONFIG_DIR"
DATABRICKS_RUNTIME_MARKER = "DATABRICKS_RUNTIME_VERSION"

# Environment variable -> dotted settings key.
ENV_VAR_OVERRIDES: dict[str, str] = {
    "WBC_LOG_LEVEL": "log_level",
    "WBC_LOCAL_DATA_ROOT": "data.local_data_root",
    "WBC_DATABRICKS_VOLUME_ROOT": "data.databricks_volume_root",
    "WBC_CATALOG": "databricks.catalog",
    "WBC_BRONZE_SCHEMA": "databricks.bronze_schema",
    "WBC_SILVER_SCHEMA": "databricks.silver_schema",
    "WBC_GOLD_SCHEMA": "databricks.gold_schema",
    "WBC_SOURCE_VOLUME": "databricks.source_volume",
    "WBC_ARTIFACT_VOLUME": "databricks.artifact_volume",
    "WBC_VECTOR_SEARCH_ENDPOINT": "vector_search.endpoint",
    "WBC_VECTOR_SEARCH_INDEX": "vector_search.index",
    "WBC_EMBEDDING_ENDPOINT": "models.embedding_endpoint",
    "WBC_LLM_ENDPOINT": "models.llm_endpoint",
    "WBC_MLFLOW_EXPERIMENT": "mlflow.experiment",
    "WBC_LOCAL_OUTPUT_ROOT": "data.local_output_root",
    "WBC_PROJECTS_WORKBOOK": "sources.projects_workbook",
    "WBC_LOANS_SNAPSHOT": "sources.loans_snapshot",
    "WBC_PROCUREMENT_CONTRACT_AWARDS": "sources.procurement_contract_awards",
}
_KEY_TO_ENV_VAR = {key: var for var, key in ENV_VAR_OVERRIDES.items()}


class Environment(StrEnum):
    LOCAL = "local"
    DATABRICKS = "databricks"


class Layer(StrEnum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        # An empty WBC_* variable (as in .env.example) means "not configured".
        if isinstance(value, str) and not value.strip():
            return None
        return value


class DataSettings(_Section):
    local_data_root: str | None = None
    databricks_volume_root: str | None = None
    structured_subdir: str = "."
    documents_subdir: str = "."
    local_output_root: str | None = None


class SourcesSettings(_Section):
    """File names or glob patterns, relative to the structured-data directory."""

    projects_workbook: str | None = None
    loans_snapshot: str | None = None
    procurement_contract_awards: str | None = None
    # Relative to the configs directory.
    document_manifest: str | None = None


class DatabricksSettings(_Section):
    catalog: str | None = None
    bronze_schema: str | None = None
    silver_schema: str | None = None
    gold_schema: str | None = None
    source_volume: str | None = None
    artifact_volume: str | None = None


class VectorSearchSettings(_Section):
    endpoint: str | None = None
    index: str | None = None


class ModelSettings(_Section):
    embedding_endpoint: str | None = None
    llm_endpoint: str | None = None


class MlflowSettings(_Section):
    experiment: str | None = None


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    environment: Environment
    repo_root: Path
    config_dir: Path
    log_level: str = "INFO"
    data: DataSettings = DataSettings()
    sources: SourcesSettings = SourcesSettings()
    databricks: DatabricksSettings = DatabricksSettings()
    vector_search: VectorSearchSettings = VectorSearchSettings()
    models: ModelSettings = ModelSettings()
    mlflow: MlflowSettings = MlflowSettings()

    def require(self, dotted_key: str) -> Any:
        """Return a configured value or raise ``ConfigurationError`` if it is unset."""
        value: Any = self
        for part in dotted_key.split("."):
            if not hasattr(value, part):
                raise ConfigurationError(f"Unknown settings key {dotted_key!r}")
            value = getattr(value, part)
        if value is None:
            hint = _KEY_TO_ENV_VAR.get(dotted_key)
            how = f"set {hint} or " if hint else ""
            raise ConfigurationError(
                f"Required setting {dotted_key!r} is not configured for environment "
                f"{self.environment.value!r}; {how}add it to "
                f"configs/environments/{self.environment.value}.yaml"
            )
        return value

    @property
    def data_root(self) -> Path | str:
        """Root holding the raw source files for the active environment.

        Local roots are resolved against the repository root and returned as a
        ``Path``. Databricks volume roots (``/Volumes/...``) are returned as-is.
        """
        if self.environment is Environment.DATABRICKS:
            return self.data.databricks_volume_root or f"{self.source_volume_path}/data"
        root = Path(self.require("data.local_data_root"))
        return root if root.is_absolute() else (self.repo_root / root).resolve()

    @property
    def structured_root(self) -> Path:
        """Directory holding the structured source files."""
        return Path(self.data_root) / self.data.structured_subdir

    @property
    def documents_root(self) -> Path:
        """Directory holding one sub-directory of documents per project."""
        return Path(self.data_root) / self.data.documents_subdir

    @property
    def local_output_root(self) -> Path:
        """Where local runs write derived artefacts (never inside the source data)."""
        if self.environment is Environment.DATABRICKS and not self.data.local_output_root:
            return Path(self.artifact_volume_path)
        root = Path(self.require("data.local_output_root"))
        return root if root.is_absolute() else (self.repo_root / root).resolve()

    @property
    def source_volume_path(self) -> str:
        """``/Volumes/<catalog>/<bronze_schema>/<source_volume>`` (original source files)."""
        return self.volume_path("bronze", self.require("databricks.source_volume"))

    @property
    def artifact_volume_path(self) -> str:
        """``/Volumes/<catalog>/<silver_schema>/<artifact_volume>`` (derived artefacts)."""
        return self.volume_path("silver", self.require("databricks.artifact_volume"))

    def volume_path(self, layer: Layer | str, volume: str) -> str:
        layer = Layer(layer)
        catalog = self.require("databricks.catalog")
        schema = self.require(f"databricks.{layer.value}_schema")
        return f"/Volumes/{catalog}/{schema}/{volume}"

    def table_name(self, layer: Layer | str, table: str) -> str:
        """Fully qualified Unity Catalog name: ``<catalog>.<schema>.<table>``."""
        layer = Layer(layer)
        catalog = self.require("databricks.catalog")
        schema = self.require(f"databricks.{layer.value}_schema")
        return f"{catalog}.{schema}.{table}"


def detect_environment(env: Mapping[str, str]) -> Environment:
    """Pick the runtime environment from environment variables."""
    explicit = env.get(ENV_VAR_ENVIRONMENT, "").strip().lower()
    if explicit:
        try:
            return Environment(explicit)
        except ValueError:
            allowed = ", ".join(e.value for e in Environment)
            raise ConfigurationError(
                f"{ENV_VAR_ENVIRONMENT}={explicit!r} is invalid; expected one of: {allowed}"
            ) from None
    if env.get(DATABRICKS_RUNTIME_MARKER):
        return Environment.DATABRICKS
    return Environment.LOCAL


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upwards from ``start`` to the directory containing ``pyproject.toml``."""
    start = (start or Path(__file__)).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ConfigurationError(
        f"Could not locate the repository root (pyproject.toml) above {start}; "
        f"set {ENV_VAR_CONFIG_DIR} to the configs directory."
    )


def load_settings(
    environment: Environment | str | None = None,
    *,
    config_dir: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    use_dotenv: bool = True,
) -> Settings:
    """Load settings for the selected environment.

    Args:
        environment: Force an environment instead of detecting it.
        config_dir: Directory containing ``environments/`` and ``projects.yaml``.
        env: Environment variables to use. Defaults to ``os.environ`` layered over
            ``<repo_root>/.env`` (when ``use_dotenv``). When given explicitly,
            ``.env`` is ignored, which keeps tests hermetic.
        use_dotenv: Whether to read ``.env`` when ``env`` is not given.
    """
    if env is None:
        env = _process_env(use_dotenv)

    if config_dir is None and env.get(ENV_VAR_CONFIG_DIR):
        config_dir = env[ENV_VAR_CONFIG_DIR]
    if config_dir is None:
        repo_root = find_repo_root()
        config_dir = repo_root / "configs"
    else:
        config_dir = Path(config_dir).resolve()
        repo_root = config_dir.parent

    selected = Environment(environment) if environment is not None else detect_environment(env)

    env_dir = Path(config_dir) / "environments"
    merged = _read_yaml(env_dir / "base.yaml", required=True)
    merged = _deep_merge(merged, _read_yaml(env_dir / f"{selected.value}.yaml", required=True))
    for var, dotted_key in ENV_VAR_OVERRIDES.items():
        if var in env:
            _set_dotted(merged, dotted_key, env[var])

    try:
        return Settings(
            environment=selected,
            repo_root=repo_root,
            config_dir=Path(config_dir),
            **merged,
        )
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid configuration in {env_dir}:\n{exc}") from exc


def _process_env(use_dotenv: bool) -> dict[str, str]:
    values: dict[str, str] = {}
    if use_dotenv:
        try:
            dotenv_path = find_repo_root() / ".env"
        except ConfigurationError:
            dotenv_path = None
        if dotenv_path is not None and dotenv_path.is_file():
            values.update({k: v for k, v in dotenv_values(dotenv_path).items() if v is not None})
    values.update(os.environ)  # real environment always wins over .env
    return values


def _read_yaml(path: Path, *, required: bool) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise ConfigurationError(f"Configuration file not found: {path}")
        return {}
    with path.open(encoding="utf-8") as fh:
        content = yaml.safe_load(fh)
    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ConfigurationError(f"{path} must contain a YAML mapping at the top level")
    return content


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _set_dotted(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    *parents, leaf = dotted_key.split(".")
    node = target
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value
