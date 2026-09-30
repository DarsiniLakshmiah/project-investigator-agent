from pathlib import Path

import pytest
from pydantic import ValidationError

from worldbank_copilot.common.config import (
    ENV_VAR_OVERRIDES,
    Environment,
    Layer,
    load_settings,
)
from worldbank_copilot.common.exceptions import ConfigurationError

REQUIRED_PLACEHOLDERS = [
    "data.local_data_root",
    "data.databricks_volume_root",
    "databricks.catalog",
    "databricks.bronze_schema",
    "databricks.silver_schema",
    "databricks.gold_schema",
    "vector_search.endpoint",
    "vector_search.index",
    "models.embedding_endpoint",
    "models.llm_endpoint",
    "mlflow.experiment",
]


def _get(settings, dotted):
    value = settings
    for part in dotted.split("."):
        value = getattr(value, part)
    return value


def test_repo_config_loads_with_local_defaults(repo_config_dir):
    settings = load_settings(config_dir=repo_config_dir, env={})

    assert settings.environment is Environment.LOCAL
    assert settings.config_dir == repo_config_dir
    assert settings.data.local_data_root == "data"
    assert settings.data_root == (repo_config_dir.parent / "data").resolve()
    assert settings.databricks.bronze_schema == "bronze"
    assert settings.databricks.silver_schema == "silver"
    assert settings.databricks.gold_schema == "gold"
    # Unity Catalog layout approved for Phase 6; endpoints etc. stay placeholders.
    assert settings.databricks.catalog == "worldbank_ai"
    assert settings.databricks.source_volume == "sources"
    assert settings.databricks.artifact_volume == "pipeline_artifacts"
    assert settings.vector_search.endpoint is None
    assert settings.models.llm_endpoint is None
    assert settings.mlflow.experiment is None


def test_default_config_dir_is_found_from_package(repo_config_dir):
    settings = load_settings(env={})
    assert settings.config_dir.resolve() == repo_config_dir.resolve()


@pytest.mark.parametrize("dotted", REQUIRED_PLACEHOLDERS)
def test_every_placeholder_is_overridable_by_env_var(repo_config_dir, dotted):
    var = next(v for v, key in ENV_VAR_OVERRIDES.items() if key == dotted)
    settings = load_settings(config_dir=repo_config_dir, env={var: "configured-value"})
    assert _get(settings, dotted) == "configured-value"


def test_blank_env_var_means_not_configured(repo_config_dir):
    settings = load_settings(config_dir=repo_config_dir, env={"WBC_CATALOG": "   "})
    assert settings.databricks.catalog is None


def test_require_raises_clear_error_for_unset_value(repo_config_dir):
    settings = load_settings(config_dir=repo_config_dir, env={})
    with pytest.raises(ConfigurationError, match="WBC_VECTOR_SEARCH_INDEX"):
        settings.require("vector_search.index")


def test_require_rejects_unknown_key(repo_config_dir):
    settings = load_settings(config_dir=repo_config_dir, env={})
    with pytest.raises(ConfigurationError, match="Unknown settings key"):
        settings.require("vector_search.nope")


def test_table_name_uses_configured_catalog_and_schema(repo_config_dir):
    settings = load_settings(
        config_dir=repo_config_dir,
        env={"WBC_CATALOG": "wb_dev", "WBC_SILVER_SCHEMA": "silver_v2"},
    )
    assert settings.table_name(Layer.BRONZE, "projects_raw") == "wb_dev.bronze.projects_raw"
    assert settings.table_name("silver", "projects") == "wb_dev.silver_v2.projects"


def test_table_name_fails_without_catalog(repo_config_dir):
    settings = load_settings(config_dir=repo_config_dir, env={"WBC_CATALOG": ""})
    with pytest.raises(ConfigurationError, match="WBC_CATALOG"):
        settings.table_name(Layer.GOLD, "project_360")


def test_table_name_rejects_unknown_layer(repo_config_dir):
    settings = load_settings(config_dir=repo_config_dir, env={"WBC_CATALOG": "c"})
    with pytest.raises(ValueError):
        settings.table_name("platinum", "x")


def test_absolute_local_data_root_is_kept(repo_config_dir, tmp_path):
    settings = load_settings(config_dir=repo_config_dir, env={"WBC_LOCAL_DATA_ROOT": str(tmp_path)})
    assert settings.data_root == tmp_path


def test_settings_are_immutable(repo_config_dir):
    settings = load_settings(config_dir=repo_config_dir, env={})
    with pytest.raises(ValidationError, match="frozen"):
        settings.log_level = "DEBUG"


def test_environment_file_is_deep_merged_over_base(write_config):
    config_dir = write_config(
        base={"log_level": "INFO", "databricks": {"catalog": None, "bronze_schema": "bronze"}},
        local={"databricks": {"catalog": "local_cat"}},
    )
    settings = load_settings(config_dir=config_dir, env={})
    assert settings.databricks.catalog == "local_cat"
    assert settings.databricks.bronze_schema == "bronze"  # untouched sibling survives merge


def test_env_var_wins_over_environment_file(write_config):
    config_dir = write_config(local={"databricks": {"catalog": "from_yaml"}})
    settings = load_settings(config_dir=config_dir, env={"WBC_CATALOG": "from_env"})
    assert settings.databricks.catalog == "from_env"


def test_unknown_yaml_key_is_rejected(write_config):
    config_dir = write_config(base={"databricks": {"catalgo": "typo"}})
    with pytest.raises(ConfigurationError, match="catalgo"):
        load_settings(config_dir=config_dir, env={})


def test_missing_environment_file_is_an_error(write_config):
    config_dir = write_config()
    (config_dir / "environments" / "local.yaml").unlink()
    with pytest.raises(ConfigurationError, match="local.yaml"):
        load_settings(config_dir=config_dir, env={})


def test_non_mapping_yaml_is_an_error(write_config):
    config_dir = write_config()
    (config_dir / "environments" / "base.yaml").write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="mapping"):
        load_settings(config_dir=config_dir, env={})


def test_config_dir_can_come_from_env_var(write_config):
    config_dir = write_config(local={"log_level": "DEBUG"})
    settings = load_settings(env={"WBC_CONFIG_DIR": str(config_dir)})
    assert settings.config_dir == Path(config_dir).resolve()
    assert settings.log_level == "DEBUG"
