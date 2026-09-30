import pytest

from worldbank_copilot.common.config import Environment, detect_environment, load_settings
from worldbank_copilot.common.exceptions import ConfigurationError


def test_defaults_to_local():
    assert detect_environment({}) is Environment.LOCAL


def test_databricks_runtime_is_auto_detected():
    assert detect_environment({"DATABRICKS_RUNTIME_VERSION": "15.4"}) is Environment.DATABRICKS


def test_explicit_env_var_beats_auto_detection():
    env = {"DATABRICKS_RUNTIME_VERSION": "15.4", "WBC_ENV": "local"}
    assert detect_environment(env) is Environment.LOCAL


def test_env_var_is_case_insensitive():
    assert detect_environment({"WBC_ENV": " Databricks "}) is Environment.DATABRICKS


def test_invalid_env_var_fails_clearly():
    with pytest.raises(ConfigurationError, match="WBC_ENV='staging'"):
        detect_environment({"WBC_ENV": "staging"})


def test_explicit_argument_beats_env_var(repo_config_dir):
    settings = load_settings("databricks", config_dir=repo_config_dir, env={"WBC_ENV": "local"})
    assert settings.environment is Environment.DATABRICKS


def test_databricks_data_root_requires_volume(repo_config_dir):
    env = {"WBC_ENV": "databricks", "WBC_SOURCE_VOLUME": ""}
    settings = load_settings(config_dir=repo_config_dir, env=env)
    with pytest.raises(ConfigurationError, match="WBC_SOURCE_VOLUME"):
        _ = settings.data_root


def test_databricks_paths_default_from_unity_catalog_settings(repo_config_dir):
    settings = load_settings(config_dir=repo_config_dir, env={"WBC_ENV": "databricks"})
    assert settings.data_root == "/Volumes/worldbank_ai/bronze/sources/data"
    assert str(settings.local_output_root).replace("\\", "/") == (
        "/Volumes/worldbank_ai/silver/pipeline_artifacts"
    )
    local = load_settings("local", config_dir=repo_config_dir, env={})
    assert local.local_output_root == (repo_config_dir.parent / ".local_output").resolve()


def test_databricks_data_root_is_volume_path(repo_config_dir):
    env = {"WBC_ENV": "databricks", "WBC_DATABRICKS_VOLUME_ROOT": "/Volumes/wb/raw/sources"}
    settings = load_settings(config_dir=repo_config_dir, env=env)
    assert settings.data_root == "/Volumes/wb/raw/sources"


def test_environment_specific_file_is_selected(write_config):
    config_dir = write_config(
        base={"databricks": {"catalog": None}},
        local={"databricks": {"catalog": "dev_local"}},
        databricks={"databricks": {"catalog": "prod_uc"}},
    )
    assert load_settings("local", config_dir=config_dir, env={}).databricks.catalog == "dev_local"
    assert (
        load_settings("databricks", config_dir=config_dir, env={}).databricks.catalog == "prod_uc"
    )
