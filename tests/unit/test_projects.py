import pytest

from worldbank_copilot.common.exceptions import ConfigurationError, UnknownProjectError
from worldbank_copilot.common.project_registry import (
    load_project_registry,
    validate_project_id_format,
)

EXPECTED_IDS = ["P130544", "P179039", "P506272"]


@pytest.fixture
def registry(repo_config_dir):
    return load_project_registry(repo_config_dir)


def _project(project_id="P000001", **overrides):
    project = {
        "project_id": project_id,
        "name": "Test",
        "instrument": "Investment Project Financing",
        "lifecycle_stage": "early",
        "purpose": "p",
        "analytical_question": "q",
        "documents_dir": project_id,
        "procurement_coverage": {"covered_by_ipf_contract_awards": True, "note": "n"},
    }
    project.update(overrides)
    return project


def test_exactly_the_three_prototype_projects(registry):
    assert registry.project_ids == EXPECTED_IDS


def test_instruments_and_procurement_coverage(registry):
    ipf = registry.get("P130544")
    assert ipf.instrument == "Investment Project Financing"
    assert ipf.procurement_coverage.covered_by_ipf_contract_awards is True
    for pid in ("P179039", "P506272"):
        pforr = registry.get(pid)
        assert pforr.instrument == "Program-for-Results Financing"
        assert pforr.procurement_coverage.covered_by_ipf_contract_awards is False
        assert "must not be interpreted as no procurement activity" in (
            pforr.procurement_coverage.note
        )


def test_lifecycle_stages(registry):
    stages = {p.project_id: p.lifecycle_stage for p in registry.projects}
    assert stages == {"P130544": "mature", "P179039": "mid", "P506272": "early"}


def test_documents_dir_matches_project_id(registry):
    for project in registry.projects:
        assert project.documents_dir == project.project_id


def test_lookup_normalises_case_and_whitespace(registry):
    assert registry.get(" p179039 ").project_id == "P179039"
    assert "p506272" in registry
    assert registry.validate_project_id("p130544") == "P130544"


def test_unknown_project_is_rejected(registry):
    with pytest.raises(UnknownProjectError) as exc:
        registry.get("P999999")
    assert exc.value.allowed == EXPECTED_IDS
    assert "P999999" not in registry


@pytest.mark.parametrize("bad", ["", "130544", "P13054", "P1305440", "X130544", "P13O544"])
def test_malformed_project_id_format(bad):
    with pytest.raises(ValueError):
        validate_project_id_format(bad)


def test_non_string_is_not_contained(registry):
    assert 130544 not in registry


def test_duplicate_project_ids_rejected(write_config):
    config_dir = write_config(projects={"projects": [_project(), _project()]})
    with pytest.raises(ConfigurationError, match="Duplicate project IDs"):
        load_project_registry(config_dir)


def test_invalid_project_id_in_yaml_rejected(write_config):
    config_dir = write_config(projects={"projects": [_project("P12")]})
    with pytest.raises(ConfigurationError, match="Invalid project ID"):
        load_project_registry(config_dir)


def test_unknown_project_field_rejected(write_config):
    config_dir = write_config(projects={"projects": [_project(risk_probability=0.75)]})
    with pytest.raises(ConfigurationError, match="risk_probability"):
        load_project_registry(config_dir)


def test_empty_registry_rejected(write_config):
    config_dir = write_config(projects={"projects": []})
    with pytest.raises(ConfigurationError, match="At least one project"):
        load_project_registry(config_dir)


def test_missing_projects_file(write_config):
    config_dir = write_config()
    with pytest.raises(ConfigurationError, match="projects.yaml"):
        load_project_registry(config_dir)
