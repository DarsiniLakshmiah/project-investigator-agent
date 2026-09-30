import importlib
import os
import subprocess
import sys

import pytest

PACKAGES = [
    "worldbank_copilot",
    "worldbank_copilot.common",
    "worldbank_copilot.common.config",
    "worldbank_copilot.common.exceptions",
    "worldbank_copilot.common.logging",
    "worldbank_copilot.common.project_registry",
    "worldbank_copilot.ingestion",
    "worldbank_copilot.parsing",
    "worldbank_copilot.transformations",
    "worldbank_copilot.retrieval",
    "worldbank_copilot.tools",
    "worldbank_copilot.agents",
    "worldbank_copilot.guardrails",
    "worldbank_copilot.observability",
    "worldbank_copilot.api",
    "worldbank_copilot.api.routes",
]


@pytest.mark.parametrize("module", PACKAGES)
def test_package_imports(module):
    assert importlib.import_module(module) is not None


def test_version_is_exposed():
    import worldbank_copilot

    assert worldbank_copilot.__version__ == "0.1.0"


def test_import_and_load_config_without_databricks():
    """Importing the package and loading settings must not pull in Databricks/Spark."""
    code = (
        "import sys\n"
        "import worldbank_copilot, worldbank_copilot.common\n"
        "from worldbank_copilot.common import load_settings, load_project_registry\n"
        "s = load_settings(env={})\n"
        "load_project_registry(s.config_dir)\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
        "{'pyspark', 'databricks', 'mlflow', 'langgraph', 'docling'})\n"
        "assert not bad, bad\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DATABRICKS", "WBC_"))}
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
