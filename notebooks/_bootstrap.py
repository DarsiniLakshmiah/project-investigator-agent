# Databricks notebook source
# MAGIC %md
# MAGIC ### Bootstrap (shared)
# MAGIC Included by every pipeline notebook via `%run ./_bootstrap`.
# MAGIC Makes `src/` importable from the Git folder, then loads settings and the project scope.
# MAGIC Contains no business logic.

# COMMAND ----------

import sys
from pathlib import Path

# In a Databricks Git folder the notebook's working directory is notebooks/.
_SRC = str(Path.cwd().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from worldbank_copilot.common import load_project_registry, load_settings  # noqa: E402
from worldbank_copilot.common.logging import configure_logging  # noqa: E402

settings = load_settings()
configure_logging(settings.log_level)
registry = load_project_registry(settings.config_dir)

print(f"environment={settings.environment.value} projects={registry.project_ids}")
