"""Notebook 07 cell independence after install -> restartPython -> bootstrap -> Steps 0/1.

Static analysis of the Databricks notebook source: every phase cell after Step 1 may use
only (a) Databricks globals, (b) names defined by ``_bootstrap`` and Steps 0/1, (c) names
it defines or imports itself, and (d) names it declares as a deliberate dependency with
``require_state("...", step=...)`` (a clear error instead of a raw NameError).
"""

import ast
import builtins
import re

import pytest
from tests.conftest import REPO_CONFIG_DIR, REPO_ROOT

from worldbank_copilot.common import load_settings
from worldbank_copilot.common.exceptions import ReconciliationError
from worldbank_copilot.retrieval import pipeline as rp
from worldbank_copilot.retrieval.config import load_retrieval_settings

NOTEBOOKS = REPO_ROOT / "notebooks"
DATABRICKS_GLOBALS = {"spark", "dbutils", "display", "displayHTML"}
BUILTINS = set(dir(builtins))


def cells(path):
    return [c for c in path.read_text(encoding="utf-8").split("# COMMAND ----------") if c.strip()]


def defined_and_used(source):
    tree = ast.parse(source)
    defined, used = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (used if isinstance(node.ctx, ast.Load) else defined).add(node.id)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            defined.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.FunctionDef | ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
    return defined, used


def top_level_defined(source):
    """Names a cell/script leaves in the notebook namespace (not function parameters)."""
    out = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import | ast.ImportFrom):
            out.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.FunctionDef | ast.ClassDef):
            out.add(node.name)
        else:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                    out.add(sub.id)
    return out


def declared(source):
    """Names declared with require_state("a", "b", step=...)."""
    out = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "require_state":
            out.update(a.value for a in node.args if isinstance(a, ast.Constant))
    return out


def step_of(cell):
    match = re.search(r"^# Step (\d+)([ab]?)", cell, re.M)
    return (int(match.group(1)), match.group(2)) if match else None


NB07 = "07_build_retrieval_and_evaluate.py"
NB07B = "07b_rerank_experiment.py"
# Deliberate (legitimate experiment-state) dependencies, per notebook and cell.
EXPECTED_DECLARED = {
    NB07: {"Step 5": {"selected"}},
    NB07B: {"Step 3": {"decisions"}, "Step 4": {"selected"}, "Step 5": {"selected"}},
}
EXPECTED_STEPS = {
    NB07: ["Step 2", "Step 3a", "Step 3b", "Step 4", "Step 5", "Step 6", "Step 7"],
    NB07B: ["Step 2", "Step 3", "Step 4", "Step 5"],
}


def baseline(notebook):
    """Names available after install -> restartPython -> bootstrap -> Steps 0/1."""
    names = set(DATABRICKS_GLOBALS)
    names |= top_level_defined((NOTEBOOKS / "_bootstrap.py").read_text(encoding="utf-8"))
    for cell in cells(NOTEBOOKS / notebook):
        step = step_of(cell)
        if step is not None and step[0] <= 1:
            names |= top_level_defined(cell)
    return names


def phase_cells(notebook):
    out = []
    for cell in cells(NOTEBOOKS / notebook):
        step = step_of(cell)
        if step is not None and step[0] >= 2:
            out.append((f"Step {step[0]}{step[1]}", cell))
    return out


def hidden_dependencies(source, base):
    defined, used = defined_and_used(source)
    return used - defined - BUILTINS - base - declared(source)


CASES = [(nb, step, src) for nb in (NB07, NB07B) for step, src in phase_cells(nb)]


@pytest.mark.parametrize("notebook", [NB07, NB07B])
def test_every_phase_has_a_cell(notebook):
    assert [name for name, _ in phase_cells(notebook)] == EXPECTED_STEPS[notebook]


@pytest.mark.parametrize(
    ("notebook", "step", "source"), CASES, ids=[f"{nb[:3]}-{step}" for nb, step, _ in CASES]
)
def test_phase_cell_needs_nothing_but_baseline_or_declared_state(notebook, step, source):
    free = hidden_dependencies(source, baseline(notebook))
    assert free == set(), f"{notebook} {step} depends on state from another cell: {sorted(free)}"


@pytest.mark.parametrize("notebook", [NB07, NB07B])
def test_only_legitimate_experiment_state_is_declared(notebook):
    deps = {name: declared(source) for name, source in phase_cells(notebook)}
    expected = EXPECTED_DECLARED[notebook]
    assert {k: v for k, v in deps.items() if v} == expected


@pytest.mark.parametrize("notebook", [NB07, NB07B])
def test_declared_state_is_defined_by_an_earlier_cell(notebook):
    defined_so_far: set[str] = set()
    for step, source in phase_cells(notebook):
        missing = declared(source) - defined_so_far
        assert not missing, f"{step} declares {missing} that no earlier cell defines"
        defined_so_far |= defined_and_used(source)[0]


def test_baseline_excludes_function_parameters():
    base = baseline(NB07B)
    assert {"settings", "registry", "require_state", "rp", "rs", "provider", "capabilities"} <= base
    assert "names" not in base and "step" not in base  # parameters of require_state


def test_hidden_dependencies_are_detected():
    base = baseline(NB07B)
    accidental = """
from worldbank_copilot.retrieval import report
for qid in ("q01",):
    result = retriever.retrieve(qid, "P1", strategy=cfg.chunk_strategy, method="dense",
                                reranker=rerankers["none"], candidate_k=10, final_k=5)
    print(report.format_evidence(result))
"""
    assert hidden_dependencies(accidental, base) == {"retriever", "cfg", "rerankers"}
    declared_ok = """
require_state("selected", step="Step 3")
cfg = selected.config
"""
    assert hidden_dependencies(declared_ok, base) == set()
    undeclared = "cfg = selected.config\n"
    assert hidden_dependencies(undeclared, base) == {"selected"}


def test_notebook_07_validated_fixes_are_preserved():
    text = (NOTEBOOKS / NB07).read_text(encoding="utf-8")
    cell = dict(phase_cells(NB07))
    assert "PILOT_REQUESTS = None" in cell["Step 3a"] and "<<<<<<<" not in text
    assert "client.list_indexes" not in text
    assert "rp.open_retriever(" in cell["Step 4"] and "vs_index" not in cell["Step 4"]
    assert declared(cell["Step 5"]) == {"selected"}
    assert "rp.persisted_corpus_profile(" in cell["Step 6"] and "corpus." not in cell["Step 6"]
    assert hidden_dependencies(cell["Step 7"], baseline(NB07)) == set()


def test_notebook_07b_reads_persistent_resources_and_never_builds_them():
    text = (NOTEBOOKS / NB07B).read_text(encoding="utf-8")
    for forbidden in (
        "update_embedding_cache",
        "build_index_source",
        "ensure_vector_index",
        "build_corpus",
        "create_delta_sync_index",
    ):
        assert forbidden not in text
    assert text.count("rp.open_retriever(") == 4  # Steps 2-5 open the existing index


@pytest.mark.parametrize("notebook", [NB07, NB07B])
def test_notebook_has_no_debug_cells_or_hard_coded_endpoint(notebook):
    text = (NOTEBOOKS / notebook).read_text(encoding="utf-8")
    assert "worldbank-gep-ai-search" not in text and "client.list_indexes" not in text
    for cell in cells(NOTEBOOKS / notebook):
        code = "\n".join(line for line in cell.splitlines() if not line.startswith("# MAGIC"))
        if code.strip() and "dbutils.library.restartPython" not in code:
            assert step_of(cell) is not None or "Databricks notebook source" in cell, cell[:80]


def test_require_state_gives_a_clear_error():
    source = (NOTEBOOKS / "_bootstrap.py").read_text(encoding="utf-8")
    function = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "require_state"
    )
    namespace: dict = {}
    exec(compile(ast.Module([function], []), "bootstrap", "exec"), namespace)
    namespace["selected"] = object()
    namespace["require_state"]("selected", step="Step 4")  # present: no error
    with pytest.raises(RuntimeError, match="vs_index not defined in this session: run Step 3b"):
        namespace["require_state"]("vs_index", step="Step 3b")


def test_no_merge_conflict_markers_in_the_repository():
    marker = re.compile(r"^(<{7}|>{7})( |$)", re.M)
    patterns = (
        "src/**/*.py",
        "tests/**/*.py",
        "notebooks/*.py",
        "configs/**/*.yaml",
        "sql/*.sql",
        "*.md",
        "*.txt",
        "evaluation/*.yaml",
    )
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for pattern in patterns
        for path in REPO_ROOT.glob(pattern)
        if marker.search(path.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert offenders == []


# -- read-only helpers used by Steps 4 and 5 ------------------------------------------------

RS = load_retrieval_settings(REPO_CONFIG_DIR)
SETTINGS = load_settings("databricks", config_dir=REPO_CONFIG_DIR, env={})


class Index:
    def __init__(self, status):
        self.status = status

    def describe(self):
        return {"status": self.status}


class Client:
    def __init__(self, index=None):
        self.index = index

    def get_index(self, endpoint_name, index_name):
        if self.index is None:
            raise Exception("NotFound")
        return self.index


def test_open_vector_index_requires_an_existing_ready_index_and_creates_nothing():
    ready = rp.open_vector_index(SETTINGS, RS, client=Client(Index({"ready": True})))
    assert ready.index_name == "worldbank_copilot.silver.document_chunk_index_qwen3_v1"
    with pytest.raises(ReconciliationError, match="not ready"):
        rp.open_vector_index(
            SETTINGS,
            RS,
            client=Client(
                Index({"ready": False, "detailed_state": "PROVISIONING_INITIAL_SNAPSHOT"})
            ),
        )
    with pytest.raises(ReconciliationError, match="run Step 3b first"):
        rp.open_vector_index(SETTINGS, RS, client=Client(None))
