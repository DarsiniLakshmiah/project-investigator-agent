"""Dependency specification and the notebook dependency-health check."""

from pathlib import Path
from types import SimpleNamespace

from tests.conftest import REPO_CONFIG_DIR, REPO_ROOT

from worldbank_copilot.common.dependency_health import (
    HealthReport,
    Shadow,
    check_environment,
    load_policy,
    pin_mismatches,
    pip_check,
    read_pins,
    shadowed_distributions,
)

NOTEBOOK_FILES = [
    "requirements-databricks.txt",
    "requirements-retrieval.txt",
    "requirements-reranker.txt",
]


def ok_runner(*_a, **_k):
    return SimpleNamespace(returncode=0, stdout="No broken requirements found.", stderr="")


def broken_runner(*_a, **_k):
    return SimpleNamespace(
        returncode=1,
        stderr="",
        stdout="grpcio-status 1.76.0 has requirement protobuf<7.0.0,>=6.31.1, "
        "but you have protobuf 5.29.6.",
    )


def test_notebook_requirement_files_use_exact_pins_only():
    pins, unpinned = read_pins(REPO_ROOT / f for f in NOTEBOOK_FILES)
    assert unpinned == []
    assert pins["databricks-ai-search"] == "0.78"
    assert pins["sentence-transformers"] == "5.5.1"


def test_deprecated_and_runtime_packages_are_not_installed_by_us():
    pins, _ = read_pins(REPO_ROOT / f for f in NOTEBOOK_FILES)
    # databricks-vectorsearch pins protobuf<6 (the root cause of the protobuf downgrade).
    assert "databricks-vectorsearch" not in pins
    # Provided by the Databricks runtime; installing them would replace runtime versions.
    for runtime_owned in ("databricks-sdk", "protobuf", "mlflow-skinny", "grpcio-status"):
        assert runtime_owned not in pins
    core, _ = read_pins([REPO_ROOT / "requirements-retrieval.txt"])
    assert set(core) == {"databricks-ai-search", "deprecation"}


def test_constraints_protect_the_runtime_protobuf():
    text = (REPO_ROOT / "constraints-databricks.txt").read_text(encoding="utf-8")
    lines = [line.split("#")[0].strip() for line in text.splitlines()]
    assert "protobuf>=6.33.5,<7" in lines
    assert "grpcio-status>=1.76.0,<2" in lines


def test_notebooks_install_with_constraints_and_isolate_the_reranker():
    notebooks = REPO_ROOT / "notebooks"
    nb07 = (notebooks / "07_build_retrieval_and_evaluate.py").read_text(encoding="utf-8")
    nb07b = (notebooks / "07b_rerank_experiment.py").read_text(encoding="utf-8")
    install07 = next(line for line in nb07.splitlines() if "%pip install" in line)
    assert "-c ../constraints-databricks.txt" in install07
    assert "requirements-reranker.txt" not in install07
    assert "check_reranker=False" in nb07
    assert nb07.index("check_environment(") < nb07.index("probe_capabilities(")
    install07b = next(line for line in nb07b.splitlines() if "%pip install" in line)
    assert "requirements-reranker.txt" in install07b
    assert "-c ../constraints-databricks.txt" in install07b
    assert "check_environment(" in nb07b


def test_unpinned_lines_are_reported(tmp_path):
    req = tmp_path / "r.txt"
    req.write_text("# comment\nfoo==1.2\nbar>=2\nbaz\n\n", encoding="utf-8")
    pins, unpinned = read_pins([req])
    assert pins == {"foo": "1.2"} and unpinned == ["r.txt: bar>=2", "r.txt: baz"]


def test_pin_mismatches_ignore_local_build_tags():
    installed = {"torch": "2.12.0+cpu", "foo": "1.0", "bar": None}.get
    out = pin_mismatches({"torch": "2.12.0", "foo": "1.1", "bar": "3"}, installed)
    assert out == ["bar==3 required, not installed", "foo==1.1 required, 1.0 installed"]


def test_pip_check_result_is_parsed():
    assert pip_check(runner=ok_runner)[0] is True
    ok, output = pip_check(runner=broken_runner)
    assert ok is False and "protobuf 5.29.6" in output


def _dist(root: Path, name: str, version: str) -> None:
    info = root / f"{name}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8"
    )


def test_shadowed_runtime_packages_are_detected(tmp_path):
    notebook, runtime = tmp_path / "notebook", tmp_path / "runtime"
    _dist(notebook, "protobuf", "5.29.6")
    _dist(runtime, "protobuf", "6.33.5")
    _dist(notebook, "pydantic", "2.13.5")
    _dist(runtime, "pydantic", "2.13.3")
    _dist(runtime, "requests", "2.32.5")
    _dist(notebook, "same", "1.0")
    _dist(runtime, "same", "1.0")
    shadows = {s.name: s for s in shadowed_distributions([str(notebook), str(runtime)])}
    assert set(shadows) == {"protobuf", "pydantic"}
    assert shadows["protobuf"].active_version == "5.29.6"
    assert shadows["protobuf"].hidden[0][0] == "6.33.5"


def test_health_fails_only_for_protected_shadowing_and_broken_requirements():
    protobuf = Shadow("protobuf", "5.29.6", "nb", (("6.33.5", "rt"),))
    pydantic = Shadow("pydantic", "2.13.5", "nb", (("2.13.3", "rt"),))
    base = {"pip_check_ok": True, "pip_check_output": "", "protected": ("protobuf",)}
    assert HealthReport(**base, shadowed=[pydantic]).ok
    bad = HealthReport(**base, shadowed=[protobuf, pydantic])
    assert not bad.ok and [s.name for s in bad.protected_shadowed] == ["protobuf"]
    assert "PROTECTED runtime package replaced: protobuf 5.29.6 (runtime had 6.33.5)" in (
        bad.format()
    )
    assert not HealthReport(pip_check_ok=False, pip_check_output="x").ok
    assert not HealthReport(pip_check_ok=True, pip_check_output="", pin_mismatches=["m"]).ok


def test_policy_and_check_environment(tmp_path):
    policy = load_policy(REPO_CONFIG_DIR)
    assert {"protobuf", "grpcio-status", "databricks-sdk", "mlflow-skinny"} <= set(
        policy["protected"]
    )
    report = check_environment(
        REPO_ROOT,
        ["requirements-retrieval.txt"],
        REPO_CONFIG_DIR,
        runner=broken_runner,
        paths=[str(tmp_path)],
    )
    assert not report.ok and not report.pip_check_ok
    assert report.unpinned_lines == []
    assert set(report.versions) == set(policy["report"])
