"""Offline @3 protocol regressions; historical @2 and frozen inputs stay intact."""

import itertools
import json
from pathlib import Path

import pytest

from worldbank_copilot.validation import phase10d_models as h
from worldbank_copilot.validation import phase10d_notebook_v3 as v3

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / h.NOTEBOOK_FILE).read_text("utf-8")
IMPORT = "from worldbank_copilot.validation.phase10d_models import run_databricks_validation"
WIDGETS = [line for line in SOURCE.splitlines() if line.startswith("dbutils.widgets.text(")]
EXPECTED = {
    "scheme": "databricks_wrapper_ast@3",
    "sha256": "754e42ac71c378df721b258b23a2c9b9b1d4aef7c8c97ef2da1d942ad957c349",
}
START = SOURCE.index(IMPORT)
END = SOURCE.index("artifact = run_databricks_validation(")


def wrapper(statements):
    return SOURCE[:START] + "\n".join(statements) + "\n" + SOURCE[END:]


def test_frozen_phase10c_and_phase9_boundaries():
    assert h.canonical_sha256(ROOT / "src/worldbank_copilot/validation/phase10c_evidence.py") == (
        "d427bb79ceb52a7c5f4991bb02c7f47e1dfb7e37a77167555f4143e99831ce9d"
    )
    assert h.canonical_sha256(ROOT / h.accepted.LOCK_FILE) == (
        "77601f14674b28f77da04c3f2b638c8633df5c301b84fed081ed3c329244ace6"
    )
    assert h.accepted.notebook_identity((ROOT / h.accepted.NOTEBOOK_FILE).read_text("utf-8")) == {
        "scheme": "databricks_wrapper_ast@2",
        "sha256": "0a30c8873fdd16625774afb33c75bc6acae4ca81a86f1e20f08fbec386ac5fd9",
    }
    assert h.accepted.prepare(ROOT)
    assert h.prior.prepare_protocol(ROOT, dependency_ok=True)


def test_exact_equivalence_keeps_payload_and_historical_v2_behavior():
    a = wrapper([IMPORT, *WIDGETS])
    b = wrapper([*WIDGETS, IMPORT])
    assert (
        v3.notebook_identity(SOURCE)
        == v3.notebook_identity(a)
        == v3.notebook_identity(b)
        == EXPECTED
    )
    assert v3.notebook_program(b) == h.accepted.notebook_program(a)
    assert h.accepted.notebook_identity(b) == {
        "scheme": "databricks_wrapper_ast@2",
        "sha256": "08b8e65b44ed4a14a74889a417613cfa17d54006c4c49a41b211ab9f9751b8ec",
    }


@pytest.mark.parametrize("order", list(itertools.permutations(range(4))))
def test_only_two_approved_placements(order):
    nodes = [IMPORT, *WIDGETS]
    source = wrapper([nodes[i] for i in order])
    if order in ((0, 1, 2, 3), (1, 2, 3, 0)):
        assert v3.notebook_identity(source) == EXPECTED
    else:
        assert v3.notebook_program(source) == h.accepted.notebook_program(source)
        assert v3.notebook_identity(source) != EXPECTED


@pytest.mark.parametrize(
    "work",
    [
        "print('work')",
        "x = 1",
        "adapter.invoke(request)",
        "run_databricks_validation(settings)",
        "import os",
        'dbutils.widgets.text("unknown", "", "unknown")',
        "# COMMAND ----------\n# MAGIC %run ./other\n# COMMAND ----------",
    ],
)
@pytest.mark.parametrize("position", range(4))
def test_no_commutation_across_work(work, position):
    nodes = [*WIDGETS, IMPORT]
    nodes.insert(position, work)
    source = wrapper(nodes)
    assert v3.notebook_program(source) == h.accepted.notebook_program(source)
    assert v3.notebook_identity(source) != EXPECTED


@pytest.mark.parametrize(
    "nodes",
    [
        [*WIDGETS, IMPORT.replace("phase10d_models", "unknown_validation")],
        [*WIDGETS, IMPORT.replace("run_databricks_validation", "unknown_symbol")],
        [WIDGETS[0].replace('"commit_sha"', '"unknown"'), *WIDGETS[1:], IMPORT],
        [*WIDGETS[:2], IMPORT],
        [*WIDGETS, IMPORT, 'dbutils.widgets.text("extra", "", "extra")'],
        [*WIDGETS, 'dbutils.widgets.text("extra", "", "extra")', IMPORT],
        [WIDGETS[0].replace('"",', "side_effect(),"), *WIDGETS[1:], IMPORT],
    ],
)
def test_unknown_incomplete_extra_or_effectful_setup_not_normalized(nodes):
    source = wrapper(nodes)
    assert v3.notebook_program(source) == h.accepted.notebook_program(source)
    assert v3.notebook_identity(source) != EXPECTED


@pytest.mark.parametrize(
    "old,new",
    [
        ('"endpoints", ""', '"endpoints", "changed-default"'),
        ("One or two approved chat endpoint names, comma separated", "changed description"),
        ('widgets.get("endpoints").strip()', 'widgets.get("endpoints")'),
        ('widgets.get("commit_sha").strip()', 'widgets.get("commit_sha")'),
        ('widgets.get("run_id").strip()', 'widgets.get("run_id")'),
        ("    settings,", "    other_settings,"),
        ('!= "PASS"', '== "PASS"'),
        ("%run ./_bootstrap", "%run ./other_bootstrap"),
    ],
)
def test_semantic_changes_remain_visible(old, new):
    assert old in SOURCE
    assert v3.notebook_identity(SOURCE.replace(old, new)) != EXPECTED


def test_import_before_bootstrap_and_arbitrary_work_order_remain_sensitive():
    b = wrapper([*WIDGETS, IMPORT])
    before = b.replace(IMPORT, "").replace(
        "# MAGIC %run ./_bootstrap", IMPORT + "\n# COMMAND ----------\n# MAGIC %run ./_bootstrap"
    )
    assert v3.notebook_identity(before) != EXPECTED
    assert v3.notebook_identity(wrapper(["x = 1", "y = 2", *WIDGETS, IMPORT])) != (
        v3.notebook_identity(wrapper(["y = 2", "x = 1", *WIDGETS, IMPORT]))
    )


def test_revised_lock_lineage_sources_and_unchanged_capability_protocol():
    previous = json.loads((ROOT / h.PREVIOUS_LOCK_FILE).read_text("utf-8"))
    current = json.loads((ROOT / h.LOCK_FILE).read_text("utf-8"))
    assert h.canonical_sha256(ROOT / h.PREVIOUS_LOCK_FILE) == (
        "ef9f5f7c59de938389933d9837252df2e08b2bf001f68a40423af73a999f43e3"
    )
    assert current == h.build_lock(ROOT)
    assert current["notebook_semantic_identity"] == EXPECTED
    assert previous["notebook_semantic_identity"] == {
        **EXPECTED,
        "scheme": "databricks_wrapper_ast@2",
    }
    changed = {
        key for key in current.keys() | previous.keys() if current.get(key) != previous.get(key)
    }
    assert changed == {
        "schema",
        "supersedes_lock_sha256_lf",
        "files_sha256_lf",
        "notebook_semantic_identity",
    }
    changed_files = {
        name
        for name in current["files_sha256_lf"]
        if current["files_sha256_lf"][name] != previous["files_sha256_lf"].get(name)
    }
    assert changed_files == {
        "src/worldbank_copilot/validation/phase10d_models.py",
        "src/worldbank_copilot/validation/phase10d_notebook_v3.py",
        "tests/unit/test_phase10d_notebook_v3.py",
    }
    for name, digest in current["files_sha256_lf"].items():
        assert h.canonical_sha256(ROOT / name) == digest


def test_production_prepare_accepts_databricks_order_without_using_v2_identity(monkeypatch):
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path == ROOT / h.NOTEBOOK_FILE:
            return wrapper([*WIDGETS, IMPORT])
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)

    def forbidden(*args, **kwargs):
        raise AssertionError("10D must own its identity")

    # Shared prepare still legitimately checks the historical 10C identity.
    cases, lock = h.prepare(ROOT)
    assert len(cases) == 11 and lock["notebook_semantic_identity"] == EXPECTED
    monkeypatch.setattr(h.accepted, "notebook_identity", forbidden)
    assert h.build_lock(ROOT)["notebook_semantic_identity"] == EXPECTED
