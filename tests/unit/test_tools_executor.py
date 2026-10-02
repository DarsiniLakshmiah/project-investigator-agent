"""Phase 9 reader and executor: allowlist, typed arguments, scope before any read,
explicit error statuses, and no generic fallback to document search."""

import time

import pytest
from tests.conftest import REPO_CONFIG_DIR
from tests.support.tool_fixtures import IPF, PFORR, context, tables

from worldbank_copilot.retrieval.models import ScopeViolation
from worldbank_copilot.tools import registry
from worldbank_copilot.tools.base import DataIntegrityError, ToolArgs, ToolSpec
from worldbank_copilot.tools.executor import ToolExecutor
from worldbank_copilot.tools.models import MechanicalCode, ToolOutcome, ToolStatus
from worldbank_copilot.tools.reader import (
    MAX_ROWS,
    TABLES,
    Filter,
    InMemoryReader,
    ReadError,
    ReadRequest,
    check_rows,
    sort_rows,
    validate_request,
)

EXECUTOR = registry.default_executor()


def run(name, args, ctx=None, scope=IPF, **kw):
    ctx = ctx or context()
    return EXECUTOR.run(name, args, ctx, scope_project_id=scope, **kw), ctx


# -- reader -----------------------------------------------------------------------------


def test_only_allowlisted_tables_and_contract_columns_are_readable():
    with pytest.raises(ReadError, match="not readable"):
        validate_request(ReadRequest("silver.document_chunks", IPF, ("chunk_id",)))
    with pytest.raises(ReadError, match="contract drift"):
        validate_request(ReadRequest("gold.project_360", IPF, ("no_such_column",)))
    with pytest.raises(ScopeViolation, match="scoped to a project_id"):
        validate_request(ReadRequest("gold.project_360", " ", ("project_name",)))
    assert "silver.procurement_awards" not in TABLES  # deferred tool, no access


def test_rows_of_another_project_and_unbounded_results_are_errors():
    req = ReadRequest("gold.project_360", IPF, ("project_name",))
    with pytest.raises(ScopeViolation):
        check_rows(req, [{"project_id": PFORR}])
    with pytest.raises(ReadError, match="more than"):
        check_rows(req, [{"project_id": IPF}] * (MAX_ROWS + 1))


def test_in_memory_reader_always_applies_the_project_predicate():
    reader = InMemoryReader(tables())
    rows = reader.read(ReadRequest("silver.loans", IPF, ("raw_loan_number",)))
    assert {r["project_id"] for r in rows} == {IPF} and len(rows) == 2
    rows = reader.read(
        ReadRequest(
            "silver.loans",
            IPF,
            ("raw_loan_number",),
            (Filter("raw_loan_number", "in", ["IBRD93240"]),),
        )
    )
    assert [r["raw_loan_number"] for r in rows] == ["IBRD93240"]


def test_sorting_puts_nulls_last_in_both_directions_and_is_stable():
    rows = [{"a": None, "b": 1}, {"a": 2, "b": 2}, {"a": 1, "b": 3}, {"a": 2, "b": 0}]
    assert [r["b"] for r in sort_rows(rows, [("a", "asc")])] == [3, 2, 0, 1]
    assert [r["b"] for r in sort_rows(rows, [("a", "desc")])] == [2, 0, 3, 1]
    assert [r["b"] for r in sort_rows(rows, [("a", "desc"), ("b", "asc")])] == [0, 2, 3, 1]


# -- executor -----------------------------------------------------------------------------


def test_unknown_tool_and_unexpected_arguments_are_invalid():
    res, ctx = run("run_sql", {"project_id": IPF, "sql": "select 1"})
    assert res.status == ToolStatus.INVALID_ARGUMENT and "allowlisted" in res.error
    res, ctx = run("get_project_overview", {"project_id": IPF, "anything": 1})
    assert res.status == ToolStatus.INVALID_ARGUMENT and ctx.reader.requests == []


@pytest.mark.parametrize(
    ("project", "scope", "authorized", "code"),
    [
        ("not-a-project", IPF, None, MechanicalCode.PROJECT_INVALID),
        ("P000000", "P000000", None, MechanicalCode.PROJECT_INVALID),
        (PFORR, IPF, None, MechanicalCode.PROJECT_OUT_OF_SCOPE),
        (IPF, IPF, [PFORR], MechanicalCode.PROJECT_OUT_OF_SCOPE),
    ],
)
def test_scope_is_refused_before_any_table_is_read(project, scope, authorized, code):
    for spec in registry.TOOL_SPECS:
        args = {"project_id": project}
        if spec.name == "search_project_documents":
            args["query"] = "closing date"
        res, ctx = run(spec.name, args, scope=scope, authorized_projects=authorized)
        assert res.status == ToolStatus.SCOPE_REFUSED, spec.name
        assert res.mechanical[0].code == code
        assert ctx.reader.requests == [] and ctx.reader.pinned == {}


def test_every_structured_tool_reads_only_its_declared_tables_for_the_scoped_project():
    calls = {
        "get_project_overview": {},
        "get_project_timeline": {},
        "get_rating_history": {},
        "get_financial_status": {"as_of_isr": "latest", "include_events": True},
        "get_results_progress": {},
        "get_risk_register": {},
        "get_attention_signals": {"status": "ALL"},
    }
    assert set(calls) == set(registry.STRUCTURED_TOOLS)
    for name, extra in calls.items():
        spec = EXECUTOR.specs[name]
        res, ctx = run(name, {"project_id": IPF, **extra})
        assert res.status == ToolStatus.OK, (name, res.error, res.mechanical)
        assert ctx.reader.requests and {r.project_id for r in ctx.reader.requests} == {IPF}
        assert {r.table for r in ctx.reader.requests} <= set(spec.tables)
        assert set(res.data_snapshot) == set(spec.tables)  # pinned at request start
        assert res.request_id == "req-test" and res.tool_version == spec.version


def _spec(fn):
    return ToolSpec("probe", "1", "d", ToolArgs, ToolArgs, ("gold.project_360",), fn)


def test_failures_map_to_explicit_statuses_and_are_never_swallowed():
    def integrity(ctx, args):
        raise DataIntegrityError("two rows")

    def boom(ctx, args):
        raise KeyError("missing")

    def foreign(ctx, args):
        raise ScopeViolation("foreign row")

    for fn, status, text in (
        (integrity, ToolStatus.DATA_INTEGRITY_ERROR, "two rows"),
        (boom, ToolStatus.ERROR, "KeyError"),
        (foreign, ToolStatus.SCOPE_REFUSED, "foreign row"),
    ):
        res = ToolExecutor([_spec(fn)]).run(
            "probe", {"project_id": IPF}, context(), scope_project_id=IPF
        )
        assert res.status == status and text in res.error and res.items == []


def test_the_deadline_is_checked_before_every_read():
    ctx = context(deadline=time.monotonic() - 1)
    res = EXECUTOR.run("get_project_overview", {"project_id": IPF}, ctx, scope_project_id=IPF)
    assert res.status == ToolStatus.TIMEOUT and ctx.reader.requests == []


class SpyDocuments:
    """Any access to document search from a structured tool would be recorded."""

    def __init__(self):
        self.touched = []

    def __getattr__(self, name):
        self.touched.append(name)
        raise AssertionError(f"structured tool touched document search ({name})")


def empty_tables():
    data = tables()
    for table in (
        "gold.project_timeline",
        "gold.risk_register",
        "gold.attention_signals",
        "gold.result_progress",
        "silver.isr_snapshots",
    ):
        data[table] = []
    return data


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("get_project_timeline", {}),
        ("get_risk_register", {}),
        ("get_attention_signals", {}),
        ("get_results_progress", {}),
        ("get_rating_history", {}),
        ("get_rating_history", {"isr_sequence_from": 9, "isr_sequence_to": 9}),
        ("get_financial_status", {"loan_number": "IBRD00000"}),
        ("get_financial_status", {"as_of_isr": 99}),
    ],
)
def test_empty_or_not_found_never_falls_back_to_documents(name, args):
    spy = SpyDocuments()
    ctx = context(InMemoryReader(empty_tables()), documents=spy)
    res = EXECUTOR.run(name, {"project_id": IPF, **args}, ctx, scope_project_id=IPF)
    assert res.status in (ToolStatus.EMPTY, ToolStatus.NOT_FOUND), (name, res.status)
    assert res.items == [] and spy.touched == []
    assert all(r.table in EXECUTOR.specs[name].tables for r in ctx.reader.requests)


def test_the_executor_has_no_fallback_path():
    """Structurally: a tool outcome is returned as-is; the executor calls exactly one tool."""
    seen = []

    def empty(ctx, args):
        seen.append("empty")
        return ToolOutcome(status=ToolStatus.NOT_COVERED)

    res = ToolExecutor([_spec(empty), *registry.TOOL_SPECS]).run(
        "probe", {"project_id": IPF}, context(documents=SpyDocuments()), scope_project_id=IPF
    )
    assert res.status == ToolStatus.NOT_COVERED and seen == ["empty"]


def test_registry_is_the_approved_catalog_without_procurement():
    assert registry.STRUCTURED_TOOLS == (
        "get_project_overview",
        "get_project_timeline",
        "get_rating_history",
        "get_financial_status",
        "get_results_progress",
        "get_risk_register",
        "get_attention_signals",
    )
    assert registry.DOCUMENT_TOOLS == ("search_project_documents",)
    assert not any("procurement" in s.name or "sql" in s.name for s in registry.TOOL_SPECS)


class FakeHistory:
    def __init__(self, version):
        self.version = version

    def select(self, column):
        assert column == "version"
        return self

    def collect(self):
        return [(self.version,)]


class FakeSpark:
    def __init__(self):
        self.statements = []

    def sql(self, statement):
        self.statements.append(statement)
        return FakeHistory(7)


def test_spark_reader_pins_one_delta_version_per_table_per_request():
    from worldbank_copilot.common import load_settings
    from worldbank_copilot.tools.reader import SparkTableReader

    settings = load_settings("databricks", config_dir=REPO_CONFIG_DIR, env={})
    spark = FakeSpark()
    reader = SparkTableReader.for_settings(spark, settings)
    reader.pin(["gold.project_360", "silver.loans", "gold.project_360"])
    assert spark.statements == [
        "DESCRIBE HISTORY `worldbank_copilot`.`gold`.`project_360` LIMIT 1",
        "DESCRIBE HISTORY `worldbank_copilot`.`silver`.`loans` LIMIT 1",
    ]
    assert reader.snapshot() == {"gold.project_360": 7, "silver.loans": 7}
    with pytest.raises(ReadError, match="not readable"):
        reader.pin(["silver.procurement_awards"])
    unpinned = SparkTableReader(FakeSpark(), lambda t: t, pin_versions=False)
    unpinned.pin(["gold.project_360"])
    assert unpinned.snapshot() == {"gold.project_360": None} and unpinned.spark.statements == []
