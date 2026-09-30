import pytest

from worldbank_copilot.common.exceptions import SourceContractError
from worldbank_copilot.ingestion.contracts import (
    FINANCERS,
    PROJECTS,
    ColumnSpec,
    FieldKind,
    SourceContract,
)
from worldbank_copilot.ingestion.tabular import (
    normalize_column_name,
    read_tabular_source,
    resolve_header,
    to_raw_text,
)

SIMPLE = SourceContract(
    name="simple",
    bronze_table="bronze_simple",
    project_id_field="source_project_id",
    columns=(
        ColumnSpec("source_project_id", ("Project ID",), True, FieldKind.IDENTIFIER),
        ColumnSpec("amount", ("Amount (USD)",), True, FieldKind.AMOUNT),
        ColumnSpec("note", ("Note",)),
    ),
)


def rows(*values):
    return list(enumerate(values, start=1))


def read(source_rows, contract=SIMPLE, ids=("P130544",)):
    return read_tabular_source(
        source_rows, contract, ids, source_file="f.csv", ingested_at="t", run_id="r"
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Project Development Objective ", "Project Development Objective"),
        ("  Amount   (USD) ", "Amount (USD)"),
        ("Line\nbreak", "Line break"),
        (None, ""),
    ],
)
def test_column_whitespace_normalization(raw, expected):
    assert normalize_column_name(raw) == expected


def test_raw_text_preserves_values():
    assert to_raw_text("USD  ") == "USD  "
    assert to_raw_text(" Water") == " Water"
    assert to_raw_text(100000000.0) == "100000000.0"
    assert to_raw_text(95) == "95"
    assert to_raw_text("") == ""
    assert to_raw_text(None) is None


def test_header_detected_below_title_rows():
    source = rows(
        ("World Bank Projects, data as of 09/26/2026", None, None),
        (None, None, None),
        ("Project ID", "Amount (USD)", "Note"),
        ("P130544", "10", "x"),
    )
    table = read(source)
    assert table.metadata.header_row == 3
    assert table.records[0]["_source_row"] == 4


def test_header_whitespace_and_case_are_tolerated():
    table = read(rows((" project id", "Amount  (USD) ", "NOTE"), ("P130544", "1", "")))
    assert table.metadata.column_mapping == {
        "source_project_id": " project id",
        "amount": "Amount  (USD) ",
        "note": "NOTE",
    }


def test_column_mapping_by_position_independent_of_order():
    table = read(rows(("Note", "Amount (USD)", "Project ID"), ("hello", "5", "P130544")))
    record = table.records[0]
    assert (record["source_project_id"], record["amount"], record["note"]) == (
        "P130544",
        "5",
        "hello",
    )


def test_alias_project_header_for_financers():
    header = resolve_header(
        rows(
            ("title", None, None, None, None, None, None),
            ("Project", "Name", "Current Amount", "Amount (USD)", "Financer ID", "Currency",
             "Project Financial Type"),
        ),
        FINANCERS,
    )  # fmt: skip
    assert header.column_mapping["source_project_id"] == "Project"


def test_secondary_api_header_is_skipped_and_recorded():
    headers = [c.source_names[0] for c in PROJECTS.columns]
    api = ["id", "regionname", "countryshortname", "status", "last_stage_reached_name",
           "project_name", "pdo", "impagency", "public_disclosure_date", "boardapprovaldate",
           "loan_effective_date", "closingdate", "curr_project_cost", "curr_ibrd_commitment",
           "idacommamt", "grantamt", "curr_total_commitment", "borrower", "lendinginstr",
           "envassesmentcategorycode", "esrc_ovrl_risk_rate", "supplementprojectflg",
           "cons_serv_reqd_ind", "proj_last_upd_date", "projectfinancialtype"]  # fmt: skip
    data = ["P130544"] + ["v"] * (len(headers) - 1)
    table = read(rows(["title"], headers, api, data), contract=PROJECTS)
    assert table.metadata.header_row == 2
    assert table.metadata.secondary_header_row == 3
    assert table.metadata.api_field_names["ibrd_commitment"] == "curr_ibrd_commitment"
    assert len(table.records) == 1 and table.records[0]["_source_row"] == 4


def test_data_row_is_not_mistaken_for_api_header():
    headers = [c.source_names[0] for c in PROJECTS.columns]
    data = ["P130544"] + ["value"] * (len(headers) - 1)
    table = read(rows(headers, data), contract=PROJECTS)
    assert table.metadata.secondary_header_row is None
    assert len(table.records) == 1


def test_missing_required_column_fails_clearly():
    with pytest.raises(SourceContractError) as exc:
        read(rows(("Project ID", "Note"), ("P130544", "x")))
    message = str(exc.value)
    assert "Amount (USD)" in message and "row 1" in message


def test_no_header_in_scan_window_fails():
    with pytest.raises(SourceContractError, match="no header row"):
        read(rows(*[("junk",)] * 20))


def test_two_columns_mapping_to_one_field_fails():
    contract = SourceContract(
        name="dup", bronze_table="b", project_id_field="pid",
        columns=(ColumnSpec("pid", ("Project", "Project ID"), True, FieldKind.IDENTIFIER),),
    )  # fmt: skip
    with pytest.raises(SourceContractError, match="both map"):
        read(rows(("Project", "Project ID"), ("P130544", "P130544")), contract=contract)


def test_project_filtering_and_canonical_id():
    table = read(
        rows(
            ("Project ID", "Amount (USD)", "Note"),
            ("P000001", "1", ""),
            (" p130544 ", "2", ""),
            ("", "3", ""),
            (None, None, None),
            ("P179039", "4", ""),
        ),
        ids=("P130544", "P179039"),
    )
    assert [r["project_id"] for r in table.records] == ["P130544", "P179039"]
    assert table.records[0]["source_project_id"] == " p130544 "  # raw preserved
    meta = table.metadata
    assert (
        meta.rows_scanned,
        meta.rows_matched,
        meta.rows_without_project_id,
        meta.empty_rows,
    ) == (
        5,
        2,
        1,
        1,
    )


def test_unmapped_columns_and_missing_optional_are_preserved():
    table = read(rows(("Project ID", "Amount (USD)", "Extra Col"), ("P130544", "1", "keep me")))
    record = table.records[0]
    assert record["_extra_fields"] == {"Extra Col": "keep me"}
    assert record["note"] is None
    assert table.metadata.missing_optional_columns == ["note"]
    assert table.metadata.unmapped_columns == ["Extra Col"]


def test_short_rows_are_padded_with_none():
    table = read(rows(("Project ID", "Amount (USD)", "Note"), ("P130544", "1")))
    assert table.records[0]["note"] is None


def test_lineage_fields_present():
    table = read(rows(("Project ID", "Amount (USD)"), ("P130544", "1")))
    record = table.records[0]
    assert record["_source_file"] == "f.csv"
    assert record["_ingested_at"] == "t" and record["_ingestion_run_id"] == "r"
    assert table.fields[:6] == [
        "_source_file", "_source_sheet", "_source_row", "_ingested_at", "_ingestion_run_id",
        "project_id",
    ]  # fmt: skip
