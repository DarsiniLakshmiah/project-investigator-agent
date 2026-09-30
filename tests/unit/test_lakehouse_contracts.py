"""Phase 6: explicit table contracts, validation, identities and generated SQL."""

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum

import pytest
from pydantic import BaseModel

from worldbank_copilot.lakehouse.contracts import (
    LOADED_AT,
    RECORD_HASH,
    RECORD_ID,
    Column,
    ContractError,
    DType,
    Role,
    build_contract,
    model_columns,
    require_valid,
    validate_rows,
)
from worldbank_copilot.lakehouse.identity import canonical, content_hash, stable_id
from worldbank_copilot.lakehouse.records import (
    enrichment_contract,
    events_contract,
    isr_contract,
    quality_contract,
    results_contract,
    risks_contract,
    silver_contract,
)
from worldbank_copilot.lakehouse.review import candidates_contract
from worldbank_copilot.lakehouse.sql import (
    create_table_sql,
    create_volume_sql,
    snapshot_merge_sql,
)


class Colour(StrEnum):
    RED = "RED"
    BLUE = "BLUE"


class Sample(BaseModel):
    name: str
    amount: Decimal | None
    when: date | None = None
    colour: Colour
    tags: list[str]
    pages: list[int]
    nested: dict = {}


def _contract():
    return build_contract(
        "sample",
        "silver",
        "sample",
        [
            Column("key", DType.STRING, False, Role.KEY),
            Column("amount", DType.DECIMAL),
            Column("when", DType.DATE),
            Column("colour", DType.STRING, vocabulary=("RED", "BLUE")),
        ],
        ("key",),
        "test",
    )


def _row(**kw):
    row = {
        RECORD_ID: "id1",
        "key": "k1",
        "amount": Decimal("1.50"),
        "when": date(2024, 1, 2),
        "colour": "RED",
        RECORD_HASH: "h",
        "_source_snapshot_id": "s",
        "_load_run_id": "r",
        LOADED_AT: datetime(2026, 9, 30, tzinfo=UTC),
        "_pipeline_version": "v",
    }
    row.update(kw)
    return row


def test_model_columns_map_types_nullability_and_vocabulary():
    columns = {c.name: c for c in model_columns(Sample)}
    assert columns["name"].dtype is DType.STRING and not columns["name"].nullable
    assert columns["amount"].dtype is DType.DECIMAL and columns["amount"].nullable
    assert columns["amount"].sql_type == "DECIMAL(38,6)"
    assert columns["when"].dtype is DType.DATE
    assert columns["colour"].vocabulary == ("RED", "BLUE")
    assert columns["tags"].dtype is DType.ARRAY_STRING
    assert columns["pages"].dtype is DType.ARRAY_BIGINT
    assert "nested_json" in columns and columns["nested_json"].dtype is DType.STRING


def test_contract_has_standard_identity_and_operational_columns():
    contract = _contract()
    assert contract.column_names[0] == RECORD_ID
    assert contract.column_names[-5:] == [
        RECORD_HASH,
        "_source_snapshot_id",
        "_load_run_id",
        LOADED_AT,
        "_pipeline_version",
    ]
    assert LOADED_AT not in contract.hashed_columns and RECORD_HASH not in contract.hashed_columns
    with pytest.raises(ContractError, match="natural key"):
        build_contract("x", "silver", "x", [], ("missing",), "d")


def test_valid_row_passes_and_float_is_rejected():
    assert validate_rows(_contract(), [_row()]) == []
    problems = validate_rows(_contract(), [_row(amount=1.5)])
    assert any("never float" in p for p in problems)


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"amount": Decimal("1.1234567")}, "would require rounding"),
        ({"when": datetime(2024, 1, 2)}, "expected date"),
        ({LOADED_AT: datetime(2026, 9, 30)}, "timezone-aware"),
        ({"colour": "GREEN"}, "not in vocabulary"),
        ({"key": None}, "is required"),
    ],
)
def test_contract_violations_are_reported(change, fragment):
    problems = validate_rows(_contract(), [_row(**change)])
    assert any(fragment in p for p in problems), problems


def test_duplicate_protection_and_column_order():
    rows = [_row(), _row(), _row(**{RECORD_ID: "id2"})]
    problems = validate_rows(_contract(), rows)
    assert any("duplicate record_id" in p for p in problems)
    assert any("duplicate natural key" in p for p in problems)
    reordered = dict(reversed(list(_row().items())))
    assert any("columns differ" in p for p in validate_rows(_contract(), [reordered]))
    with pytest.raises(ContractError):
        require_valid(_contract(), rows)


def test_stable_id_and_content_hash_semantics():
    assert stable_id("t", "P1", 5) == stable_id("t", "P1", 5)
    assert stable_id("t", "P1", 5) != stable_id("t", "P1", 6) != stable_id("u", "P1", 5)
    contract = _contract()
    base = content_hash(contract, _row())
    # Operational metadata and Decimal representation do not change the content hash.
    assert (
        content_hash(contract, _row(_load_run_id="other", **{LOADED_AT: datetime.now(UTC)})) == base
    )
    assert content_hash(contract, _row(amount=Decimal("1.500000"))) == base
    assert content_hash(contract, _row(amount=Decimal("1.51"))) != base
    assert canonical(Decimal("100")) == canonical(Decimal("100.000000")) == "100.000000"
    assert canonical(date(2025, 5, 31)) == "2025-05-31"


def test_generated_ddl_and_merge_statements():
    contract = _contract()
    ddl = create_table_sql("worldbank_ai", "silver", contract)
    assert ddl.startswith("CREATE TABLE IF NOT EXISTS `worldbank_ai`.`silver`.`sample`")
    assert "`amount` DECIMAL(38,6)" in ddl and "`record_id` STRING NOT NULL" in ddl
    assert "USING DELTA" in ddl and "'worldbank.natural_key' = 'key'" in ddl
    merge = snapshot_merge_sql("worldbank_ai", "silver", contract, "stage")
    assert "ON t.`record_id` = s.`record_id`" in merge
    assert "WHEN MATCHED AND t.`record_hash` <> s.`record_hash` THEN UPDATE" in merge
    assert "WHEN NOT MATCHED THEN INSERT" in merge
    assert merge.endswith("WHEN NOT MATCHED BY SOURCE THEN DELETE")
    assert "IF NOT EXISTS `worldbank_ai`.`bronze`.`sources`" in create_volume_sql(
        "worldbank_ai", "bronze", "sources", "c"
    )


STATIC_CONTRACTS = {
    "silver.isr_snapshots": isr_contract,
    "silver.project_results": results_contract,
    "silver.appraisal_risks": risks_contract,
    "silver.project_events": events_contract,
    "silver.project_enrichment": enrichment_contract,
    "silver.data_quality_observations": quality_contract,
    "silver.indicator_match_candidates": candidates_contract,
    "silver.loans": lambda: silver_contract("silver_loans"),
    "silver.projects": lambda: silver_contract("silver_projects"),
    "silver.procurement_awards": lambda: silver_contract("silver_procurement_awards"),
}


@pytest.mark.parametrize("name", sorted(STATIC_CONTRACTS))
def test_contracts_match_the_committed_lock(repo_config_dir, name):
    """Schema changes must be explicit: regenerate the lock with --write-expected."""
    lock = json.loads((repo_config_dir / "delta_contracts.lock.json").read_text(encoding="utf-8"))
    assert STATIC_CONTRACTS[name]().to_dict() == lock[name]


def test_document_contracts_keep_queryable_provenance_and_decimal_precision():
    results = results_contract()
    for column in (
        "evidence_document_id",
        "evidence_page_number",
        "evidence_table_id",
        "evidence_row",
        "evidence_column",
        "evidence_text",
        "evidence_hash",
        "evidence_extraction_method",
        "status",
        "quality_issue_codes",
    ):
        assert column in results.column_names
    events = events_contract()
    assert {
        "event_date",
        "candidate_event_date",
        "candidate_date_basis",
        "candidate_date_status",
    } <= set(events.column_names)
    assert events.column("candidate_date_status").vocabulary is not None
    loans = silver_contract("silver_loans")
    assert loans.column("original_principal_usd").sql_type == "DECIMAL(38,6)"
    assert loans.column("board_approval_date").dtype is DType.DATE
    isr = isr_contract()
    assert {"header_date", "archive_date", "canonical_report_date", "isr_sequence"} <= set(
        isr.column_names
    )
