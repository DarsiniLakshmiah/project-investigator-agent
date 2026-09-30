"""Phase 6: source-hash verification, reconciliation, review artefact, environment boundaries."""

import subprocess
import sys
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from tests.support.extraction_builders import isr_doc

from worldbank_copilot.common import load_project_registry, load_settings
from worldbank_copilot.common.exceptions import (
    ReconciliationError,
    SourceIntegrityError,
)
from worldbank_copilot.common.quality import CheckCode
from worldbank_copilot.extraction.models import ResultObservation
from worldbank_copilot.extraction.provenance import (
    ExtractionMethod,
    ExtractionStatus,
    evidence,
    issue,
)
from worldbank_copilot.lakehouse.pipeline import reconcile_with_expected
from worldbank_copilot.lakehouse.reconcile import compare, profile, require_match
from worldbank_copilot.lakehouse.records import LoadContext, silver_dataset
from worldbank_copilot.lakehouse.review import (
    REVIEW_COLUMNS,
    candidate_bodies,
    candidates_dataset,
    write_review_csv,
)
from worldbank_copilot.lakehouse.sources import (
    SourceSnapshot,
    build_snapshot,
    require_integrity,
    verify_snapshot,
)
from worldbank_copilot.transformations.silver_models import SilverLoan, SourceRef

CTX = LoadContext("snap", "run", datetime(2026, 9, 30, tzinfo=UTC), "v", {})


def _files(tmp_path):
    (tmp_path / "P1").mkdir()
    (tmp_path / "loans.csv").write_bytes(b"a,b\n1,2\n")
    (tmp_path / "P1" / "isr.pdf").write_bytes(b"%PDF-1.4 synthetic")
    return build_snapshot(tmp_path, [tmp_path / "loans.csv"], [("P1", tmp_path / "P1" / "isr.pdf")])


def test_snapshot_verifies_and_is_deterministic(tmp_path):
    snapshot = _files(tmp_path)
    assert [f.relative_path for f in snapshot.files] == ["P1/isr.pdf", "loans.csv"]
    report = verify_snapshot(snapshot, tmp_path)
    assert report.ok and report.counts() == {"MATCH": 2, "MISMATCH": 0, "MISSING": 0}
    assert SourceSnapshot.from_dict(snapshot.to_dict()) == snapshot
    tampered = snapshot.to_dict()
    tampered["files"][0]["sha256"] = "0" * 64
    with pytest.raises(SourceIntegrityError, match="does not match its file list"):
        SourceSnapshot.from_dict(tampered)


def test_hash_mismatch_and_missing_files_are_hard_failures(tmp_path):
    snapshot = _files(tmp_path)
    (tmp_path / "loans.csv").write_bytes(b"a,b\n1,3\n")  # one byte changed
    (tmp_path / "P1" / "isr.pdf").unlink()
    (tmp_path / "extra.csv").write_bytes(b"x")
    report = verify_snapshot(snapshot, tmp_path)
    by_path = {c.relative_path: c for c in report.checks}
    assert by_path["loans.csv"].status == "MISMATCH"
    assert by_path["loans.csv"].expected_sha256 != by_path["loans.csv"].actual_sha256
    assert by_path["P1/isr.pdf"].status == "MISSING"
    assert report.unexpected_files == ("extra.csv",)
    with pytest.raises(SourceIntegrityError, match="2 source file"):
        require_integrity(report)


def _loan(amount="24937499.48"):
    return SilverLoan(
        project_id="P1",
        raw_loan_number="IBRD1",
        normalized_loan_number=None,
        loan_base_number=None,
        loan_suffix=None,
        lender=None,
        loan_type=None,
        loan_status=None,
        borrower=None,
        country_code=None,
        currency_of_commitment=None,
        original_principal_usd=Decimal("150000000"),
        cancelled_amount_usd=Decimal(amount),
        disbursed_amount_usd=None,
        undisbursed_amount_usd=None,
        repaid_to_ibrd_usd=None,
        due_to_ibrd_usd=None,
        exchange_adjustment_usd=None,
        borrowers_obligation_usd=None,
        loans_held_usd=None,
        principal_components_total_usd=None,
        principal_components_difference_usd=None,
        board_approval_date=date(2021, 12, 21),
        agreement_signing_date=None,
        effective_date=None,
        closing_date=None,
        last_disbursement_date=None,
        first_repayment_date=None,
        last_repayment_date=None,
        snapshot_date=date(2026, 8, 31),
        valuation_caveats=[],
        source_file="l.csv",
        source_refs=[SourceRef(bronze_table="b", source_file="l.csv", ingestion_run_id="r")],
    )


def _readback(rows):
    """What Spark returns: DECIMAL(38,6) values and naive timestamps."""
    out = []
    for row in rows:
        copy = dict(row)
        for key, value in copy.items():
            if isinstance(value, Decimal):
                copy[key] = value.quantize(Decimal("0.000001"))
        copy["_loaded_at"] = row["_loaded_at"].replace(tzinfo=None)
        out.append(copy)
    return out


def test_readback_reconciles_and_value_changes_are_detected():
    dataset = silver_dataset("silver_loans", [_loan()], CTX)
    built = profile(dataset.contract, dataset.rows)
    assert compare(built, profile(dataset.contract, _readback(dataset.rows))) == []
    altered = _readback(dataset.rows)
    altered[0]["cancelled_amount_usd"] = Decimal("24937499.00")  # e.g. precision lost
    diffs = compare(built, profile(dataset.contract, altered))
    assert any(d.startswith("fingerprint") for d in diffs)
    assert any(d.startswith("stored_hash_mismatches") for d in diffs)
    with pytest.raises(ReconciliationError):
        require_match("silver.loans", built, profile(dataset.contract, altered), "Delta")
    assert any(d.startswith("row_count") for d in compare(built, profile(dataset.contract, [])))


def test_expected_profiles_must_be_for_the_same_snapshot():
    class Build:  # minimal stand-in for PlatformBuild
        snapshot = SourceSnapshot("new-snapshot", ())

    diffs = reconcile_with_expected(Build(), {"source_snapshot_id": "old", "profiles": {}}, {})
    assert "regenerate" in diffs["*"][0]
    assert "no committed expected" in reconcile_with_expected(Build(), None, {})["*"][0]


DOC = isr_doc([], document_id="doc")
REF = evidence(DOC, 3, ExtractionMethod.DOCLING_TABLE)


def _obs(key, name, seq):
    return ResultObservation(
        project_id="P130544",
        indicator_key=key,
        indicator_name_raw=name,
        indicator_name_normalized=name.lower(),
        unit="Number",
        isr_sequence=seq,
        source_document="d.pdf",
        source_page=3,
        layout="WIDE",
        extraction_method=ExtractionMethod.DOCLING_TABLE,
        status=ExtractionStatus.EXACT,
        source_ref=REF,
        identity_basis="EXACT_NAME",
        indicator_type="PDO",
    )


def test_indicator_candidates_are_pending_review_and_never_merge(tmp_path):
    observations = [
        _obs("k1", "Direct beneficiaries (Number, Core)", 1),
        _obs("k1", "Direct beneficiaries (Number, Core)", 3),
        _obs("k2", "Direct beneficiaries (Number)", 20),
    ]
    pim = issue(
        CheckCode.POSSIBLE_INDICATOR_MATCH,
        "INFO",
        "possible",
        REF,
        indicator_keys=["k2", "k1"],
        reason="same name apart from unit",
    )
    bodies = candidate_bodies([pim, pim], observations, {})
    assert len(bodies) == 1
    body = bodies[0]
    assert (body["indicator_key_a"], body["indicator_key_b"]) == ("k1", "k2")
    assert (body["first_isr_sequence_a"], body["last_isr_sequence_a"]) == (1, 3)
    assert body["isr_sequences_b"] == [20] and body["review_status"] == "PENDING_REVIEW"
    assert body["source_refs_a"] == [REF.label, REF.label]
    # Identity stays separate: the observations keep their own keys.
    assert {o.indicator_key for o in observations} == {"k1", "k2"}
    approved = candidate_bodies(
        [pim],
        observations,
        {"P130544": {"direct beneficiaries (number, core)": "direct beneficiaries (number)"}},
    )
    assert approved[0]["review_status"] == "APPROVED_ALIAS"
    dataset = candidates_dataset([pim], observations, {}, CTX)
    path = write_review_csv(dataset, tmp_path / "review.csv")
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert tuple(header) == REVIEW_COLUMNS
    assert "PENDING_REVIEW" in path.read_text(encoding="utf-8")


def test_core_modules_import_without_spark_and_local_is_a_dry_run():
    code = (
        "import sys; import worldbank_copilot.lakehouse.pipeline, "
        "worldbank_copilot.lakehouse.spark_store; "
        "print('pyspark' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_databricks_settings_use_configured_unity_catalog_names(repo_config_dir):
    settings = load_settings("databricks", config_dir=repo_config_dir, env={})
    assert settings.table_name("silver", "isr_snapshots") == "worldbank_ai.silver.isr_snapshots"
    assert settings.table_name("bronze", "loans_raw") == "worldbank_ai.bronze.loans_raw"
    assert settings.source_volume_path == "/Volumes/worldbank_ai/bronze/sources"
    assert settings.artifact_volume_path == "/Volumes/worldbank_ai/silver/pipeline_artifacts"
    assert load_project_registry(repo_config_dir).project_ids == ["P130544", "P179039", "P506272"]
