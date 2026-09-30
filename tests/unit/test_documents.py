from datetime import date

import pytest

from worldbank_copilot.common import load_project_registry
from worldbank_copilot.common.exceptions import ConfigurationError
from worldbank_copilot.ingestion.documents import (
    ClassificationMethod,
    DocumentManifest,
    DocumentType,
    ManifestStatus,
    build_document_inventory,
    check_isr_completeness,
    classify_filename,
    load_document_manifest,
)

KW = {"ingested_at": "t", "run_id": "r"}


# --- filename classification -----------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "doc_type", "sequence"),
    [
        ("Disclosable-Version-of-the-ISR-IN-Karnataka-P130544-Sequence-No-05.pdf", "ISR", 5),
        ("Disclosable-Version-of-the-ISR-IN-Karnataka-P130544-Sequence-No-14.pdf", "ISR", 14),
        ("Disclosable0Ve04000Sequence0No00015.pdf", "ISR", 15),
        ("Disclosable-Restructuring-Paper-IN-Karnataka-P130544.pdf", "RESTRUCTURING_PAPER", None),
        ("India-Karnataka-Urban-Water-Supply-Additional-Financing.pdf", "ADDITIONAL_FINANCING",
         None),
        ("PAD440-PAD-P130544-R2016-0052-1-Box394870B-OUO-9.pdf", "APPRAISAL_DOCUMENT", None),
        ("Program-ESSA-final.pdf", "ESSA", None),
        ("Integrated-Fiduciary-Systems-Assessment.pdf", "FIDUCIARY_ASSESSMENT", None),
        ("Technical-Assessment-2025.pdf", "TECHNICAL_ASSESSMENT", None),
        ("Loan-Agreement-9835.pdf", "LOAN_AGREEMENT", None),
        ("Notice-of-Cancellation.pdf", "CANCELLATION", None),
        ("Supplemental-Letter-PMI.pdf", "PERFORMANCE_INDICATORS", None),
    ],
)  # fmt: skip
def test_filename_rules(filename, doc_type, sequence):
    result = classify_filename(filename)
    assert result.document_type == DocumentType(doc_type)
    assert result.isr_sequence == sequence


def test_dated_isr_filename_gives_date_not_sequence():
    result = classify_filename("ISR-Disclosable-P130544-04-10-2017-1491820303083.pdf")
    assert result.document_type is DocumentType.ISR
    assert result.isr_sequence is None
    assert result.document_date == date(2017, 4, 10)
    assert result.date_basis == "filename_date"


@pytest.mark.parametrize(
    "filename",
    [
        "P130544-074e478e-72dd-4be8-ac96-57b86f9ef893.pdf",
        "P13054406d7c1a0330bb350d7693976372e.pdf",
        "BOSIB-0866500e-7445-4d25-8aba-065bea3defd6.pdf",
        "RAD696381150.pdf",
        "Padding-notes.pdf",
        "Sequence-No-3-of-something.pdf",  # 'Sequence No' without an ISR marker
    ],
)
def test_opaque_names_are_not_guessed(filename):
    assert classify_filename(filename).document_type is None


# --- ISR completeness ------------------------------------------------------------


def test_isr_complete():
    result = check_isr_completeness("P1", [3, 1, 2], expected_count=3)
    assert result.is_complete and result.missing == [] and result.duplicates == []


def test_isr_missing_sequences():
    result = check_isr_completeness("P1", [1, 2, 5], expected_count=5)
    assert result.missing == [3, 4]
    assert result.found_count == 3 and not result.is_complete


def test_isr_missing_beyond_highest_found():
    result = check_isr_completeness("P1", [1, 2], expected_count=4)
    assert result.missing == [3, 4]


def test_isr_duplicates_detected():
    result = check_isr_completeness("P1", [1, 2, 2, 3], expected_count=3)
    assert result.duplicates == [2]
    assert result.isr_documents == 4 and not result.is_complete


def test_isr_without_sequence_is_incomplete():
    result = check_isr_completeness("P1", [1, None], expected_count=1)
    assert result.isr_without_sequence == 1 and not result.is_complete


def test_isr_more_than_expected_is_flagged():
    result = check_isr_completeness("P1", [1, 2, 3], expected_count=2)
    assert result.missing == [] and not result.is_complete


def test_isr_no_expectation():
    assert check_isr_completeness("P1", [1, 2], expected_count=None).is_complete


# --- manifest + inventory ----------------------------------------------------------


def _manifest(entries):
    return DocumentManifest.model_validate({"documents": entries})


@pytest.fixture
def registry(repo_config_dir):
    return load_project_registry(repo_config_dir)


def _write(root, pid, name, content=b"abc"):
    folder = root / pid
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(content)


def _inventory(tmp_path, registry, manifest):
    return build_document_inventory(tmp_path, tmp_path, registry, manifest, **KW)


def test_manifest_classifies_opaque_file(tmp_path, registry):
    _write(tmp_path, "P179039", "P179039-hash.pdf", b"12345")
    manifest = _manifest(
        {
            "P179039": [
                {
                    "filename": "P179039-hash.pdf",
                    "size_bytes": 5,
                    "document_type": "ISR",
                    "isr_sequence": 4,
                    "document_date": "2024-03-08",
                    "date_basis": "isr_archived_date",
                    "evidence": "e",
                }
            ]
        }
    )
    record = _inventory(tmp_path, registry, manifest).table.records[0]
    assert record["document_type"] == "ISR"
    assert record["isr_sequence"] == 4
    assert record["document_date"] == "2024-03-08"
    assert record["classification_method"] == ClassificationMethod.MANIFEST
    assert record["manifest_status"] == ManifestStatus.MATCHED
    assert record["relative_path"] == "P179039/P179039-hash.pdf"
    assert record["extension"] == ".pdf" and record["file_size_bytes"] == 5
    assert record["document_id"].startswith("P179039-") and len(record["sha256"]) == 64


def test_filename_and_manifest_agree(tmp_path, registry):
    name = "ISR-Disclosable-P130544-04-10-2017-1.pdf"
    _write(tmp_path, "P130544", name)
    manifest = _manifest(
        {
            "P130544": [
                {
                    "filename": name,
                    "size_bytes": 3,
                    "document_type": "ISR",
                    "isr_sequence": 3,
                    "document_date": "2017-04-10",
                    "date_basis": "isr_archived_date",
                    "evidence": "e",
                }
            ]
        }
    )
    record = _inventory(tmp_path, registry, manifest).table.records[0]
    assert record["classification_method"] == ClassificationMethod.FILENAME_PATTERN_AND_MANIFEST
    assert (record["isr_sequence"], record["document_date"]) == (3, "2017-04-10")
    assert record["classification_conflicts"] == []


def test_conflict_is_not_silently_resolved(tmp_path, registry):
    name = "Disclosable-Version-of-the-ISR-P130544-Sequence-No-05.pdf"
    _write(tmp_path, "P130544", name)
    manifest = _manifest(
        {
            "P130544": [
                {
                    "filename": name,
                    "size_bytes": 3,
                    "document_type": "ISR",
                    "isr_sequence": 6,
                    "evidence": "e",
                }
            ]
        }
    )
    record = _inventory(tmp_path, registry, manifest).table.records[0]
    assert record["isr_sequence"] is None
    assert record["classification_conflicts"] == ["isr_sequence: filename=5 manifest=6"]


def test_changed_file_ignores_manifest(tmp_path, registry):
    _write(tmp_path, "P179039", "P179039-hash.pdf", b"different size")
    manifest = _manifest(
        {
            "P179039": [
                {
                    "filename": "P179039-hash.pdf",
                    "size_bytes": 5,
                    "document_type": "ISR",
                    "evidence": "e",
                }
            ]
        }
    )
    record = _inventory(tmp_path, registry, manifest).table.records[0]
    assert record["manifest_status"] == ManifestStatus.SIZE_MISMATCH
    assert record["document_type"] == "OTHER"
    assert record["classification_method"] == ClassificationMethod.UNCLASSIFIED


def test_manifest_loan_reference_is_normalized(tmp_path, registry):
    _write(tmp_path, "P130544", "RAD1.pdf")
    manifest = _manifest(
        {
            "P130544": [
                {
                    "filename": "RAD1.pdf",
                    "size_bytes": 3,
                    "document_type": "LOAN_AGREEMENT",
                    "loan_number_raw": "8601-IN",
                    "evidence": "e",
                }
            ]
        }
    )
    record = _inventory(tmp_path, registry, manifest).table.records[0]
    assert (record["raw_loan_number"], record["normalized_loan_number"]) == ("8601-IN", "8601-IN")


def test_missing_manifest_files_and_dirs_reported(tmp_path, registry):
    _write(tmp_path, "P130544", "x.pdf")
    manifest = _manifest(
        {
            "P130544": [
                {"filename": "gone.pdf", "size_bytes": 1, "document_type": "ISR", "evidence": "e"}
            ]
        }
    )
    inventory = _inventory(tmp_path, registry, manifest)
    assert inventory.missing_manifest_files == [{"project_id": "P130544", "filename": "gone.pdf"}]
    assert inventory.missing_project_dirs == ["P179039", "P506272"]


def test_filename_project_ids_recorded(tmp_path, registry):
    _write(tmp_path, "P130544", "P179039-misfiled.pdf")
    record = _inventory(tmp_path, registry, _manifest({})).table.records[0]
    assert record["filename_project_ids"] == ["P179039"]


def test_manifest_validation(tmp_path):
    bad = tmp_path / "m.yaml"
    bad.write_text("documents:\n  P1: []\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_document_manifest(bad)
    with pytest.raises(ConfigurationError, match="not found"):
        load_document_manifest(tmp_path / "missing.yaml")


def test_repo_manifest_is_valid_and_complete(repo_config_dir):
    manifest = load_document_manifest(repo_config_dir / "document_manifest.yaml")
    isr = {pid: sorted(e.isr_sequence for e in entries if e.document_type is DocumentType.ISR)
           for pid, entries in manifest.documents.items()}  # fmt: skip
    assert isr == {
        "P130544": list(range(1, 25)),
        "P179039": list(range(1, 9)),
        "P506272": list(range(1, 4)),
    }
    assert sum(len(v) for v in manifest.documents.values()) == 53
