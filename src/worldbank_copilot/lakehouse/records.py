"""Persisted row builders: Phase 2-5 outputs -> contract-shaped Delta rows.

No business logic is re-implemented here: rows are built from the validated
Bronze tables, structured Silver rows and Phase 5 extraction models exactly as
they are, with three additions:

* identity: ``record_id`` from the natural key, ``record_hash`` from the content;
* queryable provenance: the primary ``EvidenceRef`` of document-derived records is
  flattened into ``evidence_*`` columns (document, page, section, table, row, column,
  text, extraction method, source hash). The full nested structures are kept as
  canonical JSON next to them, so nothing is lost;
* operational load metadata (``_load_run_id``, ``_loaded_at``, ...), which never
  enters the content hash.

Delta table names drop the repository's layer prefix, because the Unity Catalog
schema is the layer: ``silver_loans`` -> ``silver.loans``, ``bronze_loans_raw`` ->
``bronze.loans_raw``. Phase 5 dataset names are used unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

from worldbank_copilot.common.quality import DataQualityReport, Severity
from worldbank_copilot.extraction.events import EVENT_TYPES
from worldbank_copilot.extraction.models import (
    AppraisalRisk,
    ExtractedRating,
    IsrSnapshot,
    LoanDisbursement,
    LoanKeyDates,
    ProjectEnrichment,
    ProjectEvent,
    ResultObservation,
    SortRating,
)
from worldbank_copilot.extraction.provenance import (
    EvidenceRef,
    ExtractionIssue,
    ExtractionMethod,
    ExtractionStatus,
)
from worldbank_copilot.lakehouse.contracts import (
    JSON_SUFFIX,
    LOAD_RUN_ID,
    LOADED_AT,
    PIPELINE_VERSION,
    RECORD_HASH,
    RECORD_ID,
    SOURCE_SNAPSHOT_ID,
    Column,
    DType,
    Role,
    TableContract,
    build_contract,
    model_columns,
)
from worldbank_copilot.lakehouse.identity import canonical_json, content_hash, stable_id
from worldbank_copilot.transformations.bronze import LINEAGE_FIELDS, BronzeTable
from worldbank_copilot.transformations.silver_models import (
    SilverLoan,
    SilverProcurementAward,
    SilverProcurementCoverage,
    SilverProcurementSupplier,
    SilverProject,
    SilverProjectFinancialSummary,
    SilverProjectSector,
    SilverProjectTheme,
    SilverRow,
)
from worldbank_copilot.transformations.silver_quality import ISSUE_CHECK


@dataclass(frozen=True)
class LoadContext:
    """Operational metadata of one load (never part of a record's content hash)."""

    source_snapshot_id: str
    load_run_id: str
    loaded_at: datetime
    pipeline_version: str
    source_hashes: dict[str, str]  # relative source path -> sha256


@dataclass
class Dataset:
    contract: TableContract
    rows: list[dict[str, Any]]


def finalize(
    contract: TableContract, bodies: Iterable[dict[str, Any]], ctx: LoadContext
) -> Dataset:
    """Add identity, content hash and load metadata; order columns by the contract."""
    rows = []
    body_names = [
        n
        for n in contract.column_names
        if n
        not in {
            RECORD_ID,
            RECORD_HASH,
            SOURCE_SNAPSHOT_ID,
            LOAD_RUN_ID,
            LOADED_AT,
            PIPELINE_VERSION,
        }
    ]
    for body in bodies:
        extra = sorted(set(body) - set(body_names))
        if extra:
            raise KeyError(f"{contract.name}: values for undeclared columns {extra}")
        row: dict[str, Any] = {
            RECORD_ID: stable_id(contract.name, *[body.get(k) for k in contract.natural_key])
        }
        for name in body_names:
            row[name] = body.get(name)
        row[RECORD_HASH] = content_hash(contract, row)
        row[SOURCE_SNAPSHOT_ID] = ctx.source_snapshot_id
        row[LOAD_RUN_ID] = ctx.load_run_id
        row[LOADED_AT] = ctx.loaded_at
        row[PIPELINE_VERSION] = ctx.pipeline_version
        rows.append(row)
    return Dataset(contract, rows)


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def model_values(instance: BaseModel, columns: Iterable[Column]) -> dict[str, Any]:
    """Values for model-derived columns (``<field>_json`` for nested structures)."""
    fields = type(instance).model_fields
    out: dict[str, Any] = {}
    json_dump: dict[str, Any] | None = None
    for column in columns:
        name = column.name
        if name in fields:
            out[name] = _plain(getattr(instance, name))
        elif name.endswith(JSON_SUFFIX) and name[: -len(JSON_SUFFIX)] in fields:
            if json_dump is None:
                json_dump = instance.model_dump(mode="json")
            value = json_dump[name[: -len(JSON_SUFFIX)]]
            out[name] = None if value is None else canonical_json(value)
    return out


def _vocab(columns: list[Column], vocabularies: dict[str, Iterable[str]]) -> list[Column]:
    out = []
    for c in columns:
        if c.name in vocabularies:
            c = Column(
                c.name, c.dtype, c.nullable, c.role, tuple(vocabularies[c.name]), c.description
            )
        out.append(c)
    return out


def _roles(columns: list[Column], roles: dict[str, Role]) -> list[Column]:
    return [
        Column(c.name, c.dtype, c.nullable, roles.get(c.name, c.role), c.vocabulary, c.description)
        for c in columns
    ]


# ---------------------------------------------------------------------------
# Bronze
# ---------------------------------------------------------------------------

# Every Bronze value is source text except these (explicit, from the Phase 2 builders).
BRONZE_TYPES: dict[str, DType] = {
    "_source_row": DType.BIGINT,
    "_extra_fields": DType.MAP_STRING,
    "file_size_bytes": DType.BIGINT,
    "isr_sequence": DType.BIGINT,
    "filename_isr_sequence": DType.BIGINT,
    "classification_conflicts": DType.ARRAY_STRING,
    "filename_project_ids": DType.ARRAY_STRING,
    "covered_by_dataset": DType.BOOLEAN,
    "row_count": DType.BIGINT,
}
BRONZE_NATURAL_KEYS = {
    "bronze_document_inventory": ("relative_path",),
    "bronze_procurement_coverage": ("project_id",),
}
BRONZE_OPERATIONAL = {"_ingested_at", "_ingestion_run_id"}
SOURCE_SHA_COLUMN = "_source_sha256"


def delta_name(dataset: str, layer: str) -> str:
    prefix = f"{layer}_"
    return dataset[len(prefix) :] if dataset.startswith(prefix) else dataset


def bronze_contract(table: BronzeTable) -> TableContract:
    columns = []
    for name in table.fields:
        role = Role.DATA
        if name in LINEAGE_FIELDS:
            role = Role.OPERATIONAL if name in BRONZE_OPERATIONAL else Role.PROVENANCE
        nullable = name not in {"_source_file", "project_id"}
        columns.append(Column(name, BRONZE_TYPES.get(name, DType.STRING), nullable, role))
    columns.append(
        Column(
            SOURCE_SHA_COLUMN,
            DType.STRING,
            False,
            Role.PROVENANCE,
            description="SHA-256 of the source file (validated snapshot).",
        )
    )
    key = BRONZE_NATURAL_KEYS.get(table.name, ("_source_file", "_source_sheet", "_source_row"))
    return build_contract(
        delta_name(table.name, "bronze"),
        "bronze",
        table.name,
        columns,
        key,
        f"Source-aligned Bronze records from {table.metadata.source_name}.",
        profile_groups=[("project_id",)],
    )


def bronze_dataset(table: BronzeTable, ctx: LoadContext) -> Dataset:
    contract = bronze_contract(table)
    bodies = []
    for record in table.records:
        body = {name: record.get(name) for name in table.fields}
        source = (
            record.get("relative_path")
            if table.name == "bronze_document_inventory"
            else (record["_source_file"])
        )
        if source not in ctx.source_hashes:
            raise KeyError(f"{table.name}: source {source!r} is not in the source snapshot")
        body[SOURCE_SHA_COLUMN] = ctx.source_hashes[source]
        bodies.append(body)
    return finalize(contract, bodies, ctx)


# ---------------------------------------------------------------------------
# Structured Silver (Phase 3)
# ---------------------------------------------------------------------------

SILVER_STRUCTURED: dict[
    str, tuple[type[SilverRow], tuple[str, ...], tuple[tuple[str, ...], ...]]
] = {
    "silver_projects": (SilverProject, ("project_id",), ()),
    "silver_project_sectors": (
        SilverProjectSector,
        ("project_id", "major_sector", "sector"),
        (("project_id",),),
    ),
    "silver_project_themes": (
        SilverProjectTheme,
        ("project_id", "level", "theme_path"),
        (("project_id", "level"),),
    ),
    "silver_loans": (SilverLoan, ("raw_loan_number",), (("project_id",),)),
    "silver_project_financial_summary": (
        SilverProjectFinancialSummary,
        ("project_id", "snapshot_date"),
        (),
    ),
    "silver_procurement_awards": (
        SilverProcurementAward,
        ("project_id", "contract_id"),
        (("project_id",), ("amount_basis",)),
    ),
    "silver_procurement_suppliers": (
        SilverProcurementSupplier,
        ("project_id", "contract_id", "supplier_name", "supplier_id", "supplier_row_amount_usd"),
        (("project_id",),),
    ),
    "silver_procurement_coverage": (
        SilverProcurementCoverage,
        ("project_id", "dataset"),
        (("coverage_status",),),
    ),
}

_SILVER_PROVENANCE = (
    Column(
        "source_record_keys",
        DType.ARRAY_STRING,
        False,
        Role.PROVENANCE,
        description="Bronze records the row was built from (table:file:sheet:row).",
    ),
)
_QUALITY_CODES = Column("quality_issue_codes", DType.ARRAY_STRING, False, Role.QUALITY)


def silver_contract(dataset: str) -> TableContract:
    model, key, groups = SILVER_STRUCTURED[dataset]
    columns = model_columns(
        model, roles={"source_refs": Role.PROVENANCE, "quality_issues": Role.QUALITY}
    )
    return build_contract(
        delta_name(dataset, "silver"),
        "silver",
        dataset,
        [*columns, *_SILVER_PROVENANCE, _QUALITY_CODES],
        key,
        (model.__doc__ or dataset).strip().splitlines()[0],
        profile_groups=groups,
    )


def _bronze_key(ref: Any) -> str:
    return ":".join(
        str(p)
        for p in (
            ref.bronze_table,
            ref.source_file,
            ref.source_sheet or "",
            ref.source_row if ref.source_row is not None else "",
        )
    )


def silver_dataset(dataset: str, rows: list[SilverRow], ctx: LoadContext) -> Dataset:
    contract = silver_contract(dataset)
    bodies = []
    for row in rows:
        body = model_values(row, contract.columns)
        body["source_record_keys"] = [_bronze_key(r) for r in row.source_refs]
        # Same codes the Phase 3 Silver quality report uses for these issues.
        body["quality_issue_codes"] = sorted(
            {ISSUE_CHECK[i.kind].value for i in row.quality_issues}
        )
        bodies.append(body)
    return finalize(contract, bodies, ctx)


# ---------------------------------------------------------------------------
# Document-derived Silver (Phase 5)
# ---------------------------------------------------------------------------

_METHODS = tuple(m.value for m in ExtractionMethod)
_STATUSES = tuple(s.value for s in ExtractionStatus)

EVIDENCE_FIELDS: tuple[tuple[str, str, DType], ...] = (
    ("document_id", "document_id", DType.STRING),
    ("document_type", "document_type", DType.STRING),
    ("document_date", "document_date", DType.DATE),
    ("filename", "filename", DType.STRING),
    ("hash", "source_hash", DType.STRING),
    ("page_number", "page_number", DType.BIGINT),
    ("section", "section", DType.STRING),
    ("table_id", "table_id", DType.STRING),
    ("row", "row", DType.BIGINT),
    ("column", "column", DType.BIGINT),
    ("block_id", "block_id", DType.STRING),
    ("text", "source_text", DType.STRING),
    ("extraction_method", "extraction_method", DType.STRING),
    ("label", "label", DType.STRING),
)


def evidence_columns(prefix: str = "evidence", *, required: bool = True) -> list[Column]:
    out = []
    for suffix, attr, dtype in EVIDENCE_FIELDS:
        nullable = not (
            required
            and attr
            in {"document_id", "filename", "source_hash", "page_number", "extraction_method"}
        )
        vocab = _METHODS if attr == "extraction_method" else None
        out.append(Column(f"{prefix}_{suffix}", dtype, nullable, Role.PROVENANCE, vocab))
    return out


def evidence_values(ref: EvidenceRef | None, prefix: str = "evidence") -> dict[str, Any]:
    return {
        f"{prefix}_{suffix}": (None if ref is None else _plain(getattr(ref, attr)))
        for suffix, attr, _ in EVIDENCE_FIELDS
    }


def _rating_columns(name: str) -> list[Column]:
    return [
        Column(f"{name}_raw", DType.STRING, True, Role.DATA),
        Column(name, DType.STRING, True, Role.DATA),
        Column(f"{name}_status", DType.STRING, True, Role.QUALITY, _STATUSES),
        Column(f"{name}_page_number", DType.BIGINT, True, Role.PROVENANCE),
        Column(f"{name}_table_id", DType.STRING, True, Role.PROVENANCE),
        Column(f"{name}_extraction_method", DType.STRING, True, Role.PROVENANCE, _METHODS),
    ]


def _rating_values(name: str, rating: ExtractedRating | None) -> dict[str, Any]:
    ev = rating.evidence if rating else None
    return {
        f"{name}_raw": rating.raw_rating if rating else None,
        name: rating.normalized_rating if rating else None,
        f"{name}_status": rating.status.value if rating else None,
        f"{name}_page_number": ev.page_number if ev else None,
        f"{name}_table_id": ev.table_id if ev else None,
        f"{name}_extraction_method": ev.extraction_method.value if ev else None,
    }


def _issue_codes(issues: list[ExtractionIssue]) -> list[str]:
    return sorted({str(i.code) for i in issues})


ISR_RATINGS = (
    "pdo_rating",
    "implementation_progress_rating",
    "overall_risk_rating",
    "previous_pdo_rating",
    "previous_implementation_rating",
    "previous_overall_risk_rating",
)


def isr_contract() -> TableContract:
    columns = model_columns(
        IsrSnapshot,
        exclude=ISR_RATINGS,
        roles={
            "source_refs": Role.PROVENANCE,
            "restructuring_history": Role.PROVENANCE,
            "source_document": Role.PROVENANCE,
            "source_pages": Role.PROVENANCE,
            "quality_issues": Role.QUALITY,
        },
    )
    ratings = [c for r in ISR_RATINGS for c in _rating_columns(r)]
    extra = [
        Column("source_hash", DType.STRING, False, Role.PROVENANCE),
        Column("document_type", DType.STRING, False, Role.PROVENANCE),
        Column("sort_category_count", DType.BIGINT, False),
        Column("loan_count", DType.BIGINT, False),
        _QUALITY_CODES,
    ]
    columns = _vocab(columns, {"canonical_date_basis": ("header_date", "archive_date")})
    return build_contract(
        "isr_snapshots",
        "silver",
        "isr_snapshots",
        [*columns, *ratings, *extra],
        ("project_id", "isr_sequence", "document_id"),
        "One row per ISR: ratings, dates (header/archive/canonical), loans, narratives.",
        profile_groups=[
            ("project_id",),
            ("canonical_date_basis",),
            ("pdo_rating",),
            ("pdo_rating_extraction_method",),
        ],
    )


def isr_dataset(
    snapshots: list[IsrSnapshot], documents: dict[str, Any], ctx: LoadContext
) -> Dataset:
    contract = isr_contract()
    bodies = []
    for snap in snapshots:
        doc = documents[snap.document_id]
        body = model_values(snap, contract.columns)
        for name in ISR_RATINGS:
            body.update(_rating_values(name, getattr(snap, name)))
        body.update(
            source_hash=doc.source_hash,
            document_type=doc.document_type,
            sort_category_count=len(snap.sort_ratings),
            loan_count=len(snap.loan_disbursements),
            quality_issue_codes=_issue_codes(snap.quality_issues),
        )
        bodies.append(body)
    return finalize(contract, bodies, ctx)


def _isr_child_contract(
    name: str,
    model: type[BaseModel],
    key: tuple[str, ...],
    exclude: tuple[str, ...],
    extra: list[Column],
    description: str,
) -> TableContract:
    parent = [
        Column(
            "isr_record_id",
            DType.STRING,
            False,
            Role.KEY,
            description="record_id of the parent silver.isr_snapshots row.",
        ),
        Column("project_id", DType.STRING, False),
        Column("document_id", DType.STRING, False, Role.PROVENANCE),
        Column("isr_sequence", DType.BIGINT, True),
        Column("canonical_report_date", DType.DATE, True),
    ]
    columns = model_columns(model, exclude=exclude)
    return build_contract(
        name,
        "silver",
        f"isr_snapshots.{name[4:]}",
        [*parent, *columns, *extra, *evidence_columns()],
        key,
        description,
        profile_groups=[("project_id",)],
    )


def isr_child_datasets(snapshots: list[IsrSnapshot], ctx: LoadContext) -> list[Dataset]:
    isr = isr_contract()
    sort_c = _isr_child_contract(
        "isr_sort_ratings",
        SortRating,
        ("isr_record_id", "risk_category"),
        ("rating_at_approval", "previous_rating", "current_rating"),
        [
            c
            for r in ("rating_at_approval", "previous_rating", "current_rating")
            for c in _rating_columns(r)
        ],
        "SORT rows printed in each ISR (rating at approval / previous / current).",
    )
    disb_c = _isr_child_contract(
        "isr_loan_disbursements",
        LoanDisbursement,
        ("isr_record_id", "loan_number"),
        ("evidence",),
        [],
        "Per-loan financial line printed in each ISR (US$ millions).",
    )
    dates_c = _isr_child_contract(
        "isr_loan_key_dates",
        LoanKeyDates,
        ("isr_record_id", "loan_number"),
        ("evidence",),
        [],
        "Per-loan key dates printed in each ISR.",
    )
    sort_rows, disb_rows, date_rows = [], [], []
    for snap in snapshots:
        parent = {
            "isr_record_id": stable_id(
                isr.name, snap.project_id, snap.isr_sequence, snap.document_id
            ),
            "project_id": snap.project_id,
            "document_id": snap.document_id,
            "isr_sequence": snap.isr_sequence,
            "canonical_report_date": snap.canonical_report_date,
        }
        for item in snap.sort_ratings:
            body = {**parent, **model_values(item, sort_c.columns)}
            for r in ("rating_at_approval", "previous_rating", "current_rating"):
                body.update(_rating_values(r, getattr(item, r)))
            ref = next(
                (
                    getattr(item, r).evidence
                    for r in ("current_rating", "previous_rating", "rating_at_approval")
                    if getattr(item, r)
                ),
                None,
            )
            sort_rows.append({**body, **evidence_values(ref)})
        for item in snap.loan_disbursements:
            disb_rows.append(
                {**parent, **model_values(item, disb_c.columns), **evidence_values(item.evidence)}
            )
        for item in snap.loan_key_dates:
            date_rows.append(
                {**parent, **model_values(item, dates_c.columns), **evidence_values(item.evidence)}
            )
    # SORT rows with no rating at all have no evidence; relax the requirement for them.
    sort_c = _relax_evidence(sort_c)
    return [
        finalize(sort_c, sort_rows, ctx),
        finalize(disb_c, disb_rows, ctx),
        finalize(dates_c, date_rows, ctx),
    ]


def _relax_evidence(contract: TableContract) -> TableContract:
    columns = tuple(
        Column(
            c.name,
            c.dtype,
            True if c.name.startswith("evidence_") else c.nullable,
            c.role,
            c.vocabulary,
            c.description,
        )
        for c in contract.columns
    )
    return TableContract(
        contract.name,
        contract.layer,
        contract.source_dataset,
        columns,
        contract.natural_key,
        contract.description,
        contract.write_mode,
        contract.profile_groups,
    )


def results_contract() -> TableContract:
    columns = model_columns(
        ResultObservation,
        exclude=("source_ref",),
        roles={
            "source_document": Role.PROVENANCE,
            "source_page": Role.PROVENANCE,
            "source_table": Role.PROVENANCE,
            "name_source_ref": Role.PROVENANCE,
            "extraction_method": Role.PROVENANCE,
            "status": Role.QUALITY,
            "quality_issues": Role.QUALITY,
        },
    )
    columns = _vocab(
        columns,
        {
            "layout": ("BLOCK", "WIDE", "DLI"),
            "indicator_type": ("PDO", "INTERMEDIATE", "DLI", "UNKNOWN"),
            "identity_basis": ("EXACT_NAME", "SOURCE_INDICATOR_ID", "ALIAS"),
        },
    )
    return build_contract(
        "project_results",
        "silver",
        "project_results",
        [
            *columns,
            Column("document_id", DType.STRING, False, Role.PROVENANCE),
            *evidence_columns(),
            _QUALITY_CODES,
        ],
        ("project_id", "indicator_key", "document_id"),
        "Results-framework observations: project x indicator x ISR.",
        profile_groups=[
            ("project_id",),
            ("extraction_method",),
            ("status",),
            ("layout",),
            ("identity_basis",),
        ],
    )


def results_dataset(observations: list[ResultObservation], ctx: LoadContext) -> Dataset:
    contract = results_contract()
    bodies = []
    for obs in observations:
        body = model_values(obs, contract.columns)
        body.update(
            document_id=obs.source_ref.document_id,
            **evidence_values(obs.source_ref),
            quality_issue_codes=_issue_codes(obs.quality_issues),
        )
        bodies.append(body)
    return finalize(contract, bodies, ctx)


def risks_contract() -> TableContract:
    columns = model_columns(
        AppraisalRisk,
        exclude=("risk_rating",),
        roles={
            "source_document": Role.PROVENANCE,
            "source_document_type": Role.PROVENANCE,
            "source_page": Role.PROVENANCE,
            "source_section": Role.PROVENANCE,
            "source_text": Role.PROVENANCE,
            "extraction_method": Role.PROVENANCE,
            "source_refs": Role.PROVENANCE,
            "status": Role.QUALITY,
            "quality_issues": Role.QUALITY,
        },
    )
    columns = _vocab(columns, {"framing": ("FORMAL_RISK_RATING", "ASSESSMENT_FINDING")})
    rating = [
        c
        for c in _rating_columns("risk_rating")
        if not c.name.endswith(("_page_number", "_table_id", "_extraction_method"))
    ]
    return build_contract(
        "appraisal_risks",
        "silver",
        "appraisal_risks",
        [*columns, *rating, *evidence_columns(), _QUALITY_CODES],
        ("risk_id",),
        "Appraisal-stage risks: formal SORT ratings and assessment findings.",
        profile_groups=[
            ("project_id",),
            ("framing",),
            ("project_id", "framing"),
            ("source_document_type",),
            ("status",),
            ("extraction_method",),
        ],
    )


def risks_dataset(risks: list[AppraisalRisk], ctx: LoadContext) -> Dataset:
    contract = risks_contract()
    bodies = []
    for risk in risks:
        body = model_values(risk, contract.columns)
        rating = _rating_values("risk_rating", risk.risk_rating)
        body.update({k: v for k, v in rating.items() if k in contract.column_names})
        body.update(
            evidence_values(risk.source_refs[0] if risk.source_refs else None),
            quality_issue_codes=_issue_codes(risk.quality_issues),
        )
        bodies.append(body)
    return finalize(contract, bodies, ctx)


def events_contract() -> TableContract:
    columns = model_columns(
        ProjectEvent,
        roles={
            "source_document": Role.PROVENANCE,
            "source_page": Role.PROVENANCE,
            "source_section": Role.PROVENANCE,
            "source_text": Role.PROVENANCE,
            "extraction_method": Role.PROVENANCE,
            "source_refs": Role.PROVENANCE,
            "status": Role.QUALITY,
            "quality_issues": Role.QUALITY,
        },
    )
    columns = _vocab(columns, {"event_type": EVENT_TYPES})
    return build_contract(
        "project_events",
        "silver",
        "project_events",
        [*columns, *evidence_columns(), _QUALITY_CODES],
        ("event_id",),
        "Formal project events. event_date is only ever a source-stated date; "
        "candidate_event_date is derived (see candidate_date_status).",
        profile_groups=[
            ("project_id",),
            ("project_id", "event_type"),
            ("status",),
            ("candidate_date_status",),
        ],
    )


def events_dataset(events: list[ProjectEvent], ctx: LoadContext) -> Dataset:
    contract = events_contract()
    bodies = []
    for event in events:
        body = model_values(event, contract.columns)
        body.update(
            evidence_values(event.source_refs[0] if event.source_refs else None),
            quality_issue_codes=_issue_codes(event.quality_issues),
        )
        bodies.append(body)
    return finalize(contract, bodies, ctx)


def enrichment_contract() -> TableContract:
    columns = model_columns(
        ProjectEnrichment, roles={"evidence": Role.PROVENANCE, "status": Role.QUALITY}
    )
    return build_contract(
        "project_enrichment",
        "silver",
        "project_enrichment",
        [*columns, *evidence_columns(required=False)],
        ("project_id", "target_table", "target_field", "loan_number"),
        "Reviewable document-derived enrichment (original closing dates). Not applied to "
        "silver.projects.",
        profile_groups=[("target_table", "status")],
    )


def enrichment_dataset(records: list[ProjectEnrichment], ctx: LoadContext) -> Dataset:
    contract = enrichment_contract()
    bodies = [
        {
            **model_values(r, contract.columns),
            **evidence_values(r.evidence[0] if r.evidence else None),
        }
        for r in records
    ]
    return finalize(contract, bodies, ctx)


# ---------------------------------------------------------------------------
# Data-quality observations
# ---------------------------------------------------------------------------

QUALITY_LAYERS = ("bronze", "silver", "parsing", "silver_documents")


def quality_contract() -> TableContract:
    body = [
        Column("observation_fingerprint", DType.STRING, False, Role.KEY),
        Column("layer", DType.STRING, False, vocabulary=QUALITY_LAYERS),
        Column("check_code", DType.STRING, False),
        Column("severity", DType.STRING, False, vocabulary=tuple(s.value for s in Severity)),
        Column("project_id", DType.STRING, True),
        Column("source", DType.STRING, False, Role.PROVENANCE),
        Column("message", DType.STRING, False),
        Column("explanation", DType.STRING, True),
        Column(
            "related_table",
            DType.STRING,
            True,
            Role.PROVENANCE,
            description="Table of the record this observation is attached to (if any).",
        ),
        Column("related_record_id", DType.STRING, True, Role.PROVENANCE),
        Column("document_id", DType.STRING, True, Role.PROVENANCE),
        Column("page_number", DType.BIGINT, True, Role.PROVENANCE),
        Column("table_id", DType.STRING, True, Role.PROVENANCE),
        Column("extraction_method", DType.STRING, True, Role.PROVENANCE, _METHODS),
        Column("details_json", DType.STRING, False),
        Column("occurrence_count", DType.BIGINT, False),
    ]
    return build_contract(
        "data_quality_observations",
        "silver",
        "data_quality_observations",
        body,
        ("observation_fingerprint",),
        "Every known quality observation (Bronze, Silver, parsing, document extraction), "
        "optionally linked to the record it concerns.",
        profile_groups=[("layer", "severity"), ("check_code",), ("related_table",)],
    )


@dataclass(frozen=True)
class QualityInput:
    layer: str
    check_code: str
    severity: str
    project_id: str | None
    source: str
    message: str
    explanation: str | None = None
    related_table: str | None = None
    related_record_id: str | None = None
    evidence: EvidenceRef | None = None
    details: dict | None = None


def issue_input(
    layer: str,
    issue: ExtractionIssue,
    related_table: str | None = None,
    related_record_id: str | None = None,
    project_id: str | None = None,
) -> QualityInput:
    details = dict(issue.details)
    project = (
        details.pop("project_id", None)
        or project_id
        or (issue.evidence.project_id if issue.evidence else None)
    )
    return QualityInput(
        layer,
        str(issue.code),
        issue.severity,
        project,
        issue.evidence.label if issue.evidence else layer,
        issue.message,
        None,
        related_table,
        related_record_id,
        issue.evidence,
        details,
    )


def report_inputs(layer: str, report: DataQualityReport) -> list[QualityInput]:
    out = []
    for o in report.observations:
        details = dict(o.details)
        evidence = None
        raw = details.pop("evidence", None)
        if isinstance(raw, dict):
            try:
                evidence = EvidenceRef.model_validate(raw)
            except ValueError:
                details["evidence"] = raw
        out.append(
            QualityInput(
                layer,
                o.check.value,
                o.severity.value,
                o.project_id,
                o.source,
                o.message,
                o.explanation,
                None,
                None,
                evidence,
                details,
            )
        )
    return out


def quality_dataset(inputs: list[QualityInput], ctx: LoadContext) -> Dataset:
    """Deduplicated observations; record-linked ones win over identical report copies."""
    contract = quality_contract()
    linked = {
        (i.check_code, i.message, i.evidence.label if i.evidence else i.source)
        for i in inputs
        if i.related_record_id
    }
    grouped: dict[str, dict[str, Any]] = {}
    for item in inputs:
        key3 = (
            item.check_code,
            item.message,
            item.evidence.label if item.evidence else item.source,
        )
        if item.related_record_id is None and key3 in linked:
            continue
        ev = item.evidence
        details_json = canonical_json(item.details or {})
        fingerprint = stable_id(
            "dq",
            item.layer,
            item.check_code,
            item.severity,
            item.project_id,
            item.source,
            item.message,
            item.related_table,
            item.related_record_id,
            ev.model_dump(mode="json") if ev else None,
            details_json,
        )
        if fingerprint in grouped:
            grouped[fingerprint]["occurrence_count"] += 1
            continue
        grouped[fingerprint] = {
            "observation_fingerprint": fingerprint,
            "layer": item.layer,
            "check_code": item.check_code,
            "severity": item.severity,
            "project_id": item.project_id,
            "source": item.source,
            "message": item.message,
            "explanation": item.explanation,
            "related_table": item.related_table,
            "related_record_id": item.related_record_id,
            "document_id": ev.document_id if ev else None,
            "page_number": ev.page_number if ev else None,
            "table_id": ev.table_id if ev else None,
            "extraction_method": ev.extraction_method.value if ev else None,
            "details_json": details_json,
            "occurrence_count": 1,
        }
    return finalize(contract, grouped.values(), ctx)


def record_quality_inputs(
    datasets: dict[str, Dataset], sources: dict[str, list[BaseModel]]
) -> list[QualityInput]:
    """Phase 5 issues attached to individual records, linked by record_id.

    (Phase 3 row issues are already reported by the Silver quality report.)
    """
    out: list[QualityInput] = []
    for table, models in sources.items():
        dataset = datasets[table]
        for model, row in zip(models, dataset.rows, strict=True):
            for issue in getattr(model, "quality_issues", []):
                out.append(
                    issue_input(
                        "silver_documents",
                        issue,
                        table,
                        row[RECORD_ID],
                        project_id=row.get("project_id"),
                    )
                )
    return out
