# World Bank Project Implementation Intelligence Copilot

Evidence-grounded decision support for World Bank project officers and analysts.

> **Status:** Phases 1–4 of 17 are complete: repository foundation, configuration,
> source-preserving **Bronze ingestion**, typed **Silver** structured tables (projects,
> loans, financial summary, procurement) with provenance, and **Docling parsing** of the
> 53 project PDFs into page-cited parsed documents with validated metadata. Document
> extraction, Gold, retrieval, agents, API and UI do not exist yet. See
> [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the plan and current status.

## The problem

A project officer monitoring an operation has to piece together its story from
scattered sources: portfolio data, loan snapshots, procurement records, and
dozens of PDFs (appraisal documents, a long series of Implementation Status &
Results Reports, restructuring papers, additional-financing papers). The
questions that matter are:

1. What currently deserves attention in this project?
2. What changed during implementation?
3. Why does it deserve attention?
4. What structured data or document evidence supports that?
5. Were risks identified at appraisal later observed during implementation?

Answering them today means reading dozens of documents by hand and cross-checking
numbers across datasets.

## Product boundary

**This is not a project-failure prediction system.** It never produces
statements like "P130544 has a 75% probability of failure."

Instead it surfaces **transparent, rule-based implementation attention signals**
and investigates them with evidence, for example:

> *Financial execution warrants review. The project has undergone additional
> financing, restructuring and partial cancellation. The following structured
> records and project documents support the finding…*

* Every material factual claim is traceable to structured data or a source
  document (document, date, section, page).
* Every statement carries a provenance label: `FACT`, `DOCUMENTED_FINDING`,
  `SYSTEM_DERIVED_SIGNAL`, `AI_INTERPRETATION` or `UNKNOWN`.
* "Insufficient evidence" is a valid answer; results are never fabricated.
* All analytical tools are read-only. The model never runs arbitrary SQL.

## Prototype scope: three projects

V1 covers exactly three Karnataka water-sector operations
([configs/projects.yaml](configs/projects.yaml)):

| Project | Name | Instrument | Role | Analytical question |
|---|---|---|---|---|
| P130544 | Karnataka Urban Water Supply Modernization Project | Investment Project Financing | Mature, historical validation | What implementation signals appeared before restructuring, additional financing and cancellation? |
| P179039 | Karnataka Sustainable Rural Water Supply Program | Program-for-Results | Mid-lifecycle monitoring | Which appraisal risks later appeared during implementation, and what new issues emerged? |
| P506272 | Karnataka Water Security and Resilience Program | Program-for-Results | Early-lifecycle monitoring | What early signals are emerging relative to appraisal expectations? |

The IPF contract-awards dataset covers P130544 only. For the two
Program-for-Results operations, zero procurement rows **do not** mean there was
no procurement activity.

## High-level architecture

Deterministic facts where we can be exact. Retrieval where evidence lives in
documents. Agents only where reasoning is genuinely required. Validation before
trust.

```
 Source files: Projects workbook · IBRD loans snapshot · IPF contract awards · project PDFs
        │                                          │
        ▼  ingestion (Bronze: raw + lineage)       ▼  Docling parsing + classification
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ SILVER  projects · loans · procurement · isr_snapshots · project_results ·  │
 │         project_events · appraisal_risks · document_chunks                  │
 └─────────────────────────────────────────────────────────────────────────────┘
        │  deterministic transformations            │ structure-aware chunks
        ▼                                           ▼
 GOLD  project_360 · project_timeline ·       Databricks Vector Search
       result_progress · risk_register ·            │
       attention_signals (documented rules)         │ hybrid retrieval (lexical + dense
        │                                           │ + metadata filters) → CrossEncoder
        ▼                                           ▼ rerank → context builder
   Typed read-only tools (Pydantic) ◄───────────────┘
        │
   Router ─► STRUCTURED     → tool → answer
          ├► DOCUMENT       → metadata-filtered RAG → cited answer
          └► INVESTIGATION  → Investigator → tools + Researcher → synthesis
                              → Critic (PASS / FAIL, max 2 retries) → answer
        │
   Guardrails (NeMo adapter + deterministic validators) · provenance · MLflow tracing
        │
   FastAPI  ──►  React: Portfolio · Project 360 · Finance · Results · Timeline ·
                        Attention Signals · Evidence Drawer · Copilot
```

**Stack:** Python 3.11+, Databricks (Delta Lake, Unity Catalog, SQL, Vector
Search, Workflows, Model Serving), MLflow, LangGraph, Docling, NVIDIA NeMo
Guardrails, sentence-transformers CrossEncoder, FastAPI, React, Pydantic, pytest.

## Local → GitHub → Databricks workflow

```
 VS Code (local)                GitHub                     Databricks
 ───────────────                ──────                     ──────────
 write src/ + tests      push   repository   clone/pull    Git folder
 pytest (no creds)   ────────►  (no data,   ───────────►   notebooks/ = thin entry points
 data/ stays local               no secrets)               %run ./_bootstrap → import src/
                                                           raw files in a UC volume
                                                           Delta tables, Vector Search,
                                                           Model Serving, MLflow
```

1. **Develop locally.** All business logic lives in
   `src/worldbank_copilot/` and is unit-tested with small fixtures. No Databricks
   credentials are needed to import the package or run tests.
2. **Commit to GitHub.** Datasets, PDFs and `.env` are git-ignored.
3. **Run on Databricks.** Open the repo as a Git folder, copy the source files
   to the Unity Catalog Volume and run the notebooks (Phase 6: see "Governed Delta
   foundation" below). Catalog, schema and Volume names come from configuration.
   Notebooks only call `src/` code.

The same code selects its environment automatically (`WBC_ENV`, otherwise
`local`, or `databricks` when a Databricks runtime is detected).

## Repository structure

```
.
├── README.md
├── IMPLEMENTATION_PLAN.md          plan, data findings, decisions, status
├── Claude.md                       product/engineering specification
├── pyproject.toml
├── .env.example                    every WBC_* setting
├── configs/
│   ├── projects.yaml               the three in-scope projects
│   ├── document_manifest.yaml      curated labels for the known project PDFs
│   ├── indicator_aliases.yaml      reviewed indicator aliases (empty until reviewed)
│   ├── source_snapshot.json        validated source files + SHA-256 (Phase 6)
│   ├── delta_contracts.lock.json   frozen Delta table contracts (Phase 6)
│   ├── reconciliation/             expected profiles of the current snapshot (Phase 6)
│   ├── environments/               base.yaml · local.yaml · databricks.yaml
│   └── guardrails/                 NeMo config (Phase 12)
├── data/                           source files — git-ignored (see data/README.md)
├── src/worldbank_copilot/
│   ├── common/                     config · project_registry · dates · identifiers · logging
│   ├── ingestion/                  contracts · readers · document inventory · data quality
│   ├── parsing/                    Docling adapter, metadata, parsed representation (Phase 4)
│   ├── extraction/                 ISR snapshots · results · appraisal risks · events (Phase 5)
│   ├── lakehouse/                  contracts · identities · reconciliation · Delta store (Phase 6)
│   ├── transformations/            bronze · normalize · silver (+ models, lineage, quality); Gold in Phase 6
│   ├── retrieval/                  Phases 7–8
│   ├── tools/                      Phase 9
│   ├── agents/                     Phase 11
│   ├── guardrails/                 Phase 12
│   ├── observability/              Phase 13
│   └── api/                        Phase 14
├── notebooks/                      _bootstrap + 01–10 thin Databricks entry points
├── sql/                            Phase 6 validation queries
├── review/                         human review artefacts (indicator alias candidates)
├── requirements-databricks.txt     pinned notebook-scoped runtime dependencies
├── evaluation/                     Phase 16
├── frontend/                       Phase 15
├── scripts/                        validate_data.py (Bronze/Silver) · parse_documents.py (Docling)
│                                   · extract_document_facts.py (document-derived Silver)
│                                   · platformize.py (Phase 6 local dry run)
└── tests/
    ├── unit/
    └── integration/
```

## Local setup

```powershell
py -3.14 -m venv .venv               # any python.org Python 3.11+ (see note below)
.\.venv\Scripts\python -m pip install -e ".[dev,documents]"   # 'documents' = Docling
copy .env.example .env               # optional; edit values as needed
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\ruff check src tests scripts
```

(On macOS/Linux use `.venv/bin/python` and `cp`.)

> **Windows note (Docling / PyTorch):** do not build the venv from Anaconda's
> `python.exe` if you need Docling. Anaconda ships an older Microsoft C++ runtime
> (14.29) next to its interpreter, and PyTorch 2.14 then fails to load
> (`OSError: [WinError 1114] … c10.dll`). A python.org interpreter works. Without
> the `documents` extra, everything except PDF parsing runs on any Python 3.11+.

Place the source files as described in [data/README.md](data/README.md).

### Configuration

Settings resolve as `configs/environments/base.yaml` → `<env>.yaml` → `WBC_*`
environment variables. Workspace-specific values (catalog, volume, Vector Search
endpoint/index, embedding and LLM endpoints, MLflow experiment) have no defaults.
Code that needs one fails with an error naming the variable to set.

```python
from worldbank_copilot.common import load_settings, load_project_registry

settings = load_settings()  # environment auto-selected
registry = load_project_registry(settings.config_dir)
registry.get("P179039").instrument  # 'Program-for-Results Financing'
settings.table_name("silver", "loans")  # needs WBC_CATALOG
```

Tests that need a live workspace are marked `@pytest.mark.databricks` and are
deselected by default.

### Validating source data / Bronze ingestion

```powershell
.\.venv\Scripts\python scripts\validate_data.py                      # Bronze -> Silver -> validation
.\.venv\Scripts\python scripts\validate_data.py --layer bronze       # Bronze only
.\.venv\Scripts\python scripts\validate_data.py --json .local_output\validation_report.json
.\.venv\Scripts\python scripts\validate_data.py --write-bronze --write-silver  # JSONL under .local_output\
```

The script reads the source files (never writes to `data/`), builds Bronze and Silver
tables in memory, and prints projects, loans (one-to-many), project financial
summaries, procurement coverage and awards, the document/ISR inventory, and the
Bronze and Silver data-quality observations. It exits with code 1 if any `ERROR`
observation is found.

Every Silver row carries `source_refs` back to its Bronze records, and field lineage
explains each value down to its source column:

```python
from worldbank_copilot.transformations.silver_lineage import explain

print(explain("silver_project_financial_summary", "disbursed_total_usd"))
```

### Parsing project PDFs (Docling)

```powershell
.\.venv\Scripts\python scripts\parse_documents.py                   # all 53 PDFs (cached ones reused)
.\.venv\Scripts\python scripts\parse_documents.py --project P130544
.\.venv\Scripts\python scripts\parse_documents.py --document <document_id or filename>
.\.venv\Scripts\python scripts\parse_documents.py --force           # ignore the cache
.\.venv\Scripts\python scripts\parse_documents.py --failed-only     # retry failures only
.\.venv\Scripts\python scripts\parse_documents.py --report-only     # validate outputs, no parsing
```

Each PDF becomes `.local_output/parsed/<PROJECT_ID>/<document_id>.json`: pages,
text blocks (raw and cleaned, each with page, bounding box and section), tables
(structured grid, markdown and readable text, with page), sections with page spans,
and validated metadata. Every value carries enough provenance to cite
"Source: ISR Sequence 18, page 7". Parsing is cached by source SHA-256 plus the
parser version/configuration; one failing PDF never stops the run. A full cold run
of the 53 PDFs takes roughly 30–60 minutes on CPU; no OCR or LLM is used.

Document metadata (type, date, ISR sequence, report and loan numbers, project) is
extracted deterministically and reconciled with `configs/document_manifest.yaml`:
each field records the manifest value, the document value, the resolved value and
the reason (`CONFIRMED`, `CORRECTED_FROM_DOCUMENT`, `MANIFEST_ONLY`,
`DOCUMENT_ONLY`, `CONFLICT`, `UNKNOWN`).

### Extracting document facts (document-derived Silver)

```powershell
.\.venv\Scripts\python scripts\extract_document_facts.py                 # all projects (cache reused)
.\.venv\Scripts\python scripts\extract_document_facts.py --project P179039
.\.venv\Scripts\python scripts\extract_document_facts.py --no-cache      # re-run every extractor
```

Reads the parsed cache from the previous step (Docling is never re-run) and writes
to `.local_output/silver_documents/`:

| File | Grain |
|---|---|
| `isr_snapshots.jsonl` | one row per ISR: ratings (raw + normalized, previous/current), SORT, per-loan disbursements and key dates, narrative text, restructuring history |
| `project_results.jsonl` | project × indicator × ISR observation (baseline / previous / current / target values and dates) |
| `appraisal_risks.jsonl` | `FORMAL_RISK_RATING` (SORT rows) or `ASSESSMENT_FINDING` (explicit risk tables / TA risk blocks) |
| `project_events.jsonl` | approval, effectiveness, restructuring, additional financing, cancellation, closing-date, results-framework, component and reallocation events |
| `project_enrichment.jsonl`, `silver_projects_enriched.jsonl` | reconciled original closing dates, applied to a *copy* of `silver_projects` |
| `pdf_text_fallback_log.jsonl`, `document_extraction_quality.json` | every PDF text-layer read; summary and quality observations |

Extraction is deterministic, in this order: Docling tables, then labelled
key/value text, then regular expressions, then a **logged** read of the PDF text
layer for the specific page when a Docling table fails validation. No LLM is
used. Every value carries its document, page, section or table cell, extraction
method and a status (`EXACT`, `NORMALIZED`, `DERIVED_FROM_EXPLICIT_SOURCE`,
`AMBIGUOUS`, `MISSING`, `CONFLICT`). Values that cannot be validated stay NULL
with an observation. Disagreements with the loan snapshot are reported, never
overwritten. Extraction is cached per document by source hash, parser
configuration and extractor code hash.

### Tests

```powershell
.\.venv\Scripts\python -m pytest                  # unit tests (synthetic fixtures only)
.\.venv\Scripts\python -m pytest -m integration   # checks against the real local files
.\.venv\Scripts\python -m pytest -m docling       # real Docling model runs (slow)
```

## Governed Delta foundation (Phase 6)

Phase 6 moves the validated Bronze and Silver data into Unity Catalog as Delta
tables **without changing its meaning**. The same `src/` code runs locally and in
Databricks; only persistence differs.

### Unity Catalog layout

```
worldbank_ai                      (existing catalog; validated, never created)
├── bronze                        (schema; created only if missing)
│   ├── sources                   (Volume: original files, byte-for-byte)
│   │   └── data/                 mirrors the local data/ layout
│   │       ├── all.xlsx, *.csv
│   │       └── P130544/ P179039/ P506272/   project PDFs
│   └── <9 Bronze tables>
├── silver                        (schema; created only if missing)
│   ├── pipeline_artifacts        (Volume: derived artefacts)
│   │   ├── parsed/               Phase 4 parsed documents (uploaded)
│   │   └── silver_documents/_cache/  extraction cache (written by the run)
│   └── <18 Silver tables>
└── gold                          reserved for Phase 7 (not created in Phase 6)
```

All names come from `configs/environments/base.yaml` (`databricks.catalog`,
`bronze_schema`, `silver_schema`, `gold_schema`, `source_volume`,
`artifact_volume`) and can be overridden with `WBC_*` variables. The Volume mirrors
the local `data/` layout so relative source paths, which are part of lineage, are
identical in both environments.

### Table inventory (current snapshot)

| Layer | Tables |
|---|---|
| bronze | `projects_raw`, `themes_raw`, `sectors_raw`, `geo_locations_raw`, `financers_raw`, `loans_raw`, `procurement_raw`, `document_inventory`, `procurement_coverage` |
| silver (structured, Phase 3) | `projects`, `project_sectors`, `project_themes`, `loans`, `project_financial_summary`, `procurement_awards`, `procurement_suppliers`, `procurement_coverage` |
| silver (documents, Phase 5) | `isr_snapshots` (+ `isr_sort_ratings`, `isr_loan_disbursements`, `isr_loan_key_dates`), `project_results`, `appraisal_risks`, `project_events`, `project_enrichment` |
| silver (governance) | `data_quality_observations`, `indicator_match_candidates` |

Delta names drop the repository's layer prefix (`silver_loans` → `silver.loans`);
Phase 5 names are unchanged.

### How it works

* **Contracts** (`lakehouse/contracts.py`): every table has explicit columns, types
  (money = `DECIMAL(38,6)`, never float; dates = `DATE`), nullability, controlled
  vocabularies and a natural key. Rows are validated before anything is written.
  Columns derive from the existing Pydantic models (single source of truth) and are
  frozen in `configs/delta_contracts.lock.json`, so a schema change is always a
  reviewed diff. An existing Delta table whose schema differs stops the load.
* **Provenance:** Bronze keeps source file/sheet/row and adds `_source_sha256`.
  Document-derived rows flatten their primary evidence into queryable
  `evidence_*` columns (document, page, section, table, row, column, text,
  extraction method, source hash) and keep the full nested structures as canonical
  JSON. `status` and `quality_issue_codes` stay on every record.
* **Identity and idempotency:** `record_id` is a hash of the natural key;
  `record_hash` hashes business, provenance and quality columns (not load
  metadata). Loads are snapshot MERGEs: insert new records, update only when the
  hash changes, delete records absent from the snapshot. Rerunning the same
  snapshot changes no rows. The Bronze ingestion run id is derived from the source
  snapshot, so Silver lineage is identical across reloads.
* **Load metadata** (`_load_run_id`, `_loaded_at`, `_source_snapshot_id`,
  `_pipeline_version`) is operational and separate from source provenance.
* **Source integrity:** `configs/source_snapshot.json` lists all 56 source files
  (3 structured + 53 PDFs) with SHA-256. Every run hashes the files under the
  environment's data root; a mismatch or missing file stops everything. Parsed
  documents must carry the same source hash.
* **Reconciliation:** each dataset is profiled (row count, content fingerprint,
  null counts, distinct counts, group counts). Before writing, profiles must equal
  `configs/reconciliation/expected_profiles.json` (the current validated snapshot,
  not business rules). After writing, every table is read back and must reconcile
  exactly with the validated data.
* **Quality observations:** Bronze, Silver, parsing and extraction observations,
  plus issues attached to individual records (linked by `related_record_id`), are in
  `silver.data_quality_observations`.
* **Known unresolved items stay explicit:** undated restructuring papers keep
  `event_date = NULL` with a derived `candidate_event_date` and
  `candidate_date_status = DERIVED_FROM_EXPLICIT_SOURCE`; P179039 ISR 5 keeps its
  header date, with a `SEQUENCE_DATE_ANOMALY` observation (ISR sequence is the
  ordering key); near-identical indicators are never merged.

### Running Phase 6 locally (dry run: nothing is persisted)

```powershell
.\.venv\Scripts\python scripts\platformize.py                   # verify, build, validate, reconcile
.\.venv\Scripts\python scripts\platformize.py --print-ddl       # CREATE TABLE statements
.\.venv\Scripts\python scripts\platformize.py --upload-commands # Databricks CLI copy commands
```

`--write-snapshot` and `--write-expected` re-record the source manifest and the
expected profiles and contract lock. Use them only after reviewing a new source
snapshot or an intended schema change.

### Running Phase 6 in Databricks

1. Commit and push; in Databricks, create or pull the repository as a **Git folder**.
2. Run `notebooks/05_platformize_databricks.py` once up to "Step 1". It validates the
   catalog and creates any missing schemas and Volumes, or reports the missing privilege.
3. Copy the files into the Volumes (commands from `--upload-commands`), for example:
   `databricks fs cp --recursive data dbfs:/Volumes/worldbank_ai/bronze/sources/data`
   and `databricks fs cp --recursive .local_output/parsed dbfs:/Volumes/worldbank_ai/silver/pipeline_artifacts/parsed`.
4. Run the whole notebook on DBR 15.4 LTS or later (Python ≥ 3.11). It installs the
   pinned `requirements-databricks.txt`, then:
   - verifies every source hash;
   - rebuilds and validates all datasets and reconciles them with the expected profiles;
   - MERGEs into Delta, reads the tables back and reconciles again;
   - runs a second time and asserts that no rows changed;
   - displays the queries in `sql/phase6_validation.sql`.

Any failure stops the run before (or instead of) writing.

### Indicator alias review

`review/indicator_alias_candidates.csv` (also `silver.indicator_match_candidates`)
lists every near-identical indicator pair as `PENDING_REVIEW`. To approve one, add
an entry with evidence and reviewer to `configs/indicator_aliases.yaml`. The
pipeline never writes that file; only reviewed aliases establish identity.

## Remaining documentation

Sections for Gold, Vector Search, FastAPI,
React, evaluation and known limitations will be added as those phases are
implemented.
