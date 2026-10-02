# World Bank Project Implementation Intelligence Copilot

Evidence-grounded decision support for World Bank project officers and analysts.

> **Status:** Phases 1-9 are COMPLETE. Phase 9F-C passed real Databricks acceptance
> in run `9f3`: 10/10 cases, preflight PASS, overall PASS, final artifact and completion
> receipt verified. Phase 10 is NOT STARTED.
> Adaptive reranking was evaluated diagnostically in 9E; no adaptive policy was promoted,
> and independent validation is required before any future promotion. See
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
│   ├── intelligence/               attention rules + rating scales (Phase 7)
│   ├── gold_contracts.lock.json    frozen Gold table contracts (Phase 7)
│   ├── environments/               base.yaml · local.yaml · databricks.yaml
│   └── guardrails/                 NeMo config (Phase 12)
├── data/                           source files — git-ignored (see data/README.md)
├── src/worldbank_copilot/
│   ├── common/                     config · project_registry · dates · identifiers · logging
│   ├── ingestion/                  contracts · readers · document inventory · data quality
│   ├── parsing/                    Docling adapter, metadata, parsed representation (Phase 4)
│   ├── extraction/                 ISR snapshots · results · appraisal risks · events (Phase 5)
│   ├── lakehouse/                  contracts · identities · reconciliation · Delta store (Phase 6)
│   ├── intelligence/               Gold: timeline · results · risks · signals · 360 (Phase 7, Spark)
│   ├── transformations/            bronze · normalize · silver (+ models, lineage, quality); Gold in Phase 6
│   ├── retrieval/                  chunking · embeddings · Vector Search · BM25 · rerank · evaluation (Phase 8)
│   ├── tools/                      Phase 9
│   ├── agents/                     Phase 11
│   ├── guardrails/                 Phase 12
│   ├── observability/              Phase 13
│   └── api/                        Phase 14
├── notebooks/                      _bootstrap + 01–10 thin Databricks entry points
├── sql/                            Phase 6 and Phase 7 validation queries
├── review/                         human review artefacts (indicator alias candidates)
├── requirements-databricks.txt     pinned notebook-scoped runtime dependencies
├── evaluation/                     retrieval evaluation questions (Phase 8)
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
worldbank_copilot                      (create once: CREATE CATALOG worldbank_copilot; the pipeline validates it, never creates it)
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
   Create the catalog once (needs metastore privileges; add `MANAGED LOCATION '...'`
   if your metastore requires one): `CREATE CATALOG IF NOT EXISTS worldbank_copilot;`
2. Run `notebooks/05_platformize_databricks.py` once up to "Step 1". It validates the
   catalog and creates any missing schemas and Volumes, or reports the missing privilege.
3. Copy the files into the Volumes (commands from `--upload-commands`), for example:
   `databricks fs cp --recursive data dbfs:/Volumes/worldbank_copilot/bronze/sources/data`
   and `databricks fs cp --recursive .local_output/parsed dbfs:/Volumes/worldbank_copilot/silver/pipeline_artifacts/parsed`.
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

## Gold intelligence layer (Phase 7)

Gold turns the governed Silver Delta tables into decision-support tables. It runs **in
Databricks, from `worldbank_copilot.silver.*`, with native Spark**. It does not rebuild
Phases 1–5.

```
worldbank_copilot.silver.*  ── validated against the Phase 6 contracts
   │  Spark (native functions only: Photon / serverless friendly, no Python UDFs)
   ▼
gold.project_timeline · gold.result_progress · gold.risk_register ·
gold.attention_signals · gold.project_360 · gold.quality_observations
   │  Gold contracts + invariants (any ERROR stops before writing)
   ▼
snapshot MERGE into worldbank_copilot.gold.* → read back → reconcile
```

| Table | Grain | Key sources |
|---|---|---|
| `project_360` | project | projects, project_financial_summary, project_enrichment, isr_snapshots, events, risks, results, signals |
| `project_timeline` | project × source record × event type | projects, project_enrichment, isr_snapshots, project_events |
| `result_progress` | Silver result observation | project_results, indicator_match_candidates |
| `risk_register` | Silver risk / finding (plus the latest ISR SORT) | appraisal_risks, isr_sort_ratings |
| `attention_signals` | rule × project × subject × sequence | all of the above |
| `quality_observations` | check | Gold invariants and observations |

**Attention signals are observations, not predictions.** Each one comes from a
reviewed rule in `configs/intelligence/attention_rules.yaml`, which holds the id,
version, severity logic (INFO / WATCH / HIGH), thresholds with rationale, and the
deferred rules with reasons. Each signal keeps its primary Silver record, all
supporting record ids, and the document, page, section and extraction method, so
`sql/phase7_validation.sql` (`signal_provenance`) can trace it back to the source
file hash. Ratings are compared by ordinal rank only
(`configs/intelligence/rating_scales.yaml`).

Semantics that are preserved on purpose:

- **Dates:** `event_date` is only a source-stated date. Derived restructuring
  candidate dates stay in `candidate_event_date` (`DERIVED_CANDIDATE`).
- **ISR ordering:** ISR sequence orders the timeline. P179039 ISR 5 keeps its printed
  date and has `date_sequence_anomaly = true`.
- **Indicator identity:** indicators are joined only by the Silver key (exact name or
  reviewed alias). The 96 pending alias pairs remain separate series.
- **Progress:** progress is calculated only when it is mathematically valid.
  Otherwise it is NULL, and `calculation_status` says why:
  - DLI table rows, Yes/No and text units;
  - missing values and zero target distance;
  - non-exact extraction.

### Running Gold in Databricks

After Phase 6, run `notebooks/06_build_gold_intelligence.py`. It:
1. validates the Silver tables;
2. builds Gold and checks contracts and invariants;
3. creates `worldbank_copilot.gold` if it is missing;
4. MERGEs, reads back and reconciles;
5. runs a second time and asserts that no rows changed;
6. displays `sql/phase7_validation.sql`.

### Testing Gold transformations locally (real Spark)

Gold code is tested on a local Spark session. It uses only JVM-native Spark functions,
so no Python workers are needed. The Spark dev environment is separate from the main
`.venv`:

```powershell
py -3.12 -m venv .venv-spark
.\.venv-spark\Scripts\python -m pip install "pyspark==4.0.1" pytest -r requirements-databricks.txt
.\.venv-spark\Scripts\python -m pip install -e . --no-deps
# a JDK 17: set JAVA_HOME, or unpack one into .tools\ (git-ignored)
.\.venv-spark\Scripts\python -m pytest -m spark
```

The `spark` tests have two parts:
- **Synthetic scenario:** exercises every rule and edge case.
- **Real Silver rows:** uses the rows exported by `scripts/platformize.py`. It checks invariants and determinism across two builds, and runs every query in `sql/phase7_validation.sql`.

Delta MERGE, read-back and idempotency are proven only by the Databricks run.

## Retrieval foundation (Phase 8)

Phase 8 builds evidence retrieval only: no agents and no routing. It runs in Databricks with
`notebooks/07_build_retrieval_and_evaluate.py`.

```
Phase 4 parsed documents (artifact Volume; nothing re-parsed)
  -> silver.document_chunks       3 chunking strategies, deterministic ids, full provenance
  -> silver.chunk_embeddings      embedding cache keyed by (text hash, model)
  -> silver.document_chunk_index  RETRIEVAL chunks + vectors (Change Data Feed)
  -> Databricks Vector Search     Delta Sync index, self-managed vectors
  -> Retriever: project scope -> lexical (BM25) | dense (Vector Search) | hybrid (RRF)
                -> isolation guardrail -> de-duplication -> CrossEncoder (optional)
                -> citation-ready Evidence (or INSUFFICIENT_EVIDENCE)
```

- **Project isolation.** Every retrieval needs a `project_id`. Filters are applied before scoring. Every result is re-checked against the governed corpus, and a chunk from another project raises an error. A question that names another project is refused.
- **Governed metadata.** Text, citations and provenance always come from `silver.document_chunks`, never from the index.
- **Chunking strategies:**
  - `fixed`: recursive, with overlap;
  - `structure`: sections, paragraphs, and table row groups with the header repeated;
  - `parent_child`: small children retrieved, section parents used for context.
  - Settings live in `configs/retrieval/chunking.yaml`. A changed setting changes the strategy version, and with it the chunk ids.
- **Experiments.** They run in stages (chunking -> retrieval -> reranking -> candidate depth -> query filters) over `evaluation/retrieval_questions.yaml`. Metrics: Recall@5/10, MRR, nDCG@5, Precision@5 and latency. A more complex configuration is chosen only if it beats the simpler one by `min_improvement`. Results are logged to MLflow when it's available.
- **Capability probe.** Notebook 07 first checks Vector Search and the embedding endpoint (`configs/retrieval/embeddings.yaml`). If either is missing it stops and builds nothing. There is no local fallback vector store.

### Phase 8 dependencies (Databricks notebooks)

| File | Installed by | Content |
|---|---|---|
| `requirements-databricks.txt` | notebooks 05, 07, 07b | Phase 6 exact pins (pydantic, PyYAML, python-dotenv, openpyxl, pypdfium2) |
| `requirements-retrieval.txt` | notebooks 07, 07b | `databricks-ai-search==0.78`, `deprecation==2.1.0` |
| `requirements-reranker.txt` | notebook 07b only | `sentence-transformers==5.5.1`, `transformers==4.57.6`, `torch==2.12.0` |
| `constraints-databricks.txt` | every install (`-c`) | `protobuf>=6.33.5,<7`, `grpcio-status>=1.76.0,<2` |

**Runtime packages.** `databricks-sdk`, `mlflow-skinny`, `protobuf`, `requests` and `httpx` come from Databricks serverless environment 6 and are not reinstalled. Requirement files hold exact pins only. The constraints are ranges because they protect packages the runtime owns rather than install anything.

**Vector Search client.** It is `databricks-ai-search`, the successor of the deprecated `databricks-vectorsearch`. The old package pins `protobuf<6` and downgraded the runtime's protobuf 6.33.5 to 5.29.6.

**Dependency health.** After each install, `common/dependency_health.check_environment` runs before anything else. The notebook stops if any of these fails:
- `pip check`;
- an exact pin is not installed at its version;
- a requirement file contains a line that is not an exact pin;
- a notebook-scoped install replaces a protected runtime package (`configs/environments/dependencies.yaml`).

**Reranking is isolated.** The CrossEncoder experiment runs in `notebooks/07b_rerank_experiment.py` on serverless environment 6 **ML**, which already ships those exact versions. So the capability probe and production retrieval in notebook 07 never depend on torch.

## Routing, tools and the Phase 10 contract (Phase 9)

Phase 9 builds deterministic routing and typed tools, tests two model-based additions
experimentally, and freezes the interface the Phase 10 agent layer inherits. It contains no
agents and no answer synthesis. The handoff manifest is `configs/phase9_closure.yaml`; a test
checks every value in it against the code, the configuration and the committed artifacts.

Phase 9 is COMPLETE after the accepted `9f3` Databricks run. The earlier `9f1` Git
identity precheck failure and `9f2` Volume checkpoint failure remain preserved as audit
history; both executed zero contract cases. See the 9F-C closure checkpoint in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the accepted artifact identities.
Phase 10 may build agent execution on the frozen contracts; it must not rewrite Phase 9 results.

```
USER QUERY
  -> input guardrails -> project resolution (scope + authorisation, before any call)
  -> temporal resolution -> deterministic intent rules -> requirements
  -> ROUTER (9B.2, frozen)                 records its output as-is
  -> EXECUTION MAPPING (routing/execution.py, 9D Candidate A)
       STRUCTURED     -> typed structured tools (ToolExecutor, allowlist, no SQL)
       DOCUMENT       -> profile-owned retrieval (retrieval/contract.py)
       INVESTIGATION  -> an unexecuted InvestigationPlan for Phase 10
       CLARIFY        -> targeted clarification (nothing executes)
       REFUSE         -> safe refusal (nothing executes)
```

**How the design was reached.**
- **Foundations first.** The work did not start with agents. Governed Bronze/Silver/Gold data came first (Phases 1–7), then deterministic implementation signals (Phase 7), validated document retrieval (Phase 8), and typed tools with deterministic routing (Phase 9A–9C).
- **Hosted bounded classifier (9D Candidate C): not promoted.** A hosted model was tested for ambiguous routing. It passed the capability and operational checks but failed the preregistered semantic incremental-value / repeatability gate, so deterministic routing stays authoritative.
- **Adaptive reranking (9E): not promoted.**
  - Reranking helps some questions and hurts others, and a ground-truth Oracle shows that selective reranking has real headroom.
  - Simple deterministic triggers looked promising on the Phase 8 questions, but those questions had already been used, so no adaptive policy was promoted without independent validation.
  - 9E also separated ranking failures from candidate-generation failures: P179039's main retrieval weakness cannot be fixed by reranking.
- **Freeze before handing over (9F).** Only after these interfaces and known limitations were frozen is the system handed to the agent layer.

**Distinctions the contract keeps explicit:**
- **Router vs execution mapping.** The router's recorded output never changes. When the rules cannot resolve the intent, it records `SEMANTIC_CLASSIFICATION_REQUIRED`, and the execution mapping turns that into `CLARIFY / INTENT_NOT_RESOLVED`. The semantic fallback is disabled.
- **Phase 10 quality baseline vs production configuration.** Phase 10 retrieves with `phase8_quality_baseline@1`:
  - fixed chunks, hybrid BM25 + Qwen dense with RRF k 60, candidate_k 50;
  - the CrossEncoder on every query, final_k 5.

  This is the strongest *validated* retrieval-quality configuration. It is not a production policy, and `configs/retrieval/retrieval.yaml` `production` stays null. Agents cannot tune any retrieval parameter.
- **NO_EVIDENCE vs corpus absence.** `NO_EVIDENCE` means relevant evidence was not found in the retrieved candidates. It never means the documents do not contain it.
- **DOCUMENTED_FINDING vs AI_INTERPRETATION.** A retrieved passage is a DOCUMENTED_FINDING: the document states it at the cited page, but its relevance is not verified. AI_INTERPRETATION comes only from future Phase 10 synthesis; tools and retrieval never produce it, and it is never relabelled as FACT.

**Known limitations** (listed in the manifest):
- P179039 candidate generation (4 of 11 answerable questions have no evidence in the fused top-50);
- warm-cache latency figures (a cold query embedding adds about 1.5–2 s);
- CPU CrossEncoder latency (about 5.6 s for a reranked query);
- r033 and r060;
- temporal and document-type hints are recorded but not applied as filters;
- notebook-token authentication.

## Remaining documentation

Sections for Vector Search, FastAPI,
React, evaluation and known limitations will be added as those phases are
implemented.
