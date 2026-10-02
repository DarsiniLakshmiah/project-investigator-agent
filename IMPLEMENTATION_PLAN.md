# Implementation Plan — World Bank Project Implementation Intelligence Copilot

Source of truth for requirements: [Claude.md](Claude.md). This file records how
they are being implemented, what was learned from the real data, and what is
still open. Updated at the end of every phase.

---

## 1. Current status

| Phase | Scope | Status |
|---|---|---|
| 1 | Repository skeleton, configuration, project scope, tests | **Complete, approved** (2026-09-26) |
| 2 | Structured-source mapping + Bronze ingestion | **Complete, approved** (2026-09-26) |
| 3 | Silver structured transformations | **Complete, approved** (2026-09-26) |
| 4 | Docling parsing, document metadata validation, parsed representation | **Complete, approved** (2026-09-27) |
| 5 | Structured document extraction (ISR snapshots, results, appraisal risks, events) | **Complete, approved** (2026-09-30) |
| 6 | Databricks platformization and governed Delta foundation | **Complete, validated in Databricks, approved** (2026-09-30) |
| 7 | Deterministic Gold intelligence layer | **Complete, validated in Databricks, approved** (2026-09-30) |
| 8 | Databricks-native retrieval foundation + experiments | **Corpus, Qwen embeddings, AI Search index and notebook 07 Steps 4–7 and 07b staged experiments (incl. CrossEncoder) validated in Databricks (2026-10-01). Complete. Adaptive reranking evaluated diagnostically in Phase 9E; no adaptive policy promoted; independent validation required before any future promotion** |
| 9 | Structured tools + intelligent query routing | **9A, 9B approved** (2026-10-01); **9C CLOSED and frozen** (80 reviewed cases, 29 dev / 51 test); **9D CLOSED** (2026-10-02): Candidate A SELECTED - deterministic routing + targeted clarification; no semantic LLM fallback promoted (Candidate C GPT-OSS-20B DEV REJECTED on quality/repeatability; earlier Candidate C blocks were not quality rejections); **9E CLOSED** (2026-10-02): adaptive-rerank diagnostic VALID offline and live-validated; descriptive only, no adaptive policy promoted, `production` stays null; **9F-A approved, 9F-B implemented (Phase 10 execution contract; under review)**; 9F-C (bounded Databricks contract validation) not started |
| 10–13 | See §3 (roadmap from Claude.md §36) | Not started |

Latest verification (end of Phase 4, in the rebuilt Python 3.14 `.venv`): `pytest` → 331
passed (unit); `pytest -m integration` → 18 passed (5 Bronze + 7 Silver + 6 parsed, real
files); `pytest -m docling` → 2 passed (real Docling models); `ruff check` +
`ruff format --check` clean; `validate_data.py` → exit 0 (Bronze 0/4/16, Silver 0/1/10);
`parse_documents.py` → 53/53 parsed, 0 errors, 32 warnings, 18 info.

---

## 1a. Approved decisions (user, 2026-09-26, on approving Phase 1)

1. **Package namespace:** all code lives under `src/worldbank_copilot/`.
2. **Source-data layout:** the local layout stays as delivered (`data/*.xlsx|csv`,
   `data/<PID>/*.pdf`). Source files are never moved, renamed, deleted or
   modified. Physical layout is abstracted through configuration / path resolution.
3. **Source-of-truth policy:**
   * IBRD Statement of Loans snapshot → authoritative for **loan-level** financial values.
   * Projects & Operations workbook → authoritative for **project metadata**.
   * Disagreements are **never silently resolved**: source-specific values are
     preserved and explicit data-quality observations are generated.
4. **Project → loan is one-to-many.** Never assume one loan per project
   (P130544 → IBRD86010, IBRD93240).
5. **Loan-number normalisation:** deterministic normaliser linking e.g. `IBRD86010`
   and `8601-IN` when appropriate; both `raw_loan_number` and
   `normalized_loan_number` are preserved. **No fuzzy matching** for financial identifiers.
6. **Databricks infrastructure settings stay unset** (catalog, volume, Vector Search
   endpoint/index, embedding endpoint, LLM endpoint, MLflow experiment). None are invented.
7. **Databricks package-installation / runtime decision stays open** until tested in
   the actual workspace.

### Approved on Phase 2 sign-off (user, 2026-09-26)

8. **Document manifest stays** as deterministic bootstrap metadata. Phase 4 must
   validate/correct it against parsed content; manifest metadata is never stronger
   evidence than parsed source content.
9. **The four Phase 2 data-quality observations stay WARNINGs** (source
   inconsistencies/ambiguities, not ingestion failures).
10. **Delta writer stays unimplemented**; transformation logic is built and tested
    locally and connected to Delta when execution moves to Databricks.

### Approved on Phase 3 sign-off (user, 2026-09-26)

11. **Loan principal reconciliation rule kept:** components differ + non-zero exchange
    adjustment → INFO with FX/valuation caveat; components differ without an exchange
    adjustment → WARNING. Components are never forced to reconcile; no adjustment is
    manufactured.
12. Phase 3 Silver architecture, procurement modelling (awards + supplier
    relationships, unresolved amounts as bounds), provenance design and source-of-truth
    policy are approved.

### Approved on Phase 6 sign-off (user, 2026-09-30)

20. **Phase 6 validated in Databricks:** 56/56 source hashes matched; 27 Bronze/Silver
    Delta tables; expected-profile and read-back reconciliation passed; second run 0
    inserts / 0 updates / 0 deletes; 394 quality observations (0 / 80 / 314).
21. **Databricks is the primary execution environment from Phase 7:** Gold is built in
    Databricks directly from `worldbank_copilot.silver.*` with Spark; local execution is
    for tests and developer feedback only.

### Approved on Phase 5 sign-off (user, 2026-09-30)

16. **Unity Catalog layout:** catalog `worldbank_copilot` (renamed from `worldbank_ai`, which is already used in the workspace; created once by a user with metastore privileges), schemas `bronze` and
    `silver`; `gold` reserved for Phase 7 (supersedes §1a.6 for catalog/schemas only).
17. **Restructuring candidate dates** are preserved as `candidate_event_date` /
    `candidate_date_basis` / `candidate_date_status = DERIVED_FROM_EXPLICIT_SOURCE`;
    `event_date` stays NULL unless a source states it.
18. **P179039 ISR 5:** header-date policy unchanged; ISR sequence is the primary
    ordering dimension; the anomaly stays visible (`SEQUENCE_DATE_ANOMALY`).
19. **Indicator identity:** only exact normalized name, a genuinely stable source id
    or a reviewed alias. Near-matches are reviewed by humans, never auto-merged.

### Approved on Phase 4 sign-off (user, 2026-09-27)

13. **Dropped Docling table cells are not accepted silently.** Docling stays the
    primary parser; for important tables (ISR ratings, results framework, risk ratings,
    financial/disbursement, restructuring change tables) Phase 5 uses the Docling
    structure first and falls back to the PDF text layer **for that page/table only**
    when quality checks fail (missing cells, incomplete rows, low page coverage,
    malformed structure, truncated indicator names, inconsistent column counts). Every
    value records `extraction_method` (DOCLING_TABLE / DOCLING_TEXT / PDF_TEXT_FALLBACK)
    and provenance; unresolvable Docling-vs-fallback disagreements are quality issues,
    never guesses.
14. **No OCR.** The five image-only P179039 ESSA pages stay unparsed, recorded as
    `OCR_NOT_REQUIRED_FOR_CURRENT_SCOPE`; selective OCR only if later analysis shows they
    hold required information. No OCR dependencies in Phase 5.
15. **ISR dates:** preserve both header and archive date; implementation snapshots use a
    documented canonical date policy; material differences are kept as
    provenance/data-quality information.

---

## 2. Architecture summary

Deterministic facts where we can be exact. Retrieval where evidence lives in
documents. Agents only where reasoning is genuinely required. Validation before
trust.

```
 Source files (xlsx / csv / pdf)
        │  ingestion/ (Phase 2)          parsing/ + Docling (Phase 4-5)
        ▼                                        │
  BRONZE  raw + lineage  ───────────────────────┤
        │  transformations/silver (Phase 3)      │ structured extraction
        ▼                                        ▼
  SILVER  projects, loans, procurement, isr_snapshots, project_results,
          project_events, appraisal_risks, document_chunks
        │  transformations/gold + attention_signals (Phase 6)
        ▼
  GOLD    project_360, project_timeline, result_progress, risk_register,
          attention_signals
        │                                   document_chunks → Vector Search (Phase 7)
        ▼                                                   │
  tools/ (typed, read-only, Pydantic)  ◄──  retrieval/ hybrid + rerank + context (Phase 8)
        │
  router ──► STRUCTURED: tool → answer                     (Phase 10)
         ├─► DOCUMENT:   filtered RAG → cited answer        (Phase 10)
         └─► INVESTIGATION: investigator → tools + researcher
                            → synthesis → critic (≤2 retries) (Phase 11)
        │
  guardrails/ (NeMo adapter + deterministic validators, provenance)  (Phase 12)
  observability/ (MLflow tracing)                                    (Phase 13)
        │
  api/ FastAPI (Phase 14)  ──►  frontend/ React Project 360 (Phase 15)
```

### Package layout decision

CLAUDE.md lists `src/common`, `src/tools`, `src/api`, … as top-level packages.
Generic top-level names (`common`, `tools`, `api`, `agents`) collide with
other installed distributions and are fragile when the repo is imported into
Databricks. All subpackages therefore live under one namespaced package:
`src/worldbank_copilot/<subpackage>/`, with the same subpackage and module
names as CLAUDE.md. *(Needs confirmation — §8.)*

Phase 1 creates only the subpackages (each `__init__.py` documents its planned
modules and owning phase). Module files are created in the phase that
implements them — no empty stubs pretending to be implementations.

### Configuration

* `configs/environments/base.yaml` → `<env>.yaml` → `WBC_*` environment variables
  (later wins). `.env` is read for local convenience; real env vars win.
* Environment selection: explicit argument → `WBC_ENV` → auto-detect Databricks
  (`DATABRICKS_RUNTIME_VERSION`) → `local`.
* Unconfigured values are `null`. `Settings.require("x.y")` raises
  `ConfigurationError` naming the env var to set — no silent defaults for
  catalog, endpoints, index or experiment.
* `Settings.table_name(layer, table)` → `<catalog>.<schema>.<table>`.
* `configs/projects.yaml` + `ProjectRegistry` is the single project allow-list.
* `configs/retrieval.yaml` and `configs/models.yaml` (named in CLAUDE.md) will be
  created in Phases 8 / 10–11 when their parameters (K values, rerank K, retry
  count, prompts) exist. Endpoint *names* are environment-specific, so they live
  in the environment config now.

---

## 3. Development phases and dependencies

| # | Phase | Depends on | Main deliverables |
|---|---|---|---|
| 1 | Skeleton & configuration | — | package, config, project registry, tests |
| 2 | Source mapping + Bronze ingestion | 1 | Bronze tables, document inventory, `scripts/validate_data.py` |
| 3 | Silver structured tables | 2 | projects, loans, financial summary, procurement |
| 4 | Docling PDF parsing + classification | 1 | parsed documents, metadata validation |
| 5 | Structured extraction from PDFs | 4 | `isr_snapshots`, `project_results`, `project_events`, `appraisal_risks` |
| 6 | Databricks platformization, governed Delta foundation | 2-5 | Unity Catalog objects, Volumes, Delta tables, contracts, reconciliation, idempotency |
| 7 | Gold analytical/intelligence layer | 6 | `project_360`, `project_timeline`, `result_progress`, `risk_register`, `attention_signals` |
| 8 | RAG foundation + retrieval experiments | 6 | chunking/retrieval/reranking experiments, Vector Search |
| 9 | Structured tools + adaptive query routing experiments | 7, 8 | typed tools, routing baselines |
| 10 | Agent harness + bounded decision experiments | 9 | investigator / researcher / critic |
| 11 | Hybrid memory/cache + advanced guardrails | 10 | scoped memory/cache, layered guardrails |
| 12 | MLflow tracing + comprehensive evaluation | 9-11 | traces, benchmark, evaluation |
| 13 | API + Databricks application | 12 | API, product experience |

The roadmap follows Claude.md §36 (updated 2026-09-30) and may change with evidence.
Phases 3 and 4 can proceed in parallel once Phase 2 is done.

---

## 4. Local vs Databricks responsibilities

| Concern | Local (VS Code) | Databricks |
|---|---|---|
| Business logic | `src/worldbank_copilot/**` (developed & unit-tested here) | same code, imported from the Git folder |
| Notebooks | not executed | thin entry points: `%run ./_bootstrap`, call `src/` functions |
| Raw files | `data/` (git-ignored) | Volume `worldbank_copilot.bronze.sources` (`data/` mirrored) |
| Tables | contract-shaped rows, dry-run reconciliation (`scripts/platformize.py`) | Delta tables `worldbank_copilot.<bronze|silver>.<table>` (`notebooks/05_platformize_databricks.py`) |
| Vector Search, Model Serving, MLflow experiment | not required; tests use fixtures/mocks | real endpoints (names from config) |
| Tests | `pytest` (Databricks-marked tests deselected) | `pytest -m databricks` for workspace integration tests |
| Credentials | never needed to import or unit-test | workspace identity |

Spark access is isolated in `lakehouse/spark_store.py` (lazy `pyspark` import); every
other module is importable and testable without a cluster.

---

## 5. Source data inspection (2026-09-26)

All inspection was read-only. Files are untracked by Git.

### 5.1 Structured files

**`data/all.xlsx`** — World Bank Projects & Operations, "data as of 09/26/2026".
Every sheet has a title banner in row 1 and headers in row 2.

| Sheet | Rows | Columns | Key | Rows for P130544 / P179039 / P506272 |
|---|---|---|---|---|
| World Bank Projects | 28,154 | 25 | `Project ID` | 1 / 1 / 1 |
| Themes | 222,679 | 7 | `Project ID` | 5 / 14 / 37 |
| Sectors | 57,931 | 4 | `Project ID` | 2 / 3 / 5 |
| GEO Locations | 84,472 | 9 | `Project ID` | 2 / 0 / 0 |
| Financers | 32,179 | 7 | **`Project`** | 2 / 2 / 2 |

*World Bank Projects* has a **second header row** (row 3) of API field names
(`id`, `regionname`, `closingdate`, …) before the data starts in row 4.
Columns: Project ID, Region, Country, Project Status, Last Stage Reached Name,
Project Name, Project Development Objective (note trailing space), Implementing
Agency, Public Disclosure Date, Board Approval Date, Loan Effective Date,
Project Closing Date, Current Project Cost, IBRD Commitment, IDA Commitment,
Grant Amount, Total IBRD, IDA and Grant Commitment, Borrower, Lending
Instrument, Environmental Assessment Category, Environmental and Social Risk,
Associated Project, Consultant Services Required, Last Update Date, Financing Type.

Other sheets: *Themes* (Project ID, Level 1, Percentage 1, Level 2, Percentage 2,
Level 3, Percentage 3 — hierarchical rows prefixed with `> `); *Sectors*
(Project ID, Major Sector, Sector, Sector Percent); *GEO Locations* (Project ID,
GEO Loc ID, Place ID, WBG Country Key, GEO Loc Name, GEO Latitude Number, GEO
Longitude Number, Admin Unit1 Name, Admin Unit2 Name); *Financers* (Project,
Name, Current Amount, Amount (USD), Financer ID, Currency, Project Financial Type).

**`data/ibrd_statement_of_loans_and_guarantees_latest_available_snapshot_09-26-2026.csv`**
— 9,518 rows × 35 columns; `End of Period` = 08/31/2026 for the rows inspected.
Columns: End of Period, Loan Number, Region, Country / Economy Code, Country /
Economy, Borrower, Guarantor Country / Economy Code, Guarantor, Loan Type, Loan
Status, Interest Rate, Currency of Commitment, Project ID, Project Name, Original
Principal Amount (US$), Cancelled Amount (US$), Undisbursed Amount (US$),
Disbursed Amount (US$), Repaid to IBRD (US$), Due to IBRD (US$), Exchange
Adjustment (US$), Borrower's Obligation (US$), Sold 3rd Party (US$), Repaid 3rd
Party (US$), Due 3rd Party (US$), Loans Held (US$), First Repayment Date, Last
Repayment Date, Agreement Signing Date, Board Approval Date, Effective Date (Most
Recent), Closed Date (Most Recent), Last Disbursement Date, Board approval -
Fiscal year, Board approval - Calendar year.

| Project | Loan | Status | Original | Cancelled | Disbursed | Undisbursed | Closed (most recent) |
|---|---|---|---|---|---|---|---|
| P130544 | IBRD86010 | Repaying | 100,000,000 | 0 | 100,000,000 | 0 | 11/22/2024 |
| P130544 | IBRD93240 | Disbursing&Repaying | 150,000,000 | 24,937,499.48 | 32,713,450.09 | 92,871,483.50 | 09/30/2027 |
| P179039 | IBRD94960 | Disbursing&Repaying | 363,000,000 | 0 | 220,544,000 | 142,456,000 | 06/01/2028 |
| P506272 | IBRD98350 | Disbursing | 426,000,000 | 0 | 131,961,818.05 | 252,260,260.43 | 12/31/2030 |

**`data/contract_awards_in_investment_project_financing_india_projects_09-26-2026.csv`**
— 35,253 rows × 21 columns. Columns: As of Date, Fiscal Year, Region, Borrower
Country / Economy, Borrower Country / Economy Code, Project ID, Project Name,
Procurement Category, Procurement Method, Project Global Practice, WB Contract
Number, Contract Description, Contract Signing Date, Supplier, Supplier Country /
Economy, Supplier Country / Economy Code, Supplier Contract Amount (USD),
Borrower Contract Reference Number, Contract signed - Calendar year, Review
type, Supplier ID. **P130544: 17 rows; P179039: 0; P506272: 0** (as expected
for PforR operations).

### 5.2 Documents (53 PDFs, all text-extractable, none encrypted)

*(Phase 1 reported 57 and 31 for P130544 — a counting error. Correct figures:
P130544 32, P179039 13, P506272 8 = 53.)*

Filenames are mostly opaque hashes, so document type, date and ISR sequence must
come from content, not filenames. Identified from first-page text:

**P130544 (32 PDFs)**
* ISRs: **sequence 1–24, complete** (Jun 2016 → Aug 2026).
* PAD (Report PAD440, 10 Mar 2016).
* Additional-financing Project Paper (Report PAD4503, 22 Nov 2021, US$150M).
* Restructuring papers: RES33572, RES01944, RES00289 (restructuring of the AF
  loan), RES00355. Their dates are not on page 1; to be extracted in Phase 4–5.
* Loan Agreement 8601-IN (2016); Supplemental Letter No. 2 — Performance
  Monitoring Indicators (24 May 2016).

**P179039 (13 PDFs)**
* ISRs: **sequence 1–8, complete** (Jun 2023 → Feb 2026).
* Program Appraisal Document (PAD5226, 3 Mar 2023); Technical Assessment (2 Mar
  2023); Integrated Fiduciary Systems Assessment (Feb 2023); ESSA (20 Jan 2023);
  Supplemental Letter — Performance Monitoring Indicators (5 Jun 2023).
* No loan agreement, restructuring or AF documents present.

**P506272 (8 PDFs)**
* ISRs: **sequence 1–3, complete** (Sep 2025 → Sep 2026).
* Program Appraisal Document (PADHP00139, 30 May 2025 — loan of JPY 60,954,300,000,
  US$426M equivalent); Technical Assessment (Mar 2025); Integrated Fiduciary
  Systems Assessment (11 Mar 2025); ESSA (Apr 2025); Loan Agreement 9835-IN.

---

## 6. Differences between CLAUDE.md and the actual data

1. **File locations.** Structured files are in `data/` (not `data/structured/`);
   PDFs are in `data/<PID>/` (not `data/documents/<PID>/`). Handled by
   configuration (`structured_subdir: "."`, `documents_subdir: "."`); files were
   not moved.
2. **No original vs current closing date in structured data.** The Projects sheet
   has one `Project Closing Date`; the loans file has `Closed Date (Most Recent)`.
   Original closing dates must come from PADs / restructuring papers (Phase 5).
   The `SCHEDULE_CHANGE` signal therefore depends on document extraction.
3. **P130544 has two loans** (original IBRD86010 and AF IBRD93240). Project-level
   finance must aggregate across loans; the AF loan carries the cancellation.
4. **Workbook commitment for P130544 is stale/partial:** IBRD Commitment =
   100,000,000 and Financers IBRD = 100,000,000, while the loans snapshot shows
   250,000,000 original principal across two loans. The workbook `Last Update
   Date` for P130544 is 2024-04-04. The loans snapshot must be authoritative for
   finance; this discrepancy must be surfaced, not silently resolved.
5. **Loan number formats differ:** `IBRD86010` (CSV) vs `8601-IN` (documents).
   A deterministic normaliser is needed to link them.
6. **No `state` column.** "Karnataka" appears only in names/agency text; GEO
   Locations has rows only for P130544 (Hubli, Dharwad) with empty admin units.
7. **No `current_principal` field.** Available instead: Original Principal,
   Cancelled, Disbursed, Undisbursed, Exchange Adjustment, Borrower's Obligation.
8. **`Currency of Commitment` is empty for every row.** P506272 is JPY-denominated
   (per PAD); US$ values move with FX (Exchange Adjustment −2,613,739.59).
9. **Snapshot date:** `End of Period` (08/31/2026) differs from the download
   date in the filename (09-26-2026); `End of Period` is the correct `snapshot_date`.
10. **Mixed date formats:** workbook `YYYY-MM-DD` and `YYYY-MM-DDT00:00:00Z`;
    CSVs `MM/DD/YYYY`; PDFs `M/D/YYYY`, `Mon DD, YYYY`, `DD-Mon-YYYY`.
11. **Header quirks:** title banner row; second header row on the Projects sheet;
    `Financers` uses `Project` instead of `Project ID`; trailing space in
    `Project Development Objective `; P130544 PDO contains HTML (`<p>…</p>`).
12. **Instrument naming drift:** P130544 early ISRs say "Specific Investment Loan",
    later ones "Investment Project Financing (IPF)".
13. P130544 is described as a "mature historical-validation case", but it is
    **still Active** (AF loan closes 30 Sep 2027; latest ISR Aug 2026).
14. The repo root is `project-investigator-agent/`, not
    `worldbank-implementation-intelligence/`; the structure was created at the
    repo root.

### Additional source issues found in Phase 2

15. **Loan-number suffix is significant.** In the full snapshot, 425 of the 8,121
    `IBRD#####` numbers end in a non-zero digit, and several bases have more than
    one suffix (e.g. `IBRD13440` / `IBRD13445`). `8601-IN` therefore cannot be
    turned into `IBRD86010` by rule; it is linked only when exactly one loan of
    the same project shares the base number. The snapshot also contains
    `IBRD####A`-shaped (1,221) and `AAAAA####`-shaped (176) numbers; they are
    reported as unrecognised if they ever occur in scope.
16. **Financers sheet, P130544 borrower row:** Current Amount 100,000,000 but
    Amount (USD) 53,000,000 with Currency `USD`. (53M + 100M IBRD = the 153M
    Current Project Cost.) Both values preserved; flagged `FINANCER_AMOUNT_FIELDS_DIFFER`.
17. **Duplicate WB contract number 1657297 (P130544):** two rows, two different
    suppliers, the same amount (35,729.02) each. Summing naively in Silver would
    double count.
18. **Whitespace in values:** Currency `'USD  '` (P179039 financers), leading
    space and doubled spaces in theme labels (`' Water Institutions…'`,
    `'>  Water Institutions…'`), trailing space in a sector name. Preserved in Bronze.
19. **Blank encodings differ by source:** the workbook uses empty strings for
    some blanks (P506272 Current Project Cost `''`), empty Excel cells read as
    null; CSV blanks are `''`. Bronze keeps what the source yields.
20. **Excel read cost:** the full workbook (~28k projects, ~223k theme rows) takes
    ~35–40 s to stream with openpyxl read-only mode.
21. **Procurement amounts are contract values:** the three P130544 DBOMT works
    contracts alone total ≈ US$575M, above the US$250M IBRD principal, so contract
    amounts must not be read as Bank-financed amounts.
22. **PDF count correction:** 53 PDFs (not 57, as reported in Phase 1).
23. **Dates not on page 1:** the four P130544 restructuring papers and both loan
    agreements (OCR-garbled / "Signature Date") have no reliable page-1 date; month-only
    dates (P179039 IFSA, P506272 TA, P506272 ESSA) are left null. Phase 4 must extract them.

### Additional source issues found in Phase 3

24. **Loan principal identity does not hold generally.** In the full snapshot,
    `original = disbursed + undisbursed + cancelled` fails for 1,110 of 9,518 loans
    (737 with a non-zero exchange adjustment, 373 without). The identity
    `Borrower's Obligation = Due to IBRD + Exchange Adjustment` holds for all 9,518.
    In scope: IBRD98350 (P506272) components are 41,777,921.52 below original;
    **IBRD93240 (P130544 AF) components exceed original by 522,433.07**, and it also has
    a non-zero exchange adjustment (53,527.90). The FX caveat therefore applies to
    P130544's AF loan too, not only P506272.
25. **Multi-supplier contracts are common and structurally consistent.** India dataset:
    336 contract numbers have >1 row; supplier IDs are always distinct (no exact
    duplicate rows); award-level fields (description, signing date, category, method,
    fiscal year, review type, borrower reference) never differ within a contract.
    275 repeat an identical amount per supplier row, 61 have different amounts. The
    dataset does not say whether a row amount is the whole award or a supplier share.
26. **Borrower contract reference formats vary** (`IN KUIDFC104978` vs
    `IN-KUIDFC-97508-CW-RFB`); treated as free text, not as a strict identifier.
27. **Two sector/theme taxonomies.** P130544 and P179039 names carry an `FY17 - `
    prefix; P506272's do not (a different/newer taxonomy). Names are not merged
    across taxonomies; Silver records `taxonomy_label` (`FY17` or NULL).
28. **The workbook has no country code**; `country_code` comes from the loan snapshot
    (all loans agree on `IN`) and is labelled with its source.

### Additional document issues found in Phase 4

29. **Docling drops table cells** during table-structure matching in 32 documents:
    most in early P130544 ISRs (Seq 2–6: 60–72 cells each) and the AF paper (73 cells).
    Measured content loss is small overall (coverage ≥ 95.4%) but concentrated on
    results-framework pages (e.g. indicator-name fragments such as "people with…",
    "supporting utilities"); 12 pages fall below 85% coverage.
30. **Text inside picture regions**: 1,202 blocks were only reachable by traversing
    picture children (headings/titles in ISRs and PADs).
31. **ISR header date ≠ archive date** in 4 ISRs: P130544 Seq 8 (6/14/2019 vs
    21-Feb-2019), Seq 21 (2025-01-16 vs 2024-12-08); P179039 Seq 5 (2025-05-31 vs
    2024-09-11, ~8.5 months apart) and Seq 7 (1 day). Both dates are kept.
32. **Image-only pages**: P179039 ESSA pp. 123–127 contain pictures and no text layer
    (OCR would be needed); other empty pages are blank separators (PAD440 pp. 5, 15;
    AF pp. 42, 56, 65; P179039 IFSA pp. 20, 52; P130544 ISR 20 p. 8).
33. **No document date** can be read reliably from the 4 restructuring papers, the 2 loan
    agreements, or 3 month-only covers (P179039 IFSA, P506272 TA and ESSA).
34. **Restructuring papers carry the P130544 closing-date history in tables** (e.g.
    RES33572: original 30-Nov-2022 → proposed 22-Nov-2024; RES01944: IBRD-93240
    30-Jun-2026 → 30-Sep-2027), now captured as page-cited candidates for Phase 5.

### Additional document issues found in Phase 5

35. **ISR indicator IDs (`IN########`) are report-specific.** The same indicator gets a
    new ID in every ISR (e.g. ISR 18 prints sequential IDs IN013427xx). They are kept per
    observation but cannot link indicators across ISRs.
36. **Docling misplaces indicator IDs** in old-layout (BLOCK) results tables: the next
    indicator's ID lands in the previous indicator's Value/Date row. Docling-table IDs
    are now accepted only from the name row or its header rows.
37. **Indicator names change across ISR templates** ("(Number, Core)" → "(Number,
    Corporate)" → "(Number)"; "DLI"/"CRI" tags; DLI status words such as "Not Due"
    printed after the unit). Tags and statuses are split into `indicator_tags` /
    `reported_status`. Remaining near-identical names are reported as
    POSSIBLE_INDICATOR_MATCH observations, not merged (96 pairs; see issue 47).
38. **Restructuring papers are undated** (all four P130544 papers). Their events keep
    `event_date = NULL`. A deterministic candidate date is reported for each (§8).
39. **P179039 ISR 5 header date (2025-05-31) is later than ISR 6 (2025-03-12).** With
    the header-first policy, canonical dates are not monotonic in sequence order;
    flagged as ISR_DATE_DIFFERENCE (WARNING). Sequence order is used for ordering.
40. **P130544 ISR 15/16 key-dates header is garbled** ("Orig. Closing Date Rev."
    spans two cells) and the text layer breaks the IBRD93240 row over several lines.
    Those two rows are left unmapped (EXTRACTION_AMBIGUOUS).
41. **Loan currencies:** IBRD93240 (P130544 AF) was converted from USD to JPY in July
    2024 (RES00289); its cancellation is printed in JPY (4,011,633,250, value date
    06-Aug-2024). It is not comparable with the snapshot's USD cancelled amount.
42. **"Historical Disbursed" column** (newer ISRs) has project-dependent meaning:
    0.00 for P130544 loans, equal to "Disbursed" for P179039, and equal to the loan
    snapshot's disbursed amount for P506272. It is stored as printed; no interpretation.
43. **AF paper "Approval Date" (04-Jan-2022)** differs from the snapshot's board
    approval for IBRD93240 (21-Dec-2021). It is the date printed in the paper, labelled
    as such, and reported as CROSS_SOURCE_DIFFERENCE.
44. **SORT tables vary by template:** P130544 ISRs 20–24 print 9 categories (10
    before); P179039 ISRs 5–7 print 7 and ISR 8 prints 9. Extracted as printed (INFO).
45. **ISRs contain one narrative section** ("Implementation Status and Key
    Decisions"). No ISR has a separate Key Issues section, so `key_issues_text` is NULL.
46. **Docling cell merges in WIDE results rows** (objective text or result-area labels
    merged into value cells) leave 16 of 837 observations unvalidated (NULL values,
    AMBIGUOUS). Two old-layout indicators have Docling-vs-text disagreements (NULL,
    EXTRACTION_CONFLICT).

### Additional issues found in Phase 6

47. **Phase 5 quality report undercounted indicator candidates.** It deduplicated
    observations on (check, message, source); different pairs anchored on the same
    page share both, so 96 distinct POSSIBLE_INDICATOR_MATCH pairs appeared as 51.
    Fixed (details are part of the identity; regression test added). The 51 figure
    in the Phase 5 report is superseded by 96 (P130544 62, P179039 20, P506272 14).
48. **Silver lineage embedded a random ingestion run id** (`source_refs`), which would
    make every reload look like a content change. Phase 6 derives the ingestion run id
    from the source snapshot (`snapshot-<id>`), so lineage is stable across reloads and
    environments; the Bronze `_ingested_at` timestamp is operational metadata.
49. **Phase 3 named the repository tables with a layer prefix** (`silver_loans`); in
    Unity Catalog the schema is the layer, so Delta names drop the prefix
    (`silver.loans`). Documented mapping; no other renaming.
50. **Phase 5 WIDE-layout comments:** for some newer-ISR results rows `comments` holds
    the label "Comments on achieving targets" instead of the comment text (Docling
    repeats the label across the comment row). Not changed in Phase 6 (no semantic
    changes during platformization); to fix in a later extraction revision.

### Additional issues found in Phase 7

51. **DLI table rows are not results-framework progress.** Disbursement-Linked
    Indicator tables print DLI achievement/allocation columns (e.g. P179039
    tap connections: target 0.00, and a "current" of 2,000,000 in ISR 7 between
    6,666,030 and 7,648,665 in ISRs 6 and 8, a column misalignment). Gold marks all 114
    DLI rows `DLI_LAYOUT_NOT_EVALUATED` (no progress, trend or signals).
52. **Printed target dates before project effectiveness** (6 rows, P506272, e.g.
    "Household sewer connections" target due Jun/2025 for a program effective
    2025-09-15). Kept as printed; excluded from RESULT_TARGET_DATE_PASSED_NOT_MET;
    WARNING in gold.quality_observations.
53. **Negative progress as printed** (current on the opposite side of the baseline from
    the target, e.g. P130544 "Of which, people provided with safely managed water":
    baseline 254,700, current 51,251). Possibly a baseline copied from a parent
    indicator; reported as INFO (RESULTS_NEGATIVE_PROGRESS) for review, not changed.
54. **Some indicator units contradict values** (e.g. "Female beneficiaries (Percentage)"
    reporting 21,020). Values are used as printed; no unit is re-inferred.
55. **"Comments on achieving targets" label:** 89 Silver result rows (P130544 ISRs 20,
    22-24; P179039 ISRs 7-8; P506272 ISRs 1-3) hold the label instead of the comment
    text. No Gold calculation uses comments. A deterministic fix exists (skip label
    cells in the WIDE comment row, same row provenance) but is not applied in Phase 7
    because it changes the validated Silver foundation; see §8.
56. **Local Spark on this Windows machine cannot run Python workers** (PySpark 3.5 and
    4.0, Python 3.12). Gold therefore uses only JVM-native Spark functions (also the
    better choice for Photon/serverless); local Spark tests create DataFrames from JSON
    with the contract schemas.

### Additional issues found in Phase 8

57. **Comments fix applied (resolves 55).** `extraction/results.comment_cell` skips the
    repeated label cells and takes the merged comment text.
    - Silver: 89 rows change, in `comments` only; every other Silver table's fingerprint is unchanged.
    - `configs/reconciliation/expected_profiles.json` changes by one line.
    - Gold (local Spark): all 5 business-table fingerprints are unchanged; only
      `gold.quality_observations` changes (SILVER_COMMENTS_LABEL_EXTRACTION 89 -> 0).
    - One comment row split into word-interleaved column fragments (P130544 ISR 20,
      revenue collection) is stored as NULL rather than rebuilt; that row is not in
      Silver, so no data changes.
58. **Template markers.** Operations-portal markers (`@#&OPS~...`) in restructuring
    papers are excluded from chunks.
59. **Rating glyphs.** Table cells contain rating glyphs (e.g. U+F06C, U+26AB).
    Evaluation relevance compares alphanumeric tokens only.
60. **Embedding input limit (corrected).** databricks-gte-large-en has an 8192-token
    window (Databricks supported-models page), not 512; chunks of 1,200-1,500
    characters are far below it, so no truncation is expected.
61. **Local lexical baseline** (BM25 on the real corpus, no reranker, k=10): Recall@10
    0.75–0.80 across strategies. Every miss is a paraphrase question; dense, hybrid and
    reranked results exist only after the Databricks run.
62. **Dependency fix (notebook 07 install failure).**
    - **Root cause:** `databricks-vectorsearch` 0.75 declares `protobuf<6.0,>=5.29.5`. Installing it notebook-scoped replaced the runtime's protobuf 6.33.5 with 5.29.6, breaking `grpcio-status` 1.76.0 (needs protobuf `>=6.31.1,<7`). Environment 6 ships `googleapis-common-protos` 1.71.0, which accepts protobuf `<7` and was not affected. The additional `googleapis-common-protos` conflict seen in the local reproduction came from the locally resolved 1.75.5, not from Databricks.
    - **Reproduced locally:** in a Python 3.12 environment seeded with serverless environment 6 versions.
    - **Fix:** migrated the adapter to `databricks-ai-search==0.78` (same client and index API, typed `NotFound`); `databricks-sdk` is no longer installed (runtime 0.122.0 is used); constraints protect protobuf and grpcio-status; the reranker moved to notebook 07b (ML base); a dependency-health check runs after every install.
63. **Embedding build timeout (notebook 07 Step 3).**
    - **Error:** `Timed out after 0:05:00` after 10,965 texts were queued.
    - **Root cause:** the Databricks SDK. `serving_endpoints.query` runs inside the SDK's `retried(timeout=retry_timeout_seconds)` loop (default 300 s = `0:05:00`) with a 60 s per-attempt HTTP timeout, and keeps re-sending a request that times out or is throttled (Retry-After) until 5 minutes pass. Our loop then retried that whole request 3 more times (up to about 20 minutes per request).
    - **Contributing factors:** the configured `timeout_seconds` (120) was never wired in; requests carried 64 long texts and ran strictly one at a time; results were persisted only after each 640-text slice, so nothing was saved.
    - **Fix:** an explicit HTTP transport (one POST per attempt with our own timeout; auth from the SDK config); bounded requests (16 inputs, 24,000 characters); classified retries (timeout, connection, 429, 5xx) with exponential backoff, jitter and Retry-After; validation before caching; insert-only checkpoints every 25 requests and before any error; resume through the existing (text_sha256, model) cache.
64. **Embedding rate limiting (HTTP 429 REQUEST_LIMIT_EXCEEDED, workspace QPS).**
    - **What happened:** the first Step 3 request was throttled; all 5 transient-style retries (about 1 minute) were used up; nothing was cached; the job stayed resumable.
    - **Not known:** whether that 429 carried Retry-After, because it was not logged.
    - **Fix:** 429 now has its own policy:
      - paced sequential requests (>= 3 s between starts, doubling to 30 s after each 429, recovering x0.9 per success; the probe shares the pacing state);
      - Retry-After honoured in full;
      - otherwise a 60/120/240/300 s cooldown;
      - at most 4 rate-limited retries per request and 20 minutes of rate-limit waiting per run, then a clean, checkpointed stop.
    - **Logging:** each 429 logs status, Databricks error code, Retry-After, the chosen sleep, the attempt number and the cumulative wait.
65. **GTE kept as an experimental failure; Qwen load probe.**
    - **GTE result:** `databricks-gte-large-en` failed in this workspace: 5 rate-limited attempts and 720 s of cooldown, 0 texts embedded. Recorded in `configs/retrieval/embedding_candidates.yaml`; the cooldown is not increased further.
    - **Qwen probe:** `notebooks/07a_embedding_endpoint_probe.py` probes `databricks-qwen3-embedding-0-6b`: 1 input, a batch of 4, a 15 s pause, a batch of 4, each one attempt with no retries. It reports status, error code, Retry-After, latency, the returned dimension and a verdict.
    - **No writes:** the probe writes nothing, and production `embeddings.yaml` stays unchanged until a separate decision.
66. **Qwen selected as the embedding candidate.**
    - **Probe:** `databricks-qwen3-embedding-0-6b` probe (notebook 07a): USABLE, dimension 1024, no 429.
    - **07a reporting error, explained:** `[CANNOT_DETERMINE_TYPE]` came from schema inference. On a fully successful probe, `error_code`, `retry_after` and `error` are NULL in every row (reproduced on local Spark); fixed with an explicit result schema.
    - **Configuration:** `embeddings.yaml` now points to Qwen (1024). Settings are conservative because only 4-input requests were proven: 4 inputs and <= 6,000 characters per request, sequential, >= 2 s between requests, the same 429/retry protections, checkpoints every 50 requests.
    - **Pilot:** notebook 07 Step 3a stops cleanly after PILOT_REQUESTS = 50 (200 texts) for review.
    - **Model isolation:** cache key (text_sha256, embedding_model). Cache lookup and index source filter on model and dimension; a post-MERGE check allows no foreign-model rows; an existing index with another dimension is refused.
    - **GTE:** the failure history stays in `embedding_candidates.yaml`.
67. **AI Search endpoint quota (Step 3b).**
    - **What happened:** creating `worldbank-copilot-vs` failed with "Maximum number of AI Search endpoints per workspace exceeded quota of 1" (documented limit: 500).
    - **Already done before the failure:** the index-source table had been built (12,797 rows), and the Qwen embedding cache was complete (10,965 texts; 0 retries, 0 failures, 0 rate limits).
    - **Change:**
      - reuse the existing ONLINE endpoint `worldbank-gep-ai-search` (configuration, `create_endpoint: false`) with a new project index `worldbank_copilot.silver.document_chunk_index_qwen3_v1` on `worldbank_copilot.silver.document_chunk_index`;
      - a read-only pre-flight (endpoint ONLINE, existing indexes listed, no name collision, project namespace and source table, Qwen model, dimension 1024, no foreign-model rows, full coverage);
      - a quota error at index creation stops cleanly.
    - **GEP:** the existing indexes (`worldbank_ai.rag.*`) are never modified or deleted.

---

## 7. Assumptions

* Python 3.11+ (local venv built with Python 3.12.4 from Anaconda).
* The three files and 57 PDFs listed above are the full V1 corpus.
* Source files are read-only; the pipeline never writes into `data/`.
* The loans snapshot is the authoritative source for disbursement/cancellation.
* Bronze/Silver/Gold schema names default to `bronze` / `silver` / `gold`;
  the catalog has no default.
* Notebooks run from a Databricks Git folder where the working directory is
  `notebooks/` (used by `_bootstrap` to put `src/` on `sys.path`).

---

## 8. Unresolved issues / decisions needed

Resolved 2026-09-26 (see §1a): package namespace, data layout, finance
source-of-truth policy.

Open:

1. **Endpoint names:** Vector Search endpoint/index, embedding and LLM endpoints,
   MLflow experiment path stay unset (§1a.6). Catalog/schemas/Volumes: decided (§1a.16).
2. **Databricks runtime / dependency install:** Phase 6 proposes DBR 15.4 LTS+ with the
   notebook-scoped, pinned `requirements-databricks.txt`; to be confirmed by the first
   real run in the workspace.
3. **[RESOLVED in Phase 8] Lexical retrieval implementation:** in-process BM25 fused
   with dense results by RRF (`hybrid`) won the retrieval stage in 07b (+0.023
   recall@10 over dense; see the Phase 8 07b results).
10. **Adaptive reranking trigger (Phase 8 decision, open design):** the selected direction
    is adaptive reranking, not "rerank every query". Which queries are reranked is
    not yet defined, implemented or evaluated. It must be measured against the two
    07b baselines: never rerank (`fixed|hybrid|none|k10`) and always rerank
    (`fixed|hybrid|cross_encoder|k50`).
11. **P179039 retrieval quality (Phase 8):** recall@10 is 0.545 for P179039
    (6 of 11 questions), against 0.917 for P130544 and 1.000 for P506272. Five of the
    seven misses of the selected configuration are P179039 questions. The cause is
    not established; see the Phase 8 07b results.
12. **Abstention threshold (Phase 8):** `min_rerank_score` stays `null`. The 07b
    calibration was computed on the k20 CrossEncoder run, not the selected k50, and the
    score ranges overlap. Review it before any threshold is applied.
4. **JPY loan (P506272):** US$ figures fluctuate with FX; decide in Silver/tools
   whether to show the exchange adjustment explicitly. Bronze only preserves source values.
5. **Restructuring paper dates (Phase 5):** each undated P130544 paper has one
   deterministic candidate ISR restructuring date: the earliest ISR "Restructuring
   History" approval on or after the latest "dated/on <date>" the paper cites.
   RES33572 → 20-May-2021, RES00289 → 23-Jul-2024, RES00355 → 10-Dec-2024,
   RES01944 → 29-Jun-2026. These are reported only, not applied. Decide whether Gold
   may link paper content to these dated restructurings.
6. **Canonical ISR date for P179039 ISR 5 (Phase 5):** keep the header-first policy
   (canonical 2025-05-31, out of sequence order) or use the archive date for this
   case. Currently: header-first, flagged; timelines should order ISRs by sequence.
8. **[RESOLVED in Phase 8] Comments label fix (issue 55):** apply the deterministic WIDE
   comment-row fix in extraction and reload Silver through the Phase 6 notebook (89 rows
   change; expected profiles must be regenerated) before comments are used for RAG
   or display (Phase 8).
9. **Attention thresholds (Phase 7):** schedule (12 / 24 months), rating runs (3 / 6
   ISRs), disbursement gap (25 / 40 points, IPF only) and results (HIGH below 50%) are
   project-team review markers, not World Bank policy; confirm or adjust in
   `configs/intelligence/attention_rules.yaml`.
7. **Indicator aliases (Phase 5/6):** `configs/indicator_aliases.yaml` is empty. The 96
   candidate pairs in `review/indicator_alias_candidates.csv` (issue 47) (mostly unit/type-suffix changes across ISR
   templates) need review before Gold result-progress series are built across templates.

---

## 9. Phase log

### Phase 1 — Repository foundation (complete)

Created: `pyproject.toml`, `.gitignore`, `.env.example`, `configs/` (environment
configs, `projects.yaml`, `guardrails/`), `src/worldbank_copilot/` package
skeleton with implemented `common/` (config, project registry, logging,
exceptions), thin notebooks (`_bootstrap` + 01–08 raising
`NotImplementedError`), unit tests (imports, config loading, environment
selection, project configuration), README, this plan.

Verification: `pytest` → 72 passed; `ruff check src tests` → clean; notebook
bootstrap exercised locally from `notebooks/` (loads settings and the three projects).

### Phase 2 — Structured Bronze ingestion (complete)

Flow: **SOURCE → source contract → Bronze source-preserving table** (no Silver).

**Modules**

| Module | Responsibility |
|---|---|
| `common/dates.py` | Parse dates only with formats a contract declares (no guessing). |
| `common/identifiers.py` | Loan-number parsing/normalisation and project-scoped linking. |
| `ingestion/contracts.py` | Source contracts: header aliases → Bronze fields, required flags, kinds, date formats, known gaps. |
| `ingestion/tabular.py` | Header detection (title rows, API-name row), whitespace-tolerant mapping, streaming project filter, raw-text preservation. |
| `ingestion/projects.py` | Workbook → 5 Bronze tables (one per sheet). |
| `ingestion/loans.py` | Loans CSV → `bronze_loans_raw` + `normalized_loan_number`. |
| `ingestion/procurement.py` | Contract awards → `bronze_procurement_raw`; coverage semantics. |
| `ingestion/documents.py` | Document inventory, filename rules + manifest, ISR completeness. |
| `ingestion/data_quality.py` | Typed `Observation` / `DataQualityReport`. |
| `ingestion/pipeline.py` | Source-file resolution (glob must match exactly one file), orchestration, `write_bronze`. |
| `ingestion/validation_report.py` | Text and JSON renderings. |
| `transformations/bronze.py` | `BronzeTable`, `SourceMetadata`, `BronzeWriter` protocol, `LocalJsonlBronzeWriter`. |
| `configs/document_manifest.yaml` | Curated labels for the 53 known PDFs (page-1 evidence, size guard). |
| `scripts/validate_data.py` | CLI: text report, `--json`, `--write-bronze`; exit 1 on ERROR. |

**Bronze conventions.** Every record carries lineage (`_source_file`, `_source_sheet`,
`_source_row`, `_ingested_at`, `_ingestion_run_id`), the canonical `project_id`, the
raw source key (`source_project_id`), every contracted field as raw text (no trimming,
casting or HTML removal; numeric Excel cells use Python's round-trip text, e.g.
`100000000.0`), and `_extra_fields` for any uncontracted column. Table metadata records
header row, API-name row, header→field mapping, API names, missing optional /
unmapped columns, known gaps and row counts.

**Bronze tables**

| Table | Grain | Key fields beyond lineage |
|---|---|---|
| `bronze_projects_raw` | 1 row per project (sheet row) | 25 workbook columns |
| `bronze_themes_raw` | 1 row per theme row | level_1..3, percentage_1..3 |
| `bronze_sectors_raw` | 1 row per sector | major_sector, sector, sector_percent |
| `bronze_geo_locations_raw` | 1 row per location | geo_loc_id, name, lat/long, admin units |
| `bronze_financers_raw` | 1 row per financer | financer_name/id, current_amount, amount_usd, currency |
| `bronze_loans_raw` | 1 row per **loan** (1:N per project) | raw_loan_number, normalized_loan_number, 33 snapshot columns incl. end_of_period |
| `bronze_procurement_raw` | 1 row per contract-award row | 21 dataset columns |
| `bronze_procurement_coverage` | 1 row per registered project | covered_by_dataset, coverage_status, row_count, interpretation |
| `bronze_document_inventory` | 1 row per file | document_id, relative_path, size, sha256, document_type, classification_method, isr_sequence, document_date + date_basis, report_number, raw/normalized loan number, manifest_status, filename-derived values, conflicts |

**Decisions made in Phase 2**

* `normalized_loan_number` keeps every component the source provides:
  `IBRD86010` → `IBRD-8601-0`, `8601-IN` → `8601-IN`. Linking uses the base number
  (`8601`) **within one project only** and succeeds only with exactly one candidate
  (otherwise `AMBIGUOUS` / `NO_MATCH`); unrecognised notations are never corrected.
* Document classification: filename patterns + curated manifest. When both give a value
  and disagree, the field is left empty and a `CLASSIFICATION_CONFLICT` is reported.
  A manifest entry is ignored if the file size changed. ISR `document_date` uses the
  **archive date** (`date_basis = isr_archived_date`); the ISR header date can differ
  (e.g. Seq 8: header 6/14/2019, archived 21-Feb-2019).
* Procurement coverage statuses: `RECORDS_PRESENT`, `NO_RECORDS_IN_DATASET`,
  `NOT_COVERED_BY_THIS_DATASET`; zero PforR rows produce no warning.
* `expected_isr_count` added to `configs/projects.yaml` (24 / 8 / 3 as of 2026-09-26).
* The Delta `BronzeWriter` is not implemented (waits on §1a.7); notebook 01 runs
  ingestion + report and stops explicitly before the write.
* Default `pytest` runs unit tests only; `pytest -m integration` checks the real files
  (skipped when absent) and asserts their SHA-256 is unchanged by ingestion.

**Results on the real files:** see §5 and §6 (items 15–23). All three projects present
once; loans 2 / 1 / 1; procurement 17 / not covered / not covered; documents 32 / 13 / 8;
ISR 24/24, 8/8, 3/3 complete; all three document loan references link uniquely
(`8601-IN`→IBRD86010, `9496-IN`→IBRD94960, `9835-IN`→IBRD98350).

**Open items for later phases:** Silver reconciliation of the duplicate contract rows;
workbook-vs-loan commitment presentation under the source-of-truth policy; restructuring
paper / loan agreement dates (Phase 4); Delta writer (after runtime decision).
*(The first two were resolved in Phase 3.)*

### Phase 3 — Silver structured transformations (complete)

Flow: **BRONZE (what the source said) → SILVER (what the value means)**. Bronze is read,
never modified (tested with deep-copy comparison and on the real files).

**Modules**

| Module | Responsibility |
|---|---|
| `common/quality.py` | Shared `Severity` / `CheckCode` / `Observation` / `DataQualityReport` (Bronze + Silver); `ingestion.data_quality` re-exports them. |
| `transformations/normalize.py` | Blank→NULL, whitespace, HTML removal, categories, identifiers, exact `Decimal` money, years, declared-format dates, taxonomy label, percentages. Malformed values raise `NormalizationError`; never coerced. |
| `transformations/silver_models.py` | Typed Pydantic row schemas; `SourceRef`, `ValueIssue`. |
| `transformations/silver.py` | Transforms, `RowNormalizer`, `SilverTable` / `SilverResult`, `LocalJsonlSilverWriter`, `known_award_count`. |
| `transformations/silver_lineage.py` | Field-level lineage (`FIELD_LINEAGE`, `explain()`, lineage table). |
| `transformations/silver_quality.py` | Silver validation. |
| `transformations/silver_report.py` | Text/JSON rendering. |
| `scripts/validate_data.py` | Now Bronze → Silver → Silver validation (`--layer`, `--write-silver`). |

**Silver tables**

| Table | Grain | Notes |
|---|---|---|
| `silver_projects` | 1 per project (3) | Workbook metadata; typed dates; PDO HTML stripped; `state` and `original_closing_date` always NULL; `workbook_*_usd` commitments kept as-is; `country_code` from loan snapshot (labelled); `sectors`/`major_sectors`/`themes` lists + `primary_sector` (strict max, NULL on tie). |
| `silver_project_sectors` | 1 per sector row | `sector_percent` Decimal, `taxonomy_label`. |
| `silver_project_themes` | 1 per distinct hierarchy node | Level 1–3; `>` ancestor markers removed; path, parent, percentage; structure/conflict issues recorded. |
| `silver_loans` | 1 per loan (4) | All US$ amounts as Decimal; typed dates; `loan_base_number`/`suffix`/`lender`; `principal_components_total/difference_usd` reported, not assumed zero; `valuation_caveats` only when the source signals FX (non-zero exchange adjustment); `currency_of_commitment` NULL (never inferred); no `current_principal`. |
| `silver_project_financial_summary` | 1 per project per snapshot (3) | Decimal totals (NULL if any input NULL, never partial); `net_principal_after_cancellation_usd`; `disbursement_vs_original_principal_pct` and `disbursement_vs_net_principal_pct` (2 dp half-even; documented as financial only, not physical progress); `workbook_ibrd_commitment_usd` beside the loan total, `loan_principal_minus_workbook_commitment_usd`, `commitment_sources_agree`. |
| `silver_procurement_awards` | 1 per (project, WB contract number) | Award fields must agree across supplier rows (else NULL + conflict issue); `supplier_count`; `amount_basis`; `contract_amount_usd` only for single-supplier awards; `amount_lower_bound_usd` (max row amount) / `amount_upper_bound_usd` (sum). |
| `silver_procurement_suppliers` | 1 per contract–supplier relationship (= Bronze row) | `supplier_row_amount_usd` documented as not summable across suppliers. |
| `silver_procurement_coverage` | 1 per project | `award_count` / `supplier_relationship_count` are **NULL when NOT_COVERED_BY_THIS_DATASET** (unknown, not zero); `known_award_count()` enforces this for downstream code. |

**Provenance design.** Row-level `source_refs` (Bronze table, source file, sheet, source
row, ingestion run, readable record key); derived rows reference every input's Bronze
records (the P130544 summary references both loan rows and the workbook row).
Field-level `FIELD_LINEAGE` maps every Silver field to its Bronze field(s), source
column and transform; `explain(table, field)` prints the chain down to the source column,
and `silver_field_lineage.json` is written with the Silver outputs. Values that fail
normalisation are NULL in Silver and listed in the row's `quality_issues` (raw value,
reason, source ref). Bronze metadata is not copied into Silver columns.

**Decision: contract 1657297.** Two Bronze rows, same contract number, same award-level
fields (description "Support Organisations for … CSIS Activities In Hubballi-Dharawad
City", 06/05/2021, QCBS, borrower ref IN-KUIDFC-200456-CS-QCBS), two distinct suppliers
(IDs 522100, 539003), 35,729.02 each. Not duplicates (different supplier IDs), and the
dataset cannot tell whether the amount is the award value repeated per supplier (as in
joint ventures) or each supplier's own contract (the description says "Organisations",
plural). Modelled as **one award + two supplier relationships**, `amount_basis =
MULTI_SUPPLIER_UNRESOLVED`, `contract_amount_usd = NULL`, bounds 35,729.02–71,458.04, and a
`PROCUREMENT_AWARD_AMOUNT_UNRESOLVED` warning. The split is supported by the whole dataset
(item 25). Neither row was dropped and nothing was summed into a single amount.

**Decision: reconciliation severity.** `LOAN_PRINCIPAL_COMPONENTS_DIFFERENCE` is INFO when
the loan has a non-zero exchange adjustment (consistent with, not proof of, revaluation)
and WARNING otherwise (the source offers no explanation).

**Decision: expected loans.** `expected_loan_numbers` added to `configs/projects.yaml`;
a missing expected loan is an ERROR, an additional loan a WARNING.

**Local outputs:** `.local_output/{bronze,silver}/` JSONL (+ `*.schema.json`,
`silver_field_lineage.json`). No Parquet (pyarrow is not a project dependency) and no
Delta layer.

**Not done (by design):** Delta writer; document-derived Silver tables (Phase 5);
original closing dates; attention signals.

### Phase 4 — Document parsing, metadata validation, parsed representation (complete)

Flow: **PDF → `DocumentParser` (Docling) → `ParsedContent` → metadata handlers →
manifest reconciliation → `ParsedDocument` JSON** (no LLM, no RAG, no extraction of
business entities).

**Modules** (`src/worldbank_copilot/parsing/`)

| Module | Responsibility |
|---|---|
| `models.py` | Internal schema: `ParsedDocument`, `ParsedPage`, `TextBlock`, `ParsedTable`, `Section`, `FieldValidation`, `EvidenceLocation` (+ `evidence_location()` → "Source: ISR Sequence 18, page 7"). |
| `parser.py` | `DocumentParser` protocol, `ParserError`, config hash. |
| `docling_parser.py` | The only Docling-aware module (lazy import): converter setup, warning capture, `convert_docling_document()` (split multi-provenance items per page via charspans, picture-nested text, furniture layer, table grids). |
| `assembly.py` | Parser-independent cleaning, furniture marking, sections, pages, table text rendering; `ASSEMBLY_VERSION` in the cache key. |
| `cleaning.py` | Conservative: NFC + ligatures + invisible chars, whitespace, lowercase-only hyphen repair; furniture = parser label/layer, page numbers, disclosure-stamp-only blocks, repeated text in the top/bottom margin band. Raw text always kept. |
| `metadata.py` | Content-based type detection (priority-ordered cover patterns) and per-type handlers; ISR (seq, archive date, header date, ISR number, title), report numbers, cover dates, loan numbers; closing-date *candidates* and loan references with page provenance. |
| `reconcile.py` | Manifest vs document per field → CONFIRMED / CORRECTED_FROM_DOCUMENT / MANIFEST_ONLY / DOCUMENT_ONLY / CONFLICT / UNKNOWN with manifest, document and resolved values and reason. |
| `document_builder.py` | Builds `ParsedDocument`; per-document checks incl. **PDF text-layer coverage** (pypdfium2, independent of the parser). |
| `pipeline.py` | Cache (document_id incl. SHA prefix + source SHA-256 + parser config hash), `--project/--document/--force/--failed-only/--report-only`, isolated failures, SHA before/after, index + JSON schema. |
| `report.py` | Portfolio validation and aggregation by project / type / status. |
| `scripts/parse_documents.py`, `notebooks/02_parse_documents.py` | CLI and thin notebook. |

**Parser configuration (validated):** Docling 2.130.0, OCR off (every PDF has a text
layer), TableFormer ACCURATE with cell matching, all content layers,
`traverse_pictures=True`, threads = CPU count (16; output-neutral, ~35% faster than 4).

**Decisions made in Phase 4**

* **Picture-nested text is included.** Docling puts text found inside picture regions
  under the picture item. Skipping it lost real content (e.g. ISR "Risks" and "Systematic
  Operations Risk-rating Tool"). Controlled experiment on ISR Seq 8, content coverage
  with running headers excluded: 97.7% (default) → **99.0% (with picture text)**;
  cell matching off → 88.2% (rejected). Such blocks are tagged `@picture`.
* **Text coverage is measured, not assumed:** for each page, the share of the PDF text
  layer's word tokens (running header/footer lines excluded) present in the parsed page.
  Document < 95% → `LOW_TEXT_COVERAGE` warning.
* **ISR dates:** `document_date` = archive date (`isr_archived_date`, consistent with
  the manifest); the header report date is kept separately in `report_date`. Neither
  overwrites the other.
* **Restructuring papers have no document date on the cover.** "APPROVED ON …" is the
  approval date of the operation being restructured; it is stored as
  `metadata.extras.approved_on_date` with that meaning, never as the document date.
* **Closing dates** are recorded only as candidates with page/table provenance
  (`metadata.extras.closing_date_candidates`); Silver `original_closing_date` stays NULL.
  Text candidates require "Label: date" (a key/value layout had paired an approval
  date with the "Current Closing Date" label; fixed and tested).
* **Month-only cover dates** ("February 2023") are kept as `extras.cover_month`, not
  turned into a day.
* **Manifest precedence:** document content outranks the manifest only when clearly
  identified; every field records manifest/document/resolved values and reason.
* **Environment:** the project `.venv` is now built from python.org Python 3.14.7.
  Anaconda's `python.exe` loads its bundled MSVC runtime 14.29, which breaks PyTorch
  2.14 (`WinError 1114`, `c10.dll`). The package still targets Python ≥ 3.11
  (Databricks). `documents` extra pinned to `docling>=2.130,<3`.
* Default `pytest` excludes `docling` (real-model, slow) tests; `pytest -m docling`.

**Results on the real corpus (2026-09-27)**

* 53/53 documents parsed, 0 failed; 1,140 pages, 1,197 tables, 1,546 sections.
  Total Docling time 2,848 s (47.5 min) on CPU; ISRs 23–98 s each, largest documents
  (PAD440 91 pp.) ~200 s. Cached refresh of all 53: ~3 s.
* ISRs: P130544 1–24, P179039 1–8, P506272 1–3; all 35 parsed; every ISR has sequence,
  archive date, header report date and ISR number.
* Metadata validation (318 field checks): 188 CONFIRMED, 35 DOCUMENT_ONLY (ISR numbers),
  8 MANIFEST_ONLY (project ID not printed in loan agreements/letters/assessments),
  87 UNKNOWN (not applicable, e.g. ISR sequence on non-ISRs), **0 CORRECTED, 0 CONFLICT**:
  the Phase 2 manifest was accurate.
* All 53 source SHA-256 hashes unchanged; page counts match the PDFs; every block and
  table has valid page provenance.
* PDF text coverage: min 95.4% (ISR Seq 2), median 99.4%; 12 pages below 85%.
* Observations: 0 errors, 32 warnings (`TABLE_CELLS_DROPPED`), 18 info (empty pages,
  ISR date differences, documents without a reliable date).

**Document issues discovered in Phase 4** (see §6 items 29–34).

**Open items for Phase 5:** results-framework tables in early P130544 ISRs (Seq 2–6)
lose some cells during table-structure matching; key/value "BASIC DATA" boxes are
parsed as imperfect tables; all headings come back at one level (hierarchy only via
numbering); five image-only ESSA pages would need OCR.

### Phase 5 — Structured document extraction (complete, awaiting approval)

Flow: **Phase 4 parsed cache → deterministic extractors → document-derived Silver
datasets + quality report**. Docling is not re-run. The PDF text layer is opened
only when an extractor's validation fails, and every read is logged. No LLM is used:
none was needed.

**Modules** (`src/worldbank_copilot/extraction/`)

| Module | Responsibility |
|---|---|
| `provenance.py` | `EvidenceRef` (project, document, type, date, file, source hash, page, section, table/row/column or block, ≤240-char snippet, method, label), `ExtractionMethod` (DOCLING_TABLE / DOCLING_TEXT / PDF_TEXT_FALLBACK / DERIVED), `ExtractionStatus` (EXACT / NORMALIZED / DERIVED_FROM_EXPLICIT_SOURCE / AMBIGUOUS / MISSING / CONFLICT), `ExtractionIssue`. No float confidence scores. |
| `ratings.py` | Controlled vocabularies (performance HS…HU, risk Low…High). Raw value kept; glyphs stripped → NORMALIZED; unknown → AMBIGUOUS with UNKNOWN_RATING_VALUE; abbreviations and wrong-scale values are never guessed. |
| `text_source.py` | `PdfTextLayer` (lazy pypdfium2), `LoggedTextSource` (every page read → `FallbackLog`), `DictTextSource` for tests. |
| `tables.py` | Cell cleaning, numbers (wrap-injected spaces removed), DMY / Mon-YYYY dates, loan IDs in Statement notation. |
| `isr.py` | `silver.isr_snapshots`: date policy, ratings (table → label block → text lines), SORT (split across pages), per-loan disbursements (anchored on "% Disbursed"; "Historical Disbursed" kept as printed), per-loan key dates (header-mapped, stray-date rejection, placeholder-aware text fallback), narratives, restructuring history. |
| `results.py` | `silver.project_results`: BLOCK (old) and WIDE/DLI (new) layouts, header-driven column mapping (7- and 9-column), text-layer BLOCK grammar fallback per failing page, Docling-vs-text reconciliation (disagreement → NULL + CONFLICT), wrapped-name completion, DLI status/tag split, text values for (Text)/(Yes/No) indicators, loan tables never taken as results continuations. |
| `identity.py` | Indicator keys: exact normalized name → documented alias (`configs/indicator_aliases.yaml`). Source IDs are not used for linking (issue 35). Near-matches → POSSIBLE_INDICATOR_MATCH. |
| `risks.py` | `silver.appraisal_risks`: SORT rows → FORMAL_RISK_RATING (datasheet/annex copies deduplicated; differing copies → CONFLICT); explicit risk tables (strict header labels, continuations only on the next page with the same columns, page-split fragments merged, mid-sentence rows AMBIGUOUS) and TA "Risk - n" blocks (table, or logged text fallback) → ASSESSMENT_FINDING. |
| `events.py` | `silver.project_events`: restructuring papers (PROPOSED CHANGES flags, Loan Closing and Cancellations tables, rationale and description stored verbatim), AF paper (amount, printed approval date, Changed/Not Changed flags, closing table), ISR key dates and restructuring history; flag-vs-table consistency; restructuring-date candidates (reported, not applied). |
| `closing_dates.py` | Original closing reconciliation per loan (all explicit candidates must agree) and per project (first loan by board approval); `apply_project_enrichment` returns an enriched **copy** of `silver_projects` plus lineage. Existing structured values are never overwritten. |
| `crosscheck.py` | Latest ISR and events vs `silver_loans`: loan numbers, original principal, cancelled/disbursed (INFO: observed at different dates), approval/signing/effective/closing dates, AF amount; JPY cancellation → NOT_COMPARABLE. |
| `pipeline.py` | Per-document cache (`_cache/`, key = source hash + parser config hash + extractor version + extractor source-code hash + config), project assembly, project checks (ISR completeness, canonical-date order, SORT size, picture-only pages, appraisal documents without risk tables), quality report (layer `silver_documents`), outputs. |
| `scripts/extract_document_facts.py`, `notebooks/04_extract_document_facts.py` | CLI and thin notebook. Later notebooks renumbered 05–09. |

**Decisions made in Phase 5**

* **ISR date policy:** `canonical_report_date` = header/report date when identified,
  else archive date; both are kept with `date_difference_days`. Differences ≤ 30 days
  → INFO, > 30 days → WARNING.
* **Docling first, text layer second:** a Docling results table is used only if every
  row validates (4 values, 4 dates, name with unit suffix). Otherwise that page is read
  from the PDF text layer with the BLOCK grammar. If both sources validate but differ,
  the values stay NULL (EXTRACTION_CONFLICT). Unvalidated WIDE rows stay NULL.
* **Formal events only** from restructuring papers, the AF paper and ISR key-date /
  restructuring-history lines. The paper's stated rationale is stored verbatim as
  `reason_text`; no causality is inferred.
* **ISR-derived closing changes** are dated with the first ISR that reports them
  (`event_date_basis` says so). This is not the approval date.
* **AF kept separate** from the original loan: its own event with amount (US$ M,
  as printed) and the paper's printed approval date, labelled as such.
* **Original closing date** established for all three projects (P130544 30-Nov-2022
  from ISRs + RES33572 + AF paper + RES01944; P179039 01-Jun-2028; P506272
  31-Dec-2030) via `project_enrichment.jsonl`. `silver_projects` itself is unchanged.
* **Assessment findings** only from explicit tables and TA risk blocks. Narrative
  risk discussion is a documented limitation (EXTRACTION_LIMITATION). The P506272
  fiduciary assessment has no risk table and yields no records.
* **Image-only ESSA pages** (P179039 pp. 123–127) and picture-only separator pages
  are recorded as OCR_NOT_REQUIRED_FOR_CURRENT_SCOPE.
* **Python 3.11 compatibility** is kept (ruff flagged an f-string with a backslash in
  an expression; it was replaced by a module-level pattern).

**Results on the real corpus (2026-09-30)**

* 53 documents processed from the parsed cache. Second run: 53/53 cache hits. Source
  hashes unchanged.
* `isr_snapshots`: **35** (P130544 24, P179039 8, P506272 3). PDO, implementation
  progress and overall risk ratings found for **35/35**. SORT extracted for 35/35.
* `project_results`: **837 observations**, 237 indicator keys (P130544 465 obs / 89
  keys; P179039 193 / 58; P506272 179 / 90). 821 EXACT, 16 AMBIGUOUS (values NULL).
  314 observations come from the PDF text-layer fallback (P130544 ISRs 1–19 and
  P179039 ISRs 1–4: the old BLOCK layout).
* `appraisal_risks`: **89**: 39 FORMAL_RISK_RATING (PAD440, AF PAD4503, PAD5226:
  10 SORT rows each; PADHP00139: 9); 50 ASSESSMENT_FINDING (P179039: PAD
  technical risks 4, TA 4, IFSA procurement 6, ESSA 10; P506272: TA 2, ESSA 24, of which
  8 are AMBIGUOUS because rows are split across pages).
* `project_events`: **27**: P130544 RESTRUCTURING 8 (4 dated from ISR history, 4
  undated papers), CLOSING_DATE_CHANGE 5, APPROVAL 2, EFFECTIVENESS 2,
  RESULTS_FRAMEWORK_CHANGE 2, ADDITIONAL_FINANCING 1, COMPONENT_CHANGE 1,
  CANCELLATION 1, FUND_REALLOCATION 1; P179039 and P506272 APPROVAL 1 + EFFECTIVENESS 1
  each (no formal change documents in the corpus).
* PDF text fallback: 204 page reads in 33 documents (results 151, key dates 31,
  ratings 14, SORT 6, TA risk 1, name completion 1), all in `pdf_text_fallback_log.jsonl`.
* Quality observations: **0 errors**, 27 warnings, 216 info.
* Cross-source: loan numbers, original principal, approval/signing/effective/closing
  dates agree for every loan. Differences: AF printed approval date; disbursed/cancelled
  amounts at different dates (INFO); "Historical Disbursed" as printed (INFO); JPY
  cancellation not comparable.
* Tests: 380 unit (49 new for Phase 5), 23 integration (5 new). `ruff check` and
  `ruff format --check` pass.

**Not done (by design):** Gold, attention signals, embeddings, Vector Search, RAG,
routing, LangGraph, agents, NeMo, MLflow, FastAPI, React; Delta writer for the
document-derived tables (pending the runtime decision).

### Phase 6 — Databricks platformization and governed Delta foundation (local implementation complete; real Databricks validation pending)

**Status boundary:** everything up to persistence is implemented and validated
locally. **No Databricks operation has been executed from this environment**: there is
no workspace, cluster or credential here, and none are faked. The Databricks run
(`notebooks/05_platformize_databricks.py`) must be executed after pushing the
repository and opening it as a Git folder.

**Modules** (`src/worldbank_copilot/lakehouse/`)

| Module | Responsibility |
|---|---|
| `contracts.py` | `TableContract` / `Column` (type, nullability, role, vocabulary), `DECIMAL(38,6)`, standard identity + operational columns, `model_columns` (derived from the Phase 1-5 Pydantic models), `validate_rows` (types, float rejection, scale, tz-aware timestamps, required, vocabulary, duplicate record ids and natural keys). |
| `identity.py` | `stable_id` (natural key → record_id), canonical values (Decimals at contract scale), `content_hash` (excludes operational columns). |
| `records.py` | Row builders: 9 Bronze tables (source text as read + `_source_sha256`), 8 structured Silver tables, ISR snapshots (+ SORT, loan disbursement and key-date child tables), results, risks, events (candidate dates), enrichment, data-quality observations (report-level and record-linked). |
| `review.py` | `silver.indicator_match_candidates` and `review/indicator_alias_candidates.csv` (PENDING_REVIEW; never writes aliases). |
| `sources.py` | Source snapshot manifest (`configs/source_snapshot.json`: 56 files, SHA-256, snapshot id) and verification (MATCH / MISMATCH / MISSING; hard stop). |
| `reconcile.py` | Profiles (row count, fingerprint, stored-vs-recomputed hash check, null / distinct / group counts), comparison, committed expected profiles. |
| `sql.py` | CREATE SCHEMA / VOLUME / TABLE (explicit DDL) and the snapshot MERGE. |
| `spark_store.py` | Only Spark-dependent module: validates the catalog (never creates it), creates missing schemas/Volumes, creates or schema-checks tables, MERGE with version-aware metrics, read-back. |
| `pipeline.py` | Orchestration: verify sources → rebuild with Phase 2-5 code → check parsed-document lineage → build + validate → reconcile with expected → (Databricks) persist → read back → reconcile. |
| `validation_sql.py` + `sql/phase6_validation.sql` | 14 named validation queries (not Gold analytics). |
| `scripts/platformize.py`, `notebooks/05_platformize_databricks.py`, `requirements-databricks.txt` | Local dry run; thin Databricks entry point; pinned notebook-scoped dependencies. Placeholder notebooks renumbered 06–10 to follow the new roadmap. |

**Decisions made in Phase 6**

* **Write mode = snapshot MERGE on `record_id`** for every table: insert new, update
  only when `record_hash` changes, delete records absent from the snapshot (`WHEN NOT
  MATCHED BY SOURCE`). All tables hold the full current snapshot of the three projects;
  partial-project Delta loads are not supported. Reruns of an unchanged snapshot change no
  rows; load metadata then keeps the values of the load that last changed the record.
* **Natural keys:** Bronze = source file + sheet + row (document inventory: relative
  path; coverage: project); Silver per table (e.g. loans: raw loan number; ISR:
  project + sequence + document; results: project + indicator key + document; risks:
  risk_id; events: event_id; quality: observation fingerprint).
* **Guard before any write:** the in-memory profiles must equal the committed expected
  profiles for the same source snapshot; otherwise nothing is written.
* **Volume layout mirrors `data/`** so relative source paths (lineage) are identical;
  parsed documents are uploaded to the artefact Volume. Docling is not run in Databricks.
* **Pinned runtime dependencies** equal the local versions (PDF text and XLSX readers
  must be identical for identical results; reconciliation proves it).
* **Configuration:** `databricks.catalog = worldbank_copilot`, schemas, `source_volume`,
  `artifact_volume` in `base.yaml`; Databricks data/artefact roots derive from them.
  Updated two Phase 1 configuration tests that asserted "catalog unset" (the catalog is
  now an approved value); their failure behaviour is still tested with the value unset.
* **Phase 5 changes (defects / required semantics only):** `SEQUENCE_DATE_ANOMALY`
  replaces ISR_DATE_DIFFERENCE for the out-of-order canonical date; candidate date
  fields on events; report dedupe fix (issue 47); `ExtractionRun.parse_run` kept for
  the Phase 4 observations. No extracted value changed.

**Local results (2026-09-30)**

* Source integrity: 56/56 files MATCH (53 PDFs + 3 structured), snapshot
  `71c36ffe262398175a02e9cc3cae94c08a40f8f3b7be717fc7ef96b8f41fcf9a`.
* 27 tables built and contract-validated; two independent builds produce identical
  fingerprints for every table (different operational run ids).
* Document-derived Silver: `isr_snapshots` 35 (24/8/3), `project_results` 837 (237
  indicators), `appraisal_risks` 89, `project_events` 27, `project_enrichment` 7,
  `indicator_match_candidates` 96, plus child tables `isr_sort_ratings` 332,
  `isr_loan_disbursements` 45, `isr_loan_key_dates` 42.
* `data_quality_observations`: 394 (0 errors, 80 warnings, 314 info).
* Expected-profile reconciliation: OK.
* Tests: 423 unit (Phase 6: contracts, records, integrity/reconciliation/review/boundaries,
  store), 23 integration previously + 4 new Phase 6 real-data checks; `ruff check` and
  `ruff format --check` pass.

**Pending (requires the real workspace):** Unity Catalog validation/creation, Volume
upload + hash verification in Databricks, Delta persistence, read-back reconciliation,
idempotent second run, validation SQL results.

**Not done (by design):** Gold, attention signals, RAG, Vector Search, reranking, Jev,
SLM/LLM routing, agents, memory, cache, MLflow AI evaluation, API, UI.

### Phase 6 — Databricks validation (user, 2026-09-30)

Executed in Databricks by the user and approved:
- 56/56 source hashes matched;
- 27 tables were created;
- expected-profile and read-back reconciliation passed;
- the idempotent second run changed no rows;
- all validation SQL succeeded.

### Phase 7 — Deterministic Gold intelligence layer (validated in Databricks)

**Execution model:** production runs in Databricks from `worldbank_copilot.silver.*`
(`notebooks/06_build_gold_intelligence.py`), with native Spark only. Locally, the same
code runs on a local Spark session against small synthetic fixtures and the Phase 6
Silver export (development rehearsal). **Nothing has been written to Databricks from
this environment.**

**Modules** (`src/worldbank_copilot/intelligence/`)

| Module | Responsibility |
|---|---|
| `rules.py` | Loads and validates `configs/intelligence/attention_rules.yaml` (14 rules, 8 deferred) and `rating_scales.yaml` (ordinal ranks only). Pure Python. |
| `contracts.py` | Explicit Gold contracts (Phase 6 mechanism, `DECIMAL(38,6)`, vocabularies, keys, provenance classes); frozen in `configs/gold_contracts.lock.json`. |
| `frames.py` | Spark expressions for record_id / record_hash, printed-number and date parsing (`try_*`, ANSI-safe), contract validation and profiling inside Spark. |
| `timeline.py`, `results.py`, `risks.py`, `signals.py`, `project_360.py` | Gold transformations (DataFrame API). |
| `checks.py` | 16 ERROR invariants (block the write) and 9 WARNING/INFO observations. |
| `pipeline.py` | Reads and validates Silver → builds → validates → MERGE (`SparkDeltaStore.merge_frame`) → read-back reconciliation; `silver_from_json` only for local rehearsal. |

**Decisions**

* **Native Spark only (no Python UDFs):** portable across classic, shared and
  serverless compute and Photon. It is also required for local tests here (issue 56).
* **Signals:**
  - A signal is CURRENT or HISTORICAL; each has a deterministic `signal_id` and neutral wording.
  - Wording is checked by the SIGNALS_NEUTRAL_WORDING invariant.
  - Evidence is structured: `source_table`, `source_record_id`, `supporting_record_ids`, document, page, section and method.
* **Restructurings:** counted once. Dated ISR-history approvals are counted; the undated
  papers are supporting evidence only. Closing-date changes are counted as distinct
  (loan, old, new) combinations.
* **Result progress:**
  - Direction comes from each indicator's baseline → target; no "higher is better" assumption is made.
  - NULL-safe trend logic: a regression test guards against a NULL comparison falling through to "away from target".
  - A missing target is not a "change".
* **Financial rule:** IPF only (PforR disburses against DLIs).
* **Write mode:** snapshot MERGE on `record_id`, the same as Phase 6. The `gold` schema is
  created only if missing.

**Local results on the real Silver data (local Spark, 2026-09-30)**

| Gold table | Rows | Fingerprint (prefix) |
|---|---|---|
| project_360 | 3 | e7ff05ab849f |
| project_timeline | 74 (9 milestones + 3 original closing + 35 ISRs + 27 events) | 2c8d1a4ffc3f |
| result_progress | 837 (493 OK; 344 not evaluable with reason) | 353156a7e9c7 |
| risk_register | 116 (39 formal + 50 findings + 27 latest-ISR SORT) | 2f884ecdf2b4 |
| attention_signals | 69 | 9aa8663cf0ae |
| quality_observations | 25 checks (16 ERROR, all PASS) | — |

* **Signals per rule** (69 in total):

  | Rule | Signals | Severity / status | Projects |
  |---|---|---|---|
  | SCHEDULE_CLOSING_DATE_EXTENDED | 1 | HIGH | P130544 |
  | SCHEDULE_REPEATED_CLOSING_DATE_CHANGES | 1 | WATCH | P130544 |
  | RATING_DOWNGRADE | 7 | 4 HIGH, 3 WATCH, all historical | P130544, P179039 |
  | RATING_BELOW_SATISFACTORY_PERSISTENT | 4 | 2 HIGH, 2 WATCH, historical | P130544 |
  | RATING_RECOVERY | 4 | INFO | P130544 |
  | RESULT_TARGET_DATE_PASSED_NOT_MET | 14 | 8 HIGH current, 4 HIGH historical, 2 WATCH | P130544, P506272 |
  | RESULT_MOVED_AWAY_FROM_TARGET | 5 | WATCH | P130544 |
  | RESULT_NO_CHANGE | 21 | INFO | P130544 |
  | RESULT_TARGET_CHANGED | 6 | INFO | P130544, P179039 |
  | FINANCE_DISBURSEMENT_LAG | 1 | WATCH | P130544 |
  | CHANGE_RESTRUCTURING | 1 | WATCH (4 restructurings) | P130544 |
  | CHANGE_ADDITIONAL_FINANCING | 1 | INFO | P130544 |
  | CHANGE_CANCELLATION | 1 | INFO | P130544 |
  | RISK_OVERALL_RATING_ELEVATED | 2 | WATCH | P130544, P506272 |

* **Identity:**
  - 237 series in total: 110 exact-identity series and 127 indicators in pending alias pairs.
  - 0 reviewed aliases were used; no pending pair was merged.
  - 131 series have at least one evaluable observation.
* **Determinism:** two builds with different run metadata give identical fingerprints and ids for every table.
* **Tests:**
  - unit tests: 432 passed;
  - Spark tests: 15 synthetic plus 6 on real data (including all 17 Phase 7 validation queries executed on local Spark);
  - `ruff check` and `ruff format --check` clean.

**Databricks validation (notebook 06, run by the user, 2026-09-30):**
- The `worldbank_copilot.gold` schema and all 6 tables were created.
- The Silver input fingerprint matched the local rehearsal (`ab355c39…`).
- All 5 business-table fingerprints matched the local Spark build exactly.
- First run: 1,124 rows inserted (3 / 74 / 837 / 116 / 69 / 25).
- Read-back reconciliation: OK.
- All 16 ERROR invariants passed.
- Second run: 0 inserted, 0 updated and 0 deleted in every table; fingerprints unchanged; idempotent = True.
- All 17 validation queries executed.
- Fix after the run: in `signal_provenance`, ISR-sourced signals showed the PDO rating's page and method. The query now takes page, section and method from the signal itself.

**Not done (by design):** RAG, chunking, embeddings, Vector Search, reranking, routing,
Jev/SLM, agents, memory, cache, MLflow AI evaluation, API, UI.

### Phase 7 — approved by the user (2026-09-30)

### Phase 8 — Databricks-native retrieval foundation (implemented; validated in Databricks through notebook 07b)

**Modules** (`src/worldbank_copilot/retrieval/`):
- `config` (configs/retrieval/*.yaml);
- `chunking`: 3 strategies, deterministic ids;
- `corpus`: `silver.document_chunks` contract;
- `embeddings`: pluggable provider; Databricks Model Serving, batching, retries, dimension check;
- `vector_search`: endpoint and Delta Sync index lifecycle, normalised `(chunk_id, score)` results;
- `lexical`: BM25 and RRF;
- `rerank`: none or CrossEncoder;
- `query`: deterministic normalisation, acronyms, ISR references, foreign-project refusal;
- `models` and `retriever`: scope, guardrails, citation-ready Evidence, abstention;
- `evaluation`: relevance, metrics, staged experiments, miss classification, abstention calibration, MLflow;
- `pipeline`: capability probe, corpus MERGE, embedding cache, index source, index;
- `report`: isolation checks and formatting.

**Notebooks:** `notebooks/07_build_retrieval_and_evaluate.py` (it replaces the 07/08 placeholders; no reranker) and `notebooks/07b_rerank_experiment.py` (CrossEncoder; ML base environment).
**SQL:** `sql/phase8_validation.sql`.
**Questions:** `evaluation/retrieval_questions.yaml`.

**Decisions**
* **One corpus table.** All strategies live in one table (`chunk_strategy` column) behind
  one Vector Search index filtered by strategy. That means one endpoint and one index,
  which suits workspace limits.
* **Self-managed vectors.** The embedding model can be swapped, and unchanged text is never
  re-embedded (cache keyed by `text_sha256` and model).
* **Bounded driver work** (documented):
  - chunking reads the 53 parsed JSON files;
  - only uncached texts are sent to the embedding endpoint;
  - BM25 and evidence metadata use the governed corpus loaded once (about 14k rows).
* **Dense vs governed metadata.** Dense hits are only chunk ids. Text and citations always
  come from `silver.document_chunks`, and out-of-scope or unknown ids raise.
* **Type filters.** Document-type filters inferred from question wording are an experiment
  dimension; they are not applied by default.
* **No answer generation** (optional in the brief): retrieval comes first. Abstention is a
  calibrated reranker-score threshold, reported and not auto-applied.

**Local results (real parsed corpus):**
- 53 documents.
- Retrieval chunks: fixed 2,620; structure 3,979; parent_child 6,198 (+1,530 parents).
- All 44 answerable questions are representable in `structure` chunks.
- Lexical isolation: 49 questions x 3 project scopes, 0 foreign chunks.
- Tests: 479 unit, 34 integration and 22 Spark tests pass; ruff clean.

### Phase 8 — Databricks validation of the embedding build and index (user, 2026-10-01)

Executed in Databricks:

- **Qwen embedding build (Step 3a)** with `databricks-qwen3-embedding-0-6b`:
  - 10,965 of 10,965 texts complete;
  - the full run embedded 10,565 after 400 had been cached by the pilot;
  - 0 retries, 0 failures, 0 rate limits;
  - 53 checkpoints, 5,360.8 seconds.
- **Index source (Step 3b):** `worldbank_copilot.silver.document_chunk_index` holds 12,797 rows. The rerun was idempotent (0 inserted, 0 updated, 0 deleted).
- **Endpoint:** `worldbank-gep-ai-search` was ONLINE and was reused, not created. Its 3 pre-existing GEP indexes (`worldbank_ai.rag.*`) were left untouched.
- **Pre-flight:** passed every check.
- **New index:** `worldbank_copilot.silver.document_chunk_index_qwen3_v1` is ready, in state ONLINE_NO_PENDING_UPDATE, with 12,797 indexed rows (equal to the source table).
- **Orchestration fix after validation:** Step 3b raised `NameError: strategies` when Step 3a was skipped. `build_index_source` and `update_embedding_cache` now default to every configured strategy (`configured_strategies`), so Steps 3b and 6 run independently after restart and bootstrap.
- **At the time of this run, not yet run:** retrieval experiments (Steps 4–7, notebook 07b). Notebook 07b was run later the same day (see "Phase 8 — notebook 07b results"). Phase 9 has not been started.
- **Step 6 idempotency (Databricks):** the rerun showed no change in any table up to its last assertion:

  | Table | Rows | Inserted / updated / deleted | Embeddings |
  |---|---|---|---|
  | `silver.document_chunks` | 14,327 | 0 / 0 / 0 | – |
  | Qwen cache | – | – | 10,965 of 10,965 cached, 0 newly embedded |
  | `silver.document_chunk_index` | 12,797 | 0 / 0 / 0 | – |

  The cell then raised `NameError: corpus`, because its fingerprint assertion used a variable from Step 2. (Superseded: the fixed Step 6 later passed every assertion; see the notebook 07 Steps 4–7 results.)
- **Notebook orchestration fix (after that run):**
  - **Committed problems:** notebook 07 contained committed merge-conflict markers in Step 3a (resolved to the validated `PILOT_REQUESTS = None`) and a debugging cell (an undefined `client` and a hard-coded endpoint), now removed.
  - **Step 4:** opens the existing, ready index and the corpus itself (`open_retriever` / `open_vector_index`, read-only).
  - **Step 5:** builds its own retriever and declares its only deliberate dependency, the Step 4 selection, with `require_state` (a clear error, not a NameError).
  - **Step 6:** compares against the persisted `silver.document_chunks` profile, read before the rebuild (`persisted_corpus_profile`).
  - **Tests:** `tests/unit/test_notebook_cells.py` checks, by static analysis, that every phase cell after Steps 0/1 is independent. Data, model, cache, index, retrieval logic and experiment methodology are unchanged.
- **Notebook 07b orchestration audit:** the committed 07b had these hidden dependencies:
  - Step 3: decisions, ev, names, questions, report, retriever;
  - Step 4: questions, report, rerankers, retriever, selected;
  - Step 5: cfg, questions, report, rerankers, retriever.

  The retriever (existing index + corpus), questions, imports, names and reranker objects are now rebuilt in each cell (read-only). The experiment state is declared with `require_state`: `decisions` (Step 3), `selected` (Steps 4 and 5). `tests/unit/test_notebook_cells.py` covers both notebooks, including detection of accidental dependencies.

### Phase 8 — notebook 07 Steps 4–7 results (Databricks, user run, 2026-10-01)

**Source and scope of the evidence**
- The record comes from the user's export of the orchestration-fixed `notebooks/07_build_retrieval_and_evaluate.py`. Every cell finished.
- Steps 2–3b show no output in this export, so it adds no evidence for them; their validation is recorded above.
- The workspace copy had one extra unnamed cell (`strategies = sorted(rs.chunking.strategies)`) between Steps 3a and 3b. That cell is not in the repository, and no step depends on it.

**Step 4, staged experiments without a reranker.** 44 answerable questions; recall@10 primary; minimum gain 0.02.

| Stage | Configuration | R@5 | R@10 | MRR | nDCG@5 | p50 ms | Decision |
|---|---|---|---|---|---|---|---|
| chunking | fixed\|dense\|none\|k10 | 0.602 | 0.773 | 0.476 | 0.493 | 1,996.8¹ | kept |
| chunking | structure\|dense\|none\|k10 | 0.568 | 0.773 | 0.471 | 0.453 | 99.9 | +0.000 |
| chunking | parent_child\|dense\|none\|k10 | 0.636 | 0.727 | 0.510 | 0.520 | 98.1 | −0.045 |
| retrieval | fixed\|dense\|none\|k10 | 0.602 | 0.773 | 0.476 | 0.493 | 99.3 | |
| retrieval | fixed\|lexical\|none\|k10 | 0.682 | 0.750 | 0.565 | 0.604 | 2.1 | −0.023 |
| retrieval | fixed\|hybrid\|none\|k10 | 0.773 | 0.795 | 0.590 | 0.642 | 98.8 | **+0.023, winner** |
| candidate_depth | fixed\|hybrid\|none\|k10 | 0.773 | 0.795 | 0.590 | 0.642 | 102.8 | kept |
| candidate_depth | fixed\|hybrid\|none\|k20 | 0.773 | 0.773 | 0.592 | 0.636 | 104.3 | −0.023 |
| candidate_depth | fixed\|hybrid\|none\|k50 | 0.773 | 0.795 | 0.597 | 0.650 | 108.0 | +0.000 |
| query_filters | fixed\|hybrid\|none\|k10+typefilter | 0.773 | 0.795 | 0.630 | 0.669 | 103.2 | +0.000, not preferred |

¹ Same first-run effect as in 07b (identical configuration: 99.3 ms in the retrieval stage).

- **Reranking stage:** the CrossEncoder run is `UNAVAILABLE` by design, because notebook 07 installs no reranker. The CrossEncoder is evaluated only in 07b.
- **Reproducibility:** the chunking and retrieval stages produced the same quality metrics in 07 and 07b.
- **Depth without a reranker adds nothing:** k20 lost one question and k50 equalled k10. Depth helps only together with the CrossEncoder (07b).
- **Type filter:** without a reranker it gave R@10 +0.000 (MRR +0.040, nDCG@5 +0.027). With the CrossEncoder at k50 (07b) it gave −0.023. It is not adopted in either case.

**Selected without a reranker (measured baseline): `fixed|hybrid|none|k10`**
- R@5 0.7727, R@10 0.7955, MRR 0.5904, nDCG@5 0.6415, P@5 0.1909.
- leakage@5 and leakage@10 both 0.
- p50 100.5 ms, p95 143.2 ms.
- **By project (R@10):** P130544 0.875, **P179039 0.545** (MRR 0.351), P506272 0.889.
- **By kind (R@10):** paraphrase 0.571, exact 0.850, temporal 1.000.
- **Misses:** 9 questions, all `retrieval_miss`: q03, q13, q22, q28, q29, q30, q35, q36, q42.
- **Compared with the always-rerank winner (07b):**
  - q03, q22 and q42 were recovered;
  - q35 moved from `retrieval_miss` to `ranking`;
  - q10 became a new `ranking` miss.
  - The five P179039 misses (q28, q29, q30, q35, q36) are the same with and without the reranker.

**Step 5, isolation (no reranker):** 7 of 7 PASS.
- Under each project scope, all 49 questions returned only that project's evidence: 490, 450 and 420 items for P130544, P179039 and P506272.
- The 3 cross-project questions and the unknown project P000000 were refused.

**Step 6, idempotency:** the cell completed and every assertion passed, including the corpus fingerprint and row count against the persisted profile. This supersedes the earlier `NameError: corpus` run.

| Table | Rows in snapshot | Inserted / updated / deleted | Delta version |
|---|---|---|---|
| `silver.document_chunks` | 14,327 | 0 / 0 / 0 | 5 |
| `silver.document_chunk_index` | 12,797 | 0 / 0 / 0 | 5 |

- **Embedding cache:** 10,965 texts needed, 10,965 already cached, 0 embedded now, complete.
- **Warning:** Spark Connect reported "'verifySchema' is ignored" (informational).

**Step 7, read-only SQL (`sql/phase8_validation.sql`):** every violation check was 0.
- **Corpus:**
  - fixed: 2,620 chunks;
  - structure: 3,979 chunks;
  - parent_child: 6,198 retrieval chunks plus 1,530 parents.
  - That is 14,327 rows in total; the 12,797 retrieval rows equal the index source.
- **Identity:** 0 duplicate chunk or record ids; 0 chunks without pages or scope.
- **Provenance:** 0 chunks without an inventory document; 0 hash mismatches; 0 project mismatches. The sample chains show page, section, element ids, extraction method and a matching `inventory_sha256`.
- **Parent-child:** 0 orphans; 0 parents in another document; 0 wrong roles.
- **Index alignment:** 0 retrieval chunks not indexed; 0 index rows missing from the corpus; 0 disagreeing.
- **Embedding cache:** `databricks-qwen3-embedding-0-6b`, 10,965 vectors, every one of dimension 1024.
- **Comments fix (issue 55):** 0 label-instead-of-comment rows. Rows with comments: P130544 95, P179039 59, P506272 36.

**MLflow:** 13 runs logged by notebook 07, in addition to the 13 runs from 07b.

### Phase 8 — notebook 07b results (Databricks, user run, 2026-10-01)

Source: the user's export of `notebooks/07b_rerank_experiment.py` (the orchestration-fixed version), run on serverless environment 6, ML base. Every cell finished; none raised an error. The run read the existing corpus and the index `document_chunk_index_qwen3_v1`. It created, embedded and wrote nothing to Delta.

**Environment**
- **Step 0, dependency health:** OK. `pip check` OK; protobuf 6.33.5, grpcio-status 1.76.0, googleapis-common-protos 1.71.0, databricks-ai-search 0.78, sentence-transformers 5.5.1, transformers 4.57.6, torch 2.12.0.
- **Step 1, probe:** every check OK.
  - AI Search: endpoint `worldbank-gep-ai-search` ONLINE.
  - Qwen embeddings: dimension 1024, single-call latency 166 ms.
  - CrossEncoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
  - MLflow: available.

**Step 2, staged experiments.** 44 answerable questions; primary metric recall@10; a more complex option must gain at least 0.02.

| Stage | Configuration | R@5 | R@10 | MRR | nDCG@5 | P@5 | p50 ms | p95 ms | Winner |
|---|---|---|---|---|---|---|---|---|---|
| chunking | fixed\|dense\|none\|k10 | 0.602 | 0.773 | 0.476 | 0.493 | 0.141 | 2,001.4¹ | 2,098.2 | ✔ |
| chunking | structure\|dense\|none\|k10 | 0.568 | 0.773 | 0.471 | 0.453 | 0.136 | 98.3 | 158.9 | |
| chunking | parent_child\|dense\|none\|k10 | 0.636 | 0.727 | 0.510 | 0.520 | 0.150 | 91.9 | 124.4 | |
| retrieval | fixed\|dense\|none\|k10 | 0.602 | 0.773 | 0.476 | 0.493 | 0.141 | 94.5 | 128.5 | |
| retrieval | fixed\|lexical\|none\|k10 | 0.682 | 0.750 | 0.565 | 0.604 | 0.177 | 3.6 | 170.8 | |
| retrieval | fixed\|hybrid\|none\|k10 | 0.773 | 0.795 | 0.590 | 0.642 | 0.191 | 96.3 | 142.9 | ✔ |
| reranking | fixed\|hybrid\|none\|k20 | 0.773 | 0.773 | 0.592 | 0.636 | 0.191 | 100.5 | 150.7 | |
| reranking | fixed\|hybrid\|cross_encoder\|k20 | 0.773 | 0.818 | 0.643 | 0.681 | 0.191 | 2,405.7 | 2,924.5 | ✔ |
| candidate_depth | fixed\|hybrid\|cross_encoder\|k10 | 0.727 | 0.795 | 0.637 | 0.658 | 0.177 | 1,054.2 | 1,596.4 | |
| candidate_depth | fixed\|hybrid\|cross_encoder\|k20 | 0.773 | 0.818 | 0.643 | 0.681 | 0.191 | 2,341.5 | 2,981.1 | |
| candidate_depth | fixed\|hybrid\|cross_encoder\|k50 | 0.795 | 0.841 | 0.683 | 0.717 | 0.196 | 5,541.3 | 6,433.8 | ✔ |
| query_filters | fixed\|hybrid\|cross_encoder\|k50 | 0.795 | 0.841 | 0.683 | 0.717 | 0.196 | 5,383.2 | 6,606.8 | ✔ |
| query_filters | fixed\|hybrid\|cross_encoder\|k50+typefilter | 0.773 | 0.818 | 0.683 | 0.717 | 0.191 | 5,109.4 | 6,510.9 | |

¹ This was the first configuration the run executed. The identical configuration measured 94.5 ms p50 in the retrieval stage, so the 2,001 ms figure is not representative of dense retrieval. The table does not show the cause.

**Stage decisions (as printed by the run)**
- **Chunking:** `fixed` was kept. `structure` gained +0.000; `parent_child` lost 0.045.
- **Retrieval:** `hybrid` beat dense by +0.023 recall@10. Lexical alone was −0.023.
- **Reranking:** the CrossEncoder beat no reranker at k20 by +0.045 recall@10. MRR rose by +0.051 and nDCG@5 by +0.046.
- **Candidate depth:** k20 beat k10 by +0.023, and k50 beat k20 by +0.023.
- **Query filters:** the inferred document-type filter was not preferred (−0.023).

On 44 questions, one question is 0.0227 of recall. Each depth step and the hybrid gain therefore equal exactly one more question found; the reranker gain at k20 equals two.

**Experiment winner (Step 3):** `fixed|hybrid|cross_encoder|k50`.
- recall@5 0.7955, recall@10 0.8409, MRR 0.6827, nDCG@5 0.7169, P@5 0.1955.
- leakage@5 and leakage@10 both 0.0.
- p50 5,383.2 ms, p95 6,606.8 ms.

**Selected configuration: adaptive reranking, not "rerank every query" (user decision, 2026-10-01)**
- **What the winner is:** in the 07b methodology, `fixed|hybrid|cross_encoder|k50` applies the CrossEncoder to every query. It is recorded as the measured quality ceiling of reranking, not as the production path.
- **The selected production direction is adaptive reranking.** The default path is hybrid retrieval without a reranker. The CrossEncoder over a deeper candidate set (k50) is applied only to the queries where the harness decides it is needed.
- **Fixed parts:** `fixed` chunks, `hybrid` (BM25 + dense, RRF k=60), no type filter, final_k 5.
- **Not yet implemented or evaluated:** the trigger policy, meaning which queries are reranked. 07b did not test any trigger. The code has no adaptive mode: `RunConfig` applies one reranker to every query, and `configs/retrieval/retrieval.yaml` `production` is unchanged (all `null`).
- **No adaptive figure exists yet.** No quality or latency number for adaptive reranking may be reported until a trigger is implemented and evaluated against both measured bounds:

  | Bound | Configuration | R@10 | MRR | nDCG@5 | p50 ms | p95 ms |
  |---|---|---|---|---|---|---|
  | never rerank | fixed\|hybrid\|none\|k10 | 0.7955 | 0.5904 | 0.6415 | 96.3 | 142.9 |
  | always rerank | fixed\|hybrid\|cross_encoder\|k50 | 0.8409 | 0.6827 | 0.7169 | 5,383.2 | 6,606.8 |

- **Trigger candidates:** these are hypotheses to be evaluated, not decided.
  - a deterministic rule, such as hybrid score margin or a lexical/dense disagreement;
  - question kind (paraphrase questions have the lowest recall);
  - a bounded decision model, owned by harness policy per CLAUDE.md §15.

  Each must report recall, MRR, the fraction of queries reranked, and latency.

**Latency/quality tradeoff (measured)**
- **Always-on CrossEncoder at k50 vs. hybrid without a reranker at k10:**
  - recall@10 +0.045 (2 of 44 questions);
  - MRR +0.092; nDCG@5 +0.075; recall@5 +0.023;
  - p50 latency about 56× (96 ms → 5,383 ms); p95 about 46× (143 ms → 6,607 ms).
- **Depth drives the cost:** reranking 10, 20 and 50 candidates took 1,054, 2,342 and 5,541 ms p50. The cost is roughly linear in candidates scored, on CPU (torch 2.12.0+cpu).
- **The last depth step (k20 → k50)** found one more question for about +3.2 s p50.
- **Lexical BM25 alone is the fastest** (3.6 ms p50) and has the best R@5 among the non-hybrid methods. It does not win on recall@10.
- **Latency is not part of the selection rule.** The staged winner is chosen on recall@10 alone. The adaptive decision above is the response to that gap. Latency was measured on serverless CPU, with no GPU serving tried.

**Breakdown of the experiment winner**

| Group | Questions | R@10 | R@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|
| P130544 | 24 | 0.917 | 0.875 | 0.733 | 0.752 |
| **P179039** | **11** | **0.545** | **0.545** | **0.462** | **0.592** |
| P506272 | 9 | 1.000 | 0.889 | 0.818 | 0.777 |
| kind: exact | 20 | 0.850 | 0.800 | 0.680 | 0.721 |
| kind: paraphrase | 14 | 0.714 | 0.714 | 0.523 | 0.542 |
| kind: temporal | 8 | 1.000 | 0.875 | 0.889 | 0.954 |
| kind: multi_evidence | 2 | 1.000 | 1.000 | 1.000 | 0.960 |

Weakest categories: finance (R@10 0.60, 5 questions), ratings (0.67, 3), and procurement (R@10 0.75, but nDCG@5 0.311, 4).

**Misses of the experiment winner.** There are 7 questions; the classification comes from `classify_misses`.

| Question | Project | Category | Cause | Labeled evidence |
|---|---|---|---|---|
| q10 | P130544 | ratings | ranking | 1 restructuring paper, p. 6 |
| q13 | P130544 | procurement | retrieval_miss | 3 documents, p. 1 ("re-bid(ding) the contract on a Design Build Operate Transfer") |
| q28 | P179039 | implementation_issues | retrieval_miss | 4 documents, p. 1 ("integrated helpline accessible") |
| q29 | P179039 | project_status | retrieval_miss | 3 documents, p. 1 (Operation and Maintenance Policy) |
| q30 | P179039 | finance | retrieval_miss | 4 documents, p. 1 ("US$363 million Karnataka Sustainable Rural Water Supply Program") |
| q35 | P179039 | assessment_findings | ranking | 1 document, p. 6 |
| q36 | P179039 | finance | retrieval_miss | 1 document, p. 23 ("disbursed based on the achievement of eight DLIs") |

**P179039 limitation (recorded, not resolved)**
- **Scale:** P179039 accounts for 5 of the 7 misses. Its recall@10 is 0.545 (6 of 11 questions) and its MRR 0.462. The other two projects reach 0.917 and 1.000.
- **Bias in the overall figure:** the overall 0.841 is not representative of P179039, the Program-for-Results water project. Results for this project must be treated as the weakest retrieval coverage in the corpus.
- **Observed facts:**
  - 4 of the 5 are `retrieval_miss`: chunks containing the evidence phrase exist in scope but were not retrieved. The other one (q35) is `ranking`: a matching chunk was retrieved but ranked below 10.
  - No P179039 miss was classified `bad_chunking` or `metadata_filter`. The evidence phrases are present in in-scope `fixed` chunks.
  - 3 of the misses (q28, q29, q30) have labeled evidence only on page 1 of several documents. Each is phrased as a paraphrase of that page-1 text.
  - Sample evidence for q28 (Step 5) returned a grievance-redress passage from the Technical Assessment (pp. 25–26: complainants "register and track their complaints through internet, phone, or social media"). It was not in the labeled evidence. All five reranker scores were negative (top −2.834).
- **Not established:**
  - whether the cause is how page-1 chunks are represented (for example, front-matter text diluting the passage), vocabulary mismatch in paraphrased questions, or incomplete relevance labels;
  - whether the q28 passage should count as supporting evidence. That is a labeling decision for review, not something the system resolves. No label was changed.
- **Not attempted in Phase 8:** no project-specific tuning, re-chunking or label edits were made to raise this number.

**Abstention calibration (reported, not applied)**
- **Which run:** computed on the first CrossEncoder run, `fixed|hybrid|cross_encoder|k20` (reranking stage). It was not computed on the selected k50.
- **Scores:**
  - answerable: minimum −4.788, median 3.804;
  - no-answer: maximum 1.770, median −3.777.
- **Best threshold:** −2.834, with accuracy 0.939, which is 46 of 49 questions classified correctly.
- **The ranges overlap,** so no threshold separates them perfectly.
- **Sample:** the no-answer question in Step 5 (q25, "metro rail stations") had a top reranker score of −5.897, below that threshold.
- **Config:** `min_rerank_score` remains `null`.

**Step 4, isolation with the experiment winner:** 7 of 7 PASS.
- All 49 questions under each project scope returned only that project's evidence: P130544 2,051, P179039 2,071 and P506272 2,000 evidence items.
- Three cross-project questions were refused before retrieval.
- The unsupported project P000000 was refused.

**Step 5, sample evidence:** 8 questions; each result carries project, document, ISR sequence and date, pages, section and reranker score. The CrossEncoder took 4,221–6,903 ms per query.

**MLflow:** 13 runs logged (one per stage configuration).

**Warnings, none blocking:**
- **Authentication notice:** the AI Search client printed "Using a notebook authentication token. Recommended for development only" on every query. A service principal is needed before any non-development use.
- **MLflow tag warnings:** 13 × "Encountered unexpected error during resolving tags: getContext().extraContext()…" from tag resolution. The runs were still logged.

**Not done:** no trigger policy was implemented, no configuration was changed, no tests were added, and no Databricks operation was run by Claude. Phase 9 has not been started.

### Phase 8 — validation summary (notebooks 07 and 07b, Databricks, 2026-10-01)

**Measured results (executed in Databricks)**
- **Data foundation:**
  - corpus 14,327 rows;
  - index source and AI Search index 12,797 rows;
  - Qwen cache 10,965 of 10,965 vectors, dimension 1024;
  - every Step 7 violation check 0;
  - Step 6 idempotent (0/0/0 and 0 newly embedded).
- **Measured no-reranker baseline, `fixed|hybrid|none|k10`:** R@10 0.7955, MRR 0.5904, nDCG@5 0.6415, p50 about 100 ms, p95 about 143 ms.
- **Measured always-rerank quality winner and quality ceiling, `fixed|hybrid|cross_encoder|k50`:** R@10 0.8409, MRR 0.6827, nDCG@5 0.7169, p50 5,383 ms, p95 6,607 ms.
- **Isolation:** 7 of 7 PASS in both notebooks, with leakage 0. MLflow logged 13 + 13 runs.

**Experimentally selected configurations (by the staged recall@10 rule)**
- **Without a reranker (07):** `fixed|hybrid|none|k10`. Fixed chunks, hybrid BM25+dense with RRF, k10, and no type filter all won or were kept.
- **With the CrossEncoder (07b):** `fixed|hybrid|cross_encoder|k50`. This means reranking every query; latency is not part of the rule.

**Production design direction (user decision; not implemented)**
- **Adaptive reranking:** the no-reranker baseline is the default path, and the CrossEncoder at k50 runs only on queries a trigger selects.
- **Not implemented or evaluated:** adaptive reranking has **not** been implemented or evaluated, and **no adaptive performance numbers exist**.
- **Config:** `configs/retrieval/retrieval.yaml` `production` stays unset (all `null`) until a trigger has been evaluated against both measured bounds.

**Unresolved / open issues** (also listed in §8, items 10–12)
- **Adaptive reranking trigger:** undefined. Candidates such as rules, question kind or a bounded decision model are hypotheses only.
- **P179039:** R@10 0.545 with or without the reranker; the same 5 questions are missed in both. The cause is **not established**.
- **q28 is a ground-truth-label review question:** the Technical Assessment grievance-redress passage it retrieves is not labeled as evidence. The label is unchanged pending review.
- **Abstention threshold:** experimental only (−2.834, computed on the k20 run; the score ranges overlap). `min_rerank_score` stays `null`.
- **Notebook-token authentication:** AI Search used a notebook token ("development only"). A service principal is required before production use.
- **Latency:** the always-rerank latency was measured on serverless CPU only.

### Phase 9 — Structured tools + intelligent routing (in progress)

**Approved decisions (user, 2026-10-01)**
1. **Retriever split:** approved as an integration refactor only. `retrieve()` stays as a backward-compatible wrapper. A golden regression test must pass before Phase 9 uses the split.
2. **Adaptive reranking:** no X%/Y% production thresholds. 9E is a diagnostic experiment (P0 NEVER, P1 ALWAYS, P2–P6) that reports a quality-versus-rerank-rate frontier. `retrieval.yaml` `production` stays null until the evidence is reviewed and acceptance criteria are approved.
3. **Procurement:** `get_procurement_awards` is deferred. Procurement-award coverage exists only for P130544 (IPF); the two PforR projects are outside the dataset; no Phase 9 evaluation requirement justifies a dedicated tool.
4. **Derived arithmetic:** no sixth provenance class. A derived value keeps the most conservative class of its inputs, plus derivation metadata:
   - FACT-only arithmetic stays FACT;
   - any DOCUMENTED_FINDING input keeps DOCUMENTED_FINDING;
   - deriving from a signal stays SYSTEM_DERIVED_SIGNAL;
   - an UNKNOWN input produces UNKNOWN;
   - arithmetic over facts or findings is never promoted to a signal.
5. **Routing labels:** start as DRAFT. Human review comes before any label becomes REVIEWED or the test split is frozen (9C).

**Additional requirements recorded:**
- **Retrieved passages:** DOCUMENTED_FINDING with `relevance_verified=false`. Retrieval establishes source provenance, not truth or relevance.
- **Fallback:** STRUCTURED → DOCUMENT fallback is allowlisted per intent (9B); generic fallback is impossible.
- **Sufficiency:** mechanical sufficiency (Phase 9) is separate from semantic sufficiency (Phase 10).
- **P179039 is not tuned:** q28 stays DOCUMENT and label-review; q29, q30 (pending human labelling of programme-size semantics), q35 and q36 stay DOCUMENT.

**Checkpoints:**
- 9A: tools and Retriever split;
- 9B: query understanding and deterministic routing;
- 9C: draft routing set, then HUMAN REVIEW;
- 9D: bounded semantic classifier;
- 9E: adaptive-rerank diagnostic;
- 9F: Databricks end-to-end validation and closure.

#### Checkpoint 9A — tool layer and Retriever split (local; awaiting review)

**Retriever split** (`retrieval/retriever.py`):
- `retrieve()` is now exactly `finish(first_stage(...))`.
- `FirstStage` holds the processed query, scope, guarded and de-duplicated candidates, and the start time.
- `finish` reranks or not, selects `final_k`, applies the abstention threshold and the scope check, and builds evidence. It never mutates the first stage, so one stage can be finished both ways.
- No other Phase 8 code, configuration, label, index or artifact changed.

**Golden regression:** `tests/support/retrieval_golden.py` holds a verbatim copy of the Phase 8 `retrieve` (git `351b5a0`). Results are compared on everything except wall-clock latency: evidence ids and order, scope, metadata, citations, statuses, notes, and retrieval and rerank scores.
- **Synthetic corpus:** 2,160 comparisons:
  - 2 strategies × {lexical, dense, hybrid} × {none, CrossEncoder} × k {3, 10, 50} × type filters on/off;
  - × 10 questions × 3 projects (two approved, one unknown);
  - plus an abstention threshold, and every status and error path (OK, INSUFFICIENT_EVIDENCE, ScopeViolation, ValueError).
- **Mutation check:** two deliberate regressions (one fewer evidence item; reversed first-stage order) were both caught.
- **Real parsed corpus:** all 49 questions × {fixed, structure, parent_child} × 5 configurations, including the two measured bounds (hybrid/none/k10 and hybrid/CrossEncoder/k50): 735 comparisons, all equal. Foreign-project questions fail identically.
- **Local stand-ins:** dense retrieval and the CrossEncoder are deterministic local stand-ins here. The same comparison against the real index and CrossEncoder is a 9F Databricks step.

**Tool layer** (`src/worldbank_copilot/tools/`):
- **`models`:**
  - the five-class vocabulary, identical to Gold `PROVENANCE_CLASSES`;
  - `Fact` (NULL → UNKNOWN with a reason), `Derivation` and `derive()` (Decision 4);
  - the `ToolResult` envelope: status, items, filters, argument candidates, `data_snapshot`, mechanical findings, `semantic_sufficiency=NOT_ASSESSED`, caveats, notices. Only OK carries items; AI_INTERPRETATION is rejected.
- **`reader`:**
  - an allowlist of 10 governed tables (5 Gold, 5 Silver);
  - columns are validated against the contracts;
  - the `project_id` predicate is mandatory and every returned row is re-checked;
  - results are bounded (2,000 rows).
  - `SparkTableReader` uses the DataFrame API, pins one Delta version per table per request with `DESCRIBE HISTORY` and `VERSION AS OF`, and records it in `data_snapshot`.
  - `InMemoryReader` has identical semantics, for tests.
- **`executor`:**
  - allowlist, then Pydantic arguments, then scope (approved, authorised, equal to the request scope) before any read;
  - explicit statuses: SCOPE_REFUSED, DATA_INTEGRITY_ERROR, TIMEOUT (cooperative deadline before each read), ERROR (recorded and logged);
  - no fallback path: one call runs exactly one tool.
- **The 8 tools:**

  | Tool | Sources |
  |---|---|
  | `get_project_overview` | gold.project_360, silver.projects |
  | `get_project_timeline` | gold.project_timeline |
  | `get_rating_history` | silver.isr_snapshots |
  | `get_financial_status` | gold.project_360, silver.loans, silver.isr_loan_disbursements, silver.project_events |
  | `get_results_progress` | gold.result_progress |
  | `get_risk_register` | gold.risk_register |
  | `get_attention_signals` | gold.attention_signals |
  | `search_project_documents` | Phase 8 retriever, via `retrieval/rerank_policy.py` |

  `rerank_policy.py` contains only the measured NEVER and ALWAYS bounds; there is no default policy. `DocumentSearchConfig` must be passed explicitly because `production` is null.
- **Not built (by design):** a free-form SQL tool, write tools, the procurement tool, the router, notebooks.

**Verification (local, 2026-10-01):**
- **Unit tests:** `pytest` 656 passed, including 6 golden, 18 model and derivation, 23 reader and executor, 24 structured-tool and 7 document-tool tests.
- **Integration tests:** `pytest -m integration` 38 passed. This includes the real-corpus golden; the Spark tests are skipped in the main environment.
- **Spark tests:** `.venv-spark` `pytest tests/spark -m spark`: 37 passed. This includes 12 new tool tests on the real Silver export plus locally built Gold:
  - `SparkTableReader` output equals `InMemoryReader` output for every tool and project;
  - tool totals reconcile with Gold: timeline 74, results 837, risk register 116, signals 69;
  - the validated Phase 7 facts reproduce: P130544 `days_extended` 1765 (DOCUMENTED_FINDING); latest ISR sequences 24 / 8 / 3; P179039 ratings ordered by sequence 1–8;
  - every signal is SYSTEM_DERIVED_SIGNAL;
  - DLI-layout observations never get a progress value.
- **Lint:** `ruff check` and `ruff format --check` are clean.

**Corrected while testing (my test assumptions, not tool defects):**
- The Phase 7 "no DLI progress" invariant applies to `layout == "DLI"`, not to `indicator_type == "DLI"`.
- Gold gives `EXTRACTION_NOT_EXACT` precedence over `DLI_LAYOUT_NOT_EVALUATED`.

**Not yet validated in Databricks:** Delta version pinning and the tools on the governed tables; the Retriever split against the real AI Search index and CrossEncoder. Both are 9F.

#### Checkpoint 9A — approved by the user (2026-10-01); not committed

#### Checkpoint 9B — query understanding and deterministic routing (local; awaiting review)

**Package** `src/worldbank_copilot/routing/`:
- `models`: typed, frozen state for every stage;
- `config`: loaders and validation;
- `entities`: input and project resolution;
- `temporal`, `intents`, `requirements`;
- `router`: the decision table;
- `service`: the harness.

**Configuration** (reviewable, `configs/routing/`):
- `routing.yaml`: router version, input limits, injection flags;
- `intents.yaml`: 25 rules in 5 precedence groups, plus a document-type lexicon;
- `requirements.yaml`: intent → route, tools, default time and supported time, plus investigation plans; no fallback key;
- `project_aliases.yaml`: 11 aliases. `reviewed_by: null`, so it is pending human review.

**Behaviour:**
- **Entities resolve deterministically:** P-ids, corpus document ids (by project prefix), manifest report numbers, registry loan numbers, and reviewed aliases.
- **Scope is never switched:** a foreign reference of any kind is CONFLICT; two or more projects is MULTI; unknown entities are UNSUPPORTED.
- **Temporal:**
  - the grammar covers LATEST, ISR (single, set, range), APPRAISAL, YEAR, DATE, DATE_RANGE, HISTORY and EVENT_ANCHORED;
  - event anchors resolve only against source-stated timeline dates, and an ambiguous anchor gives CLARIFY with candidates;
  - relative periods and invalid dates are UNRESOLVED; the wall clock is never used.
- **Intent:** rules plus a fixed combination order. Unresolved questions are SEMANTIC_CLASSIFICATION_REQUIRED (left for 9D).
- **Execution:**
  - STRUCTURED runs only the requirement calls;
  - DOCUMENT runs search only after scope passes, with an explicitly supplied policy;
  - INVESTIGATION returns a dry-run-validated plan and executes nothing;
  - there is no structured → document fallback;
  - `semantic_sufficiency` stays NOT_ASSESSED.

**Verification (local, 2026-10-01):**
- **Unit tests:** `pytest` 1,014 passed. 358 are new routing tests: config 5, entities 83, temporal 44, intents 57, service 169.
- **Integration tests:** `pytest -m integration` 38 passed (18 Spark tests skipped in the main environment).
- **Lint and whitespace:** `ruff check`, `ruff format --check` and `git diff --check` are clean.
- **Spark suite:** not re-run in 9B because no Spark-path code changed; it was 37 passed at 9A.

**Not done:** semantic classifier (9D), routing evaluation set (9C), adaptive reranking (9E), Databricks validation (9F), Phase 10.

#### Checkpoint 9B — approved by the user (2026-10-01); not committed

#### Checkpoint 9C — routing evaluation dataset (DRAFT; awaiting human label review)

**Files:**
- `evaluation/routing_cases.yaml`: 80 cases, all `label_status: DRAFT`. Human expected labels only.
- `evaluation/routing_baseline_9b.yaml`: the current 9B router output, generated and kept separate from the human labels.
- `evaluation/routing_review_9c.md`: the review artifact.
- `routing/evaluation.py`: dataset schema, baseline and comparison.
- `routing/review.py`: renders the review artifact.
- `scripts/routing_review_9c.py`: local-Spark rehearsal; real Gold is built in memory, nothing is persisted, and DOCUMENT routes are recorded but not executed.
- `tests/unit/test_routing_dataset.py`: structure only.

**Composition:**
- 28 cases derived from Phase 8 and 52 newly written.
- Expected routes: STRUCTURED 27, DOCUMENT 21, REFUSE 19, INVESTIGATION 8, CLARIFY 5.
- Proposed split, not frozen: 27 dev / 53 test, in families.

**9B baseline** (DRAFT labels; this is not the routing accuracy):

| Metric | Value |
|---|---|
| Deterministic coverage | 0.825 |
| Exact route agreement | 0.725 (0.879 on cases the router resolved) |
| Semantic classification required | 0.175 |
| Project-resolution agreement | 0.988 |
| Temporal exact match | 0.867 (60 cases) |
| Isolation cases that executed nothing | 14 of 15 |

The exception is r073 ("the other Karnataka program"). It is an implicit foreign reference: 9B answered from the active scope P130544 and read no other project's data, but it answered the wrong question.

**Unchanged:** the router, its rules and its configuration (no tuning); the Phase 8 retrieval labels; the aliases (`reviewed_by: null`).

**Tests:** `pytest` 1,026 passed; `pytest -m integration` 38 passed; `ruff check` and `ruff format --check` clean.

#### Checkpoint 9C — human review round 1 applied (2026-10-01; labels still DRAFT)

**Decisions recorded** (`human_decision`):
- r001/q09, r003/q16, r004/q17, r005/q32, r011/q33 → STRUCTURED.
- r008/q44, r009/q46, r010/q22 → DOCUMENT_CONTENT.
- r007/q43 → CLARIFY TIME_REQUIRED.
- q03, q05, q07 → DOCUMENT/EXPLANATION; q37 → INVESTIGATION.
- P179039 q28, q29, q30, q35, q36 → DOCUMENT. q30 means the total program size, not the IBRD principal. q28 stays LABEL_REVIEW.
- The routing principle for document references and the WHY principle are written into the label rationales.

**r005 verification:**
- `silver.appraisal_risks` and the locally rebuilt `gold.risk_register` (via `get_risk_register`) hold a P179039 FORMAL_RISK_RATING row: 'Overall', 'Moderate', APPRAISAL_DOCUMENT page 8, EXACT/DOCLING_TABLE.
- A Spark test asserts this row.

**r073 fix** (approved project-resolution correctness fix; router 9B.2):
- Relative project references ("the other … program", "another project", "both/all projects", …) are configured in `routing.yaml` and are not aliases.
- An unresolvable reference → CLARIFY `AMBIGUOUS_PROJECT_REFERENCE`, decided at the project stage before any read. The active project is never substituted and no foreign project is selected.
- If the active scope leaves exactly one other authorised project, the referent is unique and foreign → REFUSE `CROSS_PROJECT`. This case is tested explicitly.
- Explicit foreign references (CONFLICT/MULTI) and NOT_AUTHORIZED keep precedence.

**Split:** the `overview` and `ood-weather` families moved from test to dev, giving 29 dev / 51 test, with 4 P506272 cases in dev. Leakage checks pass (highest dev/test token similarity 0.67).

**Baselines:**
- The original 9B.1 baseline is kept unchanged in `routing_baseline_9B.1.yaml`.
- The new baseline is `routing_baseline_9B.2.yaml`.
- Isolation zero-execution went from 14/15 to 15/15, and project resolution from 0.988 to 1.0. No other rule changed.

**Still open:** 12 labels without a decision, the alias confirmation, freezing the split, and marking labels REVIEWED.

#### Checkpoint 9C — final human review applied; labels REVIEWED, split FROZEN (2026-10-01)

**Dataset** (`evaluation/routing_cases.yaml`):
- 80 cases, all `label_status: REVIEWED`; `review_status: REVIEWED`; `reviewed_on: 2026-10-01`.
- `split_status: FROZEN`, 29 dev / 51 test, with families never crossing the split. The dev list is pinned in a test.
- Labelled routes: STRUCTURED 28, DOCUMENT 21, REFUSE 18, INVESTIGATION 8, CLARIFY 5.
- Projects: P130544 43, P179039 19, P506272 15, plus 3 with no valid scope.
- **Dev** (29): P130544 16, P179039 8, P506272 4, 1 with no scope. Routes: STRUCTURED 11, DOCUMENT 8, REFUSE 6, INVESTIGATION 3, CLARIFY 1.
- **Test** (51): P130544 27, P179039 11, P506272 11, 2 with no or an invalid scope. Routes: STRUCTURED 17, DOCUMENT 13, REFUSE 12, INVESTIGATION 5, CLARIFY 4.

**Human semantic routing rules** (recorded in the dataset):
1. Named/source-document wording that asks what the document says → DOCUMENT.
2. A governed factual value that can be answered deterministically → STRUCTURED.
3. Documented rationale of an already uniquely identified event or change → DOCUMENT / EXPLANATION.
4. A "why" question where the change, state or event must first be established → INVESTIGATION.
5. Ambiguous project, entity or time references are resolved or clarified before execution, never guessed.
6. Broad current-state questions default to PROJECT_OVERVIEW; attention signals are not equivalent to project status.
7. Relative time scopes use a governed, reproducible as-of date, never the wall clock.

**Final decisions:**
- r018/q06 → TIMELINE_EVENTS (deterministic month arithmetic).
- r020/q10 → DOCUMENT_CONTENT.
- r044 → RISKS (extracted ESSA findings; differs from r008/q44).
- r046 → STRUCTURED with `get_project_overview` only. Verified: it holds `project_status`, `disbursed_usd` and `financial_snapshot_date`.
- r050 → EXPLANATION.
- r051, r052, r058, r056 → INVESTIGATION.
- r057 → CLARIFY AMBIGUOUS_TIME.
- r060 → STRUCTURED with a RELATIVE time scope: as-of 2026-08-31 (loan-statement snapshot, End of Period), resolved interval 2025-09-01 to 2026-08-31.
- r061 → PROJECT_OVERVIEW.

**Aliases:** exactly the 11 reported aliases, approved (`reviewed_by: phase-9c-human-review-2026-10-01`); none added.

**Baselines:**
- `routing_baseline_9B.1.yaml` is kept unchanged as the before-fix baseline.
- `routing_baseline_9B.2.yaml` is router 9B.2 (r073 fix only) against the reviewed labels:

| Metric | Value |
|---|---|
| Deterministic coverage | 0.825 |
| Exact route agreement | 0.725 (0.879 on resolved cases) |
| Semantic classification required | 0.175 |
| CLARIFY rate | 0.075 |
| REFUSE rate | 0.225 |
| Project resolution | 1.0 |
| Temporal exact match | 0.867 (60 cases) |
| Isolation cases that executed nothing | 15 of 15 |

**Remaining mismatches** (measured baseline errors or semantic-classification gaps; deliberately not tuned):
- **Semantic classification required** (14):
  - human DOCUMENT: r015, r021, r022, r023, r024, r026, r047, r048, r049;
  - human STRUCTURED: r018, r046, r061;
  - human CLARIFY: r041, r057.
- **Route differs:**
  - document-referenced values: r003, r011, r044 (9B DOCUMENT, human STRUCTURED) and r008 (9B STRUCTURED, human DOCUMENT);
  - r027 (9B DOCUMENT, human INVESTIGATION);
  - r055 (event-anchor grammar → CLARIFY, human INVESTIGATION);
  - r033 and r060 (9B CLARIFY, human STRUCTURED). These are implementation limitations: the rating tool has no date filter, and relative periods have no governed as-of resolution in 9B.2.
- **Temporal differs:** r002 ("current value"), r007, r025 ("trends"), r053, r056.
- **Tool arguments differ:** r040 (an extra RESTRUCTURING event type).

**No semantic tuning was performed against the frozen test labels.** The router changed only by the approved r073 project-resolution fix (9B.2), which was made before labels were finalised. No 9D or classifier code exists. Phase 8 retrieval labels are unchanged.

#### Checkpoint 9C — APPROVED AND CLOSED (user, 2026-10-01)

The following are ground truth and frozen:
- the 80 REVIEWED labels;
- the 29 dev / 51 test split and its families;
- the 11 reviewed aliases;
- the Phase 8 retrieval labels;
- the 9B.1 historical baseline;
- the 9B.2 deterministic behaviour.

#### Checkpoint 9D — bounded semantic routing: first stop (DEVELOPMENT only; TEST not evaluated)

**Contract and boundary** (`routing/semantic.py`, `routing/service.py`, `routing/models.py`):
- The fallback runs only when the rules return SEMANTIC_CLASSIFICATION_REQUIRED, after project scope, authorisation and every pre-execution refusal have been decided.
- It receives a `SemanticQueryContext`: question, resolved project (audit only), time kind, rule hits.
- It returns a `SemanticDecision`: one non-refusal intent (route taken from `requirements.yaml`), a confidence, the nearest development examples, or ABSTAIN.
- Abstention, or any out-of-contract intent, becomes CLARIFY `SEMANTIC_ABSTAIN` with nothing executed.
- A predicted intent continues through the unchanged deterministic requirements, route and tool stages, so the classifier never selects tools, projects or retrieval.
- With no classifier configured, 9B.2 behaviour is unchanged.

**Candidate B:**
- nearest-neighbour classifier over the frozen DEVELOPMENT examples only (22 dev cases with an executable intent);
- leave-one-family-out evaluation;
- a pre-registered grid of 108 configurations and a selection rule (`configs/routing/semantic.yaml`: route precision ≥ 0.9 at coverage ≥ 0.3);
- embedders: `LexicalEmbedder` (deterministic, local) and `ProviderEmbedder` (validated Qwen endpoint, own `routing-question` cache namespace).

**Development results, lexical embedder:**
- No configuration qualifies.
- Unthresholded route accuracy 0.23–0.32, against a majority-route baseline of 0.50.
- Confidence is not predictive: the most confident predictions are wrong.
- Only 2 dev cases are fallback-eligible under 9B.2 (r015, r048); the hypothetical fallback got 0/2.
- The lexical candidate is REJECTED.

**Provisional selection:** Candidate A (rules only, fallback disabled).
- The Qwen embedder must be run on DEVELOPMENT in Databricks (`notebooks/08_semantic_routing_dev.py`) before anything is FROZEN.
- Candidate C (a bounded schema-constrained model) is proposed, not implemented.

**False deterministic resolutions:** r003, r008, r011, r027, r044. These are rule/harness problems and are not repaired.
- Capability CLARIFYs: r033 (tool), r055, r060 (time).
- Analysis and recommendations: `evaluation/semantic_dev_report_9d.md`.

**Guard:** `assert_split_allowed` refuses TEST without a FROZEN configuration and explicit authorisation. The experiment log records `test_evaluated: false`.

**Tests:** 1,135 unit and 38 integration passed. Spark not re-run (no Spark-path changes since 9C, 38 passed then). Ruff and the format check are clean.

#### Checkpoint 9D — approved for Databricks DEV validation (user, 2026-10-01)

Candidate B runs with the real Qwen endpoint, on DEVELOPMENT only. No TEST, no Candidate C, no change to 9B.2; Candidate A stays PROVISIONAL.

**Prepared for the run:**
- **Freeze manifest** (`evaluation/routing_freeze_9c.json`):
  - dataset hash `83c5f912…` (SHA-256 of `routing_cases.yaml` with CRLF normalised to LF, because Windows checkouts use CRLF and the Databricks Git folder LF);
  - the 29 dev / 51 test case ids;
  - router 9B.2;
  - the 11 reviewed aliases.
- **Notebook 08 sequence:**
  1. Hard guards. Any failure STOPs: hash, split, REVIEWED labels, aliases, router version, no TEST evaluation recorded, DEV-only inputs, no TEST question used as an example.
  2. Endpoint probe: model, availability, dimension, latency, errors. A mismatch STOPs.
  3. Similarity-threshold diagnostic: STOP if the registered thresholds are inert on Qwen cosines, so the grid is never changed after seeing results.
  4. Qwen and lexical runs, with per-family results and latency.
  5. A selected-for-review or rejection record, written to the Volume only.
  6. MLflow: case ids and aggregates only, no question text, `test_evaluated=false`.

**Pre-commit inspection:** no credentials, `.env`, Databricks config, caches, model binaries or oversized files among the staged files. The dev log contains no question text.

#### Checkpoint 9D — Candidate C: SemIf blocked; Databricks bounded classifier capability (2026-10-02)

**Candidate B2 (Qwen nearest-neighbour)** was rejected on DEV in Databricks:
- best selective route precision ≈ 0.45;
- unthresholded route accuracy ≈ 0.41;
- rules only 24/29, rules + Qwen 25/29 (1/2 fallback cases).

TEST was not evaluated.

**C_SEMIF_OPENJEV — BLOCKED_BY_EXECUTION_ENVIRONMENT.**
- The preregistered Serverless GPU environment cannot be attached in the available workspace.
- This is NOT a model-quality rejection: P0, P1 and every semantic evaluation never ran.
- Code, config, protocol lock (`evaluation/semif_protocol_lock.json`), notebook 08c and tests are preserved unchanged. Only the config `status` field changed, and the lock does not include it.

**C_DATABRICKS_BOUNDED_CLASSIFIER** (`configs/routing/bounded_classifier.yaml`, `routing/bounded_classifier.py`, `routing/bounded_classifier_probe.py`, `notebooks/08d_bounded_classifier_capability.py`):
- **Placement:** deterministic scope, authorisation, project, time and rules first; only unresolved semantic cases reach the classifier. It returns one reviewed non-refusal intent or ABSTAIN, and route = requirements[intent].
- **Model:** a Foundation Model API chat endpoint, preferred `databricks-gpt-5-4-nano`, called from Serverless CPU with the notebook identity. It is never silently substituted; other small chat endpoints are only listed.
- **Output contract:** a strict JSON schema `{"intent": enum}` (11 intents + ABSTAIN, no additional properties).
  - abstain is derived by the harness.
  - Confidence is not self-reported and stays 0.0 unless a derivation is later approved. Logprob support is only recorded.
  - No chain-of-thought is requested and no reasoning text is kept.
- **Fail closed:** HTTP errors, 429, missing or multiple choices, truncation, refusal, empty output, non-JSON, extra keys (`route`, `reasoning`), missing intent, out-of-enum, refusal intents and wrong case all become ABSTAIN, then CLARIFY `SEMANTIC_ABSTAIN` with nothing executed.
- **Capability probe** (synthetic, fictional project SYN-RIVERBEND, one attempt per call, no retries), 33 calls:
  1. a plain call;
  2. 5 parameter-acceptance calls (reasoning_effort none/minimal/low, temperature 0, logprobs);
  3. 7 strict-schema classifications, including one ambiguous and one prompt-injection case;
  4. 2 unconstrained JSON-mode calls;
  5. 19 malformed fixtures checked offline;
  6. 10 warm-latency calls;
  7. an 8-way concurrent burst.
- **Capability gates (revised 2026-10-02):**
  - the preferred endpoint is callable;
  - the required request configuration is accepted. `required_parameters` is empty, so no optional parameter is required;
  - strict structured output works;
  - every strict-schema reply validates to exactly one allowed label;
  - malformed and out-of-contract fixtures fail closed;
  - no execution during classification (no `tools` / `functions` keys in any request);
  - model output cannot change project, authorisation or scope (the decision is label-only, route = requirements[intent]);
  - the operational network/API failure rate of the required non-burst calls (plain + strict-schema + warm) is ≤ 0.05.
- **Diagnostic only, never gates:**
  - optional parameters (reasoning_effort, temperature 0, logprobs), each recorded as SUPPORTED / UNSUPPORTED / ERROR with the response and error metadata. A parameter gates only if it is added to `required_parameters`;
  - latency: individual, plain, structured, warm p50/p95/max and concurrent. It is judged later, in the DEV architecture trade-off;
  - synthetic label agreement;
  - the deliberate 8-way burst. Its 429s and Retry-After values are reported separately and never enter the failure rate.
- **Contract hash:** `3690fc46…b59d`. The notebook verifies the frozen routing-dataset and ambiguity-probe hashes before any call. The SemIf protocol is unchanged.

**Not run:** the Databricks capability probe (it awaits the user's run), DEV, C1, C2, ambiguity-probe predictions and TEST. No change to 9B.2, labels, the split or the Candidate A/B results. 9E and Phase 10 have not started.

**Tests:** 1,224 unit passed. Ruff and the format check are clean.

#### Checkpoint 9D — notebook 08d result: databricks-gpt-5-4-nano BLOCKED_BY_MODEL_AVAILABILITY (2026-10-02)

Real workspace run (Serverless CPU):
- notebook-identity authentication succeeded;
- `databricks-gpt-5-4-nano` was not listed, and its readiness was unavailable;
- every inference attempt returned HTTP 404 ENDPOINT_NOT_FOUND, so 0 model responses were obtained.

This is **NOT a model-quality rejection**.

The safety properties held:
- the malformed fixtures failed closed;
- no execution or tool fields were sent;
- project, authorisation and scope stayed outside the classifier output;
- the frozen routing-dataset and ambiguity-probe hashes were unchanged;
- the capability artifact was written.

No DEV, C1, C2, ambiguity-probe or TEST question was evaluated.

Small chat endpoints discovered in the workspace: `databricks-gemma-3-12b`, `databricks-gpt-oss-20b`, `databricks-gpt-oss-120b`, `databricks-meta-llama-3-1-8b-instruct`. No substitution was made; endpoint selection is a design checkpoint awaiting review.

#### Checkpoint 9D — endpoint selection: databricks-gpt-oss-20b (approved 2026-10-02)

**Selected:** `databricks-gpt-oss-20b`. It is the smallest adequate of the discovered endpoints: an MoE model with 3.6B active parameters per token, the lowest per-token price of the four, and the endpoint Databricks' structured-output documentation demonstrates. The nano result stays recorded as BLOCKED_BY_MODEL_AVAILABILITY in `experiment_history`, which is append-only and not part of the contract.

**Request configuration:**
- strict JSON schema `{"intent": enum[11 intents + ABSTAIN]}`;
- `max_tokens` 1024; 30 s timeout; no retries;
- `required_parameters: {reasoning_effort: low}`. GPT OSS always reasons, and LOW fits bounded 12-way classification. If the endpoint rejects it, the capability fails.

The `reasoning_effort: none` diagnostic was removed: Databricks treats it as unset (medium) for GPT OSS, so it can't disable reasoning.

**Reasoning content is never persisted:**
- Only the text part of the final answer is parsed. Reasoning parts and fields such as `reasoning_content` are never read beyond their type.
- Failure details never quote model text.
- Records store content part types, answer length and a SHA-256 digest, plus numeric usage only. Text excerpts were removed.
- Decisions carry the label, and the route from requirements.

**Artifact:** endpoint-specific, `capability_result_<sanitized endpoint>.json`. The notebook refuses to overwrite an existing file, so the nano result `capability_result.json` is kept.

**Contract hash:** `c64561c2…fe8b`. The gates are unchanged, and no model call has been made.

**Tests:** 1,230 unit passed. Ruff and the format check are clean.

#### Checkpoint 9D — databricks-gpt-oss-20b run 1: CAPABILITY_BLOCKED_BY_RATE_LIMIT (2026-10-02)

Real 08d run, unpaced, contract `c64561c2…fe8b`. This is **NOT a model-quality rejection**.

**Everything that bears on model capability passed:**
- the endpoint was listed and READY; notebook identity resolved; model `gpt-oss-20b-080525`;
- `reasoning_effort=low` was accepted;
- strict structured output worked, and every successful reply validated to exactly one allowed label;
- the malformed fixtures failed closed;
- no tool fields were sent, and the output could not alter project, authorisation or scope.

**The reliability gate failed:**
- operational failure rate 0.3889 against the 0.05 gate, dominated by HTTP 429 REQUEST_LIMIT_EXCEEDED;
- the diagnostic 8-way burst returned 1 × 200 and 7 × 429.

**Latency of successful calls:**
- structured: p50 0.3615 s, p95 0.3860 s;
- warm: p50 0.3857 s, p95 0.3916 s.

Synthetic sanity: 6 of the 6 answered cases were as expected. One case got no answer because of a 429; that is not a misclassification.

The frozen hashes were unchanged, and no DEV, C1, C2, ambiguity-probe or TEST evaluation took place. The artifact `capability_result_databricks_gpt_oss_20b.json` is immutable.

**Next:** a paced rerun whose only experimental variable is request scheduling. Its design is under review.

#### Checkpoint 9D — databricks-gpt-oss-20b run 2: paced rerun implemented (approved 2026-10-02)

**Only experimental variable:** request scheduling (`capability.schedule`, `run_id: run2-paced-5s`). The 5 s gap is a preregistered conservative pacing experiment, not a server-derived threshold: run 1 exposed no verifiable Retry-After value.

**Schedule:**
1. 60 s initial quiet period.
2. 18 gated calls: plain, 7 strict-schema, 10 warm.
3. 6 diagnostic calls: 4 parameter probes, 2 JSON-mode.
4. 60 s pre-burst quiet period.
5. 8-call diagnostic burst.

A fixed **5.0 s gap**, measured from the previous ordinary call's END to the next call's START on a monotonic clock, applies between every ordinary call, including the gated-to-diagnostic transition. No jitter, retries or backoff, and nothing changes in response to a 429.

**Pacer:** it checks the initial quiet period, each gap and the pre-burst quiet period *before* sending. If one can't be honoured, the run stops before the call, is marked **INVALID**, and model capability is not evaluated. The tolerance is 1 µs, for floating-point rounding only.

**Instrumentation per call:**
- sequence number;
- monotonic start and end offsets;
- the actual end-to-start gap;
- HTTP status;
- `retry_after_header` and `retry_after_body`, recorded raw and separately. They are never combined and never acted on; values above 5 s are only listed.

The burst stays excluded from operational reliability.

**Unchanged:**
- endpoint, `reasoning_effort=low`, prompt, schema, synthetic cases;
- `max_tokens` 1024, 30 s timeout, `retries: 0`;
- all gates, including the ≤ 0.05 failure rate (zero failures allowed over 18 gated calls).

Contract hash unchanged: `c64561c2…fe8b`. The schedule is outside the model-facing contract.

**Artifact:** `capability_result_databricks_gpt_oss_20b__run2_paced_5s.json`. The notebook checks that it doesn't exist *before* any call, so the run-1 artifact stays immutable.

**Tests:** 1,238 unit passed. Ruff and the format check are clean. No Databricks call was made.

#### Checkpoint 9D — databricks-gpt-oss-20b run2-paced-5s: CAPABILITY_PASSED (2026-10-02, reviewed)

This is a capability pass only. It is **not** evidence that Candidate C improves routing quality; the DEV evaluation answers that.

**Capability results:**
- the endpoint was READY and notebook identity worked;
- `reasoning_effort=low` was accepted, and strict `json_schema` output worked;
- every successful reply contained exactly one allowed intent;
- malformed responses failed closed;
- no execution capability; project, authorisation and scope stayed with the deterministic harness.

**Schedule and reliability:**
- the schedule was VALID with zero violations;
- operational failure rate 0.0;
- 7/7 synthetic sanity cases gave the expected intent (diagnostic only).

**Inference latency:**
- warm: p50 ~0.34 s, p95 ~0.47 s;
- strict: p50 ~0.60 s, p95 ~0.61 s.

Inference is sub-second. The 5 s gap is experimental request pacing to stay within observed workspace API capacity, **not** inference time.

**Diagnostics only:**
- the burst returned 7 × 200 and 1 × 429, with no Retry-After value;
- the optional `json_object` calls returned 400. Production uses strict `json_schema`, so this is not a blocker.

Artifact `capability_result_databricks_gpt_oss_20b__run2_paced_5s.json` is immutable. The contract is unchanged: `c64561c2…fe8b`.

**Candidate C experiment trail (append-only):**
- SemIf/OpenJev: BLOCKED_BY_EXECUTION_ENVIRONMENT;
- databricks-gpt-5-4-nano: BLOCKED_BY_MODEL_AVAILABILITY;
- databricks-gpt-oss-20b run 1: CAPABILITY_BLOCKED_BY_RATE_LIMIT, not a quality rejection;
- databricks-gpt-oss-20b run 2 (paced 5 s): CAPABILITY_PASSED.

**Next:** the Candidate C DEV evaluation design, under review. Not implemented, no Databricks call made, TEST and the ambiguity probe untouched.

#### Checkpoint 9D — Candidate C DEV evaluation implemented (`dev1`, not run)

The approved design is implemented locally. No Databricks call, no model prediction, TEST untouched, and the ambiguity probe has not been opened.

**Populations:** derived mechanically and locked before any prediction.
- **C1** = DEV cases where the frozen 9B.2 `RoutingService` returns SEMANTIC_CLASSIFICATION_REQUIRED. The real service runs with a context factory that raises at the first data access, which happens only after the semantic boundary. The result is checked for drift against the recorded 9B.2 rows. Result: **r015, r048**.
- **C2** (shadow, DIAGNOSTIC ONLY): a RESOLVED project, a non-refusal intent and an executable route. 22 cases.
- **Repeats:** C1 × 5 (hard gate); r002, r014, r052, r006 × 3 (diagnostic).
- **Call plan:** 44 calls, of which 12 are C1 calls.

**Protocol lock:** `evaluation/candidate_c_dev_lock.json`, sha256 `15d848642942dfd6aa200911c8d1fdfb7075739805acd357bdffe4b1df13267a`, written by `scripts/candidate_c_dev_lock.py`. It holds the dataset, manifest, 9B.2-baseline and probe-reference hashes, the populations, the call plan and its hash, the schedule, the gates, the acceptance rule and its hash `09fe8528…`, and contract `c64561c2…fe8b`.

**Outcomes:** mutually exclusive, by precedence INVALID > REJECTED_SAFETY > INCONCLUSIVE_OPERATIONAL > REJECTED > ACCEPTED_FOR_NEXT_STAGE. Every call has exactly one failure kind:
- NONE;
- OPERATIONAL: no HTTP 200;
- CONTRACT: an HTTP 200 that violates the contract.

A C1 case with an operational failure is NOT EVALUABLE for the quality gates, so it can never be both INCONCLUSIVE and REJECTED. Inference p95 above 5 s maps to INCONCLUSIVE_OPERATIONAL.

**Execution:**
- `notebooks/08e_candidate_c_dev.py` (Serverless CPU) re-derives the lock and checks that the artifact doesn't exist, both before any call. It then runs the paced 44-call plan (60 s quiet, sequential, 5.0 s END→START, no retries, backoff or burst) and writes `candidate_c_dev_predictions__dev1.json`, with no question or model text.
- `scripts/candidate_c_dev_9d.py` (local, `.venv-spark`) replays the recorded main-pass predictions through the real `RoutingService`, using the same Gold harness as the 9B.2 baseline, then writes the report. `--check-harness` already reproduces all 29 recorded 9B.2 DEV rows with **zero drift**.

**Terminology:**
- `scheduled_gated_calls` = 18; `classification_required_calls` = 17. The ambiguous `required_calls` is retired.
- Run 2: 0/18 operational failures.
- DEV: `dev_scheduled_calls` = 44, `c1_calls` = 12.

**Tests:** see the implementation report.

#### Checkpoint 9D — CLOSED (2026-10-02)

**Official frozen evaluator** (`scripts/candidate_c_dev_9d.py`, unchanged; bundled JDK + `.venv-spark`). Input: the immutable artifact `candidate_c_dev_predictions__dev1.json`, sha256 `ae2bf51e…816e`.
- Candidate A recompute: zero drift on all 29 DEV rows.
- **Outcome: REJECTED.** INVALID, REJECTED_SAFETY and INCONCLUSIVE_OPERATIONAL have no findings. This is a quality and repeatability rejection.
- Report: `evaluation/candidate_c_dev_9d.json` and `.md`.

**C1:**
- **r015:** predicted CHANGE_INVESTIGATION (expected EXPLANATION). Final hybrid route INVESTIGATION, a wrong executable route, where Candidate A gave a safe CLARIFY. Repeats: EXPLANATION, then CHANGE_INVESTIGATION × 4, so 4/5 and the stability gate fails.
- **r048:** predicted DOCUMENT_CONTENT, final route DOCUMENT. Correct and 5/5 stable.

**Hybrid:** 25/29, against Candidate A's 24/29 and the required 26/29. The r015 change is not a correction.

**Operations:** schedule VALID; 44/44 HTTP 200; 0 operational and 0 contract failures; inference p50 0.3244 s, p95 1.5515 s (pacing excluded; the p95 ≤ 5 s gate passes); no Retry-After values.

**C2 (diagnostic only):**
- intent accuracy 12/22; intent-implied route accuracy 15/22; ABSTAIN rate 0;
- versus the rules: 4 cases broken, 2 corrected, net −2;
- systematic pattern: EXPLANATION→CHANGE_INVESTIGATION in 3 of 4 cases;
- r014's main prediction was correct, but all 3 of its repeats differed from it.

**Conclusion:** for the frozen 9D DEV set and the boundary left unresolved by 9B.2, this Candidate C configuration did not show enough incremental value or repeatability for promotion. It is not a general statement about the model or about LLM routing.

**Decision** (`evaluation/semantic_config_9d.yaml`: FROZEN, `test_evaluated: false`):
- Candidate A: SELECTED.
- B1 lexical: REJECTED.
- B2 Qwen similarity: REJECTED.
- SemIf/OpenJev: BLOCKED_BY_EXECUTION_ENVIRONMENT.
- GPT-5.4 nano: BLOCKED_BY_MODEL_AVAILABILITY.
- GPT-OSS capability run 1: CAPABILITY_BLOCKED_BY_RATE_LIMIT.
- GPT-OSS capability run 2: CAPABILITY_PASSED.
- GPT-OSS DEV: REJECTED.

The three blocked or rate-limited outcomes are **not** quality rejections.

**Selected strategy: DETERMINISTIC ROUTING + TARGETED CLARIFICATION.** Context, project, temporal and deterministic intent resolution come first. CLARIFY is a safety fallback for execution-critical ambiguity, not the default path.

**Ambiguity probe:** `NOT_RUN_CANDIDATE_C_DEV_REJECTED`. Candidate C was not eligible for it, so this is not a probe failure; the probe stays frozen.

**TEST:** untouched. FROZEN does not authorise it; `assert_split_allowed("test")` also requires explicit authorisation.

**Closure decisions (user, 2026-10-02):**
- **No TEST evaluation of Candidate A.** TEST outputs were already observed during the 9C review, so a TEST run would be a confirmation measurement, not a blind holdout. `test_evaluated` stays false and TEST access is not authorised. Final Phase 9 validation prioritises real Databricks end-to-end behaviour (9F).
- **Technical debt for 9F / final Phase 9 hardening:** `scripts/semantic_dev_9d.py` can overwrite the now-FROZEN `evaluation/semantic_config_9d.yaml`. Before Phase 9 closes, add a fail-closed guard so an accidental DEV rerun cannot overwrite a FROZEN decision artifact unless an explicit, intentional override or versioning mechanism exists. Notebook 08 refusing to run against the FROZEN decision is expected and stays.

**Remaining Phase 9 work:**
- 9E: adaptive-rerank diagnostic — CLOSED 2026-10-02 (see Checkpoint 9E closure);
- 9F: Databricks end-to-end validation, frozen-file protection, Phase 9 closure.

#### Checkpoint 9E — adaptive-rerank diagnostic implemented and locked (2026-10-02)

**Framing:** a preregistered *descriptive* diagnostic over the frozen Phase 8 set (49 questions, 44 answerable). It is **not** independent validation: all 44 were used in Phase 8. There is no new split, no cross-validated selection, and no automatic winner.

**Points:**
- P0: the historical `fixed|hybrid|none|k10` baseline.
- P0@50: the k50 no-rerank counterfactual.
- P1: the same k50 candidates, all reranked by the CrossEncoder.
- **12** adaptive points:
  - P2: top-1 lexical/dense disagreement;
  - P3: head overlap < τ, τ ∈ {0.2, 0.3, 0.4, 0.5};
  - P4: distinct documents in the top-5 ≥ m, m ∈ {3, 4, 5};
  - P5: lexical coverage of the top result < τ, τ ∈ {0.3, 0.5, 0.7};
  - P6: the anchoring heuristic.

That's 15 points in total. (The approved design said 13 adaptive and 16 total; that was an arithmetic slip. The policies and thresholds are exactly as approved.)

**Policy input boundary:** policies read only `TriggerFeatures` (no labels or metadata). Retained gain is always measured against P0@50, never P0.

**Implementation:**
- `retrieval/adaptive_rerank.py`: features, policies, the live `RerankPolicy` adapter, and a dense-hit recorder (so the live path makes no second index query).
- `retrieval/adaptive_eval.py`:
  - the collector, with hybrid-equivalence, pass-consistency and drift checks;
  - the HELPED / HURT / NEUTRAL / RETRIEVAL_MISS counterfactual classes;
  - trigger diagnostics, Random(r) (exact expectation plus 10,000 draws seeded `9e1:k`), the Oracle (USES GROUND TRUTH — NOT DEPLOYABLE), Pareto sets and per-project breakdowns;
  - the fail-closed INVALID reports;
  - the rate-only live-selection rule;
  - the live run and the composed-vs-live comparison.
- Notebooks `07c_adaptive_rerank_collect.py` and `07d_adaptive_rerank_live.py`; scripts `adaptive_rerank_lock.py`, `adaptive_rerank_9e.py` and `adaptive_rerank_9e_live.py`.

**Protocol lock:** `evaluation/adaptive_rerank_9e_lock.json`, sha256 `480d0a1ec02e6e54aeb7c4a86d1122c002e1b0607d76ad74a9571a7d231b591e`. Policy definitions sha256 `dafc0bf5c80acf2491099eb95991def149df51211c29eae7ccecafafd6eb1cff`.

**Production:** `configs/retrieval/retrieval.yaml` is unchanged (`production` null); `retriever.py`, `query.yaml`, chunking, embeddings, the index and the labels are untouched.

**Fresh holdout:** NOT created. It's recorded as a requirement before any strong production-generalisation claim if 9E shows a promising policy, to be revisited in 9F after the 9E review.

**Not run at lock time:** the Databricks collection (07c), the live confirmation (07d) and the real 9E results. All three were later run once each; see the closure checkpoint below.

#### Checkpoint 9E closure — adaptive-rerank diagnostic CLOSED (2026-10-02)

**Status:** Phase 9E is closed experimentally. It was an engineering diagnostic, **not** a production-policy selection experiment. No further 9E experiment is to be run.

**Commits:**
- `bce11d4` protocol lock; `da59423` fail-closed index identity;
- `9a68159` offline frontier recorded (`evaluation/adaptive_rerank_9e.json`, `.md`, `_live_selection.json`);
- `299981b` live-validation preflight hardening (07d fails closed before any live query on dependency health, lock, policy definitions, collection artifact, frontier/selection reproducibility, authorised point, production null, endpoint / index name / row count / corpus / question identity);
- `f60ea45` live result recorded (`evaluation/adaptive_rerank_9e_live.json`, `.md`).

**Integrity:**
- protocol lock `480d0a1ec02e6e54aeb7c4a86d1122c002e1b0607d76ad74a9571a7d231b591e`;
- policy definitions `dafc0bf5c80acf2491099eb95991def149df51211c29eae7ccecafafd6eb1cff`;
- collection artifact `collection__9e1.json` `8b6763e349215899b686fff45f49eb44483f65490167b6649e88014599a63d70`;
- live artifact `live__9e1-live.json` `61a62846325685718c24c80b133d9892ce6040b2ae5d1e7630f9c92852c2c9d6`.

**Results:**
1. The offline collection and frontier were **VALID** (drift vs Phase 8 within 0.001 for P0 and P1; pass-consistent; no hybrid-reconstruction problems).
2. **P0@50:** composed p50 121.1 ms vs live p50 120.9 ms; composed p95 178.7 ms vs live p95 169.4 ms.
3. **P3(0.4):** rerank rate 0.4898 (offline and live); composed p50 178.9 ms vs live p50 196.9 ms; composed p95 6126.7 ms vs live p95 6378.3 ms.
4. Across all 294 live executions (2 points × 49 questions × 3 passes, warm-up included): **zero decision mismatches and zero top-5 mismatches**.
5. P3(0.4) live latency: reranked p50 ≈ 5604.9 ms; non-reranked p50 ≈ 113.9 ms.
6. `cuda_available=false`: the CrossEncoder ran on CPU.
7. The composed and live latencies describe the **warm-cache** path. A first-seen query's embedding adds substantial cold-start latency (≈1.5–2.0 s in the warm-up pass) that the steady-state metrics do not represent.
8. Selective reranking has theoretical value: the ground-truth Oracle (NOT DEPLOYABLE) exceeded always-reranking (MRR 0.7525 vs 0.6827; nDCG@5 0.7898 vs 0.7169) while reranking only the 17 helped questions (rate 0.347).
9. Several deterministic triggers, especially P2 and the P5 variants, showed promising **descriptive** results versus random reranking at the same rate.
10. **No adaptive policy is promoted to production**: the 44 answerable questions were already used during Phase 8 development, so no result here demonstrates generalisation.
11. **P3(0.4) is NOT the production policy.** It was chosen only by the preregistered closest-to-0.5 rerank-rate rule, as the live latency validation point.
12. P179039 remains primarily a **candidate-generation** problem: four answerable questions (q28, q29, q30, q36) had no relevant evidence in the fused top-50. Reranking cannot repair those misses.

**Production state:** `configs/retrieval/retrieval.yaml` `production` remains null (`final_k` 5). No P2/P3/P4/P5/P6 policy is promoted. The Phase 8 retrieval remains the validated retrieval baseline.

**Deferred questions (future work, not authorised now):**
- independent / fresh validation of adaptive reranking (no holdout created in 9E);
- first-stage retrieval improvement for P179039;
- cold-query embedding latency;
- faster reranking options, if production latency requires them;
- whether adaptive reranking should eventually be promoted.

No thresholds were tuned, no other reranker was benchmarked, GPU was not enabled, and candidate depth was not changed.

#### Checkpoint 9F-B — Phase 10 execution contract implemented (2026-10-02; under review, not committed)

**Purpose:** freeze the routing + retrieval contract Phase 10 inherits. No experiment, no new measurement, no Databricks call, no agent code.

**9F-A decisions (user, 2026-10-02):**
- Retrieval baseline: `phase8_quality_baseline` — fixed / hybrid (BM25 + Qwen dense, RRF k 60) / candidate_k 50 / CrossEncoder `ms-marco-MiniLM-L-6-v2` (max_length 512) on every query (`AlwaysRerank`) / final_k 5. It is the PHASE 10 QUALITY BASELINE: not a production policy, not an adaptive winner, not a deployment recommendation.
- Option A: `retrieval.yaml` `production` stays null; the baseline lives only in `configs/phase9_closure.yaml` and is constructed explicitly. The 9E lock keeps passing.
- No adaptive policy promoted (P2–P6 diagnostic only; P3(0.4) only the 9E latency point).
- P179039: NON-BLOCKING KNOWN LIMITATION (artifact-only diagnosis: q28/q29 query-vocabulary gap; q30/q36 lexical evidence pushed outside fused top-50 by RRF, dense does not recover it). Nothing tuned.

**Implementation:**
- `routing/execution.py` — the runtime execution mapping (9D Candidate A). The router output is recorded unchanged; `SEMANTIC_CLASSIFICATION_REQUIRED` → `CLARIFY / INTENT_NOT_RESOLVED`. Fails closed on an invoked semantic fallback, an unknown clarification reason, a CLARIFY/REFUSE that executed a call, or an INVESTIGATION without an unexecuted plan. `router.py` is unchanged.
- `retrieval/contract.py` — `RetrievalProfile` (read from the manifest, checked against the retrieval configuration), `build_document_search`, `RetrievalRequest` (no tunable field; `require_citations` is always true), `RetrievalResult` (`OK` / `NO_EVIDENCE` / `RETRIEVAL_ERROR` / `SCOPE_REFUSED`), `DocumentRetrieval.retrieve` (runs the allowlisted `search_project_documents` tool through the `ToolExecutor`). Temporal scope and document-type hints are recorded, not applied.
- `tools/evidence.py` — `EvidenceStatus`, the exhaustive tool-status mapping, and provenance checks (SYSTEM_DERIVED_SIGNAL only from `get_attention_signals` and the three `get_project_overview` signal counts; a violation is a TOOL_ERROR). INSUFFICIENT_EVIDENCE and CONFLICTING_EVIDENCE are Phase 10 synthesis states only.
- `InvestigationPlan` gains `clarification_state` (always `NONE`: a plan exists only after every clarification check), `provenance_requirements` and `retrieval_profile`; `executed` stays `False`. `DocumentSearch` gains an optional `profile_id`.
- `common/frozen.py` + `scripts/semantic_dev_9d.py` — the 9D debt is closed: the script refuses, before any computation or write, to overwrite a FROZEN decision.
- `configs/phase9_closure.yaml` — the Phase 9 handoff manifest; `tests/unit/test_phase9_contract.py` checks it against the code, configuration and committed artifacts.

**Remaining:** 9F-C — bounded Databricks end-to-end contract validation (integration only; protocol to be approved), then Phase 9 closure.
