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
| 8 | Databricks-native retrieval foundation + experiments | **Corpus, Qwen embeddings and AI Search index validated in Databricks (2026-10-01); retrieval experiments pending** |
| 9–13 | See §3 (roadmap from Claude.md §36) | Not started |

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
3. **Lexical retrieval implementation** (Phase 8): Databricks Vector Search hybrid
   mode vs. a separate BM25 library — decide in Phase 8 with evaluation.
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

### Phase 8 — Databricks-native retrieval foundation (implemented; Databricks validation pending)

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
- **Not yet run:** retrieval experiments (Steps 4–7, notebook 07b). Phase 9 has not been started.
