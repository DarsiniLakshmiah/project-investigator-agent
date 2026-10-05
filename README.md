# World Bank Project Implementation Intelligence Copilot

An evidence-grounded copilot for World Bank project officers, built on Databricks. For one
project, it answers *what deserves my attention, what changed, when it changed, and what
evidence supports it*. Every published claim cites governed data or a project document,
and is labelled with where it came from.

> **Not a failure-prediction model.** The system never estimates whether a project will
> fail. Such questions are refused and redirected to observed implementation signals.
> Consequential judgments stay with the human officer.

**Reviewers:** start with [docs/REVIEW_GUIDE.md](docs/REVIEW_GUIDE.md), which has a 20-minute
reading order. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) walks through one request end to end.

---

## Contents

1. [What it does](#1-what-it-does)
2. [Design principles](#2-design-principles)
3. [Architecture at a glance](#3-architecture-at-a-glance)
4. [Repository layout](#4-repository-layout)
5. [Results](#5-results)
6. [Running locally](#6-running-locally)
7. [Running on Databricks](#7-running-on-databricks)
8. [Known limitations](#8-known-limitations)
9. [Production evolution](#9-production-evolution)
10. [Project history](#10-project-history)

---

## 1. What it does

Monitoring an operation means piecing together its story from portfolio data, loan
snapshots, procurement records and dozens of PDFs: appraisal documents, Implementation
Status & Results Reports (ISRs), and restructuring and additional-financing papers. The
copilot does that assembly and keeps every step traceable.

**Demo scope:** three Karnataka water-sector operations ([configs/projects.yaml](configs/projects.yaml)).

| Project | Operation | Lifecycle |
|---|---|---|
| P130544 | Karnataka Urban Water Supply Modernization Project (IPF) | mature: restructurings, additional financing, cancellation |
| P179039 | Karnataka Sustainable Rural Water Supply Program (PforR) | mid-lifecycle |
| P506272 | Karnataka Water Security and Resilience Program (PforR) | early lifecycle |

**Example questions**

| Question | What happens |
|---|---|
| *What deserves my attention?* | Deterministic Gold attention signals, with their thresholds and source values |
| *What implementation challenges were documented?* | Document retrieval, then cited documented findings |
| *Why did the PDO rating drop to Moderately Unsatisfactory?* | Investigation that combines the rating history (governed data) with ISR passages |
| *What changed after the restructuring?* | Event-anchored temporal investigation; claims must be dated after the event |
| *Will P130544 fail?* | Refused: prediction is out of scope |
| A question about another project's data | Refused: project isolation |

Each answer separates claims by **provenance**:

| Label | Meaning |
|---|---|
| `FACT` | a value from governed structured data (Silver/Gold Delta tables) |
| `DOCUMENTED_FINDING` | what a project document states |
| `SYSTEM_DERIVED_SIGNAL` | output of a deterministic, configured Gold rule |
| `AI_INTERPRETATION` | the model connecting cited evidence; labelled so it is never mistaken for an official statement |
| `UNKNOWN` | the evidence is missing or ambiguous; never published as an answer |

## 2. Design principles

- **Use the simplest reliable capability for each problem.** Code handles exact work:
  identity, dates, arithmetic, permissions, citations and thresholds. Retrieval finds
  evidence. LLMs are used only for open-ended planning, synthesis and semantic review.
- **Models propose, code decides.** The Investigator LLM proposes tool calls and searches.
  Deterministic code validates them, injects the project, executes them, attaches
  citations and decides what gets published.
- **Project isolation before retrieval.** Every read and every search is scoped to the
  project before any data can reach a model. Foreign rows fail closed.
- **Provenance is mandatory.** Every document-derived fact keeps its document, page,
  section and extraction method.
- **Refusing is a valid outcome.** `INSUFFICIENT_EVIDENCE`, `CLARIFY` and `REFUSE` are
  preferred over a fabricated answer.
- **Every probabilistic component must beat a simpler baseline.** Chunking, retrieval,
  reranking and routing were each chosen by controlled experiment. More complex
  candidates (similarity and LLM routing fallbacks, adaptive reranking) were measured
  and **not promoted** because they did not beat the simpler baseline.

The full engineering brief this project was built against is
[Claude.md](Claude.md). It was used as the standing instruction file for AI-assisted
development.

## 3. Architecture at a glance

```
World Bank sources (portfolio workbook, loan + procurement CSVs, 53 project PDFs)
  -> Bronze      source-aligned, hashed, lineage-preserving              (ingestion/)
  -> Silver      normalized facts + document-derived facts via Docling   (transformations/, parsing/, extraction/)
  -> Gold        project_360, timeline, result_progress, risk_register,
                 attention_signals (14 deterministic rules)              (intelligence/)
  -> Retrieval   chunk corpus + Databricks Vector Search index           (retrieval/)

Online request: Copilot.investigate(query, project_id)                   (copilot/)
  -> Router            deterministic fast path: refuse / clarify / confident structured answer
  -> Investigator LLM  <= 2 rounds, <= 6 governed actions; proposes, never executes
  -> govern()          allowlist + schema validation; project injected by code
  -> Governed executor 7 read-only tools (Delta) + project-scoped hybrid document search
  -> Temporal anchor   event dates confirmed against the governed timeline
  -> Packing           bounded context; evidence as request-local handles (E1, E2, ...)
  -> Synthesizer LLM   claims over handles only; code attaches identity and citations
  -> Integrity checks  scope, provenance, citations, temporal rules (deterministic)
  -> Critic LLM        per-claim SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED / CONTRADICTED
  -> Finalizer         publishes only integrity-valid, reviewed claims
  -> InvestigationResult + allowlisted MLflow trace (counts and codes, never text)

UI: Databricks App (Streamlit) -> Databricks Job (notebook 15) -> Copilot.investigate()
```

| Component | Choice |
|---|---|
| LLM (Investigator, Synthesizer, Critic) | `databricks-qwen35-122b-a10b` via Model Serving |
| Embeddings | `databricks-qwen3-embedding-0-6b` (1024-d), self-managed vectors |
| Retrieval | BM25 top 50 + Vector Search top 50, fused with reciprocal rank fusion (k = 60) |
| Reranker | CrossEncoder `ms-marco-MiniLM-L-6-v2`; top 5 passages per search |
| Storage and governance | Unity Catalog, Delta (versions pinned per run), Volumes |
| Document parsing | Docling (TableFormer, OCR off); no LLM extraction was needed |

The details are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 4. Repository layout

```
src/worldbank_copilot/   all logic (notebooks are thin entry points)
  copilot/               ** the online runtime: Copilot.investigate() **
  tools/                 7 governed, read-only, project-scoped tools + TableReader
  retrieval/             chunking, embeddings, BM25 / Vector Search / RRF, CrossEncoder
  routing/               deterministic router (fast path, refusals, clarification)
  investigation/         shared evidence / claim contracts and the claim validator
  intelligence/          Gold layer and the attention-signal rules engine
  ingestion/ parsing/ extraction/ transformations/ lakehouse/   data pipeline
  evaluation/            50-question application evaluation harness
  common/                configuration, settings, logging, project registry
  validation/            frozen acceptance gates (see "Project history")
copilot_app/             Databricks App: Streamlit UI + Job client
configs/                 environments, projects, Gold rules, retrieval, routing, runtime
notebooks/               Databricks entry points: pipeline 01-07, runtime 14-16
evaluation/              golden datasets, experiment reports, frozen locks
tests/                   unit (offline, fake models), integration, Spark
scripts/                 local CLI entry points and experiment tooling
sql/                     read-only validation queries per layer
docs/                    architecture and review guides
```

Each top-level folder with a non-obvious role has its own README:
[notebooks/](notebooks/README.md), [copilot_app/](copilot_app/README.md),
[evaluation/](evaluation/README.md), [scripts/](scripts/README.md), [data/](data/README.md).

## 5. Results

**Data foundation** (validated in Databricks): 53/53 PDFs parsed; 35 ISR snapshots, 837
results observations, 237 indicators, 89 appraisal risks/findings and 27 project events
extracted deterministically, all with source provenance.

**Retrieval** (49 labelled questions, standalone, selected configuration):
Recall@5 0.80, Recall@10 0.84, MRR 0.68, nDCG@5 0.72.

**Application evaluation** (50 questions across 10 categories, run end to end through the
real Databricks runtime; [evaluation/app_eval_cases.jsonl](evaluation/app_eval_cases.jsonl)):

| Metric | Result |
|---|---|
| Verdicts | 35 PASS, 7 PARTIAL, 8 FAIL |
| Task success | 72% |
| Runtime success | 98% |
| Project isolation | 100% |
| Citation validity / completeness / project consistency | 100% / 100% / 100% |
| Unsupported or unverified claims published | 0 |
| Published claims fully supported by the Critic | 91% |
| Refusal correctness | 88% |
| Application-level Recall@10 | about 0.33 |

The drop from 0.84 standalone recall to about 0.33 in the application comes from three
things: the Investigator writes its own queries, each search keeps only 5 passages, and
some document-labelled questions are answered from structured data instead. Failures were
categorised and kept, not tuned away. See [evaluation/README.md](evaluation/README.md).

## 6. Running locally

Local runs need no Databricks access. Unit tests use fake models and in-memory tables.

```powershell
py -3.14 -m venv .venv                                  # Python >= 3.11
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest                        # ~2,080 offline unit tests
.\.venv\Scripts\ruff check .
.\.venv\Scripts\ruff format --check .
```

Optional suites (deselected by default):

| Command | Needs |
|---|---|
| `pytest -m integration` | source files placed under `data/` ([data/README.md](data/README.md)) |
| `pytest -m docling` | `pip install -e ".[documents]"` (real Docling models; slow) |
| `pytest -m spark` | the Spark environment below |

### Testing Gold transformations (Spark)

The Gold code uses only JVM-native Spark functions, so it is tested on a real local Spark
session in a separate environment:

```powershell
py -3.12 -m venv .venv-spark
.\.venv-spark\Scripts\python -m pip install "pyspark==4.0.1" pytest -r requirements-databricks.txt
.\.venv-spark\Scripts\python -m pip install -e . --no-deps
# a JDK 17: set JAVA_HOME, or unpack one into .tools\ (git-ignored)
.\.venv-spark\Scripts\python -m pytest -m spark
```

These tests cover a synthetic scenario for every rule, plus real Silver rows exported by
`scripts/platformize.py`. Delta MERGE and idempotency are proven only by the Databricks run.

Local CLI entry points (parse, extract, validate, dry-run platformization) are described in
[scripts/README.md](scripts/README.md). Configuration and environment overrides are in
[configs/](configs/) and [.env.example](.env.example).

## 7. Running on Databricks

1. Clone the repository as a **Databricks Git folder**.
2. Upload the source files to the Unity Catalog Volume described in [data/README.md](data/README.md).
3. Run the pipeline notebooks `01` to `07` in order. This builds Bronze, Silver, Gold,
   the chunk corpus and the Vector Search index.
4. Run `14_copilot_prototype_validation` for a live end-to-end check of the runtime.
5. Create a Job on notebook `15_copilot_app_backend` (parameters `project_id`, `question`)
   and deploy [copilot_app/](copilot_app/README.md) as a Databricks App bound to that Job.
6. Optionally run `16_application_evaluation` (the 50-question evaluation).

**Requirement files.** Notebooks install pinned, notebook-scoped dependencies with `-c constraints-databricks.txt`:

| File | Purpose |
|---|---|
| `requirements-databricks.txt` | core pins (pydantic, PyYAML, openpyxl, pypdfium2) |
| `requirements-retrieval.txt` | Vector Search client (`databricks-ai-search`) |
| `requirements-reranker.txt` | CrossEncoder (`sentence-transformers`, matching the ML runtime) |
| `requirements-phase10d.txt` | `jsonschema` for strict validation of model output |
| `constraints-databricks.txt` | stops pip from replacing runtime-owned packages (protobuf, grpcio-status) |
| `copilot_app/requirements.txt` | the App only (Streamlit + Databricks SDK) |

## 8. Known limitations

- **Latency.** Each App question runs as one serverless Job. Most of the roughly 4–5 minutes
  is environment setup (notebook-scoped `pip install`, then building the runtime), not
  reasoning.
- **Application-level retrieval recall is well below standalone recall** (see Results).
- **Model capability.** The Qwen endpoint did not pass the earlier bounded
  model-capability protocol (12/19 calls). The deterministic integrity checks and the
  Critic exist to contain this, and the App shows that note with every model-generated answer.
- One pinned router rule misses project-ID prediction phrasing ("Will P130544 fail?"). The
  Investigator still refuses it, at the cost of one model call.
- Comparison answers can be narrower than ideal when the evidence covers different issues
  on each side of an event.

## 9. Production evolution (not implemented)

- **Cut cold start:** move the notebook `%pip` installs into a cached serverless Job
  environment. Longer term, run the runtime in a warm process (Databricks App backend or
  Model Serving). The `TableReader` protocol already isolates tools from Spark, so a
  Databricks SQL reader is a drop-in.
- **Initialise once:** load the corpus, reranker and clients once per process instead of
  once per request.
- **Close the retrieval gap:** run query-rewrite and depth experiments against the
  application-level metric.
- **Add components only behind a baseline:** memory, caching and per-user ACLs would be
  added only once evaluation shows a benefit over the simpler path.

## 10. Project history

The system was built in gated phases. Each phase was tested, validated on real data and
approved before the next began. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) is the
full chronological engineering log: decisions, data findings, experiments and validation
results.

Some code exists only to keep earlier accepted results reproducible. It is **not part of
the online request path**:

- `validation/` holds the frozen Phase 9 gate and the Phase 10C harness. `build_copilot`
  re-verifies the accepted retrieval configuration by hash before it serves requests.
- `routing/` experiment modules (`semantic*`, `bounded_classifier*`, `semif_*`) and
  `retrieval/adaptive_*` hold rejected candidates, kept with their evaluation evidence.
- The empty `agents/`, `api/`, `guardrails/` and `observability/` packages are pinned by
  the Phase 10C lock.
- Notebooks `07a`–`07e`, `08*` and `09` are experiment and acceptance notebooks
  ([notebooks/README.md](notebooks/README.md)).

These files are hash-locked ([evaluation/phase10c_evidence_lock.json](evaluation/phase10c_evidence_lock.json)),
so they were documented rather than moved. Lock files contain only hashes. Datasets and
PDFs are never committed.

## License

See [LICENSE](LICENSE).
