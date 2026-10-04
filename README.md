# World Bank Project Implementation Intelligence Copilot

An evidence-grounded implementation-intelligence system for World Bank project officers,
built on Databricks. It answers *what deserves my attention, what changed, when, and what
evidence supports it*, with citations to governed data and project documents.

> **This is an evidence-grounded implementation intelligence system, not a
> failure-prediction model.** It never estimates the probability that a project fails;
> such questions are refused and redirected to observed implementation signals.

---

## 1. Problem

Monitoring an operation means piecing its story together from portfolio data, loan
snapshots, procurement records and dozens of PDFs (appraisal documents, Implementation
Status & Results Reports, restructuring and additional-financing papers). Officers need to
know what deserves attention, what changed and why, and which evidence supports it,
without reading every document by hand.

## 2. Demo scope

Three Karnataka water-sector operations ([configs/projects.yaml](configs/projects.yaml)):

| Project | Operation | Role |
|---|---|---|
| P130544 | Karnataka Urban Water Supply Modernization Project (IPF) | mature; richest history (restructurings, additional financing, cancellation) |
| P179039 | Karnataka Sustainable Rural Water Supply Program (PforR) | mid-lifecycle |
| P506272 | Karnataka Water Security and Resilience Program (PforR) | early lifecycle |

## 3. Architecture

```
User question + project
  -> Router                    deterministic fast path: refuse / clarify / confident structured answer
  -> Investigator (LLM)        <= 2 rounds, <= 6 governed actions; proposes, never executes
  -> Governed execution        allowlisted tools (Delta) and Phase 8 document retrieval
  -> Evidence                  project-owned, source-identified, date-labelled
  -> Synthesizer (LLM)         claims over request-local evidence handles
  -> Deterministic integrity   identity, provenance, citations, project scope, temporal checks
  -> Critic (LLM)              claim-level SUPPORTED / PARTIAL / UNSUPPORTED / CONTRADICTED
  -> Finalizer                 publishes only integrity-valid, reviewed claims
  -> InvestigationResult       answer + citations + limitations + MLflow trace
```

Code decides *how* (identity, project, dates, tools, publication); models decide *what*
(planning, synthesis, semantic review). The single runtime is
`Copilot.investigate(query, project_id)` in
[src/worldbank_copilot/copilot/](src/worldbank_copilot/copilot/).

## 4. Data pipeline (medallion on Unity Catalog / Delta)

| Layer | Notebook | Code |
|---|---|---|
| Sources -> Bronze | 01 | `ingestion/` |
| PDF parsing (Docling) | 02 | `parsing/` |
| Structured Silver | 03 | `transformations/` |
| Document-derived Silver (ISRs, results, risks, events) | 04 | `extraction/` |
| Governed Delta foundation | 05 | `lakehouse/` |
| Gold intelligence (project_360, timeline, results, risks, attention signals) | 06 | `intelligence/` |
| Retrieval corpus + Vector Search index | 07 | `retrieval/` |

Every document-derived fact keeps its provenance (document, page, section, extraction
method). No LLM extraction was needed.

## 5. Agent architecture

- **Investigator** ([investigator.py](src/worldbank_copilot/copilot/investigator.py)):
  states the objective, picks governed tools and document searches, and may anchor the
  question to an event (BEFORE / AFTER / COMPARE). Proposals are validated by `govern()`;
  the model never writes SQL or chooses the project.
- **Governed executor** ([governed.py](src/worldbank_copilot/copilot/governed.py)):
  runs validated calls, confirms event dates against the governed timeline, labels evidence
  periods from source dates and packs a bounded context.
- **Synthesizer / Critic** ([semantic.py](src/worldbank_copilot/copilot/semantic.py)):
  bounded contracts over evidence handles; the Critic grades each claim.
- **Finalizer** ([finalizer.py](src/worldbank_copilot/copilot/finalizer.py)): claim-level
  publication; Critic failure degrades to integrity-valid claims marked NOT_ASSESSED.

## 6. RAG (Phase 8, accepted profile)

Fixed chunking; hybrid retrieval: BM25 + Databricks Vector Search
(`databricks-qwen3-embedding-0-6b`) fused with reciprocal rank fusion (k=60) over 50
candidates; CrossEncoder rerank (`ms-marco-MiniLM-L-6-v2`); 5 passages per search; project
filter applied before retrieval. Standalone evaluation on 49 labelled questions:
**Recall@10 0.84, MRR 0.68** ([evaluation/adaptive_rerank_9e.json](evaluation/adaptive_rerank_9e.json)).

## 7. Governed tools

Seven read-only, typed, project-scoped tools ([src/worldbank_copilot/tools/](src/worldbank_copilot/tools/)):
project overview, timeline, rating history, financial status, results progress, risk
register, attention signals, plus document search. Tools describe a `ReadRequest`; the
`TableReader` enforces the table allowlist, the project predicate, contract-checked columns
and row bounds.

## 8. Guardrails and deterministic enforcement

- project isolation at every read, retrieval and claim; foreign rows fail closed;
- evidence handles instead of identifiers in model context; code attaches citations;
- provenance classes (FACT, DOCUMENTED_FINDING, SYSTEM_DERIVED_SIGNAL, AI_INTERPRETATION,
  UNKNOWN); UNKNOWN values are never published as answers;
- temporal integrity: BEFORE/AFTER claims need source-dated evidence on that side of a
  governed event; cross-event claims need both sides; undated evidence cannot establish timing;
- prediction refusal; model-written limitations publish only when the Critic grounds them;
- traces carry counts and codes only, never question, claim or evidence text.

## 9. Evaluation

**Final application evaluation** (run `app-eval-001`, 50 questions, 10 categories x 5,
through the real Databricks runtime; golden set in
[evaluation/app_eval_cases.jsonl](evaluation/app_eval_cases.jsonl), harness in
[src/worldbank_copilot/evaluation/](src/worldbank_copilot/evaluation/), results persisted
in the Databricks artefact Volume under `application_evaluation/app-eval-001/`):

| Metric | Result |
|---|---|
| Verdicts | 35 PASS, 7 PARTIAL, 8 FAIL (70% strict, 84% PASS + PARTIAL) |
| Task success | 72% |
| Runtime success | 98% |
| Fully supported published claims | 91% |
| Citation validity / completeness / project consistency (where applicable) | 100% / 100% / 100% |
| Project isolation | 100% |
| Unsupported or unverified claims published | 0 |
| Refusal correctness | 88% |
| Application-level Recall@10 | about 0.33 (vs 0.84 standalone Phase 8) |

The gap between application-level and standalone retrieval reflects Investigator-written
queries, 5 passages per search and structured answers to document-labelled questions, not a
retrieval regression. Failures were kept and categorised rather than tuned away.

## 10. UI and deployment

```
Databricks App (copilot_app/) -> Databricks Job -> notebook 15 -> build_copilot(spark) -> investigate()
```

The App is a presentation layer; each question runs as one Job on serverless compute
through the validated Spark-based Copilot. **Known limitation:** serverless Job cold start
dominates interactive latency.

## 11. Repository structure

```
copilot_app/            current Databricks App (Streamlit UI + Job client)
configs/                runtime, retrieval, routing, Gold-rule and environment configuration
data/                   placement instructions for source files (sources are not committed)
evaluation/             application golden set, Phase 8/9 golden sets, frozen locks and reports
notebooks/              pipeline (01-07), demo (14), App backend (15), evaluation (16)
review/                 human-reviewed indicator alias candidates
scripts/                local pipeline entry points and frozen-protocol lock tooling
sql/                    Phase 6-8 validation queries
src/worldbank_copilot/  all logic (notebooks are thin)
tests/                  offline unit tests (fake models), integration and Spark tests
```

**Current project story**

| Path | Role |
|---|---|
| `notebooks/01`–`07` | data and RAG pipeline |
| `notebooks/14_copilot_prototype_validation.py` | live technical demo of `copilot.investigate` |
| `notebooks/15_copilot_app_backend.py` | Job backend for the Databricks App |
| `notebooks/16_application_evaluation.py` | 50-question application evaluation |
| `copilot_app/` | current UI |
| `src/worldbank_copilot/copilot/` | current orchestration (the online runtime) |
| `src/worldbank_copilot/tools/` | governed tools and table reader |
| `src/worldbank_copilot/retrieval/` | RAG |
| `src/worldbank_copilot/routing/` | deterministic router (fast path and refusals) |
| `src/worldbank_copilot/evaluation/` + `evaluation/` | evaluation framework and evidence |

**Frozen validation / reproducibility (not part of the online architecture)**

`build_copilot` runs a build-time gate that hashes the accepted Phase 8/9 configuration
and artefacts, and the offline test runtime reuses the Phase 10C harness. These are kept
only for that reason: notebooks `07a`–`07e`, `08*` and `09_phase10c_evidence_validation`;
`validation/` (Phase 9 gate, Phase 10C harness, Phase 10D fixtures); `investigation/`
(shared claim/evidence contracts reused by the runtime and the older 10C path);
experimental routing modules (`semantic*`, `bounded_classifier*`, `semif_*`, `review`,
`evaluation`); `retrieval/adaptive_*` and `endpoint_probe`; the empty `agents/`, `api/`,
`guardrails/` and `observability/` packages; Phase 9C/9D/9E files in `evaluation/` and the
matching scripts; and [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), the phase log.

## 12. Demo notebooks

- **14** — runs scenarios through `copilot.investigate` on real data and checks
  architectural invariants (scope, routing, model calls, citations, provenance).
- **15** — the App's Job: one question in, one `InvestigationResult` JSON out.
- **16** — the application evaluation: resumable, per-case persistence, layered metrics.

## 13. Known limitations

- serverless Job cold start makes the App slow for interactive use;
- application-level retrieval recall is well below standalone retrieval;
- one pinned router rule misses project-ID prediction phrasing ("Will P130544 fail?"); the
  Investigator refuses it, at the cost of one model call;
- the Qwen endpoint did not pass the earlier bounded model-capability protocol; the
  deterministic and Critic layers exist to contain that;
- comparison answers can be narrower than ideal when evidence covers different issues on
  each side of an event.

## 14. Production evolution (not implemented)

- the existing `TableReader` protocol already isolates tools from Spark: a Databricks
  SQL-backed reader would let the Copilot run inside a long-lived App process;
- initialise the corpus, reranker and clients once per process instead of once per Job;
- per-request model adapters for concurrent sessions; an explicit MLflow experiment.

## Local development

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev,documents]"
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\ruff check .
```

Local tests use fake models and never call Databricks. Place source files as described in
[data/README.md](data/README.md).
