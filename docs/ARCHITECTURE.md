# Architecture

This document follows one question through the system, then describes the offline data
pipeline that the question depends on. Paths are relative to `src/worldbank_copilot/`
unless stated otherwise.

The guiding rule: **models decide *what* (which evidence is useful, what it means, whether a
claim is supported); deterministic code decides *how* (identity, project scope, tool
permissions, dates, citations, publication).**

---

## 1. Online request path

Entry point: `Copilot.investigate(query, project_id)` in [copilot/service.py](../src/worldbank_copilot/copilot/service.py).
It is wired against live Databricks components by `build_copilot()` in
[copilot/databricks.py](../src/worldbank_copilot/copilot/databricks.py).

```
question + project_id
  1  project check            unsupported project -> REFUSE
  2  router fast path         refuse / clarify / confident structured answer (no model)
  3  Investigator  (LLM)      objective + proposed actions, <= 2 rounds, <= 6 calls
  4  govern()                 allowlist, schema validation, project injected by code
  5  GovernedExecutor         tools (Delta) and document search (hybrid RAG)
  6  temporal anchor          event date confirmed against the governed timeline
  7  packing                  bounded context; evidence becomes handles E1..En
  8  Synthesizer   (LLM)      claims over handles + interpretation flag
  9  enrich + integrity       code resolves handles, attaches provenance and citations, validates
 10  Critic        (LLM)      per-claim support verdict
 11  finalize                 publish, qualify or drop each claim
 12  InvestigationResult      status, claims, citations, limitations, activity, trace
```

### 1.1 Router fast path ([routing/](../src/worldbank_copilot/routing/))

A deterministic `RoutingService` handles requests that need no model:

- **refusals**: failure prediction, a question naming another project, an out-of-scope request;
- **clarification**: an ambiguous indicator or event reference;
- **confident structured answers**: for example "What deserves my attention?", which reads
  `gold.attention_signals` directly and returns `EVIDENCE_ONLY`.

Document search is disabled on the fast path. Anything that is not clearly structured goes
to the Investigator. Routing was chosen by experiment: deterministic rules plus targeted
clarification were selected after similarity-based and LLM routing fallbacks were rejected on the frozen development set
(see [evaluation/README.md](../evaluation/README.md)).

### 1.2 Investigator ([copilot/investigator.py](../src/worldbank_copilot/copilot/investigator.py))

The Investigator is the runtime's only planner. It returns a typed `InvestigatorDecision`
containing:

- an objective, or a disposition (refuse or clarify);
- actions: governed tool calls and/or document searches with free-text queries;
- an optional temporal anchor (`BEFORE` / `AFTER` / `COMPARE` / `ACROSS` an event).

It never writes SQL, chooses a project or executes anything. `govern()` rejects any action
outside the tool catalog, validates its arguments against the tool schema, and injects
the project ID itself. Rejected actions are recorded and reported back in round 2.

### 1.3 Governed execution ([copilot/governed.py](../src/worldbank_copilot/copilot/governed.py), [tools/](../src/worldbank_copilot/tools/))

Eight governed actions exist:

| Tool | Reads |
|---|---|
| `get_project_overview` | `gold.project_360` |
| `get_project_timeline` | `gold.project_timeline` |
| `get_rating_history` | `silver.isr_snapshots` |
| `get_financial_status` | `silver.loans`, `silver.isr_loan_disbursements`, `silver.project_events` |
| `get_results_progress` | `gold.result_progress` |
| `get_risk_register` | `gold.risk_register` |
| `get_attention_signals` | `gold.attention_signals` |
| `search_project_documents` | hybrid retrieval over the chunk corpus |

Each tool describes a `ReadRequest`. The `TableReader` protocol
([tools/reader.py](../src/worldbank_copilot/tools/reader.py)) enforces four things:

- the table allowlist;
- the project predicate;
- contract-checked columns;
- row bounds.

Any foreign-project row raises `GovernanceViolation`; it is never filtered silently.
`SparkTableReader` is the Databricks implementation, and tests use an in-memory one.

### 1.4 Document retrieval ([retrieval/](../src/worldbank_copilot/retrieval/))

```
query (written by the Investigator) + project_id
  -> project filter (applied before scoring)
  -> BM25 top 50  ||  Vector Search top 50  (Qwen3-Embedding 0.6B, query embedded by our code)
  -> reciprocal rank fusion (k = 60)         -> 50 candidates
  -> isolation re-check against the governed corpus, de-duplication
  -> CrossEncoder ms-marco-MiniLM-L-6-v2     -> top 5 passages
```

Text, citations and provenance always come from the governed Delta corpus
(`silver.document_chunks`), never from the index payload. The configuration was selected
by staged experiments:

- chunking strategies;
- lexical vs dense vs hybrid retrieval;
- no reranker vs CrossEncoder;
- candidate depth.

The experiments are in [configs/retrieval/](../configs/retrieval/) and
[evaluation/](../evaluation/README.md).

### 1.5 Temporal anchoring

When the Investigator anchors a question to an event ("after the restructuring"), code
resolves that event against `gold.project_timeline` and confirms its date. The model's
date is never trusted. Each evidence item is then labelled with a period taken from its
**source date** (for example the ISR date or document date):

- `BEFORE` and `AFTER` claims need dated evidence on the correct side of the event;
- `COMPARE` claims need evidence from both sides;
- undated evidence cannot establish timing.

### 1.6 Packing and evidence handles

The packed context is bounded (40 KB, 1,500 characters per passage) and assembled
round-robin by relevance. The model sees evidence only as request-local handles (`E1`,
`E2`, …), never real record or document identifiers, so it cannot invent a citation.

### 1.7 Synthesizer, enrichment and integrity ([copilot/semantic.py](../src/worldbank_copilot/copilot/semantic.py))

The Synthesizer returns only semantic content: claim text, the handles it relies on, and
an `interpretation` flag. Deterministic enrichment then does the following:

- resolves handles to real evidence and attaches project, period, citations and provenance;
- assigns `AI_INTERPRETATION` when the claim connects or explains beyond what any single
  source says (the only claim type allowed to mix provenance);
- otherwise assigns the single provenance of its evidence. A factual claim that mixes
  provenance is dropped (`PROVENANCE_VIOLATION`).

Each claim is then checked on its own by `validate_claims`
([investigation/synthesis.py](../src/worldbank_copilot/investigation/synthesis.py)):

- **Response-level failures** (project isolation, invalid evidence reference, citation
  failure, temporal scope violation) mark the whole response `FAIL_CLOSED`.
- **Claim-level failures** remove only that claim. For example, an interpretation built on
  `UNKNOWN` evidence is removed.

### 1.8 Critic and finalizer ([copilot/finalizer.py](../src/worldbank_copilot/copilot/finalizer.py))

The Critic grades each integrity-valid claim against its cited evidence.

| Critic verdict | Publication |
|---|---|
| `SUPPORTED` | published ("Supported by cited evidence") |
| `PARTIALLY_SUPPORTED` | published with its qualifier |
| `UNSUPPORTED`, `CONTRADICTED` | removed |
| Critic call fails | integrity-valid claims publish as `NOT_ASSESSED`, with a limitation |

A model-written limitation is published only if the Critic finds it grounded. If no claim
survives, the result is `INSUFFICIENT_EVIDENCE`.

### 1.9 Result and observability

`InvestigationResult` ([copilot/contracts.py](../src/worldbank_copilot/copilot/contracts.py))
carries the following:

- **status:** `ANSWER`, `EVIDENCE_ONLY`, `INSUFFICIENT_EVIDENCE`, `CLARIFY`, `REFUSE` or
  `FAIL_CLOSED`;
- **content:** published claims with provenance and support, citations, evidence, attention
  signals and limitations;
- **activity:** an activity summary, model-call metrics and validation counts.

MLflow traces and request metrics ([application/observability.py](../src/worldbank_copilot/application/observability.py))
are **allowlisted**. They contain counts, codes, latencies and token usage, and never the
question, claim or evidence text.

### 1.10 Bounds

| Bound | Value | Where |
|---|---|---|
| Investigator rounds | 2 | [configs/copilot.yaml](../configs/copilot.yaml) |
| Governed calls per request | 6 | `configs/copilot.yaml` |
| Context size | 40 KB | `configs/copilot.yaml` |
| Model output tokens / timeout | 5,000 / 60 s | `configs/copilot.yaml` |
| Request deadline | 180 s | `configs/copilot.yaml` |
| Passages per search | 5 | [configs/retrieval/](../configs/retrieval/) |

There are no unbounded loops anywhere in the runtime.

---

## 2. Offline data pipeline

Each layer has a thin notebook entry point in [notebooks/](../notebooks/README.md). All logic
lives in the package and is unit-tested locally.

| Layer | Notebook | Package | Output |
|---|---|---|---|
| Bronze | 01 | `ingestion/`, `transformations/bronze.py` | source-aligned tables with source path, hash and ingestion metadata |
| Parsing | 02 | `parsing/` | Docling parse of 53 PDFs (TableFormer, OCR off), cached in a Volume |
| Structured Silver | 03 | `transformations/` | normalized project, loan, financial and procurement facts |
| Document Silver | 04 | `extraction/` | ISR snapshots, results, appraisal risks and events, with page-level provenance |
| Governed Delta | 05 | `lakehouse/` | Unity Catalog tables, data contracts, deterministic IDs, reconciliation |
| Gold | 06 | `intelligence/` | `project_360`, `project_timeline`, `result_progress`, `risk_register`, `attention_signals` |
| Retrieval | 07 | `retrieval/` | chunk corpus, embedding cache, Vector Search index, retrieval evaluation |

**Extraction is deterministic.** The method order is:

1. Docling tables;
2. labelled key/value fields;
3. regex;
4. targeted `pypdfium2` text fallback on specific failing pages.

No LLM extraction was needed. Every extracted fact records its extraction method.

**Attention signals are deterministic Gold rules**
([configs/intelligence/attention_rules.yaml](../configs/intelligence/attention_rules.yaml),
[intelligence/signals.py](../src/worldbank_copilot/intelligence/signals.py)). Each rule has a
documented threshold and emits `INFO`, `WATCH` or `HIGH` together with the source values it
used. Two examples:

- Disbursement lag compares elapsed time with disbursed share: 90% elapsed and a 31-point
  gap gives `WATCH` (thresholds 25 and 40).
- Results progress below 50% of the baseline-to-target distance gives `HIGH`.

Signals are observations, not predictions.

---

## 3. Security and isolation

- **Project scope is enforced before data reaches a model.** It applies to tool reads
  (the `TableReader` predicate), retrieval (a filter before scoring plus a post-retrieval
  re-check) and claims (an integrity check on every citation).
- **Retrieved text is untrusted evidence, not instructions.** The model gets it only
  inside the evidence view.
- **The App validates input and scope** ([copilot_app/backend.py](../copilot_app/backend.py)).
  It limits question length, rejects credential-like input and only accepts results whose
  scope matches the request.
- **Isolation is tested.** Adversarial cases in the unit suite and the application
  evaluation (Project B data while scoped to Project A) give 100% isolation.

---

## 4. Deployment

```
Databricks App (copilot_app/, Streamlit)
  -> Jobs API: run_now(project_id, question)
  -> Job: notebooks/15_copilot_app_backend (serverless)
       %pip install (pinned, constrained)  -> _bootstrap  -> build_copilot(spark)
       -> copilot.investigate()  -> InvestigationResult JSON via notebook exit
  <- App validates and renders claims, provenance badges, citations, signals and activity
```

`build_copilot` runs these steps before it serves requests:

1. a dependency-health check;
2. the frozen Phase 9 gate, which hash-verifies the accepted retrieval configuration;
3. verification of the Vector Search index;
4. pinning of the Delta table versions used for the run.

---

## 5. What was evaluated and rejected

| Hypothesis | Outcome |
|---|---|
| Lexical / embedding-similarity routing fallback (B1, B2) | rejected on the frozen DEV set |
| LLM routing fallback (GPT-OSS-20B bounded classifier) | passed the capability check, rejected on DEV quality and repeatability (net −2 vs rules) |
| SemIf / OpenJev bounded decision model | not run: blocked by the execution environment (not a quality result) |
| Selected routing | deterministic rules + targeted clarification |
| Adaptive (confidence-gated) reranking | descriptive only; no policy promoted |
| LLM extraction from PDFs | unnecessary; deterministic extraction met the targets |
| Separate agents for finance, procurement and results | implemented as tools instead |

Reports for each experiment are in [evaluation/](../evaluation/README.md). The chronology is
in [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md).
