# World Bank Project Implementation Intelligence Copilot

## 1. Project Mission

We are building a production-style, evidence-based Project Implementation Intelligence Copilot for World Bank projects.

The system should help a project officer answer:

- What deserves my attention?
- What changed?
- When did it change?
- What evidence explains the change?
- What structured facts support the conclusion?
- Which claims are facts, derived signals, documented findings, or AI interpretations?

The system is NOT a generic chatbot.

The system is NOT intended to predict whether a World Bank project will fail.

The system should surface observable implementation signals and allow users to investigate them using governed structured data and authoritative project documents.

Human users remain responsible for consequential project judgments.

---

# 2. Core Engineering Philosophy

The architecture follows one central principle:

Use the simplest reliable capability for each problem.

Do not use an LLM where deterministic software is sufficient.

The target capability hierarchy is:

DETERMINISTIC CODE
→ exact operations

RETRIEVAL
→ evidence discovery

JEV-LIKE DECISION MODEL
→ bounded decisions

SLM
→ lightweight semantic tasks

STRONG LLM
→ ambiguity, investigation, synthesis and open-ended reasoning

HUMAN
→ consequential judgment and escalation

Every probabilistic component must earn its place through evaluation.

Do not add a technique simply because it is fashionable or available in a framework.

---

# 3. Engineering Model

Think about the AI system through the following engineering layers.

## Prompt Engineering

Controls how an individual model call behaves.

Examples:

- task instructions
- output contracts
- examples
- constraints
- evidence requirements
- citation requirements

Prompt engineering should not compensate for missing context, poor retrieval or weak application control.

## Context Engineering

Controls what information a model receives.

Examples:

- retrieved evidence
- structured facts
- tool outputs
- conversation state
- memory
- project scope
- temporal scope

Treat context as a limited budget.

Do not send information simply because the model supports a large context window.

## Harness Engineering

Controls what models are allowed to do.

Examples:

- tools
- schemas
- authentication
- authorization
- ACLs
- retries
- timeouts
- validation
- state
- guardrails
- logging
- human approval

The model may decide WHEN a capability is needed.

Deterministic software controls HOW that capability executes.

## Graph Engineering

Controls which execution path runs.

Examples:

STRUCTURED QUESTION
→ deterministic tool / SQL

DOCUMENT QUESTION
→ RAG

COMPLEX INVESTIGATION
→ agent workflow

Do not route every request through the complete agent graph.

## Loop Engineering

Controls adaptation after observing results.

Examples:

retrieve
→ evaluate evidence
→ retry if necessary
→ stop or escalate

All loops must have explicit stopping conditions.

No unbounded autonomous loops.

## Evaluation Engineering

Evaluation is part of the architecture, not an afterthought.

Evaluate individual components and complete trajectories.

---

# 4. Target Architecture

The intended end-state is:

World Bank Sources
        ↓
Databricks governed storage
        ↓
Bronze
        ↓
Silver
        ↓
Gold Intelligence
        ↓
Structured Tools + RAG
        ↓
Query / Context Layer
        ↓
Adaptive Agent Harness
        ↓
Guardrails
        ↓
Grounded Response
        ↓
MLflow Tracing + Evaluation
        ↓
API / Application

---

# 5. Development and Deployment Model

Development is performed locally using:

VS Code
+
Claude
+
Git

Databricks is the target execution environment.

The workflow is:

VS Code
    ↓
Git repository
    ↓
push to remote Git
    ↓
Databricks Git Folder
    ↓
Databricks execution

Do not design the repository as a local-only application.

At the same time, do not make core logic dependent on notebooks.

The same Python modules should remain locally testable.

---

# 6. Repository Design Principle

Core logic belongs under:

src/worldbank_copilot/

Examples:

common/
ingestion/
parsing/
transformations/
extraction/
retrieval/
tools/
agents/
guardrails/
memory/
evaluation/
observability/
api/

Not every directory must exist before its phase requires it.

Do not create placeholder complexity unnecessarily.

Notebooks belong under:

notebooks/

Notebooks must remain thin orchestration layers.

Do NOT place important business logic directly inside Databricks notebooks.

Scripts may provide local/CI entry points.

Configuration belongs under:

configs/

Tests belong under:

tests/

---

# 7. Databricks-Native Target

Prefer Databricks-native capabilities when they appropriately solve the problem.

Target platform components include:

Unity Catalog
Delta Lake
Unity Catalog Volumes
Databricks SQL
Databricks Vector Search
Databricks Workflows / Jobs
Databricks Model Serving where appropriate
MLflow
Databricks Apps where appropriate

Do not introduce external infrastructure when Databricks already provides the required capability unless there is a demonstrated reason.

---

# 8. Data Architecture

Use medallion architecture.

## Bronze

Purpose:

preserve source-aligned data with lineage.

Bronze should remain close to the source.

Preserve:

- source identity
- source path
- source hash
- ingestion metadata
- source system/type

Do not place analytical interpretation in Bronze.

## Silver

Purpose:

clean, normalize and structure facts.

Structured Silver includes project/loan/financial/procurement data.

Document-derived Silver includes:

silver.isr_snapshots
silver.project_results
silver.appraisal_risks
silver.project_events

Silver records must preserve provenance.

## Gold

Gold is NOT yet implemented.

Planned Gold datasets include:

gold.project_360
gold.project_timeline
gold.result_progress
gold.attention_signals
gold.risk_register

Gold converts validated facts into analytical views and deterministic observable signals.

Gold must NOT create unsupported predictive claims.

---

# 9. Provenance Is Mandatory

For every important document-derived business fact, preserve enough information to answer:

“Where did this come from?”

Relevant provenance may include:

project_id
document_id
document_type
document_date
page_number
section
table
row
column
source text/reference
extraction method
extraction status
source hash

Never remove provenance simply to make schemas cleaner.

---

# 10. Fact / Interpretation Boundaries

Future user-facing information should distinguish between:

FACT

DOCUMENTED_FINDING

SYSTEM_DERIVED_SIGNAL

AI_INTERPRETATION

UNKNOWN

Do not present system-derived or AI-generated conclusions as official World Bank facts.

Do not infer causality unless evidence explicitly supports it.

---

# 11. Document Intelligence

Docling is the primary document parser.

Extraction priority is:

1. structured Docling tables
2. labeled key/value fields
3. deterministic text/regex
4. targeted PDF text fallback
5. constrained LLM extraction only if deterministic approaches demonstrably fail

Do not default to LLM extraction.

PDF text fallback should be targeted to specific failing pages/tables.

Preserve extraction method.

Do not enable OCR globally.

OCR should be introduced only when required evidence exists solely on image-only pages and the need is demonstrated.

---

# 12. RAG Target Architecture

RAG is NOT implemented yet.

When introduced, do not build:

question
→ vector search
→ LLM

as the entire retrieval architecture.

Target retrieval pipeline:

Question
    ↓
Project/entity resolution
    ↓
Intent understanding
    ↓
Temporal scope extraction
    ↓
Document-type inference
    ↓
Query transformation where useful
    ↓
ACL + project metadata filtering
    ↓
First-stage retrieval
    ↓
Reranking
    ↓
Evidence selection
    ↓
Context construction
    ↓
LLM

Metadata filtering must happen before unauthorized content can enter model context.

---

# 13. RAG Must Be Experimental

Do not assume one RAG technique is optimal.

Create controlled experiments.

## Chunking candidates

- fixed-size baseline
- recursive
- structure-aware
- parent-child
- semantic only if justified

## Retrieval candidates

- BM25
- dense retrieval
- hybrid BM25 + dense

## Reranking candidates

At minimum compare:

- no reranker
- CrossEncoder

Where useful also evaluate:

- late-interaction / ColBERT-style approach
- SLM-based reranking

Do not assume CrossEncoder is automatically optimal.

Strong frontier LLM reranking should not be the default.

## Retrieval depth

Experiment with candidate depth and final context size.

Examples:

retrieve 10 → rerank → top 5

retrieve 20 → rerank → top 5

retrieve 50 → rerank → top 5

Choose using evaluation.

## Metrics

Use metrics such as:

Recall@K
MRR
nDCG
Precision@K
evidence coverage
citation accuracy
groundedness
latency
cost

Evidence coverage is especially important for multi-source investigation questions.

---

# 14. Query Routing

Not every request should enter an agent.

Target routing categories include:

STRUCTURED

DOCUMENT

INVESTIGATION

Potential paths:

STRUCTURED
→ governed SQL/tool
→ answer

DOCUMENT
→ RAG
→ grounded answer

INVESTIGATION
→ adaptive agent graph

Routing itself must be evaluated.

Potential routing implementations include:

- deterministic rules
- Jev-like decision model
- SLM
- strong LLM fallback

Do not assume the strongest model is required.

---

# 15. Jev / Bounded Decision Models

Jev is a candidate component, not a mandatory dependency.

Potential bounded-decision use cases include:

- route selection
- evidence sufficiency
- retry decision
- escalation decision
- completion check
- bounded scoring
- gating

Jev should not own application policy.

Example:

Jev/model:
“RAG probability = 0.92”

Harness code:
if probability >= evaluated_threshold:
    execute RAG
else:
    escalate/fallback

The model provides judgment.

Deterministic code owns policy and execution.

Benchmark Jev against:

- rules
- SLM
- inexpensive generative model
- strong model where appropriate

Measure:

accuracy
calibration
latency
cost
failure behavior

Do not integrate Jev into production merely because it is new.

---

# 16. Small Language Models

SLMs are candidates for lightweight semantic work.

Potential tasks:

- intent classification
- query rewriting
- entity extraction
- temporal extraction
- document classification
- simple relevance classification
- claim classification
- lightweight critique

SLMs should be benchmarked against deterministic approaches and larger models.

Do not use an SLM where code solves the problem exactly.

---

# 17. Strong LLM Usage

Reserve strong LLMs primarily for genuine open-ended uncertainty.

Examples:

- investigation planning
- ambiguous user requests
- multi-source reasoning
- synthesis
- explanation
- complex evidence interpretation
- final grounded response generation
- semantic critique when simpler models are insufficient

Do not use the strongest model for:

SQL
arithmetic
permissions
schema validation
simple routing
exact date calculations
deterministic thresholds
citation existence checks

---

# 18. Agent Architecture

Do not build an excessive number of agents.

Current target roles:

## Investigator / Orchestrator

Responsible for deciding what information is required for complex investigations.

May use structured tools and document retrieval.

## Evidence Researcher

Responsible for finding relevant document evidence.

## Evidence Critic

Responsible for checking whether candidate claims are supported by supplied evidence.

Finance, procurement, results and timeline capabilities are TOOLS, not agents.

Do not convert every function into an agent.

---

# 19. Agent State

Future agent state should be explicit and structured.

Potential state:

project_id
question
intent
temporal_scope
structured_facts
evidence
candidate_claims
critic_feedback
retry_count

Do not treat the complete conversation transcript as application state.

---

# 20. Hybrid Memory

Memory must be separated by purpose.

Do not create one undifferentiated memory store.

Potential memory classes:

## Working Memory

Short-lived state for the current investigation.

Likely stored in graph state.

## Conversation Memory

Compact conversational continuity such as:

current project
current topic
current temporal scope
resolved references

Do not continuously resend the entire chat history.

## Semantic Memory

Only validated reusable knowledge.

Examples:

reviewed aliases
approved mappings
validated domain relationships

AI-generated claims must not silently become permanent semantic memory.

## Episodic Memory

Potentially useful previous execution strategies.

Do not implement episodic memory until evaluation demonstrates benefit.

---

# 21. Hybrid Cache

Cache is NOT memory.

Potential cache layers:

## Exact Cache

For identical deterministic requests.

## Retrieval Cache

Keyed by appropriate combination of:

project scope
query
filters
index version

## Embedding Cache

Key using:

document hash
chunk hash
embedding model/version

Do not recompute unchanged embeddings.

## Semantic Cache

Potentially reuse semantically equivalent answers/operations.

Semantic caching requires strict authorization and project isolation.

Do not introduce semantic cache until leakage and staleness behavior are tested.

---

# 22. Security and Data Isolation

Cross-project or cross-user data leakage is unacceptable.

Security boundaries must be enforced before retrieval and before model context construction.

Never retrieve globally and ask the LLM to ignore unauthorized records.

Scope retrieval using:

user identity
role
authorized projects
project_id
document ACL
tenant/workspace where relevant

The same isolation requirements apply to:

retrieval
memory
cache
tools
agent state
logs
evaluation datasets

Cache keys must include relevant authorization/project/data-version boundaries.

---

# 23. Guardrails

Do not build one generic “guardrail agent.”

Use layered guardrails.

## Input Boundary

- authentication
- authorization
- scope
- prompt injection detection where appropriate
- request validation

## Retrieval Boundary

- allowed corpus
- project filters
- document ACL
- metadata filters

Retrieved documents are UNTRUSTED EVIDENCE.

They are not instructions.

## Tool Boundary

- read-only where possible
- typed arguments
- allowlisted operations
- timeout
- rate limits
- schema validation

## Reasoning Boundary

Prevent unsupported:

- financial values
- dates
- project facts
- causal conclusions
- citations

## Output Boundary

Validate:

- citations exist
- sources belong to authorized scope
- numeric claims are supported
- unsupported causal language
- structured output
- provenance

---

# 24. Evaluation Architecture

Evaluation must exist at component level and end-to-end.

## Routing

Evaluate:

routing accuracy
false-route rate
calibration
latency
cost

## Retrieval

Evaluate:

Recall@K
MRR
nDCG
Precision@K
evidence coverage

## Reranking

Evaluate:

ranking quality improvement
latency
cost

## Tool Use

Evaluate:

tool selection accuracy
argument correctness
execution success

## Reasoning

Evaluate:

task success
evidence usage
causal discipline

## Claims

Evaluate:

groundedness
claim support

## Citations

Evaluate:

citation existence
citation correctness
citation completeness

## System

Evaluate:

latency
token usage
cost
retry count
failure rate

---

# 25. Data Isolation Evaluation

Explicitly test leakage boundaries.

Create adversarial evaluation cases where:

information exists in Project B

but the active session is scoped to Project A.

Verify that:

retrieval does not return Project B
memory does not expose Project B
cache does not return Project B
tools reject Project B if unauthorized
LLM context contains no Project B evidence
final answer does not expose Project B

Security controls prevent leakage.

Evaluation proves that those controls work.

---

# 26. MLflow Target

Use MLflow for tracing and evaluation when the AI layer is introduced.

Future traces should make it possible to inspect:

user request
routing
tool selection
retrieval
reranking
context assembly
model calls
critic calls
retries
guardrail decisions
final answer

Capture relevant:

latency
tokens
cost
errors
models
retrieval configuration
documents
chunks
tool calls
evaluation results

Do not introduce MLflow agent instrumentation before the relevant AI layer exists.

---

# 27. Experiment-Driven Development

Architectural techniques are hypotheses until evaluated.

Examples:

fixed chunking vs structure-aware chunking

dense vs BM25 vs hybrid retrieval

no reranker vs CrossEncoder vs alternatives

rules vs Jev vs SLM routing

SLM vs LLM query rewriting

simple RAG vs RAG + tools

RAG + tools vs adaptive agent

memory vs no memory

cache vs no cache

Each experiment should have:

hypothesis
dataset
configuration
metrics
result
decision

Do not choose architecture based only on intuition.

---

# 28. Baselines Are Mandatory

Every complex AI feature should be compared against a simpler baseline.

Examples:

agent vs deterministic workflow

agent vs simple RAG

CrossEncoder vs no reranker

Jev vs rules

SLM vs rules

semantic cache vs exact cache

memory vs no memory

Complexity must demonstrate measurable value.

---

# 29. Failure Philosophy

Prefer an explicit UNKNOWN or insufficient-evidence result over fabrication.

Examples:

not enough evidence
conflicting sources
ambiguous indicator identity
unresolved date
unsupported causal relationship

Do not force the system to produce an answer.

A trustworthy refusal to infer is a valid system outcome.

---

# 30. Testing Requirements

Every phase must preserve previous tests.

Use:

unit tests
integration tests
real-document validation where appropriate
regression fixtures
data-contract tests
security/isolation tests
evaluation datasets

Run:

pytest
ruff check
ruff format --check

before declaring a phase complete.

Do not modify tests merely to make broken implementation pass.

---

# 31. Coding Standards

Prefer:

small cohesive modules
typed interfaces
Pydantic/dataclasses where appropriate
explicit schemas
dependency injection/configuration
deterministic IDs
clear error handling
structured logging
testable pure functions

Avoid:

giant notebooks
giant orchestration classes
framework-specific business logic
hidden global state
random identifiers where stable identity is required
silent exception swallowing
silent source reconciliation
hard-coded environment paths throughout business logic

---

# 32. Configuration

Environment-specific configuration must be externalized.

Examples:

Databricks catalog
schemas
Volume paths
warehouse/compute identifiers
model endpoints
Vector Search indexes
feature flags

Local and Databricks environments should share the same core configuration model.

Do not commit secrets.

---

# 33. Current Project Status

Completed:

Phase 1 — repository foundation

Phase 2 — Bronze ingestion

Phase 3 — structured Silver transformation

Phase 4 — document parsing

Phase 5 — structured document extraction

Phase 5 validated:

35 ISR snapshots

837 results observations

237 indicators

89 appraisal risks/findings

27 project events

No LLM extraction was required.

All required source provenance is preserved.

Known unresolved items remain explicit rather than silently resolved.

---

# 34. Current Phase

The next approved phase is:

PHASE 6 — DATABRICKS PLATFORMIZATION AND GOVERNED DELTA FOUNDATION

Phase 6 should move the validated data foundation into Databricks without changing its analytical meaning.

Do NOT start Gold, RAG, agents, memory, Jev integration or application development during Phase 6.

---

# 35. Phase Discipline

Work on only the explicitly approved phase.

At the end of every phase:

1. run all tests;
2. run integration tests;
3. run lint/format checks;
4. validate real outputs;
5. report counts;
6. report warnings/errors;
7. report unresolved ambiguities;
8. report architecture changes;
9. update IMPLEMENTATION_PLAN.md;
10. update README where required;
11. STOP.

Do not automatically begin the next phase.

Wait for explicit approval.

---

# 36. Planned High-Level Roadmap

The current intended sequence is approximately:

Phase 6
Databricks platformization and governed Delta foundation

Phase 7
Gold analytical/intelligence layer

Phase 8
RAG foundation + retrieval experiments

Phase 9
Structured tools + adaptive query routing experiments

Phase 10
Agent harness + bounded decision experiments

Phase 11
Hybrid memory/cache + advanced guardrails

Phase 12
MLflow tracing + comprehensive evaluation

Phase 13
API + Databricks application / product experience

The exact roadmap may change based on evidence from each completed phase.

Do not treat future phase numbering as permission to implement it.

---

# 37. Final Design Principle

The goal is NOT:

“Build the most agentic system possible.”

The goal is:

Build the most reliable system for the task.

Use:

code for certainty;

retrieval for evidence;

bounded decision models for bounded uncertainty;

SLMs for lightweight semantic work;

strong LLMs for genuine open-ended reasoning;

humans for consequential judgment.

Measure every additional layer of autonomy against a simpler baseline.