# Review guide

A short path through the codebase for reviewers. It shows where to start, where each
guarantee is enforced, and which parts are historical.

## 1. Twenty-minute reading order

| # | File | Why |
|---|---|---|
| 1 | [README.md](../README.md) | scope, design principles, results |
| 2 | [copilot/service.py](../src/worldbank_copilot/copilot/service.py) | the single online runtime: `Copilot.investigate()` from start to finish |
| 3 | [copilot/investigator.py](../src/worldbank_copilot/copilot/investigator.py) | the LLM planner and `govern()`, the boundary between model proposal and code execution |
| 4 | [copilot/governed.py](../src/worldbank_copilot/copilot/governed.py) | governed execution, temporal anchoring, context packing |
| 5 | [tools/reader.py](../src/worldbank_copilot/tools/reader.py) + one tool, e.g. [tools/signals.py](../src/worldbank_copilot/tools/signals.py) | how project scope and the table allowlist are enforced on every read |
| 6 | [copilot/semantic.py](../src/worldbank_copilot/copilot/semantic.py) | evidence handles, Synthesizer and Critic contracts, provenance assignment |
| 7 | [copilot/finalizer.py](../src/worldbank_copilot/copilot/finalizer.py) | claim-level integrity and publication rules |
| 8 | [retrieval/contract.py](../src/worldbank_copilot/retrieval/contract.py) + [retrieval/retriever.py](../src/worldbank_copilot/retrieval/retriever.py) | project-scoped hybrid retrieval and reranking |
| 9 | [intelligence/signals.py](../src/worldbank_copilot/intelligence/signals.py) + [configs/intelligence/attention_rules.yaml](../configs/intelligence/attention_rules.yaml) | deterministic attention signals |
| 10 | [copilot_app/](../copilot_app/README.md) | the presentation layer and its Job boundary |
| 11 | [evaluation/app_metrics.py](../src/worldbank_copilot/evaluation/app_metrics.py) | how the application is measured |

[ARCHITECTURE.md](ARCHITECTURE.md) explains the same path in prose.

## 2. Where each guarantee is enforced

| Guarantee | Enforced in | Tested in (examples) |
|---|---|---|
| Model cannot pick tools, projects or SQL outside the catalog | `govern()` in `copilot/investigator.py` | `test_copilot_runtime.py::test_invalid_actions_are_rejected_not_executed` |
| Every read is project-scoped; foreign rows fail closed | `tools/reader.py`, `tools/executor.py` | `test_tools_executor.py`, `test_tools_structured.py` |
| Retrieval is filtered before scoring and re-checked after | `retrieval/retriever.py`, `retrieval/contract.py` | `test_retrieval_core.py`, `tests/spark/test_embedding_isolation_spark.py` |
| Model never sees real identifiers, so it cannot invent citations | evidence handles in `copilot/semantic.py` | `test_copilot_semantic.py`, `test_copilot_runtime.py::test_second_round_sees_handles_and_call_summaries_but_no_identities` |
| Provenance labels are assigned by code, not the model | `enrich()` in `copilot/semantic.py` | `test_copilot_semantic.py` |
| A foreign project in any claim fails the whole response | `copilot/finalizer.py`, `investigation/synthesis.py` | `test_copilot_runtime.py::test_foreign_project_in_a_claim_fails_the_whole_response` |
| Critic failure never bypasses deterministic integrity | `copilot/finalizer.py` | `test_copilot_runtime.py::test_critic_failure_never_bypasses_deterministic_integrity` |
| Temporal claims need source-dated evidence on the correct side of a governed event | `copilot/governed.py` | `test_copilot_runtime.py::test_named_dated_event_is_confirmed_and_bounds_the_evidence` |
| Failure prediction is refused | router fast path + finalizer wording check | `test_copilot_runtime.py::test_prediction_requests_are_refused_without_model_calls` |
| Traces contain no question, claim or evidence text | `application/observability.py` | `test_copilot_runtime.py::test_trace_has_named_stages_and_only_structural_metadata` |
| Every loop is bounded | `configs/copilot.yaml` read by `copilot/config.py` | `test_copilot_runtime.py::test_tool_calls_are_bounded_across_rounds` |

Unit tests live in `tests/unit/`. Run `pytest -k <keyword>` to focus on one area.

## 3. Verifying locally

```powershell
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest          # offline; fake models; no Databricks access
.\.venv\Scripts\ruff check .
.\.venv\Scripts\ruff format --check .
```

No credentials, network access or source data are needed for the default suite.

## 4. What is current and what is historical

**Current (the online system):**

| Area | Paths |
|---|---|
| Runtime | `copilot/`, `tools/`, `retrieval/`, `routing/` (router and fast path), `investigation/` (shared contracts), `application/observability.py` |
| Data pipeline | `ingestion/`, `parsing/`, `extraction/`, `transformations/`, `lakehouse/`, `intelligence/`, `common/` |
| Evaluation | `evaluation/` |
| UI | `copilot_app/` |
| Notebooks | pipeline `01`–`07`, runtime `14`–`16` |

**Historical, kept for reproducibility (not on the request path):**

| Path | Why it is still here |
|---|---|
| `validation/phase9_contract.py` | `build_copilot` re-verifies the accepted retrieval configuration by hash at startup |
| `validation/phase10c_evidence.py`, `phase10d_fixtures.py` | reused by the offline test runtime |
| `routing/semantic*`, `bounded_classifier*`, `semif_*`, `review.py`, `evaluation.py` | routing candidates that were evaluated and not promoted |
| `retrieval/adaptive_*`, `endpoint_probe.py` | adaptive-rerank and embedding-endpoint experiments |
| empty `agents/`, `api/`, `guardrails/`, `observability/` packages | pinned by the Phase 10C lock |
| notebooks `07a`–`07e`, `08*`, `09` | experiment and acceptance runs |
| `evaluation/*_9c*`, `*_9d*`, `*_9e*`, `*lock*.json` | frozen experiment datasets, reports and protocol locks |
| `scripts/*_9c*`, `*_9d*`, `*_9e*`, `*_lock.py` | tooling for those experiments |

These files are hash-pinned by [evaluation/phase10c_evidence_lock.json](../evaluation/phase10c_evidence_lock.json)
and verified by tests. That is why they were documented rather than moved or deleted.
Some docstrings in pinned files mention notebooks or files that have since been removed.
Those files cannot be edited without breaking the lock.

## 5. Conventions

- **Notebooks are thin.** They call package functions; no business logic lives in notebook cells.
- **Configuration is external.** Code reads `configs/`, and every environment value can be overridden with `WBC_*` variables ([.env.example](../.env.example)).
- **IDs are deterministic.** Record and chunk IDs are content-derived hashes, never random.
- **Contracts are typed.** Pydantic models and dataclasses are used at every boundary.
- **Failures are explicit.** `UNKNOWN`, `INSUFFICIENT_EVIDENCE` and `FAIL_CLOSED` are preferred over silent fallbacks.
- **No data is committed.** Datasets and PDFs are git-ignored. Lock files hold hashes only.
