# Evaluation

Golden datasets, experiment reports and frozen protocol locks. Each experiment followed
the same pattern: **hypothesis → frozen dataset → protocol lock (written before any
prediction) → configuration → metrics → decision.** Locks contain hashes only, so results
cannot be quietly re-tuned afterwards.

Evaluation code lives in [src/worldbank_copilot/evaluation/](../src/worldbank_copilot/evaluation/)
(the application evaluation) and [src/worldbank_copilot/retrieval/evaluation.py](../src/worldbank_copilot/retrieval/evaluation.py)
(the retrieval evaluation).

## 1. Application evaluation (end to end)

| File | Content |
|---|---|
| `app_eval_cases.jsonl` | 50 questions, 10 categories × 5, run through the real `Copilot.investigate` |

Categories: `attention`, `documents`, `ratings`, `results`, `financial`,
`restructuring`, `temporal`, `prediction` (must refuse), `cross_project` (must isolate),
and `invalid` (must refuse or clarify).

Each case states the expected **behaviour**: allowed statuses, refusal, isolation and
temporal anchoring. Golden facts are never typed values. Each one is a governed query
(tool, arguments, field path) resolved against the real Silver/Gold data at evaluation
time. Document relevance reuses the retrieval ground truth below.

Metrics are reported in separate layers and never collapsed into one score:

- retrieval;
- context;
- fact recall;
- citations;
- temporal;
- isolation;
- refusal;
- task success;
- cost and latency.

Run with `notebooks/16_application_evaluation.py`. Results are appended per run under the
artefact Volume (`application_evaluation/<run_id>/`), and earlier runs are never overwritten.

**Result (`app-eval-001`):**

| Metric | Result |
|---|---|
| Verdicts | 35 PASS / 7 PARTIAL / 8 FAIL |
| Task success | 72% |
| Runtime success | 98% |
| Project isolation | 100% |
| Citation validity | 100% |
| Unsupported claims published | 0 |
| Fully supported claims | 91% |
| Refusal correctness | 88% |
| Application Recall@10 | about 0.33 |

## 2. Retrieval

| File | Content |
|---|---|
| `retrieval_questions.yaml` | 49 questions: 44 answerable and 5 with no answer. Ground truth is the document, page(s) and a verbatim phrase taken from the sources, never from the retriever. `tests/integration/test_retrieval_real.py` checks that each phrase really appears on its page. |

The staged experiments covered:

- chunking: fixed, recursive and structure-aware;
- retrieval: BM25, dense and hybrid with RRF;
- reranking: none vs CrossEncoder;
- candidate depth.

They are configured in [configs/retrieval/evaluation.yaml](../configs/retrieval/evaluation.yaml).

**Selected configuration:** hybrid retrieval with a CrossEncoder, 50 candidates, top 5.

| Recall@5 | Recall@10 | MRR | nDCG@5 |
|---|---|---|---|
| 0.795 | 0.841 | 0.683 | 0.717 |

**Adaptive reranking** (rerank only when confidence is low):

| File | Content |
|---|---|
| `adaptive_rerank_9e_lock.json` | protocol lock |
| `adaptive_rerank_9e.{json,md}` | offline frontier |
| `adaptive_rerank_9e_live*.{json,md}` | live latency confirmation |

**Decision:** descriptive only; no adaptive policy promoted.

## 3. Routing

| File | Content |
|---|---|
| `routing_cases.yaml`, `routing_freeze_9c.json`, `routing_review_9c.md` | 80 human-reviewed routing cases (29 dev / 51 test), frozen |
| `routing_baseline_9B.1.yaml`, `routing_baseline_9B.2.yaml` | deterministic router output (baseline measurement) |
| `semantic_dev_experiments_9d.json`, `semantic_dev_report_9d.md` | lexical and embedding-similarity fallbacks on the dev split |
| `candidate_c_dev_lock.json`, `candidate_c_dev_9d.{json,md}` | LLM bounded classifier (GPT-OSS-20B) on the dev split |
| `semantic_ambiguity_probe.yaml` | separate safety probe set (not run; the candidate was rejected first) |
| `semif_protocol_lock.json` | SemIf / OpenJev candidate (blocked by the execution environment) |
| `semantic_config_9d.yaml` | the frozen routing decision |

**Decision:** deterministic routing with targeted clarification. The semantic and LLM
fallbacks did not improve on the rules (the LLM was net −2 on dev, with poor
repeatability). The test split was never used for selection.

## 4. Frozen acceptance contracts

| File | Content |
|---|---|
| `phase9_contract_cases.yaml`, `phase9_contract_lock.json` | wiring acceptance for tools and retrieval on Databricks |
| `phase10c_evidence_cases.yaml`, `phase10c_evidence_lock.json`, `phase10c_wrapper_expected_program.json` | evidence-layer acceptance |

The Phase 10C lock also pins the hashes of accepted source and config files. The test
suite verifies them, and `build_copilot` re-checks the accepted retrieval configuration
before serving requests.
