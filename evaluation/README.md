# Evaluation

* `retrieval_questions.yaml` (Phase 8): 49 retrieval questions across P130544, P179039 and
  P506272 (44 answerable and 5 with no answer). Ground truth for each question is the
  document, the page(s) and a verbatim phrase, taken from the parsed sources and validated
  Silver facts, never from the retriever. `tests/integration/test_retrieval_real.py`
  checks that every phrase really appears on its pages.
* Metrics and staged experiments: `src/worldbank_copilot/retrieval/evaluation.py`,
  configured in `configs/retrieval/evaluation.yaml`.

Agent and answer evaluation are added in later phases.

## Application evaluation (final prototype)

* `app_eval_cases.jsonl`: 50 application questions (10 categories x 5) run through the
  canonical `copilot.investigate`. Cases state expected behaviour; golden facts are
  governed queries resolved on Databricks (never typed values); document relevance reuses
  the Phase 8 ground truth above by question id.
* Code: `src/worldbank_copilot/evaluation/` (`app_cases`, `app_metrics`, `app_runner`);
  notebook `notebooks/16_application_evaluation.py`. Results are appended per run under
  the artefact Volume (`application_evaluation/<run_id>/`); historical runs are never
  overwritten. This complements, and does not replace, the component evaluations above.
