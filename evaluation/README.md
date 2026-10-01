# Evaluation

* `retrieval_questions.yaml` (Phase 8): 49 retrieval questions across P130544, P179039 and
  P506272 (44 answerable and 5 with no answer). Ground truth for each question is the
  document, the page(s) and a verbatim phrase, taken from the parsed sources and validated
  Silver facts, never from the retriever. `tests/integration/test_retrieval_real.py`
  checks that every phrase really appears on its pages.
* Metrics and staged experiments: `src/worldbank_copilot/retrieval/evaluation.py`,
  configured in `configs/retrieval/evaluation.yaml`.

Agent and answer evaluation are added in later phases.
