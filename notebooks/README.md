# Notebooks

Databricks entry points, stored as `.py` source notebooks so they diff cleanly in Git.
They are **thin**: each one loads settings with `%run ./_bootstrap`, then calls functions
in `src/worldbank_copilot/`. No business logic lives in notebook cells.

Run them from a Databricks Git folder. Notebooks that need pinned packages install them
notebook-scoped (`%pip install -r ../requirements-*.txt -c ../constraints-databricks.txt`).

## Data and retrieval pipeline (run in order)

| Notebook | Builds | Logic in |
|---|---|---|
| `01_ingest_structured` | Bronze tables from the portfolio workbook and the loan and procurement CSVs | `ingestion/`, `transformations/bronze.py` |
| `02_parse_documents` | Docling parse of the project PDFs (cached in a Volume) | `parsing/` |
| `03_build_silver` | normalized structured Silver tables | `transformations/` |
| `04_extract_document_facts` | ISR snapshots, results, appraisal risks and events with provenance | `extraction/` |
| `05_platformize_databricks` | governed Unity Catalog / Delta foundation, contracts, reconciliation | `lakehouse/` |
| `06_build_gold_intelligence` | Gold views and deterministic attention signals | `intelligence/` |
| `07_build_retrieval_and_evaluate` | chunk corpus, embeddings, Vector Search index, retrieval evaluation | `retrieval/` |

## Runtime

| Notebook | Purpose |
|---|---|
| `14_copilot_prototype_validation` | live end-to-end run of `Copilot.investigate` on real data, checking architectural invariants |
| `15_copilot_app_backend` | **the App's Job**: one question in, one `InvestigationResult` JSON out (parameters `project_id`, `question`) |
| `16_application_evaluation` | the 50-question application evaluation (resumable, results saved to a Volume) |
| `_bootstrap` | shared setup: makes `src/` importable, loads settings and project scope |

## Experiments and acceptance runs (historical)

These notebooks produced the evidence behind design decisions. They are kept for
reproducibility and are not needed to run the system.

| Notebook | Experiment |
|---|---|
| `07a_embedding_endpoint_probe` | embedding endpoint capability probe |
| `07b_rerank_experiment` | no reranker vs CrossEncoder |
| `07c_adaptive_rerank_collect`, `07d_adaptive_rerank_live` | adaptive reranking (descriptive; not promoted) |
| `07e_phase9_contract_validation` | Phase 9 contract acceptance on Databricks |
| `08_semantic_routing_dev` | semantic routing fallback (development split) |
| `08d_bounded_classifier_capability` | LLM bounded-classifier capability check |
| `08e_candidate_c_dev` | LLM routing candidate on the development split (rejected) |
| `09_phase10c_evidence_validation` | evidence-layer acceptance on Databricks |
