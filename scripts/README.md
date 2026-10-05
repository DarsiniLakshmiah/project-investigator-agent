# Scripts

Local command-line entry points. They use the same package code as the notebooks and make
no Databricks writes.

## Data pipeline (local)

| Script | Purpose |
|---|---|
| `validate_data.py` | Bronze ingestion, Silver transformation and data-quality checks on the local source files (`--layer`, `--json PATH`, `--write-bronze`, `--write-silver`) |
| `parse_documents.py` | parse the in-scope PDFs with Docling and report parsing quality (`--project`, `--document`) |
| `extract_document_facts.py` | extract document-derived Silver datasets from the parsed cache (`--project`, `--no-cache`) |
| `platformize.py` | dry run of the governed Delta foundation: build, validate and reconcile Delta-ready datasets locally |

Example:

```powershell
.\.venv\Scripts\python scripts\validate_data.py --json .local_output\quality.json
```

Outputs go to `.local_output/`, which is git-ignored. These scripts need the source files
under `data/` ([data/README.md](../data/README.md)).

## Experiment tooling (historical)

These scripts belong to frozen routing and reranking experiments. Each writes or checks a
protocol lock *before* any prediction, so results cannot be tuned after the fact.

| Script | Experiment |
|---|---|
| `routing_review_9c.py` | baseline router over the routing dataset; writes the human-review artefact |
| `semantic_dev_9d.py` | semantic routing fallback grid on the development split |
| `candidate_c_dev_lock.py`, `candidate_c_dev_9d.py` | LLM routing candidate: protocol lock, then offline replay |
| `semif_protocol_lock.py` | protocol lock for the SemIf / OpenJev candidate |
| `adaptive_rerank_lock.py`, `adaptive_rerank_9e.py`, `adaptive_rerank_9e_live.py` | adaptive reranking: lock, offline frontier, live latency check |
