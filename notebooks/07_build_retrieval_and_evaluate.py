# Databricks notebook source
# MAGIC %md
# MAGIC ## 07_build_retrieval_and_evaluate: Retrieval corpus, Vector Search and experiments (Phase 8)
# MAGIC Thin entry point. All logic lives in `worldbank_copilot.retrieval`.
# MAGIC
# MAGIC Prerequisites: Phase 6 (Silver + parsed artefacts in the Volume) and, for the comments
# MAGIC fix, a re-run of notebooks 05 and 06 (see README, Phase 8).
# MAGIC
# MAGIC Steps: 0 capability probe (stops if Vector Search or the embedding endpoint is missing)
# MAGIC → 1 corpus (`silver.document_chunks`) → 2 embeddings cache, index source, Vector Search
# MAGIC index → 3 staged experiments → 4 project-isolation checks → 5 sample evidence
# MAGIC → 6 idempotency re-run → 7 validation SQL.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-databricks.txt -r ../requirements-retrieval.txt

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821 (Databricks built-in)

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

# Step 0: capability probe. Nothing is built if a blocking capability is missing.
from worldbank_copilot.retrieval import pipeline as rp  # noqa: E402
from worldbank_copilot.retrieval.config import load_retrieval_settings  # noqa: E402
from worldbank_copilot.retrieval.embeddings import embedding_provider  # noqa: E402

rs = load_retrieval_settings(settings.config_dir)  # noqa: F821
provider = embedding_provider(rs.embeddings)
capabilities = rp.probe_capabilities(rs, provider)
print(capabilities.format())
if capabilities.blocking_failures:
    raise RuntimeError(
        "STOP: required Databricks capability unavailable; nothing was built.\n"
        + capabilities.format()
    )

# COMMAND ----------

# Step 1: governed retrieval corpus (all chunking strategies) with read-back reconciliation.
corpus = rp.build_corpus(spark, settings, rs, registry, progress=print)  # noqa: F821
print(corpus.write)
for group, counts in corpus.counts.items():
    print(group, counts)

# COMMAND ----------

# Step 2: embeddings (cache: only new text is embedded), index source, Vector Search index.
strategies = sorted(rs.chunking.strategies)
embedding_report = rp.update_embedding_cache(spark, settings, rs, provider, strategies, print)  # noqa: F821
print(embedding_report)
index_write, index_rows = rp.build_index_source(spark, settings, rs, provider.model, strategies)  # noqa: F821
print(index_write)
vs_index = rp.vector_index(settings, rs)  # noqa: F821
index_status = rp.ensure_vector_index(vs_index, index_rows, progress=print)
print(index_status)

# COMMAND ----------

# Step 3: staged retrieval experiments (configs/retrieval/evaluation.yaml).
from worldbank_copilot.retrieval import evaluation as ev  # noqa: E402
from worldbank_copilot.retrieval import report  # noqa: E402
from worldbank_copilot.retrieval.rerank import CrossEncoderReranker, NoReranker  # noqa: E402
from worldbank_copilot.retrieval.retriever import ChunkStore, Retriever  # noqa: E402

names = rp.names(settings, rs)  # noqa: F821
store = ChunkStore.from_table(spark, names["chunks"])  # noqa: F821
retriever = Retriever(store, rs, registry.project_ids, embeddings=provider, dense=vs_index)  # noqa: F821
questions = ev.load_questions(settings.repo_root / rs.evaluation.questions_file)  # noqa: F821
rerankers = {"none": NoReranker()}
if any(c.name == "cross_encoder" and c.status == "OK" for c in capabilities.checks):
    rerankers["cross_encoder"] = CrossEncoderReranker(rs.retrieval.reranker)
baseline = ev.RunConfig("structure", "dense", "none", 10)
decisions = ev.run_stages(retriever, questions, rs.evaluation, rerankers, baseline, print)
display(spark.createDataFrame(report.decisions_table(decisions)))  # noqa: F821
for d in decisions:
    print(f"[{d.stage}] winner {d.winner.name}: {d.reason}")

# COMMAND ----------

# Step 3b: selected configuration: breakdowns, failure analysis, abstention calibration, MLflow.
final = decisions[-1]
selected = next(r for r in final.runs if r.config == final.winner)
print("SELECTED:", selected.config.name, selected.summary)
display(spark.createDataFrame(report.breakdown(selected)))  # noqa: F821
misses = ev.classify_misses(retriever, questions, selected)
if misses:
    display(spark.createDataFrame(misses))  # noqa: F821
print("abstention:", ev.abstention_calibration(selected))
if rs.evaluation.mlflow.enabled:
    run_ids = ev.log_to_mlflow(
        decisions,
        {"embedding_model": provider.model, "questions": len(questions),
         "index": names["index_name"]},
        rs.evaluation.mlflow.experiment_name,
    )
    print(f"MLflow: {len(run_ids)} runs logged")

# COMMAND ----------

# Step 4: project isolation (ERROR-level). Every question under every project scope.
cfg = selected.config
checks = report.isolation_checks(retriever, questions, registry.project_ids, cfg.chunk_strategy,  # noqa: F821
                                 cfg.retrieval, rerankers[cfg.reranker], cfg.candidate_k)
for c in checks:
    print(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
assert all(c.passed for c in checks), "cross-project isolation failed"

# COMMAND ----------

# Step 5: representative evidence for manual inspection (citation-ready).
for qid in ("q01", "q04", "q07", "q28", "q33", "q39", "q44", "q25"):
    q = next(x for x in questions if x.id == qid)
    result = retriever.retrieve(q.question, q.project_id, strategy=cfg.chunk_strategy,
                                method=cfg.retrieval, reranker=rerankers[cfg.reranker],
                                candidate_k=cfg.candidate_k, final_k=5)
    print(report.format_evidence(result), "\n")

# COMMAND ----------

# Step 6: idempotency - rebuilding with unchanged inputs must change nothing.
again = rp.build_corpus(spark, settings, rs, registry)  # noqa: F821
embeddings_again = rp.update_embedding_cache(spark, settings, rs, provider, strategies)  # noqa: F821
index_again, _ = rp.build_index_source(spark, settings, rs, provider.model, strategies)  # noqa: F821
print("corpus:", again.write)
print("embeddings:", embeddings_again)
print("index source:", index_again)
assert again.profile["fingerprint"] == corpus.profile["fingerprint"]
assert (again.write.inserted, again.write.updated, again.write.deleted) == (0, 0, 0)
assert embeddings_again.embedded_now == 0
assert (index_again.inserted, index_again.updated, index_again.deleted) == (0, 0, 0)

# COMMAND ----------

# Step 7: read-only validation queries (sql/phase8_validation.sql).
from worldbank_copilot.lakehouse.validation_sql import load_queries  # noqa: E402

for name, query in load_queries(settings, settings.repo_root / "sql" / "phase8_validation.sql").items():  # noqa: F821
    print(f"-- {name}")
    display(spark.sql(query))  # noqa: F821
