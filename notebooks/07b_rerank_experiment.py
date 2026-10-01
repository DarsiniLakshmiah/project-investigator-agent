# Databricks notebook source
# MAGIC %md
# MAGIC ## 07b_rerank_experiment: Optional CrossEncoder reranking experiment (Phase 8)
# MAGIC Thin entry point. Reads what notebook 07 built (corpus, Vector Search index); writes
# MAGIC nothing to Delta.
# MAGIC
# MAGIC Environment: serverless **environment version 6, ML base**. It already provides
# MAGIC torch 2.12.0+cpu, transformers 4.57.6 and sentence-transformers 5.5.1, so installing
# MAGIC `requirements-reranker.txt` changes nothing. The dependency-health check fails if a
# MAGIC protected runtime package would be replaced.
# MAGIC
# MAGIC Steps: 0 dependency health → 1 probe (incl. CrossEncoder) → 2 full staged experiments
# MAGIC (with reranker) → 3 selected configuration, breakdowns, failure analysis, abstention
# MAGIC calibration → 4 isolation with the final configuration → 5 sample evidence.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-databricks.txt -r ../requirements-retrieval.txt -r ../requirements-reranker.txt -c ../constraints-databricks.txt

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821 (Databricks built-in)

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

# Step 0: dependency health.
from worldbank_copilot.common.dependency_health import check_environment  # noqa: E402

health = check_environment(
    settings.repo_root,  # noqa: F821
    ["requirements-databricks.txt", "requirements-retrieval.txt", "requirements-reranker.txt"],
    settings.config_dir,  # noqa: F821
)
print(health.format())
if not health.ok:
    raise RuntimeError("STOP: notebook environment is inconsistent (see dependency health)")

# COMMAND ----------

# Step 1: probe, including the CrossEncoder.
from worldbank_copilot.retrieval import pipeline as rp  # noqa: E402
from worldbank_copilot.retrieval.config import load_retrieval_settings  # noqa: E402
from worldbank_copilot.retrieval.embeddings import embedding_provider  # noqa: E402

rs = load_retrieval_settings(settings.config_dir)  # noqa: F821
provider = embedding_provider(rs.embeddings)
capabilities = rp.probe_capabilities(rs, provider, check_reranker=True)
print(capabilities.format())
if capabilities.blocking_failures:
    raise RuntimeError("STOP: required capability unavailable.\n" + capabilities.format())

# COMMAND ----------

# Step 2: full staged experiments (retrieval reads notebook 07's corpus and index).
from worldbank_copilot.retrieval import evaluation as ev  # noqa: E402
from worldbank_copilot.retrieval import report  # noqa: E402
from worldbank_copilot.retrieval.rerank import CrossEncoderReranker, NoReranker  # noqa: E402

# Existing, ready index and governed corpus (read-only; nothing is created or embedded).
retriever = rp.open_retriever(spark, settings, rs, registry.project_ids, provider)  # noqa: F821
questions = ev.load_questions(settings.repo_root / rs.evaluation.questions_file)  # noqa: F821
rerankers = {"none": NoReranker()}
if any(c.name == "cross_encoder" and c.status == "OK" for c in capabilities.checks):
    rerankers["cross_encoder"] = CrossEncoderReranker(rs.retrieval.reranker)
decisions = ev.run_stages(retriever, questions, rs.evaluation, rerankers,
                          ev.RunConfig("structure", "dense", "none", 10), print)
display(spark.createDataFrame(report.decisions_table(decisions)))  # noqa: F821
for d in decisions:
    print(f"[{d.stage}] winner {d.winner.name}: {d.reason}")

# COMMAND ----------

# Step 3: selected configuration, breakdowns, failure analysis, abstention, MLflow.
# Deliberate dependency: the experiment results of Step 2.
from worldbank_copilot.retrieval import evaluation as ev  # noqa: E402
from worldbank_copilot.retrieval import report  # noqa: E402

require_state("decisions", step="Step 2 (full staged experiments)")  # noqa: F821
selected = next(r for r in decisions[-1].runs if r.config == decisions[-1].winner)  # noqa: F821
print("SELECTED:", selected.config.name, selected.summary)
display(spark.createDataFrame(report.breakdown(selected)))  # noqa: F821
analysis_retriever = rp.open_retriever(spark, settings, rs, registry.project_ids, provider)  # noqa: F821
analysis_questions = ev.load_questions(settings.repo_root / rs.evaluation.questions_file)  # noqa: F821
misses = ev.classify_misses(analysis_retriever, analysis_questions, selected)
if misses:
    display(spark.createDataFrame(misses))  # noqa: F821
reranked = [r for d in decisions for r in d.runs  # noqa: F821
            if r.status == "OK" and r.config.reranker == "cross_encoder"]
for run in reranked[:1]:
    print("abstention calibration:", ev.abstention_calibration(run))
if rs.evaluation.mlflow.enabled:
    run_ids = ev.log_to_mlflow(
        decisions,  # noqa: F821
        {"embedding_model": provider.model, "questions": len(analysis_questions),
         "index": rp.names(settings, rs)["index_name"], "notebook": "07b"},  # noqa: F821
        rs.evaluation.mlflow.experiment_name,
    )
    print(f"MLflow: {len(run_ids)} runs logged")

# COMMAND ----------

# Step 4: project isolation with the final configuration (ERROR-level).
# Deliberate dependency: the configuration selected in Step 3.
from worldbank_copilot.retrieval import evaluation as ev  # noqa: E402
from worldbank_copilot.retrieval import report  # noqa: E402
from worldbank_copilot.retrieval.rerank import reranker_for  # noqa: E402

require_state("selected", step="Step 3 (selected configuration)")  # noqa: F821
cfg = selected.config  # noqa: F821
isolation_retriever = rp.open_retriever(spark, settings, rs, registry.project_ids, provider)  # noqa: F821
isolation_questions = ev.load_questions(settings.repo_root / rs.evaluation.questions_file)  # noqa: F821
checks = report.isolation_checks(isolation_retriever, isolation_questions, registry.project_ids,  # noqa: F821
                                 cfg.chunk_strategy, cfg.retrieval,
                                 reranker_for(cfg.reranker, rs.retrieval.reranker), cfg.candidate_k)
for c in checks:
    print(f"[{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
assert all(c.passed for c in checks), "cross-project isolation failed"

# COMMAND ----------

# Step 5: representative evidence for manual inspection (citation-ready).
# Deliberate dependency: the configuration selected in Step 3.
from worldbank_copilot.retrieval import evaluation as ev  # noqa: E402
from worldbank_copilot.retrieval import report  # noqa: E402
from worldbank_copilot.retrieval.rerank import reranker_for  # noqa: E402

require_state("selected", step="Step 3 (selected configuration)")  # noqa: F821
cfg = selected.config  # noqa: F821
evidence_retriever = rp.open_retriever(spark, settings, rs, registry.project_ids, provider)  # noqa: F821
evidence_questions = ev.load_questions(settings.repo_root / rs.evaluation.questions_file)  # noqa: F821
evidence_reranker = reranker_for(cfg.reranker, rs.retrieval.reranker)
for qid in ("q01", "q04", "q07", "q28", "q33", "q39", "q44", "q25"):
    q = next(x for x in evidence_questions if x.id == qid)
    result = evidence_retriever.retrieve(q.question, q.project_id, strategy=cfg.chunk_strategy,
                                         method=cfg.retrieval, reranker=evidence_reranker,
                                         candidate_k=cfg.candidate_k, final_k=5)
    print(report.format_evidence(result), "\n")
