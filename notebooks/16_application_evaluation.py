# Databricks notebook source
# MAGIC %md
# MAGIC ## Final application evaluation: 50 golden questions through `copilot.investigate`
# MAGIC Thin entry point; logic lives in `worldbank_copilot.evaluation`. Measurement only: the
# MAGIC runtime is not tuned to pass. Cases run sequentially through the canonical runtime with
# MAGIC real tools, retrieval, models and MLflow traces. Results are appended to the artefact
# MAGIC Volume (`application_evaluation/<RUN_ID>/results.jsonl`); a restarted notebook resumes
# MAGIC and never reruns stored cases unless `RERUN = True`. Golden facts are resolved from
# MAGIC governed data once per run (`golden.json`) with their table/record identity.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements-databricks.txt -r ../requirements-retrieval.txt -r ../requirements-reranker.txt -r ../requirements-phase10d.txt -c ../constraints-databricks.txt

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

# MAGIC %run ./_bootstrap

# COMMAND ----------

# Canonical runtime (same gates as notebook 14), with MLflow tracing on.
from worldbank_copilot.copilot.databricks import build_copilot

copilot = build_copilot(spark, settings)  # noqa: F821

# COMMAND ----------

# Run selection. Reuse RUN_ID to resume; use a new RUN_ID for a new run (never overwritten).
RUN_ID = "app-eval-001"
CASE_IDS: list[str] = []  # e.g. ["APP_001", "APP_046"]; empty = all
CATEGORIES: list[str] = []  # e.g. ["prediction"]; empty = all
RERUN = False  # True appends a new attempt for the selected cases (old lines are kept)
LOG_TO_MLFLOW = False  # optional run-level metrics; per-request traces exist regardless

# COMMAND ----------

from worldbank_copilot.evaluation.app_cases import CASES_FILE, load_cases, resolve_golden
from worldbank_copilot.evaluation.app_runner import RunStore, run_identity, select_cases
from worldbank_copilot.retrieval.evaluation import load_questions

repo = settings.repo_root  # noqa: F821
questions = {q.id: q for q in load_questions(repo / "evaluation/retrieval_questions.yaml")}
cases = load_cases(
    repo / CASES_FILE,
    project_ids=registry.project_ids,  # noqa: F821
    retrieval_questions=questions,
)
store = RunStore(settings.local_output_root / "application_evaluation", RUN_ID)  # noqa: F821
store.write_once(
    "run.json", run_identity(RUN_ID, repo, (repo / CASES_FILE).read_text("utf-8"), copilot)
)


def governed_read(tool, arguments, project_id):
    """Golden facts come from the governed, project-scoped tools (never from the Copilot)."""
    result = copilot.router.executor.run(
        tool,
        arguments,
        copilot.router.context_factory(f"golden-{RUN_ID}"),
        scope_project_id=project_id,
        authorized_projects=(project_id,),
    )
    return result.status.value, [item.model_dump(mode="json") for item in result.items]


golden = store.read("golden.json")
if golden is None:
    golden = resolve_golden(cases, governed_read)
    store.write_once("golden.json", golden)
print({cid: {f: g["status"] for f, g in facts.items()} for cid, facts in golden.items()})
print(store.dir)

# COMMAND ----------

# Run (sequential; stored cases are skipped unless RERUN). One failure never stops the run.
from worldbank_copilot.evaluation.app_runner import run_cases

selected = select_cases(cases, CASE_IDS, CATEGORIES)
run_cases(copilot, selected, store, rerun=RERUN)

# COMMAND ----------

# Score the latest attempt of every stored case; writes scored.jsonl, summary.json, report.md.
import json

from worldbank_copilot.evaluation.app_runner import load_reviews, score_run

reviews = load_reviews(store.read("reviews.json") or [])
scored, summary = score_run(cases, store, golden, questions, reviews)
print((store.dir / "report.md").read_text("utf-8"))

# COMMAND ----------

# Optional: run-level metrics in MLflow (uses the runtime's mlflow-skinny; nothing installed).
if LOG_TO_MLFLOW:
    import mlflow

    numeric = {
        f"{group}.{key}": entry["value"]
        for group in ("retrieval", "context", "generation", "citations", "agent")
        for key, entry in summary[group].items()
        if isinstance(entry, dict) and isinstance(entry.get("value"), int | float)
    }
    with mlflow.start_run(run_name=f"application-evaluation-{RUN_ID}"):
        mlflow.log_metrics(numeric)
        mlflow.log_dict(summary, "summary.json")
        mlflow.log_text((store.dir / "report.md").read_text("utf-8"), "report.md")

# COMMAND ----------

# Human review of semantic dimensions (relevance, completeness, conciseness, citation
# entailment; 0/1/2). Export once, fill in, save as reviews.json in the run folder, then
# rerun the scoring cell. Nothing here calls a model.
from worldbank_copilot.evaluation.app_runner import review_template

template = store.dir / "review_template.json"
if not template.exists():
    template.write_text(json.dumps(review_template(cases, store), indent=2), encoding="utf-8")
print(template)
