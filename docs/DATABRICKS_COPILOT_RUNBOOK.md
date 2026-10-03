# Databricks runbook

All steps are user-run. No workspace/model calls or deployment occurred locally. Review and commit the new files yourself, synchronize the exact snapshot, and use its full SHA. Do not reuse the older reviewed revision for new implementation. Preserve every reservation, checkpoint, final artifact and receipt; use only unused IDs.

## 1. Read 10d3 without rerunning

In a separate notebook use existing `%run ./_bootstrap`, then:

```python
import importlib.util, json
from pathlib import Path
spec = importlib.util.spec_from_file_location("attempt_reader", Path(settings.repo_root)/"scripts/read_phase10d_attempt.py")
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
print(json.dumps(reader.read_attempt(settings, "10d3"), indent=2, default=str))
```

The reader lists reservation/checkpoint/final/receipt files, verifies completed artifacts and reports error classes for unreadable files. Inspect preflight, failure, revision and persistence in the final or latest verified checkpoint. Return this output for exact diagnosis. Zero completed rows alone cannot identify the gate or exclude an invocation that failed before persistence. The reader performs no reservation, model setup or write.

## 2. Existing 10D validation gate

After diagnosis/review run `notebooks/09_phase10d_model_validation.py` with:

```
commit_sha=<full reviewed committed SHA>
run_id=10d4  # only if unused; otherwise next unused 10dN
endpoints=databricks-gpt-oss-20b
```

Preserve 10d1/2/3. Required: accepted 10C integrity; phase10d_capability_lock@2; databricks_wrapper_ast@3; notebook SHA 754e42ac71c378df721b258b23a2c9b9b1d4aef7c8c97ef2da1d942ad957c349; exact case/schedule identity; environment and endpoint checks; all 19 scheduled calls per endpoint and required repetitions/checks. Preflight PASS alone does not accept GPT-OSS. Workspace Git ambiguity retains accepted USER_DECLARED_REVIEWED_REVISION behavior.

## 3. Trusted runtime configuration

Reuse accepted Databricks compute, settings, governed tables, retrieval index and artifact Volume. Configure server-side values before runtime construction; notebook restartPython requires restoring Python-cell environment settings after bootstrap, or configuring compute environment variables. Never put secrets in source/request JSON.

```
WBC_APP_ALLOWED_PROJECTS=<registered authorized IDs, comma separated>
WBC_APP_COST_CEILING=<approved positive budget in pricing units>
WBC_APP_PRICING_VERSION=<actual reviewed pricing schedule identifier>
WBC_APP_MAX_COST_PER_MODEL_CALL=<approved conservative one-call upper bound>
WBC_APP_ACCEPTED_10D_RUN_ID=<preserved passing 10dN>
WBC_APP_SYNTHESIS_ENDPOINT=<endpoint covered by that receipt>
WBC_APP_CRITIC_ENDPOINT=<endpoint covered by that receipt>
WBC_APP_INVESTIGATOR_ENDPOINT=<approved proposal endpoint>
WBC_APP_ENABLE_INVESTIGATOR=false
```

Set existing `mlflow.experiment` configuration to an accessible real experiment. New notebooks install existing requirements/constraints plus requirements-copilot-runtime.txt. Live wiring requires a valid final 10D receipt, matching reviewed protocol, PASS and selected endpoints. Worst-case model slots are reserved conservatively; actual billing is unavailable rather than invented.

## 4. Proposal and comparison validation

1. Run `notebooks/13_investigator_capability.py`: `commit_sha=<full SHA>`, `run_id=10e1` if unused. At most three real proposals on synthetic contexts; zero proposed tool executions. Require schema/action/target/project/time validity, no budget violation and complete receipts. Contract PASS is not usefulness acceptance.
2. Review proposals/invariants. Set WBC_APP_ENABLE_INVESTIGATOR=true only for intentional C evaluation.
3. Run `notebooks/12_copilot_validation.py`: `commit_sha=<full SHA>`, `run_id=10f1` if unused. Require all 18 A/B/C records, zero hard invariant failures, resolvable provenance/citations and real MLflow trace IDs. Initialization failure is not a skipped case.
4. Read the final with `worldbank_copilot.validation.copilot_e2e.read_completed(Path(settings.artifact_volume_path)/"copilot_e2e"/"10f1.json")`. Review actual claims/traces for groundedness, sufficiency, task success, retrieved-content injection, critic relevance/overclaiming and temporal scope. Unsupported direct document date filtering must clarify. Register any additional quality protocol before executing it; never tune frozen cases after outcomes.
5. Compare actual latency/calls/mechanical coverage. Inspect whether C used investigation. Record billing separately; retain B unless reviewed task-quality improvement justifies C. All mechanical artifacts deliberately remain NOT_ACCEPTED.

## 5. Backend Job and App deployment

Create one notebook-task Job in workspace UI pointing to `notebooks/11_copilot_request.py`. Define Job parameter request_json with empty default, and task notebook parameter `request_json={{job.parameters.request_json}}`. Use approved compute with trusted environment above, max concurrent runs 1, task retries 0. Job run-as requires existing governed table SELECT/USE CATALOG/USE SCHEMA, approved retrieval/embedding/model query access, Volume READ for the accepted receipt, and MLflow permissions. Validation additionally requires artifact Volume WRITE. No data/index rebuild is required.

The frontend uses SDK jobs.run_now(job_parameters=...) and reads the single notebook-task output. Requests contain question/project/session only, never endpoint/cost/permission/tool controls.

Create a Databricks App from the reviewed source in workspace UI. Root app.yaml launches frontend/app.py; root requirements.txt installs UI dependencies. Add a Job resource named copilot-job for the backend Job, granting the App principal permission to manage/run that Job. WBC_APP_JOB_ID comes from that resource. Configure the App allowlist equal to or narrower than the backend's and review manifest defaults. Backend run-as owns data/model permissions; App uses platform authentication without PAT files. See [App configuration](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/app-runtime) and [SDK Jobs API](https://databricks-sdk-py.readthedocs.io/en/latest/workspace/jobs/jobs.html).

## 6. App smoke and final immutable E2E

Ask one structured, document and investigation question from the fixed cases. Inspect route, project, sources/citations, UNKNOWN/clarification, critic outcome, request ID and actual trace. Change project: old answers must not appear as current evidence. Source follow-ups show previous snapshots; new questions fetch new evidence. Backend validation must reject unsupported projects/unknown tools/project changes/time expansion, stop on budget exhaustion, and expose no chain-of-thought or credentials.

Run notebook 12 against the final reviewed snapshot with NEW e2e1 (or next unused ID). Mechanical PASS plus actual App smoke and human semantic review are required for acceptance. Missing 10D capability PASS, unresolved 10d3 diagnosis, model-quality review, runtime/permissions failures or failed App smoke remain blockers. Preserve failures; never relax checks to claim completion.
