# Copilot App

A Databricks App (Streamlit) that presents `Copilot.investigate()`. It is a
**presentation layer only**: it never routes, retrieves or calls a model.

```
app.py (Streamlit UI)
  -> backend.py: validate input -> Jobs API run_now(project_id, question)
  -> Job: notebooks/15_copilot_app_backend -> Copilot.investigate()
  <- backend.py: parse and scope-check the InvestigationResult JSON
  -> presentation.py: claims, provenance badges, support labels, citations, signals, activity
```

| File | Role |
|---|---|
| `app.py` | Streamlit page: project selector, question box, result rendering |
| `backend.py` | the App-to-Job boundary: input validation, job submission, result validation |
| `presentation.py` | pure formatting helpers (labels, badges, tables); unit-tested |
| `app.yaml` | App command and environment; binds the Job resource |
| `requirements.txt` | App-only dependencies (Streamlit, Databricks SDK) |

**Input and output checks** (`backend.py`):

- only the supported projects are accepted;
- questions are limited to 1,000 characters;
- credential-like input is rejected;
- responses are size-bounded;
- the returned project scope must match the request;
- errors are shown as safe, generic messages.

## Deploy

1. Create a Job with one notebook task: `notebooks/15_copilot_app_backend` on serverless
   compute, with parameters `project_id` and `question`.
2. Create a Databricks App with source path `<git folder>/copilot_app`.
3. Add the Job as an App resource named **`copilot-job`** with `CAN_MANAGE_RUN`.
   `app.yaml` maps it to `WBC_COPILOT_JOB_ID`, so no job ID is hard-coded.
4. Deploy. An App deployment is a snapshot of the source, so redeploy after pulling changes.

Tests: `tests/unit/test_copilot_app.py`.
