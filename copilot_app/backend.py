"""App -> Job boundary: runs ``notebooks/15_copilot_app_backend`` (``copilot.investigate``).

The App never routes, retrieves or calls a model itself. It submits the selected project
and question to one trusted notebook job and validates the returned result's scope.
No tracing happens here; the backend records its own allowlisted MLflow trace.
"""

from __future__ import annotations

import json
import re
from datetime import timedelta
from typing import Any

from presentation import PROJECTS

STATUSES = {"ANSWER", "EVIDENCE_ONLY", "INSUFFICIENT_EVIDENCE", "CLARIFY", "REFUSE", "FAIL_CLOSED"}
MAX_QUESTION = 1000
MAX_RESPONSE_BYTES = 5_000_000
_CREDENTIAL = re.compile(
    r"(?i)(Bearer\s+\S+|dapi[0-9a-f]{20,}|(?:api_key|password|access_token)\s*[:=])"
)


class BackendError(Exception):
    """Safe, user-presentable backend failure (no SDK or job internals)."""


def validate_request(question: str, project_id: str) -> str:
    question = (question or "").strip()
    if project_id not in PROJECTS:
        raise BackendError("Select one of the supported projects.")
    if not question:
        raise BackendError("Enter a question to investigate.")
    if len(question) > MAX_QUESTION:
        raise BackendError(f"Questions are limited to {MAX_QUESTION} characters.")
    if _CREDENTIAL.search(question):
        raise BackendError("The question looks like it contains a credential; remove it.")
    return question


class CopilotBackend:
    """Runs one trusted notebook job per investigation and returns its result JSON."""

    def __init__(self, job_id: str | int, client: Any = None, timeout_minutes: int = 15):
        self.job_id = int(job_id)
        self.timeout = timedelta(minutes=timeout_minutes)
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from databricks.sdk import WorkspaceClient

            self._client = WorkspaceClient()
        return self._client

    def investigate(self, question: str, project_id: str) -> dict[str, Any]:
        question = validate_request(question, project_id)
        run = self.client.jobs.run_now(
            job_id=self.job_id,
            job_parameters={"project_id": project_id, "question": question},
        ).result(timeout=self.timeout)
        tasks = run.tasks or []
        if len(tasks) != 1 or tasks[0].run_id is None:
            raise BackendError("The Copilot backend job is misconfigured (one task expected).")
        output = self.client.jobs.get_run_output(run_id=tasks[0].run_id).notebook_output
        if output is None or output.truncated or not output.result:
            raise BackendError("The Copilot backend returned no complete result.")
        if len(output.result.encode()) > MAX_RESPONSE_BYTES:
            raise BackendError("The Copilot backend result was too large to display.")
        return validate_result(json.loads(output.result), question, project_id)


def validate_result(result: Any, question: str, project_id: str) -> dict[str, Any]:
    """The result must be an InvestigationResult for exactly this request's scope."""
    if not isinstance(result, dict) or result.get("status") not in STATUSES:
        raise BackendError("The Copilot backend returned an unrecognized result.")
    if result.get("project_id") != project_id or result.get("query") != question:
        raise BackendError("The Copilot backend result does not match this request.")
    return result
