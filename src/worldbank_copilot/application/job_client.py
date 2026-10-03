"""Small App-to-notebook boundary. SDK calls occur only on an explicit Ask."""

import uuid
from datetime import timedelta

from worldbank_copilot.application.service import Answer, AnswerRequest


def ask_job(client, job_id, request):
    request = AnswerRequest.model_validate(request)
    # No notebook path, SQL, access grants, or endpoints originate in the request.
    run = client.jobs.run_now(
        job_id=int(job_id),
        job_parameters={"request_json": request.model_dump_json()},
        idempotency_token=request.session_id + "-" + uuid.uuid4().hex[:20],
    ).result(timeout=timedelta(minutes=10))
    tasks = run.tasks or []
    if len(tasks) != 1 or tasks[0].run_id is None:
        raise ValueError("one trusted notebook task required")
    output = client.jobs.get_run_output(run_id=tasks[0].run_id)
    if (
        output.notebook_output is None
        or output.notebook_output.truncated
        or not output.notebook_output.result
    ):
        raise ValueError("missing or truncated backend response")
    if len(output.notebook_output.result.encode()) > 2000000:
        raise ValueError("backend response too large")
    answer = Answer.model_validate_json(output.notebook_output.result)
    if (
        answer.project_id != request.project_id
        or answer.trace is None
        or answer.trace.session_id != request.session_id
    ):
        raise ValueError("backend response scope/session mismatch")
    return answer
