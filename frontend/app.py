"""Databricks App: Streamlit display and one trusted Spark notebook job."""

import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
from databricks.sdk import WorkspaceClient

from worldbank_copilot.application.job_client import ask_job
from worldbank_copilot.application.service import AnswerRequest, source_followup

st.title("World Bank Implementation Intelligence")
st.caption("Observable evidence and implementation signals. Human judgment remains essential.")
projects = tuple(
    p.strip() for p in os.environ.get("WBC_APP_ALLOWED_PROJECTS", "").split(",") if p.strip()
)
if not projects or not os.environ.get("WBC_APP_JOB_ID"):
    st.error("Configure the approved project list and backend notebook job before use.")
    st.stop()
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
project = st.selectbox("Project", projects)
question = st.text_area("Question", max_chars=1000)
st.caption(
    "Include dates or ISR sequences in your question. "
    "Unsupported temporal scopes require clarification."
)
if st.button("Ask", type="primary"):
    try:
        with st.spinner("Reading governed evidence?"):
            request = AnswerRequest(
                project_id=project, question=question, session_id=st.session_state.session_id
            )
            st.session_state.answer = ask_job(
                WorkspaceClient(), os.environ["WBC_APP_JOB_ID"], request
            )
    except Exception as exc:
        st.error("Request stopped safely: " + type(exc).__name__)
answer = st.session_state.get("answer")
if answer and answer.project_id == project:
    st.write(answer.message)
    if answer.final:
        for claim in answer.final.published_claims:
            st.write(f"{claim.claim_id} ? {claim.provenance_label.value}")
            st.write(claim.claim_text)
            st.caption("Evidence: " + ", ".join(claim.evidence_ids))
        for limitation in answer.final.limitations:
            st.caption(limitation)
    with st.expander("Evidence and source citations"):
        st.caption("Previous answer snapshot; a new question obtains current evidence.")
        for source in source_followup(answer, project, projects):
            st.json(source)
    if answer.trace:
        with st.expander("Request diagnostics"):
            trace = answer.trace
            st.json(
                {
                    "route": trace.route,
                    "investigator_used": trace.investigator_used,
                    "evidence_count": len(answer.evidence),
                    "critic_used": trace.critic_used,
                    "latency_ms": trace.latency_ms,
                    "request_id": trace.request_id,
                    "trace_id": trace.mlflow_trace_id,
                    "status": answer.status,
                }
            )
