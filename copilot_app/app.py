"""Databricks App: presentation layer over ``copilot.investigate`` (via one trusted job).

Layout only. Every result decision comes from the Copilot backend; ``presentation``
formats it, and ``backend`` is the only path to the Copilot.
"""

import os

import streamlit as st
from backend import BackendError, CopilotBackend
from presentation import (
    DEFAULT_PROJECT,
    EXAMPLES,
    PROJECTS,
    SIGNAL_NOTICE,
    ResultView,
    SourceView,
    present,
    signal_groups,
)

st.set_page_config(page_title="Implementation Intelligence Copilot", layout="wide")


_MARKDOWN = set("\\`*_{}[]()#+-.!|<>~:$")


def md(value: str) -> str:
    """Escape data-derived text for Markdown (titles, labels, limitations)."""
    return "".join("\\" + ch if ch in _MARKDOWN else ch for ch in str(value))


def badge(b) -> str:
    return f":{b.color}-background[{b.label}]"


def text(value: str) -> None:
    """Untrusted (model or document) text: shown verbatim, never interpreted as markup."""
    st.text(value)


def pairs(rows) -> None:
    """Two-column field/value table for technical metadata."""
    st.dataframe(
        [{"Field": label, "Value": value} for label, value in rows],
        hide_index=True,
        use_container_width=True,
    )


def source_card(source: SourceView) -> None:
    st.markdown(f"**{md(source.title)}**  {badge(source.badge)}")
    if source.details:
        st.caption(" · ".join(source.details))
    if source.excerpt:
        text(source.excerpt)


def signals_panel(groups) -> None:
    st.subheader("Implementation Attention")
    st.caption(SIGNAL_NOTICE)
    if not groups:
        st.caption("Run an investigation to load the selected project's current signals.")
    for category, signals in groups:
        st.markdown(f"**{md(category)}**")
        for s in signals:
            with st.container(border=True):
                st.markdown(f"**{md(s.title)}**")
                st.caption(" · ".join(v for v in (s.severity, s.status, s.observed) if v))
                if s.values:
                    st.caption(s.values)
                if s.description:
                    text(s.description)
                for caveat in s.caveats:
                    st.caption(caveat)


def main_result(view: ResultView) -> None:
    if view.headline:
        tone = {"REFUSE": st.info, "CLARIFY": st.info, "FAIL_CLOSED": st.warning}
        tone.get(view.status, st.warning)(view.headline)
    if view.objective:
        st.caption("Investigated: " + md(view.objective))
    for claim in view.claims:
        with st.container(border=True):
            st.markdown(f"{badge(claim.badge)}  {md(claim.support)}")
            text(claim.text)
            if claim.qualifier:
                st.caption("Not fully established: " + md(claim.qualifier))
            if claim.sources:
                st.caption("Sources: " + "; ".join(md(s) for s in claim.sources))
    if view.withheld_unknown_claims:
        st.caption(
            f"{view.withheld_unknown_claims} statement(s) withheld: underlying values unknown."
        )
    if view.governed_notice:
        st.success(view.governed_notice)
    for table in view.tables:
        st.markdown(f"**{md(table.title)}**")
        st.dataframe(
            [dict(zip(table.columns, row, strict=True)) for row in table.rows],
            hide_index=True,
            use_container_width=True,
        )
    for passage in view.passages:
        with st.container(border=True):
            source_card(passage)
    if view.status == "EVIDENCE_ONLY" and not (view.tables or view.passages):
        st.caption("See Implementation Attention for the governed signals.")
    if view.sources:
        with st.expander(f"Evidence & Sources ({len(view.sources) + view.hidden_sources})"):
            for source in view.sources:
                source_card(source)
                st.divider()
            if view.hidden_sources:
                st.caption(f"…and {view.hidden_sources} more governed records.")
    if view.limitations:
        with st.expander("Limitations"):
            for limitation in view.limitations:
                st.markdown(f"- {md(limitation)}")
    with st.expander("Technical Details"):
        st.caption(
            "Routing → Evidence → Synthesis → Deterministic Validation → Critic → Finalization"
        )
        pairs(view.technical)
        if view.stage_latency:
            st.markdown("**Stage latency**")
            pairs(view.stage_latency)
        for call in view.model_calls:
            st.markdown("**Model call**")
            pairs(call)
        if view.technical_notes:
            st.markdown("**Engineering notes**")
            for note in view.technical_notes:
                st.caption(note)


st.title("World Bank Project Implementation Intelligence Copilot")
st.markdown("Evidence-grounded implementation monitoring and investigation")
st.caption("Decision support, not project failure prediction.")

job_id = os.environ.get("WBC_COPILOT_JOB_ID", "").strip()
if not job_id:
    st.error("The Copilot backend job is not configured (WBC_COPILOT_JOB_ID).")
    st.stop()

left, right = st.columns([2, 1], gap="large")
with left:
    projects = list(PROJECTS)
    project = st.selectbox(
        "Project",
        projects,
        index=projects.index(DEFAULT_PROJECT),
        format_func=lambda pid: f"{pid} — {PROJECTS[pid]}",
    )
    st.session_state.setdefault("question", "")
    chips = st.columns(len(EXAMPLES))
    for column, example in zip(chips, EXAMPLES, strict=True):
        if column.button(example, use_container_width=True):
            st.session_state.question = example  # populates only; never runs
    question = st.text_area("Question", key="question", max_chars=1000)
    if st.button("Investigate", type="primary"):
        try:
            with st.spinner("Investigating governed evidence…"):
                result = CopilotBackend(job_id).investigate(question, project)
            st.session_state.result = result
            st.session_state.error = None
        except BackendError as exc:
            st.session_state.result, st.session_state.error = None, str(exc)
        except Exception as exc:  # UI boundary: never show SDK/job internals
            st.session_state.result = None
            st.session_state.error = (
                f"The Copilot service is unavailable right now ({type(exc).__name__})."
            )
    if st.session_state.get("error"):
        st.error(st.session_state.error)
    result = st.session_state.get("result")
    view = present(result) if result and result.get("project_id") == project else None
    if view:
        main_result(view)
with right:
    signals_panel(view.signal_groups if view else signal_groups(()))
