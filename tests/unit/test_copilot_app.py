"""Databricks App presentation layer: real InvestigationResult fixtures, no network or model.

Fixtures are produced by the real offline Copilot stack (fake model replies), so the UI is
tested against the actual result contract rather than hand-written JSON.
"""

import ast
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from tests.unit.test_copilot_runtime import (
    INVESTIGATION,
    PROJECT,
    Critic,
    Investigator,
    Synthesizer,
    copilot,
)

APP = Path(__file__).resolve().parents[2] / "copilot_app"
sys.path.insert(0, str(APP))

import backend  # noqa: E402
import presentation  # noqa: E402

# name: (question, Synthesizer changes, Investigator round-1 changes, Critic supports)
QUESTIONS = {
    "ANSWER": (INVESTIGATION, {}, {}, ("SUPPORTED", "PARTIALLY_SUPPORTED")),
    "EVIDENCE_ONLY_TIMELINE": ("Show the timeline of restructurings.", {}, {}, ("SUPPORTED",)),
    "EVIDENCE_ONLY_ATTENTION": ("What deserves my attention?", {}, {}, ("SUPPORTED",)),
    "INSUFFICIENT_EVIDENCE": (INVESTIGATION, {"abstain": True}, {}, ("SUPPORTED",)),
    "CLARIFY": ("Tell me about it.", {}, {"disposition": "CLARIFY"}, ("SUPPORTED",)),
    "REFUSE": ("Will the project fail?", {}, {}, ("SUPPORTED",)),
    "FAIL_CLOSED": (
        INVESTIGATION,
        {"text": "Unlike P179039, delays occurred."},
        {},
        ("SUPPORTED",),
    ),
}


@pytest.fixture(scope="module")
def results():
    out = {}
    for name, (question, changes, investigator, supports) in QUESTIONS.items():
        app = copilot(Synthesizer(**changes), Critic(supports), Investigator(**investigator))
        out[name] = json.loads(app.investigate(question, PROJECT).model_dump_json())
    return out


def strings(view) -> str:
    return json.dumps(asdict(view))


def identifiers(result) -> set[str]:
    """Every internal identifier present in a result that must never be displayed."""
    found = set()
    for e in result["evidence"]:
        found.update(v for v in (e.get("evidence_id"),) if v)
        for part in (e.get("citation") or {}, e.get("source") or {}, e.get("payload") or {}):
            for key in ("document_id", "record_id", "source_record_id", "chunk_id", "source_hash"):
                if isinstance(part.get(key), str):
                    found.add(part[key])
    for c in result["claims"]:
        found.update(c["evidence_ids"])
        found.update(ref["source_identity"] for ref in c["citations"])
    return found


def test_fixtures_cover_every_status(results):
    assert {r["status"] for r in results.values()} == set(backend.STATUSES)


def test_answer_renders_claims_with_provenance_and_sources(results):
    view = presentation.present(results["ANSWER"])
    assert view.headline is None and view.claims
    for claim, raw in zip(view.claims, results["ANSWER"]["claims"], strict=True):
        assert claim.text == raw["text"]
        assert claim.badge == presentation.badge(raw["provenance"])
        assert claim.sources
    assert {c.badge.label for c in view.claims} <= {
        "FACT",
        "DOCUMENTED FINDING",
        "SYSTEM-DERIVED SIGNAL",
        "AI INTERPRETATION",
    }
    assert view.governed_notice is None and view.sources


def test_evidence_only_timeline_is_governed_and_chronological(results):
    result = results["EVIDENCE_ONLY_TIMELINE"]
    assert result["model_calls"] == []
    view = presentation.present(result)
    assert view.governed_notice == "Governed data — no generative model used"
    assert not view.claims and view.headline is None
    timeline = next(t for t in view.tables if t.title == "Timeline")
    assert timeline.columns == ("Date", "Type", "Event", "Description", "ISR")
    dates = [row[0][:10] for row in timeline.rows if row[0][:1].isdigit()]
    assert dates == sorted(dates)


def test_evidence_only_attention_renders_signal_cards(results):
    view = presentation.present(results["EVIDENCE_ONLY_ATTENTION"])
    assert view.governed_notice and view.signal_groups and not view.claims
    assert not any(t.title == "Attention signals" for t in view.tables)
    raw = results["EVIDENCE_ONLY_ATTENTION"]["attention_signals"]
    shown = [s for _, group in view.signal_groups for s in group]
    assert len(shown) == len(raw)
    assert {s.severity for s in shown} == {s["severity"] for s in raw}  # never invented


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("INSUFFICIENT_EVIDENCE", "not sufficient to support a publishable answer"),
        ("CLARIFY", "couldn't determine the intended analysis"),
        ("REFUSE", "does not predict whether a project will succeed or fail"),
        ("FAIL_CLOSED", "did not pass the Copilot's validation checks"),
    ],
)
def test_non_answer_statuses_render_fixed_user_messages(results, name, message):
    view = presentation.present(results[name])
    assert view.status == name and message in view.headline and not view.claims
    # Internal routing/validation wording stays out of the main message.
    assert results[name]["message"] not in view.headline


def test_fail_closed_keeps_diagnostics_in_technical_details(results):
    view = presentation.present(results["FAIL_CLOSED"])
    technical = dict(view.technical)
    assert technical["Validation failures"] == "PROJECT_ISOLATION_VIOLATION"
    assert "PROJECT_ISOLATION_VIOLATION" not in view.headline


@pytest.mark.parametrize("name", list(QUESTIONS))
def test_internal_identifiers_and_handles_are_never_displayed(results, name):
    text = strings(presentation.present(results[name]))
    for identifier in identifiers(results[name]):
        assert identifier not in text, identifier
    assert not re.search(r"\bev_[0-9a-f]{6}|\breq_[0-9a-f]{6}|\bcitation_[0-9a-f]{6}", text)
    assert not re.search(r'"[ER][0-9]{1,3}"', text)  # local handles


def test_unknown_is_never_rendered_as_authoritative(results):
    result = json.loads(json.dumps(results["ANSWER"]))
    for claim in result["claims"]:
        claim["provenance"] = "UNKNOWN"
    view = presentation.present(result)
    assert not view.claims and view.withheld_unknown_claims == len(result["claims"])
    assert "not sufficient" in view.headline
    assert presentation.badge("UNKNOWN").label == "UNKNOWN — not available"


def test_unknown_structured_value_reads_as_not_available():
    item = {
        "evidence_id": "ev_x",
        "provenance": ["UNKNOWN"],
        "source_type": "STRUCTURED",
        "citation": None,
        "source": {"table": "gold.project_360", "record_id": "r1"},
        "payload": {"name": "closing_date", "value": None, "unknown_reason": "Not reported"},
    }
    source = presentation._source(item)
    assert source.title == "Closing date: not available (Not reported)"
    assert source.badge.label == "UNKNOWN — not available"


@pytest.mark.parametrize("name", list(QUESTIONS))
def test_technical_details_contain_safe_metadata_only(results, name):
    result = results[name]
    view = presentation.present(result)
    allowed = {
        "Route", "Intent", "Routing reason", "Result status", "Validation disposition",
        "Mechanical validity", "Semantic support", "Critic status", "Validation failures",
        "Evidence records", "Published claims", "Citations", "Attention signals", "Model calls",
        "Investigator rounds", "Governed tool calls", "Planning failure", "Rejected actions",
        "Rejection reasons",
        "Evidence shown to models", "Date anchor", "Anchor resolution", "Claims removed",
        "Total latency", "MLflow trace ID", "Request ID", "Model note",
    }  # fmt: skip
    assert {label for label, _ in view.technical} <= allowed
    technical = json.dumps([view.technical, view.model_calls, view.stage_latency])
    assert result["query"] not in technical
    for claim in result["claims"]:
        assert claim["text"] not in technical
    for item in result["evidence"]:
        excerpt = (item.get("payload") or {}).get("text")
        if excerpt:
            assert excerpt[:40] not in technical


def test_internal_limitations_move_to_technical_notes(results):
    view = presentation.present(results["ANSWER"])
    assert not any("selected by the Investigator" in x for x in view.limitations)
    assert any("selected by the Investigator" in x for x in view.technical_notes)


def test_claim_support_and_qualifier_are_shown(results):
    view = presentation.present(results["ANSWER"])
    assert [c.support for c in view.claims] == [
        "Supported by cited evidence",
        "Partially supported",
    ]
    assert view.claims[0].qualifier is None and view.claims[1].qualifier
    assert view.objective == results["ANSWER"]["objective"]
    technical = dict(view.technical)
    assert technical["Investigator rounds"] == "2" and technical["Governed tool calls"] == "2"
    assert view.objective not in json.dumps(view.technical)


# -- architecture: the UI never bypasses copilot.investigate -------------------------
def imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    return names


def test_presentation_is_pure_and_cannot_call_models():
    assert imports(APP / "presentation.py") <= {"__future__", "re", "dataclasses", "typing"}


def test_app_layer_has_no_intelligence_or_tracing_dependencies():
    assert imports(APP / "app.py") <= {"os", "streamlit", "backend", "presentation"}
    assert imports(APP / "backend.py") <= {
        "__future__",
        "json",
        "re",
        "datetime",
        "typing",
        "presentation",
        "databricks",
    }
    for name in ("app.py", "backend.py", "presentation.py"):
        names = imports(APP / name) | {
            n.id
            for n in ast.walk(ast.parse((APP / name).read_text("utf-8")))
            if isinstance(n, ast.Name)
        }
        assert not {"worldbank_copilot", "mlflow", "openai", "requests"} & names, name


def test_app_has_exactly_one_investigate_call_and_examples_only_populate():
    tree = ast.parse((APP / "app.py").read_text(encoding="utf-8"))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "investigate"
    ]
    assert len(calls) == 1
    loop = next(
        n for n in ast.walk(tree) if isinstance(n, ast.For) and "EXAMPLES" in ast.unparse(n.iter)
    )
    assert "investigate" not in ast.unparse(loop) and "session_state.question" in ast.unparse(loop)


def test_backend_notebook_calls_only_the_copilot_service():
    source = (APP.parent / "notebooks/15_copilot_app_backend.py").read_text(encoding="utf-8")
    assert "copilot.investigate(" in source and "build_copilot(spark, settings)" in source
    code = [
        line for line in source.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    assert not any("mlflow" in line.lower() for line in code)
    assert "requirements-copilot-runtime" not in source
    install = next(line for line in source.splitlines() if "%pip install" in line)
    validated = next(
        line
        for line in (APP.parent / "notebooks/14_copilot_prototype_validation.py")
        .read_text("utf-8")
        .splitlines()
        if "%pip install" in line
    )
    assert install == validated


def test_app_yaml_binds_the_backend_job_resource():
    config = yaml.safe_load((APP / "app.yaml").read_text(encoding="utf-8"))
    assert config["command"][:3] == ["streamlit", "run", "app.py"]
    assert {"name": "WBC_COPILOT_JOB_ID", "valueFrom": "copilot-job"} in config["env"]


def test_projects_match_the_registered_project_names():
    registry = yaml.safe_load((APP.parent / "configs/projects.yaml").read_text(encoding="utf-8"))
    names = {p["project_id"]: p["name"] for p in registry["projects"]}
    for project_id, name in presentation.PROJECTS.items():
        assert names[project_id].endswith(name)
    assert presentation.DEFAULT_PROJECT == "P130544" and len(presentation.EXAMPLES) == 4


# -- backend client ------------------------------------------------------------------
def fake_client(result: dict | str, *, truncated=False, tasks=1):
    client = SimpleNamespace(jobs=SimpleNamespace(run_now=Mock(), get_run_output=Mock()))
    client.jobs.run_now.return_value.result.return_value = SimpleNamespace(
        tasks=[SimpleNamespace(run_id=7)] * tasks
    )
    payload = result if isinstance(result, str) else json.dumps(result)
    client.jobs.get_run_output.return_value = SimpleNamespace(
        notebook_output=SimpleNamespace(result=payload, truncated=truncated)
    )
    return client


def test_backend_runs_the_job_with_only_project_and_question(results):
    result = results["ANSWER"]
    client = fake_client(result)
    returned = backend.CopilotBackend(42, client=client).investigate(f"  {INVESTIGATION} ", PROJECT)
    assert returned == result
    client.jobs.run_now.assert_called_once()
    kwargs = client.jobs.run_now.call_args.kwargs
    assert kwargs == {
        "job_id": 42,
        "job_parameters": {"project_id": PROJECT, "question": INVESTIGATION},
    }


@pytest.mark.parametrize(
    ("question", "project"),
    [
        ("", PROJECT),
        ("x" * 1001, PROJECT),
        ("What is it? Bearer abc.def", PROJECT),
        ("status", "P999999"),
    ],
    ids=["empty", "too-long", "credential", "unsupported-project"],
)
def test_backend_rejects_bad_requests_before_any_call(question, project):
    client = fake_client({})
    with pytest.raises(backend.BackendError):
        backend.CopilotBackend(1, client=client).investigate(question, project)
    client.jobs.run_now.assert_not_called()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: {**r, "project_id": "P179039"},
        lambda r: {**r, "query": "a different question"},
        lambda r: {**r, "status": "SOMETHING_ELSE"},
    ],
    ids=["other-project", "other-query", "unknown-status"],
)
def test_backend_rejects_results_outside_the_request_scope(results, mutate):
    client = fake_client(mutate(results["ANSWER"]))
    with pytest.raises(backend.BackendError):
        backend.CopilotBackend(1, client=client).investigate(INVESTIGATION, PROJECT)


@pytest.mark.parametrize(
    "kwargs", [{"truncated": True}, {"tasks": 2}], ids=["truncated", "two-tasks"]
)
def test_backend_rejects_incomplete_job_output(results, kwargs):
    client = fake_client(results["ANSWER"], **kwargs)
    with pytest.raises(backend.BackendError):
        backend.CopilotBackend(1, client=client).investigate(INVESTIGATION, PROJECT)
