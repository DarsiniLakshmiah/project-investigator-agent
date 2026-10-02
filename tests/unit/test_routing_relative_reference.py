"""Phase 9C human-review fix (case r073): relative project references.

"the other Karnataka program" never selects a project by itself and is never replaced by
the active scope. Unresolvable -> CLARIFY AMBIGUOUS_PROJECT_REFERENCE before any read.
Uniquely resolvable from authorised context (active scope + exactly one other authorised
project) -> that referent is a foreign project -> REFUSE CROSS_PROJECT. Both behaviours
execute nothing.
"""

import pytest
from tests.support.routing_fixtures import ALL, CONFIG, INDEX, harness
from tests.support.tool_fixtures import IPF, OTHER, PFORR

from worldbank_copilot.routing.models import (
    AccessContext,
    ExecutionOutcome,
    MentionKind,
    ProjectStatus,
    Route,
)

R073 = "How does the other Karnataka program's disbursement compare with ours?"
RELATIVE = [
    R073,
    "What did the other program disburse?",
    "Show the ratings of another project.",
    "Is the other operation also restructured?",
    "How do the other programs compare?",
    "What about the sister project?",
    "Compare both projects.",
    "Show disbursement for all three programs.",
]


def nothing_ran(h):
    return h.executor.calls == [] and h.retriever.first_stage_calls == [] and h.reads == []


@pytest.mark.parametrize("question", RELATIVE)
@pytest.mark.parametrize("active", [IPF, PFORR, OTHER, None])
def test_unresolvable_relative_reference_is_clarified_and_executes_nothing(question, active):
    h = harness()
    r = h.ask(question, active=active, authorized=ALL)
    assert (r.decision.route, r.decision.reason_code) == (
        Route.CLARIFY,
        "AMBIGUOUS_PROJECT_REFERENCE",
    )
    p = r.understanding.project
    assert p.status == ProjectStatus.AMBIGUOUS_REFERENCE
    assert p.project_id is None  # active scope NOT substituted, no foreign project selected
    assert r.outcome == ExecutionOutcome.NOT_EXECUTED and nothing_ran(h)
    assert r.understanding.intent is None  # stopped at the project stage
    assert any(m.kind == MentionKind.RELATIVE_REFERENCE for m in p.mentions)


def test_r073_before_and_after():
    """Before the fix 9B resolved r073 to the active scope and ran get_financial_status
    for P130544 (recorded in evaluation/routing_baseline_9B.1.yaml). Now: no execution."""
    h = harness()
    r = h.ask(R073, active=IPF)
    assert r.decision.reason_code == "AMBIGUOUS_PROJECT_REFERENCE"
    assert r.executed_tools == [] and r.anchor_results == [] and not r.retrieval_executed
    assert nothing_ran(h)


@pytest.mark.parametrize(("active", "other"), [(IPF, PFORR), (PFORR, OTHER), (OTHER, IPF)])
def test_uniquely_resolvable_reference_is_a_cross_project_refusal(active, other):
    """Exactly one other authorised project: the referent is unique - and foreign."""
    h = harness()
    r = h.ask(R073, active=active, authorized=(active, other))
    assert (r.decision.route, r.decision.reason_code) == (Route.REFUSE, "CROSS_PROJECT")
    p = r.understanding.project
    assert (p.status, p.project_id, p.mentioned_projects) == (
        ProjectStatus.CONFLICT,
        None,
        (other,),
    )
    assert nothing_ran(h)


def test_no_other_authorised_project_or_no_active_scope_is_unresolvable():
    h = harness()
    assert h.ask(R073, active=IPF, authorized=(IPF,)).decision.reason_code == (
        "AMBIGUOUS_PROJECT_REFERENCE"
    )
    assert h.ask(R073, active=None, authorized=(IPF, PFORR)).decision.reason_code == (
        "AMBIGUOUS_PROJECT_REFERENCE"
    )
    assert nothing_ran(h)


def test_explicit_foreign_references_and_authorisation_keep_precedence():
    h = harness()
    r = h.ask("Compare the other program with P506272", active=IPF)
    assert r.decision.reason_code == "CROSS_PROJECT"
    r = h.ask("Compare P179039 and P506272 with the other program", active=IPF)
    assert r.decision.reason_code == "MULTI_PROJECT_NOT_SUPPORTED"
    r = h.ask(R073, active=IPF, authorized=(PFORR, OTHER))
    assert r.decision.reason_code == "NOT_AUTHORIZED"
    assert nothing_ran(h)


@pytest.mark.parametrize(
    "question",
    [
        "What about the other loan?",
        "What is the program's disbursement?",
        "How do other components compare?",
        "Which other risks were identified?",
    ],
)
def test_non_project_uses_of_other_are_not_relative_project_references(question):
    access = AccessContext(authorized_projects=ALL, active_project_id=IPF)
    from worldbank_copilot.routing.entities import resolve_project

    p = resolve_project(question, access, INDEX)
    assert (p.status, p.project_id) == (ProjectStatus.RESOLVED, IPF)


def test_relative_references_are_configuration_not_aliases():
    assert CONFIG.settings.relative_project_references
    assert not any("other" in a for a in INDEX.aliases)
