"""Admission regressions use the unchanged Phase 9 router and offline fixtures."""

from dataclasses import replace
from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError
from tests.conftest import REPO_CONFIG_DIR
from tests.support.routing_fixtures import CONFIG, INDEX, harness
from tests.support.tool_fixtures import IPF, OTHER

from worldbank_copilot.investigation.gate import (
    AdmissionContext,
    AdmissionOutcome,
    AdmissionReason,
    admit,
)
from worldbank_copilot.investigation.models import (
    BASELINE_ID,
    AssessmentStatus,
    RequirementAssessment,
)
from worldbank_copilot.investigation.policy import (
    AttemptStatus,
    BudgetLedger,
    Consumption,
    InvestigationPolicy,
    Reservation,
)
from worldbank_copilot.routing.execution import ExecutionMode, execution_decision
from worldbank_copilot.routing.models import AccessContext, ExecutionOutcome


def admitted_fixture(question="What changed and why?", policy=None, enabled=True):
    h = harness()
    h._documents.profile_id = BASELINE_ID
    original_factory = h.service.context_factory

    def bound_factory(request_id):
        ctx = original_factory(request_id)
        ctx.request_id = request_id
        return ctx

    h.service.context_factory = bound_factory
    result = h.ask(question)
    context = AdmissionContext(
        result.request_id,
        question,
        result.access,
        CONFIG,
        INDEX,
        REPO_CONFIG_DIR,
        policy or InvestigationPolicy(),
        enabled,
    )
    return h, result, context


def test_valid_admission_is_detached_without_execution():
    h, source, context = admitted_fixture()
    original = source.model_dump_json()
    result = admit(source, context)
    assert result.outcome == AdmissionOutcome.ADMITTED
    state = result.state
    assert state.request_id == source.request_id
    assert state.project_id == IPF
    assert state.access == source.access
    assert state.source_plan.executed is False
    assert state.requirements == state.assessments == ()
    assert state.operation_results == () and state.final_response is None
    assert not h.executor.calls and not h.retriever.first_stage_calls and not h.reads
    source.investigation_plan.structured_calls[0].arguments["project_id"] = OTHER
    source.versions["router"] = "tampered"
    exposed = state.source
    exposed.investigation_plan.structured_calls[0].arguments["project_id"] = OTHER
    exposed.versions.clear()
    assert state.source.model_dump_json() == original
    with pytest.raises(ValidationError):
        state.source = source
    with pytest.raises(TypeError):
        state.model_copy(update={"source": source})


def test_feature_default_is_disabled():
    _, source, context = admitted_fixture(enabled=False)
    assert admit(source, context).outcome == AdmissionOutcome.DISABLED
    assert (
        AdmissionContext(
            source.request_id,
            source.question,
            source.access,
            CONFIG,
            INDEX,
            REPO_CONFIG_DIR,
            InvestigationPolicy(),
        ).feature_enabled
        is False
    )


@pytest.mark.parametrize(
    "question,mode",
    [
        ("Show project overview", ExecutionMode.EXECUTE_STRUCTURED),
        ("What does the PAD say about procurement?", ExecutionMode.EXECUTE_DOCUMENT),
        ("Why is this happening?", ExecutionMode.CLARIFY),
        ("Predict project success", ExecutionMode.REFUSE),
    ],
)
def test_other_phase9_modes_cannot_enter(question, mode):
    _, source, context = admitted_fixture(question)
    assert execution_decision(source).mode == mode
    admission = admit(source, context)
    assert admission.outcome != AdmissionOutcome.ADMITTED
    assert admission.state is None


def test_candidate_a_clarification_stays_outside():
    _, source, context = admitted_fixture("Why is this happening?")
    decision = execution_decision(source)
    assert decision.translated and decision.mode == ExecutionMode.CLARIFY
    assert admit(source, context).outcome == AdmissionOutcome.CLARIFY


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: setattr(r, "request_id", "forged"),
        lambda r: setattr(r, "question", "forged"),
        lambda r: setattr(
            r, "access", AccessContext(authorized_projects=(OTHER,), active_project_id=OTHER)
        ),
        lambda r: setattr(r, "retrieval_executed", True),
        lambda r: r.executed_tools.append("get_project_overview"),
        lambda r: setattr(r, "outcome", ExecutionOutcome.EXECUTED),
        lambda r: r.investigation_plan.structured_calls[0].arguments.update(project_id=OTHER),
        lambda r: r.investigation_plan.structured_calls[0].arguments.update(sql="select *"),
    ],
)
def test_forged_or_executed_source_fails(mutation):
    h, source, context = admitted_fixture()
    mutation(source)
    assert admit(source, context).outcome == AdmissionOutcome.FAIL
    assert not h.executor.calls and not h.retriever.first_stage_calls


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("question", "Different question", AdmissionReason.REQUEST_MISMATCH),
        ("project_id", OTHER, AdmissionReason.AUTHORIZATION_MISMATCH),
        ("retrieval_profile", "adaptive@1", AdmissionReason.PROFILE_MISMATCH),
        ("investigation_kind", "UNKNOWN", AdmissionReason.UNSUPPORTED_KIND),
        ("provenance_requirements", (), AdmissionReason.PLAN_MISMATCH),
        ("executed", True, AdmissionReason.INVALID_SOURCE),
    ],
)
def test_plan_mutation_is_revalidated(field, value, reason):
    _, source, context = admitted_fixture()
    source.investigation_plan = source.investigation_plan.model_copy(update={field: value})
    result = admit(source, context)
    assert result.outcome == AdmissionOutcome.FAIL
    assert result.reason == reason


def test_temporal_and_call_flags_are_not_authority():
    _, source, context = admitted_fixture()
    plan = source.investigation_plan
    source.investigation_plan = plan.model_copy(
        update={"temporal": plan.temporal.model_copy(update={"date_to": date(2020, 1, 1)})}
    )
    assert admit(source, context).reason == AdmissionReason.TEMPORAL_MISMATCH
    _, source, context = admitted_fixture()
    calls = source.investigation_plan.structured_calls
    forged = calls[0].model_copy(update={"tool": "web_search", "validated": True})
    source.investigation_plan = source.investigation_plan.model_copy(
        update={"structured_calls": (forged, *calls[1:])}
    )
    assert admit(source, context).reason == AdmissionReason.INVALID_CALL


def test_authorization_and_supplied_execution_are_bound():
    _, source, context = admitted_fixture()
    foreign = replace(
        context, access=AccessContext(authorized_projects=(OTHER,), active_project_id=OTHER)
    )
    assert admit(source, foreign).reason == AdmissionReason.AUTHORIZATION_MISMATCH
    decision = execution_decision(source).model_copy(
        update={"mode": ExecutionMode.EXECUTE_STRUCTURED}
    )
    assert admit(source, context, decision).reason == AdmissionReason.INVALID_SOURCE


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_initial_requirements": 1},
        {"max_initial_operations": 1},
        {"max_retrieval_operations": 0},
    ],
)
def test_admission_checks_policy_bounds(kwargs):
    _, source, context = admitted_fixture(policy=InvestigationPolicy(**kwargs))
    assert admit(source, context).reason == AdmissionReason.POLICY_BOUNDS


def test_policy_hypotheses_and_unset_live_cost():
    policy = InvestigationPolicy()
    assert policy.max_repair_cycles == 1 and policy.aggregate_token_budget == 60000
    assert policy.cost_ceiling is None and policy.status == "INITIAL_HYPOTHESIS"
    with pytest.raises(ValueError, match="live execution"):
        policy.require_live_cost_configuration()
    with pytest.raises(ValidationError):
        InvestigationPolicy(cost_ceiling=1.5, pricing_version="test")
    with pytest.raises(ValidationError):
        InvestigationPolicy(cost_ceiling=Decimal("1"))
    with pytest.raises(ValidationError):
        InvestigationPolicy(max_initial_operations=11)


def test_ledger_monotonic_failed_attempt_and_deadline():
    ledger = BudgetLedger(policy=InvestigationPolicy())
    reserved = ledger.reserve(
        Reservation(
            reservation_id="a",
            operations=1,
            initial_operations=1,
            retrieval_operations=1,
            model_calls=1,
            tokens=100,
        )
    )
    failed = reserved.consume(
        Consumption(reservation_id="a", status=AttemptStatus.FAILED, actual_tokens=20)
    )
    elapsed = failed.advance_elapsed(Decimal("10"))
    assert elapsed.extends(ledger) and elapsed.extends(reserved)
    assert elapsed.reservations[0].tokens == 100
    assert elapsed.consumptions[0].actual_tokens == 20
    assert not ledger.extends(reserved)
    with pytest.raises(ValueError):
        elapsed.advance_elapsed(Decimal("9"))
    with pytest.raises(ValueError):
        elapsed.advance_elapsed(Decimal("181"))
    with pytest.raises(ValidationError):
        reserved.consume(
            Consumption(reservation_id="a", status=AttemptStatus.SUCCEEDED, actual_tokens=101)
        )
    with pytest.raises(ValidationError):
        failed.consume(Consumption(reservation_id="a", status=AttemptStatus.FAILED))
    with pytest.raises(ValidationError):
        reserved.reserve(Reservation(reservation_id="a", tokens=1))


@pytest.mark.parametrize(
    "field,value",
    [
        ("operations", 11),
        ("initial_operations", 9),
        ("retrieval_operations", 5),
        ("model_calls", 7),
        ("tokens", 60001),
        ("repair_cycles", 2),
    ],
)
def test_each_ledger_budget_is_bounded(field, value):
    kwargs = {field: value}
    if field in ("initial_operations", "retrieval_operations"):
        kwargs["operations"] = value
    with pytest.raises(ValidationError):
        BudgetLedger(policy=InvestigationPolicy()).reserve(
            Reservation(reservation_id="a", **kwargs)
        )


@pytest.mark.parametrize(
    "field", ["operations", "retrieval_operations", "model_calls", "tokens", "repair_cycles"]
)
def test_negative_reservations_rejected(field):
    with pytest.raises(ValidationError):
        Reservation(reservation_id="a", **{field: -1})


def test_decimal_cost_requires_pricing_and_never_refunds():
    r = Reservation(reservation_id="cost", model_calls=1, cost=Decimal("0.4"))
    with pytest.raises(ValidationError):
        BudgetLedger(policy=InvestigationPolicy()).reserve(r)
    ledger = BudgetLedger(
        policy=InvestigationPolicy(cost_ceiling=Decimal("0.5"), pricing_version="fake@1")
    ).reserve(r)
    ledger = ledger.consume(
        Consumption(reservation_id="cost", status=AttemptStatus.FAILED, actual_cost=Decimal("0.2"))
    )
    assert ledger.reservations[0].cost == Decimal("0.4")
    with pytest.raises(ValidationError):
        ledger.reserve(Reservation(reservation_id="b", model_calls=1, cost=Decimal("0.2")))


def test_assessment_contract_does_not_fake_pending_evidence():
    for status in AssessmentStatus:
        assert RequirementAssessment(requirement_id="req_x", status=status).status == status
    with pytest.raises(ValidationError):
        RequirementAssessment(requirement_id="req_x", supporting_evidence_ids=("E1",))


def test_source_version_mismatch_and_anchor_records_fail_closed():
    _, source, context = admitted_fixture()
    source.versions["router"] = "forged"
    assert admit(source, context).reason == AdmissionReason.INVALID_SOURCE
    _, source, context = admitted_fixture()
    source.anchor_results.append({"forged": True})
    with pytest.warns(UserWarning, match="serializer warnings"):
        assert admit(source, context).outcome == AdmissionOutcome.FAIL


def test_policy_accounts_for_separate_template_requirements():
    _, source, context = admitted_fixture(policy=InvestigationPolicy(max_initial_requirements=3))
    assert admit(source, context).reason == AdmissionReason.POLICY_BOUNDS


def test_source_dated_anchor_is_checked_without_repeating_reads():
    h, source, context = admitted_fixture("What changed and why since the last restructuring?")
    before = (len(h.executor.calls), len(h.reads))
    admission = admit(source, context)
    assert admission.outcome == AdmissionOutcome.ADMITTED
    assert (len(h.executor.calls), len(h.reads)) == before
    assert len(admission.state.budget.reservations) == 1
    state = admission.state
    restored = type(state).model_validate_json(state.model_dump_json())
    assert restored == state
    assert restored.source.anchor_results[0].items[-1].event_date is not None
    temporal = source.investigation_plan.temporal.model_copy(update={"date_from": date(1999, 1, 1)})
    source.investigation_plan = source.investigation_plan.model_copy(update={"temporal": temporal})
    source.understanding = source.understanding.model_copy(update={"temporal": temporal})
    assert admit(source, context).reason == AdmissionReason.TEMPORAL_MISMATCH
