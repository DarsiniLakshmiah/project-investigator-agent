"""Phase 9 structured tools: filtering, ordering, provenance classes, derived values,
empty / not-found / ambiguous behaviour (synthetic rows, in-memory reader)."""

from datetime import date
from decimal import Decimal
from typing import get_args

from tests.support.tool_fixtures import IPF, OTHER, PFORR, context, project_360, tables

from worldbank_copilot.intelligence.contracts import GOLD_CONTRACTS, RISK_RECORD_TYPES
from worldbank_copilot.lakehouse.contracts import Role
from worldbank_copilot.tools import project, registry, risks
from worldbank_copilot.tools.models import MechanicalCode, ProvenanceClass, ToolStatus
from worldbank_copilot.tools.reader import InMemoryReader

EXECUTOR = registry.default_executor()
F, D, S, U = (
    ProvenanceClass.FACT,
    ProvenanceClass.DOCUMENTED_FINDING,
    ProvenanceClass.SYSTEM_DERIVED_SIGNAL,
    ProvenanceClass.UNKNOWN,
)


def call(name, data=None, scope=IPF, **args):
    ctx = context(InMemoryReader(data or tables()))
    return EXECUTOR.run(name, {"project_id": scope, **args}, ctx, scope_project_id=scope)


def facts(res):
    return {f.name: f for f in res.items[0].facts}


# -- T1 overview ----------------------------------------------------------------------------


def test_overview_classes_follow_the_gold_contract_documentation():
    contract = GOLD_CONTRACTS["project_360"]()
    data_columns = {
        c.name
        for c in contract.columns
        if c.role.value == Role.DATA.value and c.name != "project_id"
    }
    mapped = (
        set(project.DIRECT) | set(project.COUNTS) | set(project.DERIVED) | set(project.METADATA)
    )
    assert data_columns == mapped  # every Gold column is classified exactly once
    produced = facts(call("get_project_overview"))
    for column in contract.columns:
        if column.name in project.METADATA:  # returned as descriptive attributes, not claims
            continue
        doc = column.description or ""
        for cls in ("FACT", "DOCUMENTED_FINDING", "SYSTEM_DERIVED_SIGNAL"):
            if doc.startswith(cls):
                declared = project.DIRECT.get(column.name) or project.COUNTS.get(column.name)
                actual = declared[1] if declared else produced[column.name].provenance_class
                assert actual.value == cls, column.name


def test_overview_values_and_derived_classes():
    res = call("get_project_overview")
    assert res.status == ToolStatus.OK
    f = facts(res)
    assert f["current_closing_date"].provenance_class == F
    assert f["original_closing_date"].provenance_class == D
    assert (f["days_extended"].provenance_class, f["days_extended"].value) == (D, 1765)
    assert f["days_extended"].derivation.inputs == ("current_closing_date", "original_closing_date")
    pct = f["disbursement_pct_of_net_principal"]
    assert (pct.provenance_class, pct.value, pct.derivation.operation) == (
        F,
        Decimal("50"),
        "percentage",
    )
    assert f["original_principal_usd"].derivation.operation == "sum_over_loans"
    assert f["current_high_signal_count"].provenance_class == S
    assert f["number_of_isrs"].provenance_class == D and f["number_of_isrs"].derivation
    assert (f["do_rating_change"].provenance_class, f["do_rating_change"].value) == (D, "DOWNGRADE")
    assert f["project_development_objective"].source.table == "silver.projects"
    assert any("valuation caveats" in c for c in res.caveats)


def test_overview_unknown_input_withholds_the_stored_derived_value():
    data = tables()
    data["gold.project_360"][0] = project_360(
        IPF, original_closing_date=None, days_extended=1765, do_rating_change="NOT_COMPARABLE"
    )
    f = facts(call("get_project_overview", data))
    assert f["original_closing_date"].provenance_class == U
    assert f["days_extended"].provenance_class == U and f["days_extended"].value is None
    assert "withheld" in f["days_extended"].unknown_reason
    assert f["do_rating_change"].provenance_class == U


def test_overview_requires_exactly_one_row():
    data = tables()
    data["gold.project_360"] = [r for r in data["gold.project_360"] if r["project_id"] != IPF]
    assert call("get_project_overview", data).status == ToolStatus.DATA_INTEGRITY_ERROR
    data["gold.project_360"] += [project_360(IPF), project_360(IPF, record_id="dup")]
    assert call("get_project_overview", data).status == ToolStatus.DATA_INTEGRITY_ERROR


def test_pforr_overview_states_which_rules_do_not_apply():
    res = call("get_project_overview", scope=PFORR)
    assert any("FINANCE_DISBURSEMENT_LAG" in c and "not evaluated" in c for c in res.caveats)
    assert not any("FINANCE_DISBURSEMENT_LAG" in c for c in call("get_project_overview").caveats)


# -- T2 timeline ------------------------------------------------------------------------------


def test_timeline_is_in_canonical_order_with_row_provenance():
    res = call("get_project_timeline")
    assert [e.event_sequence for e in res.items] == [1, 2, 3, 4]
    assert res.items[0].provenance_class == F and res.items[1].provenance_class == D
    assert res.items[1].source.table == "gold.project_timeline"
    assert any("dated later than the next ISR" in c for c in res.caveats)


def test_timeline_date_filter_uses_source_dates_and_candidates_only_on_request():
    res = call("get_project_timeline", date_from=date(2016, 6, 1), date_to=date(2019, 1, 1))
    assert [e.event_sequence for e in res.items] == [2, 4]
    assert any("without a source-stated date were excluded" in c for c in res.caveats)
    res = call(
        "get_project_timeline",
        date_from=date(2016, 6, 1),
        date_to=date(2019, 1, 1),
        include_candidate_dates=True,
    )
    used = {e.event_sequence: e.date_used_for_filter for e in res.items}
    assert used == {2: "EVENT_DATE", 3: "CANDIDATE_DATE", 4: "EVENT_DATE"}
    assert res.items[1].event_date is None  # the candidate is never shown as the event date
    assert any("DERIVED candidate date" in c for c in res.caveats)


def test_timeline_filters_and_empty_and_invalid():
    res = call("get_project_timeline", event_types=["ISR_REPORT"], isr_sequence_from=2)
    assert [e.isr_sequence for e in res.items] == [2]
    res = call("get_project_timeline", event_types=["CANCELLATION"])
    assert res.status == ToolStatus.EMPTY and res.mechanical[0].code == MechanicalCode.NO_RECORDS
    assert (
        call("get_project_timeline", event_types=["PREDICTION"]).status
        == ToolStatus.INVALID_ARGUMENT
    )
    bad = call("get_project_timeline", date_from=date(2020, 1, 1), date_to=date(2019, 1, 1))
    assert bad.status == ToolStatus.INVALID_ARGUMENT
    limited = call("get_project_timeline", limit=2)
    assert len(limited.items) == 2 and limited.mechanical[0].code == MechanicalCode.RESULT_TRUNCATED


# -- T3 ratings -------------------------------------------------------------------------------


def ratings_of(res, seq):
    item = next(i for i in res.items if i.isr_sequence == seq)
    return {o.rating_type: o for o in item.ratings}


def test_rating_history_changes_by_sequence_with_ordinal_ranks():
    res = call("get_rating_history")
    assert [i.isr_sequence for i in res.items] == [1, 2, 3]
    first, second, third = (ratings_of(res, s) for s in (1, 2, 3))
    assert first["PDO"].change_from_previous is None
    change = second["PDO"].change_from_previous
    assert (change.value, change.provenance_class) == ("DOWNGRADE", D)
    assert change.derivation.operation == "rank_comparison"
    assert second["OVERALL_RISK"].change_from_previous.value == "INCREASED"
    assert third["OVERALL_RISK"].change_from_previous.value == "UNCHANGED"


def test_missing_rating_is_unknown_and_its_changes_are_unknown():
    res = call("get_rating_history")
    second, third = ratings_of(res, 2), ratings_of(res, 3)
    assert (
        second["IP"].rating.provenance_class == U
        and "NOT_FOUND" in second["IP"].rating.unknown_reason
    )
    assert second["IP"].change_from_previous.provenance_class == U
    assert third["IP"].change_from_previous.provenance_class == U
    assert any(m.code == MechanicalCode.REQUIRED_FIELD_NULL for m in res.mechanical)


def test_rating_range_compares_with_the_isr_before_the_range():
    res = call("get_rating_history", isr_sequence_from=3, isr_sequence_to=3, rating_types=["PDO"])
    (item,) = res.items
    assert (
        item.ratings[0].compared_with_isr == 2
        and item.ratings[0].change_from_previous.value == "DOWNGRADE"
    )
    assert [i.isr_sequence for i in call("get_rating_history", latest_n=2).items] == [2, 3]


def test_requested_isr_absent_is_not_found_and_bad_arguments_are_invalid():
    res = call("get_rating_history", isr_sequence_from=7, isr_sequence_to=7)
    assert (
        res.status == ToolStatus.NOT_FOUND
        and res.mechanical[0].code == MechanicalCode.ISR_NOT_FOUND
    )
    assert (
        call("get_rating_history", latest_n=2, isr_sequence_from=1).status
        == ToolStatus.INVALID_ARGUMENT
    )
    assert call("get_rating_history", rating_types=["SORT"]).status == ToolStatus.INVALID_ARGUMENT


# -- T4 finance -------------------------------------------------------------------------------


def test_finance_returns_statement_and_isr_sources_side_by_side_unreconciled():
    res = call("get_financial_status", as_of_isr="latest", include_events=True)
    (item,) = res.items
    assert [loan.loan_number for loan in item.loans] == ["IBRD86010", "IBRD93240"]
    assert all(loan.provenance_class == F for loan in item.loans)
    assert {line.isr_sequence for line in item.isr_reported} == {3}
    assert all(
        line.provenance_class == D and line.unit.startswith("US$ millions")
        for line in item.isr_reported
    )
    assert [e.event_type for e in item.events] == ["ADDITIONAL_FINANCING"]
    assert item.events[0].additional_financing_currency == "JPY"  # never converted
    assert any("not reconciled" in n for n in res.notices)
    pct = next(f for f in item.summary if f.name == "disbursement_pct_of_net_principal")
    assert pct.provenance_class == F and pct.derivation is not None


def test_loan_numbers_are_normalised_and_scoped():
    res = call("get_financial_status", loan_number="ibrd-9324-0", as_of_isr=2)
    (item,) = res.items
    assert [loan.loan_number for loan in item.loans] == ["IBRD93240"]
    assert [line.loan_number for line in item.isr_reported] == ["IBRD93240"]
    foreign = call("get_financial_status", loan_number="IBRD-94960")
    assert foreign.status == ToolStatus.SCOPE_REFUSED
    assert foreign.mechanical[0].code == MechanicalCode.PROJECT_OUT_OF_SCOPE
    unknown = call("get_financial_status", loan_number="IBRD11111")
    assert unknown.status == ToolStatus.NOT_FOUND
    assert [c.value for c in unknown.argument_candidates] == ["IBRD86010", "IBRD93240"]


def test_requested_isr_without_loan_lines_is_not_found():
    res = call("get_financial_status", loan_number="IBRD93240", as_of_isr=1)
    assert (
        res.status == ToolStatus.NOT_FOUND
        and res.mechanical[0].code == MechanicalCode.ISR_NOT_FOUND
    )
    assert "2-3" in res.mechanical[0].detail


# -- T5 results ------------------------------------------------------------------------------


def test_results_latest_per_indicator_with_derived_progress():
    res = call("get_results_progress", indicator="ind-a")
    (obs,) = res.items
    assert obs.isr_sequence == 2 and obs.provenance_class == D
    assert (obs.progress_percentage.value, obs.progress_percentage.provenance_class) == (
        Decimal("50"),
        D,
    )
    assert obs.change_from_previous.value == Decimal("10")
    assert any("canonical_id" in n for n in res.notices)


def test_indicator_resolution_never_guesses():
    exact = call("get_results_progress", indicator="direct project BENEFICIARIES")
    assert exact.status == ToolStatus.OK and exact.items[0].indicator_id == "ind-a"
    ambiguous = call("get_results_progress", indicator="female beneficiaries number")
    assert ambiguous.status == ToolStatus.OK  # all words only in ind-c
    ambiguous = call("get_results_progress", indicator="female")
    assert ambiguous.status == ToolStatus.AMBIGUOUS_ARGUMENT and ambiguous.items == []
    assert [c.value for c in ambiguous.argument_candidates] == ["ind-b", "ind-c"]
    assert "pending alias" in ambiguous.mechanical[0].detail
    missing = call("get_results_progress", indicator="metro rail stations")
    assert missing.status == ToolStatus.NOT_FOUND
    assert missing.mechanical[0].code == MechanicalCode.INDICATOR_NOT_FOUND


def test_results_not_evaluable_is_unknown_with_the_reason():
    res = call("get_results_progress", indicator_type="DLI")
    (obs,) = res.items
    assert obs.progress_percentage.provenance_class == U
    assert "DLI_LAYOUT_NOT_EVALUATED" in obs.progress_percentage.unknown_reason
    assert res.mechanical[0].code == MechanicalCode.NOT_EVALUABLE


def test_results_isr_history_and_absent_isr():
    assert [
        o.isr_sequence for o in call("get_results_progress", indicator="ind-a", history=True).items
    ] == [1, 2]
    res = call("get_results_progress", indicator="ind-b", isr_sequence=1)
    assert (
        res.status == ToolStatus.NOT_FOUND
        and res.mechanical[0].code == MechanicalCode.ISR_NOT_FOUND
    )
    assert (
        call("get_results_progress", history=True, isr_sequence=1).status
        == ToolStatus.INVALID_ARGUMENT
    )
    assert call("get_results_progress", scope=OTHER, data=tables()).status == ToolStatus.EMPTY


# -- T6 risks --------------------------------------------------------------------------------


def test_risk_register_order_framing_and_resolution_notice():
    res = call("get_risk_register")
    assert [r.risk_or_finding_id for r in res.items] == ["r3", "r2", "r4", "r1"]
    assert {r.resolution_status for r in res.items} == {"NOT_STATED"}
    assert any("does not infer" in n for n in res.notices)
    assert all(r.provenance_class == D for r in res.items)


def test_risk_filters_and_unknown_category_candidates():
    assert [
        r.risk_or_finding_id for r in call("get_risk_register", category="fiduciary").items
    ] == ["r2"]
    res = call("get_risk_register", category="Weather")
    assert res.status == ToolStatus.EMPTY
    assert [c.value for c in res.argument_candidates] == [
        "Environmental",
        "Fiduciary",
        "Other",
        "Political",
    ]
    assert call("get_risk_register", record_type="PREDICTED").status == ToolStatus.INVALID_ARGUMENT
    literal = get_args(risks.RiskArgs.model_fields["record_type"].annotation)[0]
    assert get_args(literal) == RISK_RECORD_TYPES


# -- T7 signals ------------------------------------------------------------------------------


def test_signals_are_system_derived_ordered_and_carry_the_notice():
    res = call("get_attention_signals", status="ALL")
    assert [s.signal_id for s in res.items] == ["s2", "s3", "s1"]
    assert {s.provenance_class for s in res.items} == {S}
    assert any("not a prediction" in n for n in res.notices)
    current = call("get_attention_signals", min_severity="WATCH")
    assert [s.signal_id for s in current.items] == ["s2"]


def test_no_signal_is_not_an_all_clear():
    res = call("get_attention_signals", scope=PFORR)
    assert res.status == ToolStatus.EMPTY
    assert any("not an assessment" in n for n in res.notices)
    assert any("FINANCE_DISBURSEMENT_LAG" in c for c in res.caveats)
    assert any("deferred rule candidate" in c for c in res.caveats)
    assert (
        call("get_attention_signals", categories=["WEATHER"]).status == ToolStatus.INVALID_ARGUMENT
    )


def test_a_signal_row_with_another_class_is_a_data_integrity_error():
    data = tables()
    data["gold.attention_signals"][0]["provenance_class"] = "FACT"
    assert call("get_attention_signals", data).status == ToolStatus.DATA_INTEGRITY_ERROR
