"""Phase 9 tool envelope, provenance classes and derivation rules (Decision 4)."""

from itertools import product

import pytest
from pydantic import BaseModel

from worldbank_copilot.intelligence.contracts import PROVENANCE_CLASSES
from worldbank_copilot.tools.models import (
    Fact,
    ProvenanceClass,
    ToolResult,
    ToolStatus,
    derive,
    derived_class,
    fact,
)

F, D, S, A, U = (
    ProvenanceClass.FACT,
    ProvenanceClass.DOCUMENTED_FINDING,
    ProvenanceClass.SYSTEM_DERIVED_SIGNAL,
    ProvenanceClass.AI_INTERPRETATION,
    ProvenanceClass.UNKNOWN,
)


def known(name, cls, value=1):
    return fact(name, value, cls)


def test_the_five_classes_are_exactly_the_gold_vocabulary():
    assert tuple(c.value for c in ProvenanceClass) == PROVENANCE_CLASSES


def test_a_missing_value_is_unknown_with_a_reason_never_dropped():
    item = fact("closing", None, F)
    assert item.provenance_class == U and item.value is None and item.unknown_reason
    with pytest.raises(ValueError, match="UNKNOWN"):
        Fact(name="x", value=None, provenance_class=F)
    with pytest.raises(ValueError, match="cannot carry a value"):
        Fact(name="x", value=1, provenance_class=U, unknown_reason="r")
    with pytest.raises(ValueError, match="needs a reason"):
        Fact(name="x", provenance_class=U)


def test_fact_only_arithmetic_stays_fact_with_derivation():
    out = derive(
        "pct",
        "percentage",
        [known("a", F, 45), known("b", F, 90)],
        compute=lambda a, b: a / b * 100,
    )
    assert (out.provenance_class, out.value) == (F, 50)
    assert out.derivation.operation == "percentage" and out.derivation.inputs == ("a", "b")


def test_any_documented_finding_input_keeps_documented_finding():
    out = derive("days", "difference", [known("current", F), known("original", D)], value=10)
    assert out.provenance_class == D and out.derivation is not None


def test_deriving_from_a_signal_stays_a_signal():
    assert derive("n", "count", [known("sig", S)], value=3).provenance_class == S
    assert derive("n", "x", [known("a", F), known("sig", S)], value=1).provenance_class == S
    assert derive("n", "x", [known("a", D), known("sig", S)], value=1).provenance_class == S


@pytest.mark.parametrize("classes", list(product([F, D], repeat=3)))
def test_ordinary_arithmetic_is_never_promoted_to_a_signal(classes):
    inputs = [known(f"i{n}", c) for n, c in enumerate(classes)]
    out = derive("x", "sum", inputs, compute=lambda *v: sum(v))
    assert out.provenance_class in (F, D) and out.provenance_class != S
    assert out.provenance_class == (D if D in classes else F)


def test_unknown_input_never_produces_a_known_value():
    out = derive("days", "difference", [known("current", F), fact("original", None, D)], value=99)
    assert out.provenance_class == U and out.value is None
    assert "original" in out.unknown_reason and "withheld" in out.unknown_reason
    computed = derive("x", "sum", [fact("a", None, F), known("b", F)], compute=lambda a, b: 1)
    assert computed.provenance_class == U and computed.value is None


def test_an_undefined_derivation_is_unknown():
    out = derive(
        "x",
        "rank",
        [known("a", D), known("b", D)],
        compute=lambda a, b: None,
        unknown_reason="off scale",
    )
    assert out.provenance_class == U and out.unknown_reason == "off scale"


def test_ai_interpretation_cannot_be_an_input_and_needs_inputs():
    with pytest.raises(ValueError, match="AI interpretations"):
        derived_class([Fact(name="ai", value="x", provenance_class=A)])
    with pytest.raises(ValueError, match="at least one input"):
        derived_class([])


class Item(BaseModel):
    facts: list[Fact]


def result(status, items):
    return ToolResult(
        tool="t", tool_version="1", request_id="r", project_id="P130544", status=status, items=items
    )


def test_tool_results_never_carry_ai_interpretation():
    ai = Fact(name="ai", value="x", provenance_class=A)
    with pytest.raises(ValueError, match="AI_INTERPRETATION"):
        result(ToolStatus.OK, [Item(facts=[ai])])


def test_only_ok_carries_items_and_ok_needs_items():
    item = Item(facts=[known("a", F)])
    for status in set(ToolStatus) - {ToolStatus.OK}:
        with pytest.raises(ValueError, match="cannot carry items"):
            result(status, [item])
    with pytest.raises(ValueError, match="at least one item"):
        result(ToolStatus.OK, [])
    ok = result(ToolStatus.OK, [item])
    assert ok.semantic_sufficiency == "NOT_ASSESSED"
    assert ok.model_dump()["items"][0]["facts"][0]["name"] == "a"  # subclass fields serialised
