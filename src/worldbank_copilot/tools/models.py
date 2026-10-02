"""Tool envelope, provenance classes, facts and derivation rules (Phase 9).

Every value a tool returns says what kind of claim it is (CLAUDE.md §10):

* FACT                  - structured source data (loan statement, project workbook);
* DOCUMENTED_FINDING    - stated by a source document (Phase 5 extraction, retrieval);
* SYSTEM_DERIVED_SIGNAL - output of a documented deterministic attention rule;
* AI_INTERPRETATION     - never produced by a tool (rejected by ``ToolResult``);
* UNKNOWN               - the value is not available; always carries a reason.

Derived arithmetic (Phase 9 Decision 4) never gets a class of its own: it keeps the most
conservative class of its inputs and carries ``Derivation`` metadata.

* all inputs FACT                     -> FACT;
* any input DOCUMENTED_FINDING        -> DOCUMENTED_FINDING;
* any input SYSTEM_DERIVED_SIGNAL     -> SYSTEM_DERIVED_SIGNAL (deriving from a signal);
* any input UNKNOWN / missing value   -> UNKNOWN (no known value is ever produced);
* AI_INTERPRETATION input             -> rejected.

Ordinary arithmetic over facts/findings is therefore never promoted to a signal.

Sufficiency: tools assess only MECHANICAL sufficiency (no records, unknown project, ISR
absent, required field NULL, not covered, zero chunks). Whether evidence semantically
supports a claim is never assessed here (``semantic_sufficiency = NOT_ASSESSED``; Phase 10).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from enum import StrEnum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, model_validator


class ProvenanceClass(StrEnum):
    FACT = "FACT"
    DOCUMENTED_FINDING = "DOCUMENTED_FINDING"
    SYSTEM_DERIVED_SIGNAL = "SYSTEM_DERIVED_SIGNAL"
    AI_INTERPRETATION = "AI_INTERPRETATION"
    UNKNOWN = "UNKNOWN"


class ToolStatus(StrEnum):
    OK = "OK"
    EMPTY = "EMPTY"  # the query ran; no record matched (not "nothing happened")
    NOT_FOUND = "NOT_FOUND"  # a specifically requested item (ISR, loan, indicator) is absent
    NOT_COVERED = "NOT_COVERED"  # the source does not cover this project/question
    AMBIGUOUS_ARGUMENT = "AMBIGUOUS_ARGUMENT"  # several candidates; never guessed
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    SCOPE_REFUSED = "SCOPE_REFUSED"  # unknown / unauthorised / foreign project
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"  # mechanical: e.g. zero chunks
    DATA_INTEGRITY_ERROR = "DATA_INTEGRITY_ERROR"  # an expected authoritative record is wrong
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


class MechanicalCode(StrEnum):
    NO_RECORDS = "NO_RECORDS"
    PROJECT_INVALID = "PROJECT_INVALID"
    PROJECT_OUT_OF_SCOPE = "PROJECT_OUT_OF_SCOPE"
    ISR_NOT_FOUND = "ISR_NOT_FOUND"
    LOAN_NOT_FOUND = "LOAN_NOT_FOUND"
    INDICATOR_NOT_FOUND = "INDICATOR_NOT_FOUND"
    INDICATOR_AMBIGUOUS = "INDICATOR_AMBIGUOUS"
    REQUIRED_FIELD_NULL = "REQUIRED_FIELD_NULL"
    NOT_EVALUABLE = "NOT_EVALUABLE"
    NOT_COVERED = "NOT_COVERED"
    NO_CHUNKS = "NO_CHUNKS"
    RESULT_TRUNCATED = "RESULT_TRUNCATED"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MechanicalFinding(_Frozen):
    code: MechanicalCode
    detail: str
    field: str | None = None


class SourceRef(_Frozen):
    """Where a value came from: the table read and the evidence behind that row."""

    table: str  # governed table the tool read, e.g. gold.risk_register
    record_id: str | None = None  # row read in that table
    source_table: str | None = None  # primary upstream record (Gold provenance)
    source_record_id: str | None = None
    supporting_record_ids: tuple[str, ...] = ()
    document_id: str | None = None
    page_number: int | None = None
    section: str | None = None
    table_id: str | None = None
    extraction_method: str | None = None
    extraction_status: str | None = None


class Derivation(_Frozen):
    operation: str  # e.g. "difference_in_days", "rank_comparison", "count"
    inputs: tuple[str, ...]  # names of the input facts
    note: str | None = None


class Fact(_Frozen):
    name: str
    value: Any = None
    unit: str | None = None
    provenance_class: ProvenanceClass
    source: SourceRef | None = None
    derivation: Derivation | None = None
    unknown_reason: str | None = None

    @model_validator(mode="after")
    def _known_or_unknown(self) -> Fact:
        if self.provenance_class == ProvenanceClass.UNKNOWN:
            if self.value is not None:
                raise ValueError(f"{self.name}: an UNKNOWN fact cannot carry a value")
            if not self.unknown_reason:
                raise ValueError(f"{self.name}: an UNKNOWN fact needs a reason")
        elif self.value is None:
            raise ValueError(f"{self.name}: a missing value must be UNKNOWN with a reason")
        return self


def fact(
    name: str,
    value: Any,
    provenance_class: ProvenanceClass,
    source: SourceRef | None = None,
    *,
    unit: str | None = None,
    unknown_reason: str = "not stated in the source",
) -> Fact:
    """A sourced value; a NULL becomes UNKNOWN with a reason (never silently dropped)."""
    if value is None or provenance_class == ProvenanceClass.UNKNOWN:
        return Fact(
            name=name,
            unit=unit,
            provenance_class=ProvenanceClass.UNKNOWN,
            source=source,
            unknown_reason=unknown_reason,
        )
    return Fact(name=name, value=value, unit=unit, provenance_class=provenance_class, source=source)


def derived_class(inputs: Sequence[Fact]) -> ProvenanceClass:
    """Most conservative class of the inputs (Decision 4)."""
    classes = {f.provenance_class for f in inputs}
    if not inputs:
        raise ValueError("a derivation needs at least one input")
    if ProvenanceClass.AI_INTERPRETATION in classes:
        raise ValueError("tools never derive values from AI interpretations")
    if ProvenanceClass.UNKNOWN in classes:
        return ProvenanceClass.UNKNOWN
    if ProvenanceClass.SYSTEM_DERIVED_SIGNAL in classes:
        return ProvenanceClass.SYSTEM_DERIVED_SIGNAL
    if ProvenanceClass.DOCUMENTED_FINDING in classes:
        return ProvenanceClass.DOCUMENTED_FINDING
    return ProvenanceClass.FACT


def derive(
    name: str,
    operation: str,
    inputs: Sequence[Fact],
    *,
    value: Any = None,
    compute: Callable[..., Any] | None = None,
    unit: str | None = None,
    source: SourceRef | None = None,
    note: str | None = None,
    unknown_reason: str | None = None,
) -> Fact:
    """A derived value with the inputs' conservative class and derivation metadata.

    Either ``compute(*input_values)`` produces the value, or ``value`` is a value the
    source already derived (e.g. a Gold column), classified by the same rule. An UNKNOWN
    input always yields UNKNOWN, even if the source holds a value.
    """
    derivation = Derivation(operation=operation, inputs=tuple(f.name for f in inputs), note=note)
    cls = derived_class(inputs)
    if cls == ProvenanceClass.UNKNOWN:
        missing = [f.name for f in inputs if f.provenance_class == ProvenanceClass.UNKNOWN]
        reason = f"input {', '.join(missing)} is UNKNOWN"
        if value is not None:
            reason += "; the stored derived value is withheld (inconsistent with its inputs)"
        return Fact(
            name=name,
            unit=unit,
            provenance_class=ProvenanceClass.UNKNOWN,
            source=source,
            derivation=derivation,
            unknown_reason=reason,
        )
    if compute is not None:
        value = compute(*(f.value for f in inputs))
    if value is None:
        return Fact(
            name=name,
            unit=unit,
            provenance_class=ProvenanceClass.UNKNOWN,
            source=source,
            derivation=derivation,
            unknown_reason=unknown_reason or "the derivation is not defined for these inputs",
        )
    return Fact(
        name=name,
        value=value,
        unit=unit,
        provenance_class=cls,
        source=source,
        derivation=derivation,
    )


class ArgumentCandidate(_Frozen):
    """A possible value for an ambiguous argument (returned, never chosen)."""

    value: str
    label: str
    note: str | None = None


T = TypeVar("T", bound=BaseModel)


def iter_provenance_classes(obj: Any) -> Iterator[ProvenanceClass]:
    if isinstance(obj, ProvenanceClass):
        yield obj
    elif isinstance(obj, BaseModel):
        for name in type(obj).model_fields:
            yield from iter_provenance_classes(getattr(obj, name))
    elif isinstance(obj, list | tuple | set):
        for item in obj:
            yield from iter_provenance_classes(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from iter_provenance_classes(item)


class ToolResult(BaseModel, Generic[T]):
    """Envelope of every tool call (read-only, typed, scope-checked)."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    tool_version: str
    request_id: str
    project_id: str | None
    status: ToolStatus
    items: list[SerializeAsAny[T]] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    argument_candidates: list[ArgumentCandidate] = Field(default_factory=list)
    data_snapshot: dict[str, int | None] = Field(default_factory=dict)
    mechanical: list[MechanicalFinding] = Field(default_factory=list)
    semantic_sufficiency: Literal["NOT_ASSESSED"] = "NOT_ASSESSED"
    caveats: list[str] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None

    @model_validator(mode="after")
    def _no_ai_interpretation(self) -> ToolResult:
        if ProvenanceClass.AI_INTERPRETATION in set(iter_provenance_classes(self.items)):
            raise ValueError("tools never return AI_INTERPRETATION")
        if self.items and self.status != ToolStatus.OK:
            raise ValueError(f"status {self.status} cannot carry items")
        if self.status == ToolStatus.OK and not self.items:
            raise ValueError("status OK needs at least one item (use EMPTY / NOT_FOUND)")
        return self


class ToolOutcome(BaseModel):
    """What a tool function returns; the executor wraps it into a ``ToolResult``."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    status: ToolStatus
    items: list[Any] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    argument_candidates: list[ArgumentCandidate] = Field(default_factory=list)
    mechanical: list[MechanicalFinding] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)
