"""Versioned, configurable policy hypotheses and conservative reservation accounting.

Reservations charge their full allowance immediately. Completion records known actual
usage (including failed attempts), without refunding the allowance. Deadline accounting
is admission accounting, not cancellation of external work. Live execution requires
explicit pricing and a Decimal cost ceiling; neither is selected here.
"""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Contract(BaseModel):
    """Detached nested values; reading a nested value returns a defensive copy.

    Frozen outer objects alone would not protect reused Phase 9 dict/list fields.
    Explicit transition methods construct new validated records. model_copy(update=)
    is disabled because Pydantic does not validate those updates. This is an application
    contract, not a defence against Python internals/model_construct tampering.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    @model_validator(mode="before")
    @classmethod
    def detach(cls, value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="python")
        return deepcopy(value)

    def __getattribute__(self, name: str) -> Any:
        value = super().__getattribute__(name)
        if not name.startswith("_") and name in type(self).model_fields:
            if isinstance(value, (BaseModel, dict, list, tuple)):
                return deepcopy(value)
        return value

    def model_copy(self, *, update=None, deep=False):
        if update:
            raise TypeError("use a validated contract transition, not model_copy(update=)")
        return super().model_copy(deep=True)


class InvestigationPolicy(Contract):
    policy_id: str = Field(default="investigation_initial@1", min_length=1, max_length=100)
    status: Literal["INITIAL_HYPOTHESIS"] = "INITIAL_HYPOTHESIS"
    max_initial_requirements: int = Field(default=6, ge=1, le=100)
    max_repair_requirements: int = Field(default=1, ge=0, le=100)
    max_initial_operations: int = Field(default=8, ge=1, le=100)
    max_total_operations: int = Field(default=10, ge=1, le=100)
    max_retrieval_operations: int = Field(default=4, ge=0, le=100)
    max_repair_cycles: int = Field(default=1, ge=0, le=10)
    max_claims_per_draft: int = Field(default=20, ge=1, le=100)
    max_model_calls: int = Field(default=6, ge=0, le=100)
    max_investigator_context_tokens: int = Field(default=4000, ge=1, le=1000000)
    max_synthesis_context_tokens: int = Field(default=12000, ge=1, le=1000000)
    max_critic_context_tokens: int = Field(default=12000, ge=1, le=1000000)
    max_planning_output_tokens: int = Field(default=2000, ge=1, le=1000000)
    aggregate_token_budget: int = Field(default=60000, ge=1, le=10000000)
    overall_deadline_seconds: int = Field(default=180, ge=1, le=86400)
    max_ledger_entries: int = Field(default=32, ge=1, le=1000)
    cost_ceiling: Decimal | None = Field(default=None, gt=0)
    pricing_version: str | None = Field(default=None, min_length=1, max_length=100)

    @field_validator("cost_ceiling", mode="before")
    @classmethod
    def exact_cost(cls, value):
        if value is not None and not isinstance(value, Decimal):
            raise ValueError("cost must be Decimal, with explicit pricing")
        return value

    @model_validator(mode="after")
    def coherent(self):
        if self.max_initial_operations > self.max_total_operations:
            raise ValueError("initial operations exceed total")
        if self.max_retrieval_operations > self.max_total_operations:
            raise ValueError("retrieval operations exceed total")
        if (self.cost_ceiling is None) != (self.pricing_version is None):
            raise ValueError("cost ceiling and pricing version must be configured together")
        return self

    def require_live_cost_configuration(self):
        if self.cost_ceiling is None or self.pricing_version is None:
            raise ValueError("live execution requires cost ceiling and pricing configuration")


class AttemptStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class Reservation(Contract):
    reservation_id: str = Field(min_length=1, max_length=100)
    operations: int = Field(default=0, ge=0)
    initial_operations: int = Field(default=0, ge=0)
    retrieval_operations: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    tokens: int = Field(default=0, ge=0)
    repair_cycles: int = Field(default=0, ge=0)
    cost: Decimal | None = Field(default=None, ge=0)

    @field_validator("cost", mode="before")
    @classmethod
    def exact_cost(cls, value):
        return InvestigationPolicy.exact_cost(value)

    @model_validator(mode="after")
    def coherent(self):
        if self.initial_operations > self.operations or self.retrieval_operations > self.operations:
            raise ValueError("operation subcounts exceed operations")
        if not any((self.operations, self.model_calls, self.tokens, self.repair_cycles)):
            raise ValueError("empty reservation")
        return self


class Consumption(Contract):
    reservation_id: str
    status: AttemptStatus
    actual_tokens: int | None = Field(default=None, ge=0)
    actual_cost: Decimal | None = Field(default=None, ge=0)

    @field_validator("actual_cost", mode="before")
    @classmethod
    def exact_cost(cls, value):
        return InvestigationPolicy.exact_cost(value)


class BudgetLedger(Contract):
    policy: InvestigationPolicy
    reservations: tuple[Reservation, ...] = ()
    consumptions: tuple[Consumption, ...] = ()
    elapsed_seconds: Decimal = Field(default=Decimal(0), ge=0)

    @model_validator(mode="after")
    def bounded(self):
        ids = [r.reservation_id for r in self.reservations]
        if len(ids) != len(set(ids)) or len(ids) > self.policy.max_ledger_entries:
            raise ValueError("duplicate or excessive ledger entries")
        checks = {
            "operations": self.policy.max_total_operations,
            "initial_operations": self.policy.max_initial_operations,
            "retrieval_operations": self.policy.max_retrieval_operations,
            "model_calls": self.policy.max_model_calls,
            "tokens": self.policy.aggregate_token_budget,
            "repair_cycles": self.policy.max_repair_cycles,
        }
        for name, ceiling in checks.items():
            if sum(getattr(r, name) for r in self.reservations) > ceiling:
                raise ValueError(f"budget exceeded: {name}")
        if self.elapsed_seconds > self.policy.overall_deadline_seconds:
            raise ValueError("deadline exceeded")
        costs = [r.cost for r in self.reservations if r.cost is not None]
        if costs:
            self.policy.require_live_cost_configuration()
            if sum(costs, Decimal(0)) > self.policy.cost_ceiling:
                raise ValueError("cost budget exceeded")
        consumed = set()
        for c in self.consumptions:
            if c.reservation_id not in ids or c.reservation_id in consumed:
                raise ValueError("unknown or repeated consumption")
            consumed.add(c.reservation_id)
            r = next(r for r in self.reservations if r.reservation_id == c.reservation_id)
            if c.actual_tokens is not None and c.actual_tokens > r.tokens:
                raise ValueError("actual tokens exceed reservation")
            if c.actual_cost is not None and (r.cost is None or c.actual_cost > r.cost):
                raise ValueError("actual cost exceeds configured reservation")
        return self

    def reserve(self, reservation: Reservation):
        return BudgetLedger(
            policy=self.policy,
            reservations=(*self.reservations, reservation),
            consumptions=self.consumptions,
            elapsed_seconds=self.elapsed_seconds,
        )

    def consume(self, consumption: Consumption):
        return BudgetLedger(
            policy=self.policy,
            reservations=self.reservations,
            consumptions=(*self.consumptions, consumption),
            elapsed_seconds=self.elapsed_seconds,
        )

    def advance_elapsed(self, seconds: Decimal):
        if seconds < self.elapsed_seconds:
            raise ValueError("elapsed accounting cannot decrease")
        return BudgetLedger(
            policy=self.policy,
            reservations=self.reservations,
            consumptions=self.consumptions,
            elapsed_seconds=seconds,
        )

    def extends(self, old: BudgetLedger) -> bool:
        return (
            self.policy == old.policy
            and self.elapsed_seconds >= old.elapsed_seconds
            and self.reservations[: len(old.reservations)] == old.reservations
            and self.consumptions[: len(old.consumptions)] == old.consumptions
        )
