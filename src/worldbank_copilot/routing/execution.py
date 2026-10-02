"""Runtime execution mapping: RouteResult -> allowed execution mode (Phase 9F).

This is the 9D Candidate A decision made real at runtime: deterministic routing plus
targeted clarification, with the semantic fallback disabled. It is NOT a second router:
it never reclassifies an intent, reads a document, calls a model, or changes project or
temporal resolution. It only translates the recorded router output into what may execute.

The router output stays exactly as recorded (9B.2 is frozen and evaluated as is):

    router output                       -> execution decision
    STRUCTURED                          -> EXECUTE_STRUCTURED
    DOCUMENT                            -> EXECUTE_DOCUMENT
    INVESTIGATION                       -> PLAN_INVESTIGATION (a plan; nothing executes)
    CLARIFY <reason>                    -> CLARIFY <reason>
    REFUSE <reason>                     -> REFUSE <reason>
    SEMANTIC_CLASSIFICATION_REQUIRED    -> CLARIFY INTENT_NOT_RESOLVED   (Candidate A)

Anything outside this contract (an invoked semantic fallback, an unknown clarification
reason, a CLARIFY / REFUSE that executed a call, an executed plan) raises
``ExecutionContractError``: fail closed, never a guessed execution.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from worldbank_copilot.common.exceptions import CopilotError
from worldbank_copilot.routing.models import Route, RouteResult

EXECUTION_POLICY = "9D_CANDIDATE_A"
INTENT_NOT_RESOLVED = "INTENT_NOT_RESOLVED"


class ExecutionContractError(CopilotError):
    """A routing result falls outside the Phase 10 execution contract."""


class ExecutionMode(StrEnum):
    EXECUTE_STRUCTURED = "EXECUTE_STRUCTURED"
    EXECUTE_DOCUMENT = "EXECUTE_DOCUMENT"
    PLAN_INVESTIGATION = "PLAN_INVESTIGATION"
    CLARIFY = "CLARIFY"
    REFUSE = "REFUSE"


ROUTE_MODES: dict[Route, ExecutionMode] = {
    Route.STRUCTURED: ExecutionMode.EXECUTE_STRUCTURED,
    Route.DOCUMENT: ExecutionMode.EXECUTE_DOCUMENT,
    Route.INVESTIGATION: ExecutionMode.PLAN_INVESTIGATION,
    Route.CLARIFY: ExecutionMode.CLARIFY,
    Route.REFUSE: ExecutionMode.REFUSE,
    Route.SEMANTIC_CLASSIFICATION_REQUIRED: ExecutionMode.CLARIFY,  # Candidate A
}

# Every reason a request is clarified before anything executes.
CLARIFICATION_REASON_CODES = frozenset(
    {
        "PROJECT_REQUIRED",
        "AMBIGUOUS_PROJECT_REFERENCE",
        "AMBIGUOUS_TIME",
        "TIME_REQUIRED",
        "TEMPORAL_NOT_SUPPORTED",
        INTENT_NOT_RESOLVED,
    }
)


class ExecutionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: ExecutionMode
    reason_code: str
    detail: str
    router_route: Route  # the recorded router output, never rewritten
    router_reason_code: str
    translated: bool  # True only for SEMANTIC_CLASSIFICATION_REQUIRED -> CLARIFY
    policy: str = EXECUTION_POLICY
    clarification_options: tuple[str, ...] = ()
    suggestion: str | None = None


def execution_decision(result: RouteResult) -> ExecutionDecision:
    """The single allowed execution mode for a routing result."""
    if result.semantic is not None:
        raise ExecutionContractError(
            "the Phase 9D semantic fallback is disabled; a result that invoked it is outside "
            "the Phase 10 contract"
        )
    decision = result.decision
    mode = ROUTE_MODES[decision.route]
    translated = decision.route == Route.SEMANTIC_CLASSIFICATION_REQUIRED
    reason = INTENT_NOT_RESOLVED if translated else decision.reason_code
    if mode == ExecutionMode.CLARIFY and reason not in CLARIFICATION_REASON_CODES:
        raise ExecutionContractError(f"clarification reason {reason!r} is not in the contract")
    if mode in (ExecutionMode.CLARIFY, ExecutionMode.REFUSE) and (
        result.executed_tools or result.tool_results or result.retrieval_executed
    ):
        raise ExecutionContractError(f"{mode} {reason}: a call executed before {mode}")
    if mode == ExecutionMode.PLAN_INVESTIGATION and (
        result.investigation_plan is None or result.investigation_plan.executed
    ):
        raise ExecutionContractError("an INVESTIGATION must carry an unexecuted plan")
    return ExecutionDecision(
        mode=mode,
        reason_code=reason,
        detail=decision.detail,
        router_route=decision.route,
        router_reason_code=decision.reason_code,
        translated=translated,
        clarification_options=decision.clarification_options,
        suggestion=decision.suggestion,
    )
