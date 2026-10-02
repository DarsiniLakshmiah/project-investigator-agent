"""Candidate C_SEMIF_OPENJEV - bounded-decision contract (Phase 9D, path B-nb).

SemIf-OpenJev (independent open implementation; NOT TypeSafe Jev) scores a FIXED set of
options in one forward pass and returns a probability distribution. It never generates
text, so it cannot emit a project, tool, route, SQL, document id, query, answer or
reasoning. This module:

* builds the SemIf input row: ``{id, state, question, options}`` - state = the normalised
  request + read-only context; question = the fixed criterion (V1, or V2 with fixed DEV
  examples, removing the scored case's own family); options = the fixed 12-option set;
* validates the SemIf ``direct`` result against the pinned contract and FAILS CLOSED
  (probabilities are UNCALIBRATED scores, never described as calibrated confidence);
* applies the harness-owned policy: argmax (direct mode returns no choice), clarification
  handling, top-probability and margin thresholds -> accepted intent or ABSTAIN, with
  harness reason codes; route = requirements[intent] (never predicted);
* computes the protocol lock (hashes of everything that must not change after predictions).

No model is loaded or called here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from worldbank_copilot.routing.models import Intent, SemanticDecision
from worldbank_copilot.routing.semantic import SEMANTIC_INTENTS

CANDIDATE = "C_SEMIF_OPENJEV"
CLARIFICATION = "CLARIFICATION_NEEDED"
ABSTAIN = "ABSTAIN"


class HarnessReason(StrEnum):
    ACCEPTED = "ACCEPTED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    LOW_MARGIN = "LOW_MARGIN"
    MODEL_CHOSE_CLARIFICATION = "MODEL_CHOSE_CLARIFICATION"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    ENDPOINT_FAILURE = "ENDPOINT_FAILURE"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SemifConfig(_Model):
    version: int
    candidate: Literal["C_SEMIF_OPENJEV"]
    path: Literal["B-nb"]
    status: str
    implementation: dict[str, str]
    model: dict[str, str]
    contract: dict[str, Any]
    options: tuple[dict[str, str], ...]
    criterion: dict[str, Any]
    state_fields: tuple[str, ...]
    policy: dict[str, Any]
    c1_eligible_dev_case_ids: tuple[str, ...]
    c2_shadow_dev_case_ids: tuple[str, ...]
    probe_set: dict[str, Any]
    acceptance_gates: dict[str, Any]
    p1_synthetic: tuple[Any, ...]
    p1_repeats_for_warm_latency: int

    @model_validator(mode="after")
    def _option_set(self) -> SemifConfig:
        ids = [o["id"] for o in self.options]
        expected = [i.value for i in SEMANTIC_INTENTS] + [CLARIFICATION]
        if sorted(ids) != sorted(expected) or len(set(ids)) != len(ids):
            raise ValueError("options must be exactly the reviewed intents + CLARIFICATION_NEEDED")
        if not 2 <= len(ids) <= 16:
            raise ValueError("SemIf accepts 2-16 options")
        return self


def load_semif_config(config_dir: Path) -> SemifConfig:
    raw = yaml.safe_load((Path(config_dir) / "routing" / "semantic_semif.yaml").read_text("utf-8"))
    return SemifConfig.model_validate(raw)


# -- input rows -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FewShotExample:
    case_id: str
    family: str
    question: str
    intent: str


def criterion_text(
    config: SemifConfig,
    variant: str,
    examples: Sequence[FewShotExample] = (),
    exclude_family: str | None = None,
) -> str:
    text = config.criterion["V1"]
    if variant == "V1":
        return text
    if variant != "V2":
        raise ValueError(f"unknown variant {variant!r}")
    kept = [e for e in examples if e.family != exclude_family]
    lines = [f'- "{e.question}" -> {e.intent}' for e in kept]
    return text + " Examples:\n" + "\n".join(lines)


def build_row(
    config: SemifConfig,
    request_id: str,
    question: str,
    project_id: str,
    time_scope: str,
    variant: str,
    examples: Sequence[FewShotExample] = (),
    exclude_family: str | None = None,
) -> dict[str, Any]:
    return {
        "id": request_id,
        "state": {"request": question, "resolved_project": project_id, "time_scope": time_scope},
        "question": criterion_text(config, variant, examples, exclude_family),
        "options": [dict(o) for o in config.options],
    }


def few_shot_examples(config: SemifConfig, dataset: Any) -> list[FewShotExample]:
    by_id = {c.case_id: c for c in dataset.cases}
    out = []
    for cid in config.criterion["V2_examples_dev_case_ids"]:
        case = by_id[cid]
        if case.split != "dev":
            raise ValueError(f"few-shot example {cid} is not a DEV case")
        out.append(FewShotExample(cid, case.family, case.question, case.expected.intents[0].value))
    return out


# -- result validation (fail closed) ----------------------------------------------------------


@dataclass(frozen=True)
class Scores:
    request_id: str
    probabilities: dict[str, float]
    prompt_sha256: str
    model_metadata: Any
    forward_seconds: float | None
    total_seconds: float | None


class InvalidSemifResult(ValueError):
    """The SemIf output violates the pinned contract."""


def validate_result(result: dict[str, Any], row: dict[str, Any], config: SemifConfig) -> Scores:
    option_ids = [o["id"] for o in row["options"]]
    if result.get("id") != row["id"]:
        raise InvalidSemifResult("result id does not match the request")
    if result.get("option_ids") != option_ids:
        raise InvalidSemifResult("option ids differ from the declared options (order matters)")
    probs = result.get("probabilities")
    if not isinstance(probs, list) or len(probs) != len(option_ids):
        raise InvalidSemifResult("probabilities must be a list aligned with option_ids")
    if not all(isinstance(p, int | float) and math.isfinite(p) and 0.0 <= p <= 1.0 for p in probs):
        raise InvalidSemifResult("every probability must be a finite number in [0, 1]")
    if abs(sum(probs) - 1.0) > config.contract["probability_sum_tolerance"]:
        raise InvalidSemifResult(f"probabilities sum to {sum(probs):.6f}, not 1")
    digest = result.get("prompt_sha256")
    if not (
        isinstance(digest, str)
        and len(digest) == 64
        and all(c in "0123456789abcdef" for c in digest)
    ):
        raise InvalidSemifResult("prompt_sha256 missing or malformed")
    if result.get("prompt_version") != config.contract["expected_prompt_version"]:
        raise InvalidSemifResult(f"unexpected prompt_version {result.get('prompt_version')!r}")
    # The model revision is infrastructure provenance, verified from the HF snapshot in P1
    # (semif_capability.hf_snapshot_provenance), not a property of the inference output.
    return Scores(
        row["id"],
        dict(zip(option_ids, map(float, probs), strict=True)),
        digest,
        result.get("model"),
        result.get("forward_seconds"),
        result.get("total_seconds"),
    )


# -- harness policy ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyOutcome:
    accepted: bool
    choice: str  # harness argmax (ties: option order)
    intent: Intent | None
    route: str | None
    top: float
    margin: float
    reason: HarnessReason


def apply_policy(
    scores: Scores,
    tau: float,
    margin: float,
    route_of: dict[Intent, str],
    option_order: Sequence[str],
) -> PolicyOutcome:
    ranked = sorted(option_order, key=lambda o: (-scores.probabilities[o], option_order.index(o)))
    top_id, second_id = ranked[0], ranked[1]
    top = scores.probabilities[top_id]
    gap = top - scores.probabilities[second_id]
    if top_id == CLARIFICATION:
        reason = HarnessReason.MODEL_CHOSE_CLARIFICATION
    elif top < tau:
        reason = HarnessReason.LOW_CONFIDENCE
    elif gap < margin:
        reason = HarnessReason.LOW_MARGIN
    else:
        intent = Intent(top_id)
        return PolicyOutcome(
            True, top_id, intent, route_of[intent], top, gap, HarnessReason.ACCEPTED
        )
    return PolicyOutcome(False, top_id, None, None, top, gap, reason)


def to_decision(
    outcome: PolicyOutcome | None,
    *,
    version: str,
    latency_ms: float = 0.0,
    failure: HarnessReason | None = None,
    detail: str = "",
) -> SemanticDecision:
    """Harness decision for the existing semantic hook (abstain -> CLARIFY, no execution)."""
    if outcome is None or failure is not None:
        reason = (failure or HarnessReason.INVALID_RESPONSE).value
        return SemanticDecision(
            classifier=CANDIDATE,
            version=version,
            abstain=True,
            reason=f"{reason}: {detail}".rstrip(": "),
            latency_ms=latency_ms,
        )
    return SemanticDecision(
        classifier=CANDIDATE,
        version=version,
        abstain=not outcome.accepted,
        intent=outcome.intent,
        route=outcome.route,
        confidence=round(outcome.top, 4),
        reason=outcome.reason.value,
        latency_ms=latency_ms,
    )


# -- protocol lock ---------------------------------------------------------------------------


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def protocol_lock(config: SemifConfig, dataset_sha256_lf: str, probe_sha256_lf: str) -> dict:
    return {
        "candidate": CANDIDATE,
        "path": config.path,
        "semif_commit": config.implementation["commit"],
        "model_id": config.model["id"],
        "model_revision": config.model["revision"],
        "mode_backend_device_dtype": [
            config.model[k] for k in ("mode", "backend", "device", "dtype")
        ],
        "option_set_sha256": _sha(list(config.options)),
        "criterion_version": config.criterion["version"],
        "criterion_sha256": _sha(
            {
                "V1": config.criterion["V1"],
                "V2_examples": config.criterion["V2_examples_dev_case_ids"],
            }
        ),
        "tau_grid": config.policy["tau_grid"],
        "margin_grid": config.policy["margin_grid"],
        "threshold_grid_sha256": _sha(
            {
                "tau": config.policy["tau_grid"],
                "margin": config.policy["margin_grid"],
                "variants": config.policy["variants"],
                "selection": config.policy["selection"],
            }
        ),
        "probe_set_sha256_lf": probe_sha256_lf,
        "routing_dataset_sha256_lf": dataset_sha256_lf,
        "p1_synthetic_sha256": _sha(list(config.p1_synthetic)),
    }


def lock_sha256(lock: dict) -> str:
    """Hash of the whole protocol lock (canonical JSON), persisted with every P1/DEV run."""
    return _sha(lock)


# -- ambiguity probe set (frozen; labels are human annotations, not model outputs) ------------


class ProbeExpected(_Model):
    intent: str
    reason_code: str


class ProbeCase(_Model):
    probe_id: str
    kind: Literal["ambiguous", "control"]
    active_project_id: str
    question: str
    expected: ProbeExpected
    rationale: str

    @model_validator(mode="after")
    def _consistent(self) -> ProbeCase:
        if (self.expected.intent == ABSTAIN) != (self.kind == "ambiguous"):
            raise ValueError(f"{self.probe_id}: ambiguous cases abstain, controls do not")
        if self.expected.intent != ABSTAIN and self.expected.intent not in {
            i.value for i in SEMANTIC_INTENTS
        }:
            raise ValueError(f"{self.probe_id}: unknown intent {self.expected.intent}")
        return self


class ProbeSet(_Model):
    version: int
    status: Literal["FROZEN"]
    frozen_on: date
    label_provenance: str
    cases: tuple[ProbeCase, ...]


def load_probe_set(path: Path) -> ProbeSet:
    return ProbeSet.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
