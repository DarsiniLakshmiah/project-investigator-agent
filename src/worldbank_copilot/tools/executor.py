"""Tool executor: allowlist, typed arguments, scope enforcement, error mapping (Phase 9).

Order of operations for every call:

1. the tool must be in the registry (allowlist) - otherwise INVALID_ARGUMENT;
2. arguments are validated by the tool's Pydantic model (extra fields rejected);
3. the project must be approved (registry), authorised for this request, and equal to
   the request's resolved scope - otherwise SCOPE_REFUSED, BEFORE any table is read;
4. the tool's tables are pinned to one Delta version each (consistent snapshot);
5. the tool runs; it can only read through ``ToolContext.read`` (project-scoped);
6. known failures map to explicit statuses; unexpected exceptions become ERROR with the
   exception recorded and logged (never swallowed, never turned into an empty answer).

The executor never calls another tool and never falls back to document search: any
STRUCTURED -> DOCUMENT fallback is a separate, allowlisted routing decision (Phase 9B).
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterable
from typing import Any

from pydantic import ValidationError

from worldbank_copilot.retrieval.models import ScopeViolation
from worldbank_copilot.tools.base import DataIntegrityError, ToolContext, ToolSpec, ToolTimeout
from worldbank_copilot.tools.models import (
    MechanicalCode,
    MechanicalFinding,
    ToolResult,
    ToolStatus,
)
from worldbank_copilot.tools.reader import ReadError

log = logging.getLogger(__name__)


class ToolExecutor:
    def __init__(self, specs: Iterable[ToolSpec]):
        self.specs = {s.name: s for s in specs}

    def run(
        self,
        name: str,
        arguments: dict[str, Any],
        ctx: ToolContext,
        *,
        scope_project_id: str,
        authorized_projects: Iterable[str] | None = None,
    ) -> ToolResult:
        started = time.perf_counter()
        spec = self.specs.get(name)

        def result(status: ToolStatus, project_id: str | None, **kw: Any) -> ToolResult:
            return ToolResult(
                tool=name,
                tool_version=spec.version if spec else "unknown",
                request_id=ctx.request_id,
                project_id=project_id,
                status=status,
                data_snapshot=ctx.reader.snapshot() if spec else {},
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                **kw,
            )

        if spec is None:
            return result(
                ToolStatus.INVALID_ARGUMENT, None, error=f"tool {name!r} is not allowlisted"
            )
        try:
            args = spec.args_model.model_validate(arguments)
        except ValidationError as exc:
            pid = arguments.get("project_id") if isinstance(arguments, dict) else None
            invalid_project = any(e["loc"][:1] == ("project_id",) for e in exc.errors())
            return result(
                ToolStatus.SCOPE_REFUSED if invalid_project else ToolStatus.INVALID_ARGUMENT,
                pid if isinstance(pid, str) else None,
                error=str(exc),
                mechanical=[MechanicalFinding(code=MechanicalCode.PROJECT_INVALID, detail=str(pid))]
                if invalid_project
                else [],
            )
        pid = args.project_id
        refusal = self._scope_refusal(pid, ctx, scope_project_id, authorized_projects)
        if refusal is not None:
            code, detail = refusal
            return result(
                ToolStatus.SCOPE_REFUSED,
                pid,
                error=detail,
                mechanical=[MechanicalFinding(code=code, detail=detail)],
            )
        try:
            ctx.reader.pin(spec.tables)
            outcome = spec.run(ctx, args)
        except ScopeViolation as exc:
            return result(ToolStatus.SCOPE_REFUSED, pid, error=str(exc))
        except DataIntegrityError as exc:
            log.error("tool %s data integrity error: %s", name, exc)
            return result(ToolStatus.DATA_INTEGRITY_ERROR, pid, error=str(exc))
        except ToolTimeout as exc:
            return result(ToolStatus.TIMEOUT, pid, error=str(exc))
        except ReadError as exc:
            log.error("tool %s read error: %s", name, exc)
            return result(ToolStatus.ERROR, pid, error=f"ReadError: {exc}")
        except Exception as exc:  # recorded and logged; never an empty answer
            log.exception("tool %s failed", name)
            return result(ToolStatus.ERROR, pid, error=f"{type(exc).__name__}: {exc}")
        return result(
            outcome.status,
            pid,
            items=outcome.items,
            filters=outcome.filters,
            argument_candidates=outcome.argument_candidates,
            mechanical=outcome.mechanical,
            caveats=outcome.caveats,
            notices=outcome.notices,
        )

    @staticmethod
    def _scope_refusal(
        project_id: str,
        ctx: ToolContext,
        scope_project_id: str,
        authorized_projects: Iterable[str] | None,
    ) -> tuple[MechanicalCode, str] | None:
        if project_id not in ctx.registry:
            return (
                MechanicalCode.PROJECT_INVALID,
                f"project {project_id} is not in the approved corpus",
            )
        if authorized_projects is not None and project_id not in set(authorized_projects):
            return MechanicalCode.PROJECT_OUT_OF_SCOPE, f"project {project_id} is not authorised"
        if project_id != scope_project_id:
            return (
                MechanicalCode.PROJECT_OUT_OF_SCOPE,
                f"project {project_id} differs from the request scope {scope_project_id}",
            )
        return None


def new_request_id() -> str:
    """Request ids identify a call in logs/traces; they are not data identities."""
    return uuid.uuid4().hex
