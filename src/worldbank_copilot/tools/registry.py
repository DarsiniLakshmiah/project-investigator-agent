"""The Phase 9 tool catalog (allowlist).

Seven structured, read-only tools over governed Silver/Gold tables plus one
document-search tool. Deliberately absent: a free-form SQL tool, any write tool, and
``get_procurement_awards`` (deferred: procurement-award coverage exists only for
P130544, an IPF operation; the two PforR projects are outside the dataset, and no
Phase 9 evaluation requirement justifies a dedicated tool).
"""

from __future__ import annotations

from worldbank_copilot.tools import (
    documents,
    finance,
    project,
    ratings,
    results,
    risks,
    signals,
    timeline,
)
from worldbank_copilot.tools.base import ToolSpec
from worldbank_copilot.tools.executor import ToolExecutor

TOOL_SPECS: tuple[ToolSpec, ...] = (
    project.SPEC,
    timeline.SPEC,
    ratings.SPEC,
    finance.SPEC,
    results.SPEC,
    risks.SPEC,
    signals.SPEC,
    documents.SPEC,
)
STRUCTURED_TOOLS = tuple(s.name for s in TOOL_SPECS if s.tables)
DOCUMENT_TOOLS = tuple(s.name for s in TOOL_SPECS if not s.tables)


def default_executor() -> ToolExecutor:
    return ToolExecutor(TOOL_SPECS)
