"""Application-only guards over the unchanged governed tool executor."""

from worldbank_copilot.routing.models import TemporalKind
from worldbank_copilot.routing.temporal import parse_temporal
from worldbank_copilot.tools.executor import ToolExecutor
from worldbank_copilot.tools.models import ToolResult, ToolStatus


class ScopedTools(ToolExecutor):
    def __init__(self, delegate):
        super().__init__(tuple(delegate.specs.values()))
        self.delegate = delegate
        self.document_calls = 0

    def run(self, name, arguments, ctx, **kwargs):
        if name == "search_project_documents":
            temporal = parse_temporal(arguments["query"])
            if temporal.kind in (
                TemporalKind.DATE,
                TemporalKind.YEAR,
                TemporalKind.DATE_RANGE,
                TemporalKind.EVENT_ANCHORED,
            ):
                return ToolResult(
                    tool=name,
                    tool_version=self.specs[name].version,
                    request_id=ctx.request_id,
                    project_id=arguments.get("project_id"),
                    status=ToolStatus.AMBIGUOUS_ARGUMENT,
                    caveats=[
                        "Document date/anchor hints are not actual retrieval filters; "
                        "use an enforceable ISR scope."
                    ],
                )
            self.document_calls += 1
        return self.delegate.run(name, arguments, ctx, **kwargs)
