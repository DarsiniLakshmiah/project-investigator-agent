"""Phase 9B routing fixtures: the real registry/manifest/config, synthetic governed rows,
a tiny chunk corpus for the three real project ids, and call-recording spies."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from tests.conftest import REPO_CONFIG_DIR
from tests.support.retrieval_golden import OverlapCrossEncoder, ScopedDense, TextEmbeddings
from tests.support.tool_fixtures import IPF, OTHER, PFORR, REGISTRY, context, gold, tables

from worldbank_copilot.ingestion.documents import load_document_manifest
from worldbank_copilot.retrieval.config import load_retrieval_settings
from worldbank_copilot.retrieval.rerank_policy import NeverRerank
from worldbank_copilot.retrieval.retriever import ChunkStore, Retriever
from worldbank_copilot.routing.config import load_routing_config
from worldbank_copilot.routing.entities import EntityIndex
from worldbank_copilot.routing.models import AccessContext
from worldbank_copilot.routing.service import RoutingService
from worldbank_copilot.tools.documents import DocumentSearch, DocumentSearchConfig
from worldbank_copilot.tools.executor import ToolExecutor
from worldbank_copilot.tools.reader import InMemoryReader
from worldbank_copilot.tools.registry import TOOL_SPECS

CONFIG = load_routing_config(REPO_CONFIG_DIR)
MANIFEST = load_document_manifest(REPO_CONFIG_DIR / "document_manifest.yaml")
INDEX = EntityIndex.build(REGISTRY, MANIFEST, CONFIG)
RS = load_retrieval_settings(REPO_CONFIG_DIR)
ALL = (IPF, PFORR, OTHER)
TEXTS = [
    "The closing date was extended because distribution works were delayed.",
    "The restructuring paper states the cancellation was requested by the borrower.",
    "Procurement delays affected the water treatment plant contract.",
    "Grievance redress improved through an integrated helpline.",
]


def chunk_rows() -> list[dict[str, Any]]:
    rows = []
    for project in ALL:
        for i, text in enumerate(TEXTS):
            rows.append(
                {
                    "chunk_id": f"{project}-c{i}",
                    "chunk_strategy": "fixed",
                    "chunk_role": "RETRIEVAL",
                    "chunk_type": "TEXT",
                    "parent_chunk_id": None,
                    "project_id": project,
                    "document_id": f"{project}-{'ab' * 6}",
                    "document_type": "RESTRUCTURING_PAPER",
                    "document_label": "Restructuring Paper",
                    "document_date": date(2021, 5, 1),
                    "isr_sequence": None,
                    "source_file": f"{project}/f.pdf",
                    "source_hash": "h" * 64,
                    "page_number": i + 1,
                    "page_numbers": [i + 1],
                    "section_title": "Rationale",
                    "element_ids": [f"b{i}"],
                    "chunk_text": f"{text} ({project})",
                    "search_text": f"{project} | {text}",
                }
            )
    return rows


class SpyExecutor(ToolExecutor):
    """Records every tool call (name, arguments, scope) before delegating."""

    def __init__(self):
        super().__init__(TOOL_SPECS)
        self.calls: list[tuple[str, dict, str]] = []

    def run(self, name, arguments, ctx, *, scope_project_id, authorized_projects=None):
        self.calls.append((name, dict(arguments), scope_project_id))
        return super().run(
            name,
            arguments,
            ctx,
            scope_project_id=scope_project_id,
            authorized_projects=authorized_projects,
        )


class SpyRetriever(Retriever):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_stage_calls: list[tuple[str, str]] = []

    def first_stage(self, question, project_id, **kwargs):
        self.first_stage_calls.append((question, project_id))
        return super().first_stage(question, project_id, **kwargs)


def routing_tables() -> dict[str, list[dict[str, Any]]]:
    """Tool fixtures plus two source-dated restructurings for anchor resolution."""
    data = tables()
    data["gold.project_timeline"] += [
        gold(
            record_id="t5",
            project_id=IPF,
            timeline_event_id="e5",
            event_type="RESTRUCTURING",
            event_title="Restructuring (approved)",
            event_date=date(2018, 6, 1),
            event_date_status="SOURCE_STATED",
            event_sequence=5,
            date_sequence_anomaly=False,
        ),
        gold(
            record_id="t6",
            project_id=IPF,
            timeline_event_id="e6",
            event_type="RESTRUCTURING",
            event_title="Restructuring (approved)",
            event_date=date(2021, 5, 20),
            event_date_status="SOURCE_STATED",
            event_sequence=6,
            date_sequence_anomaly=False,
        ),
    ]
    return data


@dataclass
class Harness:
    service: RoutingService
    executor: SpyExecutor
    retriever: SpyRetriever
    readers: list[InMemoryReader] = field(default_factory=list)

    def ask(self, question, active=IPF, authorized=ALL, documents=True):
        self.service.documents = self._documents if documents else None
        access = AccessContext(
            user_ref="u-test", authorized_projects=authorized, active_project_id=active
        )
        return self.service.handle(question, access, request_id="req-1")

    @property
    def reads(self):
        return [r for reader in self.readers for r in reader.requests]


def harness(data=None, rerank=None) -> Harness:
    executor = SpyExecutor()
    rows = chunk_rows()
    retriever = SpyRetriever(ChunkStore(rows), RS, list(ALL), TextEmbeddings(), ScopedDense(rows))
    readers: list[InMemoryReader] = []

    def factory(request_id):
        reader = InMemoryReader(data() if callable(data) else (data or routing_tables()))
        readers.append(reader)
        return context(reader)

    h = Harness(RoutingService(CONFIG, INDEX, executor, factory), executor, retriever, readers)
    h._documents = DocumentSearch(
        retriever,
        DocumentSearchConfig(chunk_strategy="fixed", method="hybrid", candidate_k=10, final_k=5),
        rerank or NeverRerank(),
        OverlapCrossEncoder(),
    )
    return h
