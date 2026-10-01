"""silver.document_chunks: the governed retrieval corpus (Phase 8).

One table holds the chunks of every configured strategy (``chunk_strategy`` column), so
experiments and production share one governed source and one Vector Search index
(filtered by strategy). Built with the Phase 6 contract mechanism: deterministic
``record_id`` (natural key: ``chunk_id``), content ``record_hash`` and snapshot MERGE.

Provenance chain: chunk -> element_ids (Phase 4 blocks / tables) + pages + section ->
document_id -> bronze.document_inventory (source_hash).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from worldbank_copilot.lakehouse.contracts import Column, DType, Role, TableContract, build_contract
from worldbank_copilot.lakehouse.records import Dataset, LoadContext, finalize
from worldbank_copilot.parsing.models import ParsedDocument, ParseStatus
from worldbank_copilot.retrieval.chunking import (
    CHUNK_TABLE,
    PARENT,
    RETRIEVAL,
    TABLE,
    TEXT,
    chunk_rows,
)
from worldbank_copilot.retrieval.config import ChunkingConfig

S, B, D = DType.STRING, DType.BIGINT, DType.DATE
P = Role.PROVENANCE
METHODS = ("DOCLING_TEXT", "DOCLING_TABLE", "DOCLING_TEXT_AND_TABLE")


def _c(name, dtype, nullable=True, role=Role.DATA, vocab=None, doc=None) -> Column:  # noqa: ANN001
    return Column(name, dtype, nullable, role, vocab, doc)


def chunks_contract(strategies: Iterable[str]) -> TableContract:
    body = [
        _c("chunk_id", S, False, Role.KEY, doc="hash(strategy, version, document, role, ordinal)"),
        _c("chunk_strategy", S, False, vocab=tuple(sorted(strategies))),
        _c("strategy_version", S, False, doc="Hash of the strategy's chunking settings."),
        _c(
            "chunk_role",
            S,
            False,
            vocab=(RETRIEVAL, PARENT),
            doc="RETRIEVAL chunks are indexed; PARENT chunks are context only.",
        ),
        _c("chunk_type", S, False, vocab=(TEXT, TABLE)),
        _c("chunk_ordinal", B, False, doc="Order within document, strategy and role."),
        _c("part_index", B, doc="Row group / split index when one element was split."),
        _c("parent_chunk_id", S, doc="PARENT chunk (parent_child strategy only)."),
        _c("project_id", S, False),
        _c("document_id", S, False),
        _c("document_type", S),
        _c("document_label", S, False, doc="Display label used in citations."),
        _c("document_date", D),
        _c("isr_sequence", B),
        _c("report_number", S),
        _c("source_file", S, False, P),
        _c("source_relative_path", S, False, P),
        _c("source_hash", S, False, P, doc="sha256 of the source PDF (bronze.document_inventory)."),
        _c("parser_config_hash", S, False, P),
        _c("page_number", B, False, P, doc="First PDF page (1-based) the chunk comes from."),
        _c("page_numbers", DType.ARRAY_BIGINT, False, P),
        _c("section_id", S, role=P),
        _c("section_title", S, role=P),
        _c("table_id", S, role=P),
        _c("element_ids", DType.ARRAY_STRING, False, P, doc="Phase 4 block / table ids."),
        _c("extraction_method", S, False, P, vocab=METHODS),
        _c("chunk_text", S, False, doc="Text exactly as parsed (cleaned)."),
        _c(
            "search_text",
            S,
            False,
            doc="Context header (project | document | section) + chunk_text; embedded/indexed.",
        ),
        _c("text_sha256", S, False, doc="sha256 of search_text (embedding cache key)."),
        _c("char_count", B, False),
    ]
    return build_contract(
        CHUNK_TABLE,
        "silver",
        "silver.document_chunks",
        body,
        ("chunk_id",),
        "Retrieval corpus built from Phase 4 parsed documents (all chunking strategies).",
        profile_groups=[
            ("chunk_strategy", "chunk_role"),
            ("project_id", "chunk_strategy"),
            ("chunk_strategy", "document_type"),
        ],
    )


def corpus_bodies(documents: Iterable[ParsedDocument], config: ChunkingConfig) -> list[dict]:
    rows: list[dict[str, Any]] = []
    for doc in sorted(documents, key=lambda d: d.document_id):
        if doc.parse_status is not ParseStatus.SUCCESS:
            continue
        for strategy in sorted(config.strategies):
            rows.extend(chunk_rows(doc, config, strategy))
    return rows


def corpus_dataset(
    documents: Iterable[ParsedDocument], config: ChunkingConfig, ctx: LoadContext
) -> Dataset:
    contract = chunks_contract(config.strategies)
    return finalize(contract, corpus_bodies(documents, config), ctx)
