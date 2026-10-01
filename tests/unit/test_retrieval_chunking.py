"""Phase 8 chunking and corpus: identity, boundaries, tables, parent-child, provenance."""

from datetime import UTC, datetime

import pytest
from tests.conftest import REPO_CONFIG_DIR
from tests.support.extraction_builders import S, T, Tbl, make_doc

from worldbank_copilot.lakehouse.contracts import validate_rows
from worldbank_copilot.lakehouse.records import LoadContext
from worldbank_copilot.retrieval.chunking import (
    chunk_document,
    chunk_rows,
    split_text,
    split_with_overlap,
    table_parts,
)
from worldbank_copilot.retrieval.config import load_retrieval_settings
from worldbank_copilot.retrieval.corpus import corpus_bodies, corpus_dataset

LONG = " ".join(f"Sentence {i} describes implementation progress in the city." for i in range(60))


@pytest.fixture(scope="module")
def cfg():
    return load_retrieval_settings(REPO_CONFIG_DIR).chunking


def doc(**kw):
    pages = [
        [
            (T, "ISR07744"),
            (S, "1. OBJECTIVE"),
            (T, "The objective is continuous piped water supply."),
        ],
        [
            (S, "4. KEY ISSUES & STATUS"),
            (T, LONG),
            (T, "@#&OPS~Doctype~OPS^dynamics@restrhybridsummarychanges#doctemplate"),
            (T, "The operator contract was signed and mobilised in all three cities."),
        ],
        [(S, "6. RESULTS"), (T, "Results are reported in the table below.")],
    ]
    rows = [["Indicator Name", "Baseline", "Current", "Target"]] + [
        [f"Indicator {i} (Number)", "0", str(i * 10), "100"] for i in range(40)
    ]
    return make_doc(pages, [Tbl(3, rows)], document_id="P130544-abc", isr_sequence=7, **kw)


def rows_for(cfg, strategy, document=None):
    return chunk_rows(document or doc(), cfg, strategy)


@pytest.mark.parametrize("strategy", ["fixed", "structure", "parent_child"])
def test_chunk_ids_are_deterministic_and_unique(cfg, strategy):
    first, second = rows_for(cfg, strategy), rows_for(cfg, strategy)
    assert [r["chunk_id"] for r in first] == [r["chunk_id"] for r in second]
    assert len({r["chunk_id"] for r in first}) == len(first)
    assert first == second


@pytest.mark.parametrize("strategy", ["fixed", "structure", "parent_child"])
def test_retrieval_chunks_respect_size_limit_and_keep_provenance(cfg, strategy):
    limit = cfg.strategies[strategy].retrieval_max_chars
    for row in rows_for(cfg, strategy):
        assert row["project_id"] == "P130544" and row["document_id"] == "P130544-abc"
        assert row["source_hash"] == "0" * 64 and row["isr_sequence"] == 7
        assert row["page_number"] == row["page_numbers"][0] and row["element_ids"]
        assert row["search_text"].startswith("P130544 | ")
        if row["chunk_role"] == "RETRIEVAL":
            assert row["char_count"] <= limit


def test_structure_chunks_never_cross_sections_and_drop_markers(cfg):
    rows = rows_for(cfg, "structure")
    for row in rows:
        assert "@#&OPS" not in row["chunk_text"]
        assert "ISR07744" not in row["chunk_text"]  # below min_chars
    by_section = {r["section_title"] for r in rows}
    assert {"4. KEY ISSUES & STATUS", "6. RESULTS"} <= by_section
    key = [r for r in rows if r["section_title"] == "4. KEY ISSUES & STATUS"]
    assert len(key) > 1  # the long paragraph was split within its section
    assert all(r["chunk_type"] == "TEXT" for r in key)


def test_tables_are_split_into_row_groups_with_repeated_header(cfg):
    tables = [r for r in rows_for(cfg, "structure") if r["chunk_type"] == "TABLE"]
    assert len(tables) > 1
    for row in tables:
        assert row["chunk_text"].startswith("Indicator Name | Baseline | Current | Target")
        assert row["table_id"] == "t0001" and row["extraction_method"] == "DOCLING_TABLE"
    body = "\n".join(r["chunk_text"] for r in tables)
    assert all(f"Indicator {i} (Number)" in body for i in range(40))  # no row lost


def test_parent_child_links_are_consistent(cfg):
    rows = rows_for(cfg, "parent_child")
    parents = {r["chunk_id"]: r for r in rows if r["chunk_role"] == "PARENT"}
    children = [r for r in rows if r["chunk_role"] == "RETRIEVAL"]
    assert parents and children
    for child in children:
        parent = parents[child["parent_chunk_id"]]
        assert parent["section_id"] == child["section_id"]
        assert child["chunk_text"] in parent["chunk_text"]
    assert all(p["parent_chunk_id"] is None for p in parents.values())
    assert all(r["parent_chunk_id"] is None for r in rows_for(cfg, "structure"))


def test_duplicate_text_within_a_document_is_indexed_once(cfg):
    pages = [
        [(S, "A"), (T, "Identical paragraph repeated in the report for emphasis.")],
        [(S, "B"), (T, "Identical paragraph repeated in the report for emphasis.")],
    ]
    rows = chunk_rows(make_doc(pages, document_id="P130544-dup"), cfg, "structure")
    assert len(rows) == 1


def test_changed_configuration_changes_identity_but_not_content(cfg):
    other = cfg.model_copy(update={"min_chars": cfg.min_chars + 1})
    assert other.version("structure") != cfg.version("structure")
    assert rows_for(other, "structure")[0]["chunk_id"] != rows_for(cfg, "structure")[0]["chunk_id"]


def test_splitting_primitives():
    assert all(len(p) <= 100 for p in split_text(LONG, 100))
    assert " ".join(split_text(LONG, 100)).split() == LONG.split()
    spans = split_with_overlap(LONG, 200, 40)
    assert spans[0][0] == 0 and spans[-1][1] == len(LONG)
    assert all(e - s <= 200 for s, e in spans)
    assert all(spans[i + 1][0] < spans[i][1] for i in range(len(spans) - 1))  # overlap
    parts = table_parts("Name | Value\n" + "\n".join(f"row {i} | {i}" for i in range(50)), 120)
    assert all(p.startswith("Name | Value\n") and len(p) <= 120 for p in parts)


def test_corpus_rows_satisfy_the_contract_and_are_reproducible(cfg):
    ctx = LoadContext("snap", "run-a", datetime(2026, 9, 30, tzinfo=UTC), "v", {})
    other = LoadContext("snap", "run-b", datetime(2027, 1, 1, tzinfo=UTC), "v", {})
    first = corpus_dataset([doc()], cfg, ctx)
    second = corpus_dataset([doc()], cfg, other)
    assert validate_rows(first.contract, first.rows) == []
    assert [(r["record_id"], r["record_hash"]) for r in first.rows] == [
        (r["record_id"], r["record_hash"]) for r in second.rows
    ]
    strategies = {r["chunk_strategy"] for r in first.rows}
    assert strategies == set(cfg.strategies)
    assert len(corpus_bodies([doc()], cfg)) == len(first.rows)


def test_failed_documents_are_excluded(cfg):
    from worldbank_copilot.parsing.models import ParseStatus

    failed = doc().model_copy(update={"parse_status": ParseStatus.FAILED})
    assert corpus_bodies([failed], cfg) == []


def test_chunk_document_returns_parents_only_for_parent_child(cfg):
    assert {c.role for c in chunk_document(doc(), cfg, "structure")} == {"RETRIEVAL"}
    assert {c.role for c in chunk_document(doc(), cfg, "parent_child")} == {"RETRIEVAL", "PARENT"}
