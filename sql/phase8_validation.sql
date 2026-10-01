-- Phase 8 validation queries (read-only). Placeholders {catalog}, {bronze}, {silver},
-- {gold} are rendered from configuration. Each query starts with "-- name: <id>".
-- Queries named *_violations must return 0 in every count column.

-- name: corpus_counts
SELECT chunk_strategy, chunk_role, chunk_type, count(*) AS chunks,
       count(DISTINCT document_id) AS documents, max(char_count) AS max_chars,
       round(avg(char_count)) AS avg_chars
FROM {catalog}.{silver}.document_chunks
GROUP BY ALL
ORDER BY chunk_strategy, chunk_role, chunk_type;

-- name: chunks_by_project_and_type
SELECT chunk_strategy, project_id, document_type, count(*) AS retrieval_chunks
FROM {catalog}.{silver}.document_chunks
WHERE chunk_role = 'RETRIEVAL'
GROUP BY ALL
ORDER BY chunk_strategy, project_id, document_type;

-- name: identity_violations
SELECT count(*) - count(DISTINCT chunk_id) AS duplicate_chunk_ids,
       count(*) - count(DISTINCT record_id) AS duplicate_record_ids,
       sum(CASE WHEN page_number IS NULL OR size(page_numbers) = 0 THEN 1 ELSE 0 END)
         AS chunks_without_pages,
       sum(CASE WHEN project_id IS NULL OR document_id IS NULL THEN 1 ELSE 0 END)
         AS chunks_without_scope
FROM {catalog}.{silver}.document_chunks;

-- name: provenance_violations
-- Every chunk resolves to its source document in bronze.document_inventory with the same hash.
SELECT sum(CASE WHEN d.document_id IS NULL THEN 1 ELSE 0 END) AS chunks_without_inventory_document,
       sum(CASE WHEN d.document_id IS NOT NULL AND d.sha256 <> c.source_hash THEN 1 ELSE 0 END)
         AS chunks_with_hash_mismatch,
       sum(CASE WHEN d.document_id IS NOT NULL AND d.project_id <> c.project_id THEN 1 ELSE 0 END)
         AS chunks_with_project_mismatch
FROM {catalog}.{silver}.document_chunks c
LEFT JOIN {catalog}.{bronze}.document_inventory d ON d.document_id = c.document_id;

-- name: provenance_chains
-- Retrieved chunk -> parsed elements/pages/section -> document -> inventory -> source hash.
SELECT c.project_id, c.document_label, c.page_numbers, c.section_title, c.element_ids,
       c.extraction_method, substr(c.chunk_text, 1, 120) AS chunk_text, d.relative_path,
       d.sha256 AS inventory_sha256, c.source_hash = d.sha256 AS hash_matches
FROM {catalog}.{silver}.document_chunks c
JOIN {catalog}.{bronze}.document_inventory d ON d.document_id = c.document_id
WHERE c.chunk_strategy = 'structure' AND c.chunk_role = 'RETRIEVAL'
  AND c.document_id IN ('P130544-9fa1d6d298ab', 'P130544-dc4201ccaf76',
                        'P179039-95c180d52c77', 'P506272-c70c73ae1a46',
                        'P506272-ac25b4911bfb')
  AND c.chunk_ordinal < 3
ORDER BY c.project_id, c.document_label, c.chunk_ordinal;

-- name: parent_child_violations
SELECT sum(CASE WHEN p.chunk_id IS NULL THEN 1 ELSE 0 END) AS children_without_parent,
       sum(CASE WHEN p.chunk_id IS NOT NULL AND p.document_id <> c.document_id THEN 1 ELSE 0 END)
         AS parent_in_other_document,
       sum(CASE WHEN p.chunk_id IS NOT NULL AND p.chunk_role <> 'PARENT' THEN 1 ELSE 0 END)
         AS parent_not_parent_role
FROM {catalog}.{silver}.document_chunks c
LEFT JOIN {catalog}.{silver}.document_chunks p ON p.chunk_id = c.parent_chunk_id
WHERE c.chunk_strategy = 'parent_child' AND c.chunk_role = 'RETRIEVAL';

-- name: index_alignment_violations
-- The Vector Search source holds exactly the RETRIEVAL chunks, with the same project.
SELECT
  (SELECT count(*) FROM {catalog}.{silver}.document_chunks c
    WHERE c.chunk_role = 'RETRIEVAL' AND NOT EXISTS (
      SELECT 1 FROM {catalog}.{silver}.document_chunk_index i WHERE i.chunk_id = c.chunk_id))
    AS retrieval_chunks_not_indexed,
  (SELECT count(*) FROM {catalog}.{silver}.document_chunk_index i
    LEFT JOIN {catalog}.{silver}.document_chunks c ON c.chunk_id = i.chunk_id
    WHERE c.chunk_id IS NULL OR c.chunk_role <> 'RETRIEVAL') AS index_rows_not_in_corpus,
  (SELECT count(*) FROM {catalog}.{silver}.document_chunk_index i
    JOIN {catalog}.{silver}.document_chunks c ON c.chunk_id = i.chunk_id
    WHERE i.project_id <> c.project_id OR i.text_sha256 <> c.text_sha256)
    AS index_rows_disagreeing_with_corpus;

-- name: embedding_cache
SELECT embedding_model, embedding_dimension, count(*) AS vectors,
       min(size(embedding)) AS min_size, max(size(embedding)) AS max_size
FROM {catalog}.{silver}.chunk_embeddings
GROUP BY ALL;

-- name: comments_fix
-- Phase 8 fix: no result comment holds the label instead of the comment text.
SELECT project_id,
       sum(CASE WHEN lower(trim(comments)) = 'comments on achieving targets' THEN 1 ELSE 0 END)
         AS label_instead_of_comment,
       sum(CASE WHEN comments IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_comments
FROM {catalog}.{silver}.project_results
GROUP BY project_id
ORDER BY project_id;
