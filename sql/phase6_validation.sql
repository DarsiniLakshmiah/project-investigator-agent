-- Phase 6 validation queries (NOT Gold analytics). Placeholders {catalog}, {bronze},
-- {silver}, {gold} are rendered from configuration by
-- worldbank_copilot.lakehouse.validation_sql. Each query starts with "-- name: <id>".

-- name: isr_history_p130544
-- ISR history ordered by ISR sequence (the primary ordering dimension).
SELECT isr_sequence, canonical_report_date, canonical_date_basis, header_date, archive_date,
       pdo_rating, implementation_progress_rating, overall_risk_rating,
       pdo_rating_extraction_method, source_document
FROM {catalog}.{silver}.isr_snapshots
WHERE project_id = 'P130544'
ORDER BY isr_sequence;

-- name: latest_ratings
-- Latest rating state per project (latest = highest ISR sequence, not latest date).
SELECT project_id, isr_sequence, canonical_report_date, pdo_rating,
       implementation_progress_rating, overall_risk_rating
FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY project_id ORDER BY isr_sequence DESC) AS rn
      FROM {catalog}.{silver}.isr_snapshots)
WHERE rn = 1
ORDER BY project_id;

-- name: p179039_isr_ordering
-- Canonical dates kept as printed; ISR 5 is later than ISR 6 by date (flagged anomaly).
SELECT isr_sequence, header_date, archive_date, canonical_report_date, date_difference_days,
       canonical_report_date < LAG(canonical_report_date) OVER (ORDER BY isr_sequence)
         AS earlier_than_previous_isr
FROM {catalog}.{silver}.isr_snapshots
WHERE project_id = 'P179039'
ORDER BY isr_sequence;

-- name: indicator_observations
-- One indicator across ISRs (same indicator_key only; near-matches are NOT merged).
SELECT isr_sequence, indicator_name_raw, baseline_value, current_value, `current_date`,
       target_value, target_date, status, extraction_method,
       evidence_filename, evidence_page_number, evidence_table_id, evidence_row
FROM {catalog}.{silver}.project_results
WHERE project_id = 'P130544'
  AND indicator_name_normalized = 'direct project beneficiaries (number, corporate)'
ORDER BY isr_sequence;

-- name: formal_risks_p179039
SELECT risk_category, risk_rating_raw, risk_rating, status, source_document_type,
       evidence_filename, evidence_page_number, evidence_table_id, evidence_row
FROM {catalog}.{silver}.appraisal_risks
WHERE project_id = 'P179039' AND framing = 'FORMAL_RISK_RATING'
ORDER BY risk_category;

-- name: event_history_p130544
-- event_date is source-stated only; candidate_event_date is DERIVED (never official).
SELECT event_type, event_date, event_date_basis, candidate_event_date, candidate_date_status,
       loan_number, old_closing_date, new_closing_date, cancelled_amount, cancelled_currency,
       additional_financing_amount, status, evidence_filename, evidence_page_number
FROM {catalog}.{silver}.project_events
WHERE project_id = 'P130544'
ORDER BY COALESCE(event_date, candidate_event_date), event_type;

-- name: pdf_text_fallback_records
SELECT 'project_results' AS table_name, project_id, evidence_filename AS document,
       evidence_page_number AS page_number, COUNT(*) AS records
FROM {catalog}.{silver}.project_results WHERE extraction_method = 'PDF_TEXT_FALLBACK'
GROUP BY ALL
UNION ALL
SELECT 'appraisal_risks', project_id, evidence_filename, evidence_page_number, COUNT(*)
FROM {catalog}.{silver}.appraisal_risks WHERE extraction_method = 'PDF_TEXT_FALLBACK'
GROUP BY ALL
UNION ALL
SELECT 'isr_snapshots (ratings)', project_id, source_document, pdo_rating_page_number, COUNT(*)
FROM {catalog}.{silver}.isr_snapshots WHERE pdo_rating_extraction_method = 'PDF_TEXT_FALLBACK'
GROUP BY ALL
ORDER BY table_name, project_id;

-- name: unresolved_records
-- Records whose values are not established (AMBIGUOUS / CONFLICT / MISSING).
SELECT 'project_results' AS table_name, status, project_id, COUNT(*) AS records
FROM {catalog}.{silver}.project_results WHERE status IN ('AMBIGUOUS', 'CONFLICT', 'MISSING')
GROUP BY ALL
UNION ALL
SELECT 'appraisal_risks', status, project_id, COUNT(*)
FROM {catalog}.{silver}.appraisal_risks WHERE status IN ('AMBIGUOUS', 'CONFLICT', 'MISSING')
GROUP BY ALL
UNION ALL
SELECT 'project_events', status, project_id, COUNT(*)
FROM {catalog}.{silver}.project_events WHERE status IN ('AMBIGUOUS', 'CONFLICT', 'MISSING')
GROUP BY ALL
ORDER BY table_name, status, project_id;

-- name: indicator_match_candidates
SELECT project_id, review_status, candidate_reason, indicator_name_raw_a, indicator_name_raw_b,
       first_isr_sequence_a, last_isr_sequence_a, first_isr_sequence_b, last_isr_sequence_b
FROM {catalog}.{silver}.indicator_match_candidates
ORDER BY project_id, indicator_name_raw_a;

-- name: cross_source_disagreements
SELECT project_id, check_code, severity, message, document_id, page_number
FROM {catalog}.{silver}.data_quality_observations
WHERE check_code IN ('CROSS_SOURCE_DIFFERENCE', 'CROSS_SOURCE_NOT_COMPARABLE')
ORDER BY project_id, check_code;

-- name: quality_summary
SELECT layer, severity, check_code, COUNT(*) AS observations
FROM {catalog}.{silver}.data_quality_observations
GROUP BY ALL
ORDER BY layer, severity, check_code;

-- name: provenance_example
-- "Where did this value come from?": result value -> page/table/row -> source file hash.
SELECT r.project_id, r.isr_sequence, r.indicator_name_raw, r.current_value, r.`current_date`,
       r.evidence_filename, r.evidence_page_number, r.evidence_table_id, r.evidence_row,
       r.evidence_extraction_method, r.evidence_text, r.evidence_hash,
       d.relative_path, d.sha256 AS bronze_inventory_sha256, d._source_sha256
FROM {catalog}.{silver}.project_results r
JOIN {catalog}.{bronze}.document_inventory d ON d.document_id = r.document_id
WHERE r.project_id = 'P130544' AND r.isr_sequence = 24
ORDER BY r.indicator_name_raw
LIMIT 5;

-- name: decimal_precision
SELECT raw_loan_number, original_principal_usd, cancelled_amount_usd, disbursed_amount_usd,
       undisbursed_amount_usd, exchange_adjustment_usd, board_approval_date, closing_date
FROM {catalog}.{silver}.loans
ORDER BY raw_loan_number;

-- name: no_gold_tables
SELECT COUNT(*) AS gold_tables
FROM {catalog}.information_schema.tables
WHERE table_schema = '{gold}';
