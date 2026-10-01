-- Phase 7 validation / demonstration queries (read-only). Placeholders {catalog},
-- {bronze}, {silver}, {gold} are rendered from configuration by
-- worldbank_copilot.lakehouse.validation_sql. Each query starts with "-- name: <id>".

-- name: project_360
SELECT project_id, project_name, instrument, project_status, approval_date,
       effectiveness_date, original_closing_date, current_closing_date, days_extended,
       disbursement_pct_of_net_principal, latest_isr_sequence, latest_isr_date,
       latest_do_rating, latest_ip_rating, do_rating_change, ip_rating_change,
       number_of_restructurings, number_of_result_indicators,
       current_attention_signal_count, current_watch_signal_count, current_high_signal_count
FROM {catalog}.{gold}.project_360
ORDER BY project_id;

-- name: timeline_p130544
-- event_date is source-stated only; candidate dates are shown separately.
SELECT event_sequence, event_type, event_title, event_date, event_date_status,
       candidate_event_date, ordering_date, ordering_basis, isr_sequence, loan_number,
       event_description, source_table, document_id, page_number
FROM {catalog}.{gold}.project_timeline
WHERE project_id = 'P130544'
ORDER BY event_sequence;

-- name: isr_chronology_p179039
-- ISR sequence is the ordering dimension; ISR 5's printed date is later than ISR 6's.
SELECT event_sequence, isr_sequence, event_date, ordering_date, ordering_basis,
       date_sequence_anomaly
FROM {catalog}.{gold}.project_timeline
WHERE project_id = 'P179039' AND event_type = 'ISR_REPORT'
ORDER BY event_sequence;

-- name: latest_ratings
SELECT project_id, latest_isr_sequence, latest_isr_date, latest_do_rating,
       previous_do_rating, do_rating_change, latest_ip_rating, previous_ip_rating,
       ip_rating_change, latest_overall_risk_rating
FROM {catalog}.{gold}.project_360
ORDER BY project_id;

-- name: closing_date_changes
SELECT project_id, event_type, event_date, event_date_status, loan_number,
       event_description, source_table, document_id, page_number
FROM {catalog}.{gold}.project_timeline
WHERE event_type IN ('ORIGINAL_CLOSING_DATE', 'CLOSING_DATE_CHANGE', 'CURRENT_CLOSING_DATE')
ORDER BY project_id, event_sequence;

-- name: result_progress_trustworthy
-- Only calculation_status = OK has a progress percentage; others explain why not.
SELECT project_id, isr_sequence, indicator_name, unit, baseline_value, current_value,
       target_value, target_date, progress_percentage, target_gap, trend_direction,
       target_status, identity_review_status, page_number
FROM {catalog}.{gold}.result_progress
WHERE project_id = 'P130544' AND isr_sequence = 24 AND calculation_status = 'OK'
ORDER BY indicator_name;

-- name: result_calculation_status
SELECT project_id, calculation_status, COUNT(*) AS observations
FROM {catalog}.{gold}.result_progress
GROUP BY ALL
ORDER BY project_id, calculation_status;

-- name: risks_and_findings
SELECT project_id, record_type, category, rating, observed_date, isr_sequence,
       resolution_status, source_document_type, page_number
FROM {catalog}.{gold}.risk_register
ORDER BY project_id, record_type, category;

-- name: signals_by_project
SELECT project_id, signal_status, severity, COUNT(*) AS signals
FROM {catalog}.{gold}.attention_signals
GROUP BY ALL
ORDER BY project_id, signal_status, severity;

-- name: signals_by_category
SELECT signal_category, rule_id, rule_version, severity, COUNT(*) AS signals
FROM {catalog}.{gold}.attention_signals
GROUP BY ALL
ORDER BY signal_category, rule_id, severity;

-- name: current_high_and_watch
SELECT project_id, severity, signal_category, signal_title, subject, signal_description,
       threshold, caveats
FROM {catalog}.{gold}.attention_signals
WHERE signal_status = 'CURRENT' AND severity IN ('HIGH', 'WATCH')
ORDER BY project_id, CASE severity WHEN 'HIGH' THEN 1 ELSE 2 END, signal_category;

-- name: signal_provenance
-- Gold signal -> Silver source record -> document page/table/text -> source file hash.
WITH evidence AS (
  SELECT record_id, evidence_filename AS filename, evidence_page_number AS page,
         evidence_section AS section, evidence_table_id AS table_id,
         evidence_extraction_method AS method, evidence_text AS text, document_id
  FROM {catalog}.{silver}.project_results
  UNION ALL
  -- ISR rows: page and method come from the signal itself (the rating that fired it).
  SELECT record_id, source_document, NULL, NULL, NULL, NULL, NULL, document_id
  FROM {catalog}.{silver}.isr_snapshots
  UNION ALL
  SELECT record_id, evidence_filename, evidence_page_number, evidence_section,
         evidence_table_id, evidence_extraction_method, evidence_text, evidence_document_id
  FROM {catalog}.{silver}.project_events
  UNION ALL
  SELECT record_id, evidence_filename, evidence_page_number, evidence_section,
         evidence_table_id, evidence_extraction_method, evidence_text, evidence_document_id
  FROM {catalog}.{silver}.project_enrichment
)
SELECT s.project_id, s.rule_id, s.severity, s.subject, s.source_table, s.source_record_id,
       e.filename, coalesce(s.page_number, e.page) AS page, coalesce(s.section, e.section) AS section,
       e.table_id, coalesce(s.extraction_method, e.method) AS method,
       substr(e.text, 1, 160) AS text,
       d.relative_path, d.sha256 AS source_sha256, size(s.supporting_record_ids) AS supporting
FROM {catalog}.{gold}.attention_signals s
LEFT JOIN evidence e ON e.record_id = s.source_record_id
LEFT JOIN {catalog}.{bronze}.document_inventory d ON d.document_id = e.document_id
WHERE s.signal_status = 'CURRENT'
ORDER BY s.project_id, s.severity, s.rule_id;

-- name: restructuring_history
SELECT project_id, event_sequence, event_date, event_date_status, candidate_event_date,
       event_description, document_id, page_number
FROM {catalog}.{gold}.project_timeline
WHERE event_type IN ('RESTRUCTURING', 'ADDITIONAL_FINANCING', 'CANCELLATION')
ORDER BY project_id, event_sequence;

-- name: financial_execution
SELECT project_id, instrument, financial_snapshot_date, original_principal_usd, cancelled_usd,
       net_principal_usd, disbursed_usd, undisbursed_usd, disbursement_pct_of_net_principal,
       loans_with_valuation_caveats
FROM {catalog}.{gold}.project_360
ORDER BY project_id;

-- name: unresolved_indicator_aliases
SELECT c.project_id, c.review_status, c.indicator_name_raw_a, c.indicator_name_raw_b,
       COUNT(DISTINCT r.canonical_indicator_id) AS gold_series
FROM {catalog}.{silver}.indicator_match_candidates c
JOIN {catalog}.{gold}.result_progress r
  ON r.canonical_indicator_id IN (c.indicator_key_a, c.indicator_key_b)
WHERE c.review_status = 'PENDING_REVIEW'
GROUP BY ALL
ORDER BY c.project_id, c.indicator_name_raw_a;

-- name: gold_quality_observations
SELECT severity, check_id, table_name, observation_count, message
FROM {catalog}.{gold}.quality_observations
ORDER BY CASE severity WHEN 'ERROR' THEN 1 WHEN 'WARNING' THEN 2 ELSE 3 END, check_id;

-- name: no_ai_interpretation
SELECT 'attention_signals' AS table_name, COUNT(*) AS ai_rows
FROM {catalog}.{gold}.attention_signals WHERE provenance_class = 'AI_INTERPRETATION'
UNION ALL
SELECT 'project_timeline', COUNT(*) FROM {catalog}.{gold}.project_timeline
WHERE provenance_class = 'AI_INTERPRETATION';
