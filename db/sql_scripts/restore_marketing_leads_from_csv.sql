-- ─────────────────────────────────────────────────────────────────────────────
-- Marketing Leads CRM — reload the CSVs produced by backup_marketing_leads_to_csv.sql
-- AFTER the tables have been dropped and recreated with the area/lead_cost
-- columns. Preserves the original `id` values (critical: marketing_lead_call_log
-- rows reference marketing_leads.id by that exact number, so a fresh SERIAL
-- renumbering on reload would silently point every restored call log at the
-- wrong lead) and resets both SERIAL sequences afterward so new rows created
-- through the app don't collide with the restored ids.
--
-- area and lead_cost are NOT in the backup CSV (they didn't exist in the old
-- schema) -- every restored lead gets NULL for both, which is correct: none
-- of the old leads have this information yet.
--
-- Run this AFTER CREATE TABLE (db/sql_scripts/create_marketing_leads_tables.sql
-- has already been run to recreate the tables with the new schema), from the
-- same directory the CSVs were written to:
--   psql -h <host> -U <user> -d <dbname> -f db/sql_scripts/restore_marketing_leads_from_csv.sql
-- ─────────────────────────────────────────────────────────────────────────────

\copy marketing_leads (id, zid, fb_lead_id, created_time, ad_id, ad_name, adset_id, adset_name, campaign_id, campaign_name, form_id, form_name, is_organic, platform, full_name, work_phone_number, company_name, street_address, job_title, inbox_url, lead_status, extra_fields, lead_stage, uploaded_by, uploaded_at) FROM 'marketing_leads_backup.csv' WITH (FORMAT csv, HEADER true)

\copy marketing_lead_call_log (id, zid, lead_id, called_at, called_by, outcome, next_visit_date, notes) FROM 'marketing_lead_call_log_backup.csv' WITH (FORMAT csv, HEADER true)

-- Advance both SERIAL sequences past the restored ids -- otherwise the next
-- lead/call-log saved through the app would try to reuse an id that already
-- exists and fail on the UNIQUE/PRIMARY KEY constraint.
SELECT setval(pg_get_serial_sequence('marketing_leads', 'id'), COALESCE((SELECT MAX(id) FROM marketing_leads), 1));
SELECT setval(pg_get_serial_sequence('marketing_lead_call_log', 'id'), COALESCE((SELECT MAX(id) FROM marketing_lead_call_log), 1));

-- Compare these against the leads_backed_up / call_logs_backed_up numbers
-- backup_marketing_leads_to_csv.sql printed -- they should match exactly.
SELECT count(*) AS leads_restored FROM marketing_leads;
SELECT count(*) AS call_logs_restored FROM marketing_lead_call_log;
