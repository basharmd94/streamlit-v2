-- ─────────────────────────────────────────────────────────────────────────────
-- Marketing Leads CRM — export existing data to CSV BEFORE dropping the tables
-- to add the area/lead_cost columns. Pair with
-- restore_marketing_leads_from_csv.sql, which reloads these CSVs afterward.
--
-- \copy runs on the CLIENT (wherever you run psql from), not the server, so
-- the CSVs land in your current working directory -- no server filesystem
-- access needed.
--
-- Run this BEFORE the DROP TABLE / CREATE TABLE step:
--   psql -h <host> -U <user> -d <dbname> -f db/sql_scripts/backup_marketing_leads_to_csv.sql
-- ─────────────────────────────────────────────────────────────────────────────

\copy marketing_leads TO 'marketing_leads_backup.csv' WITH (FORMAT csv, HEADER true)
\copy marketing_lead_call_log TO 'marketing_lead_call_log_backup.csv' WITH (FORMAT csv, HEADER true)

-- Row counts for this backup -- keep these numbers to compare against after
-- restore_marketing_leads_from_csv.sql runs.
SELECT count(*) AS leads_backed_up FROM marketing_leads;
SELECT count(*) AS call_logs_backed_up FROM marketing_lead_call_log;
