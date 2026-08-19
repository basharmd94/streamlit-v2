-- ─────────────────────────────────────────────────────────────────────────────
-- Marketing Leads CRM — add `area` and `lead_cost` to an EXISTING
-- marketing_leads table. Non-destructive: preserves every current lead and
-- call-log row. Use this instead of dropping/recreating the table if
-- marketing_leads already has real data on it.
--
-- No IF NOT EXISTS on the ADD COLUMN clauses -- this server predates
-- Postgres 9.6 (confirmed elsewhere: CREATE INDEX IF NOT EXISTS and
-- ON CONFLICT both throw syntax errors on it), and ADD COLUMN IF NOT EXISTS
-- is itself a 9.6+ feature. Safe to run once; running it a second time will
-- error with "column already exists" rather than silently no-op.
--
-- Run this once against the live app database:
--   psql -h <host> -U <user> -d <dbname> -f db/sql_scripts/add_marketing_leads_area_lead_cost_columns.sql
-- ─────────────────────────────────────────────────────────────────────────────

ALTER TABLE marketing_leads ADD COLUMN area       VARCHAR(255);
ALTER TABLE marketing_leads ADD COLUMN lead_cost  NUMERIC(12,2);
