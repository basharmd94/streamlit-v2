-- Adds columns to campaign_recipients for manually-pulled WhatsFly status
-- reconciliation -- views/marketing.py's Campaign History "🔄 Reconcile
-- Unconfirmed" section, for a recipient whose webhook never arrived at
-- all (see CLAUDE.md's 2026-09-15 incident: WhatsFly silently stopped
-- calling the webhook for a whole 37-recipient campaign, though the
-- messages themselves sent fine and had real status visible on
-- WhatsFly's own dashboard). Pulled directly via WhatsFly's
-- GET/POST /whatsapp/get/message-status (core/whatsfly.py::
-- get_message_status) instead of waiting on a webhook that may never
-- come.
--
-- whatsfly_raw is always stored regardless of whether whatsfly_status/
-- whatsfly_error_detail were extracted correctly -- the response shape
-- for this endpoint isn't confirmed against the live account yet, so
-- the parsed columns are a best-effort guess and the raw payload is the
-- fallback source of truth to re-parse later without another live call.
--
-- Non-destructive ALTER, same pattern as every other incremental
-- migration in this repo (add_marketing_leads_area_lead_cost_columns.sql,
-- add_item_attribute_source_type.sql) -- pre-9.5 Postgres server, no
-- ADD COLUMN IF NOT EXISTS (a 9.6+ feature). Safe to run once; errors
-- (not a silent no-op) if run twice.
--
-- No new grants needed -- streamlit_campaign_writer already has UPDATE
-- on campaign_recipients (see add_streamlit_campaign_role.sql).
--
--   psql -h <host> -U <admin> -d whatsapp_webhooks -f add_campaign_recipients_whatsfly_reconciliation_columns.sql

ALTER TABLE campaign_recipients ADD COLUMN whatsfly_status TEXT;
ALTER TABLE campaign_recipients ADD COLUMN whatsfly_checked_at TIMESTAMPTZ;
ALTER TABLE campaign_recipients ADD COLUMN whatsfly_error_detail TEXT;
ALTER TABLE campaign_recipients ADD COLUMN whatsfly_raw JSONB;
