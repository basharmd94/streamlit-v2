-- Narrow write-scoped Postgres role for Streamlit's Bulk Messaging send
-- step — INSERT/UPDATE/SELECT on ONLY the campaign tables (campaigns,
-- campaign_recipients, contact_opt_outs, template_variable_mappings).
-- Deliberately does NOT touch the existing streamlit-side SELECT-only
-- role documented in CLAUDE.md (core/whatsapp_webhook_db.py) — that role
-- keeps its existing read-only access to messages/contacts/webhook_events/
-- etc. unchanged; this is a SEPARATE, additional role, used only by the
-- new campaign code, so nothing about the existing WhatsFly Messaging
-- conversation history feature's access model changes.
--
-- STATUS: already applied for real, WITHOUT template_variable_mappings
-- (added to this file after that role was created) — see
-- add_template_mapping_grants.sql for the separate grant that needs to
-- run for real against the live database. This file is kept accurate as
-- a from-scratch reference for a fresh database, where all four grants
-- can be applied in one pass.
--
-- Run once, after add_bulk_campaign_tables.sql + add_template_variable_
-- mappings_table.sql (or schema.sql on a fresh database) have created
-- the tables this grants against:
--   psql -h <host> -U <admin> -d whatsapp_webhooks -f add_streamlit_campaign_role.sql
--
-- Replace REPLACE_ME_WITH_A_REAL_PASSWORD before running, then record the
-- real password in config/whatsapp_webhook_db.ini's own credentials (never
-- commit it) — same convention as every other *.ini in this repo.

CREATE ROLE streamlit_campaign_writer LOGIN PASSWORD 'REPLACE_ME_WITH_A_REAL_PASSWORD';

GRANT CONNECT ON DATABASE whatsapp_webhooks TO streamlit_campaign_writer;
GRANT USAGE ON SCHEMA public TO streamlit_campaign_writer;

GRANT SELECT, INSERT, UPDATE ON campaigns, campaign_recipients, contact_opt_outs, template_variable_mappings
    TO streamlit_campaign_writer;

-- BIGSERIAL id columns need explicit sequence USAGE — without this, INSERT
-- (which relies on the column default nextval(...)) fails even with table
-- INSERT privilege granted above.
GRANT USAGE, SELECT ON SEQUENCE
    campaigns_id_seq, campaign_recipients_id_seq, contact_opt_outs_id_seq, template_variable_mappings_id_seq
    TO streamlit_campaign_writer;
