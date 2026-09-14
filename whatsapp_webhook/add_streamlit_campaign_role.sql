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
-- ALSO STATUS: the real deployed role is missing DELETE on
-- contact_opt_outs entirely — a genuine gap found while first adding a
-- curated-list feature (Phase 5's "Remove Opt-Out" does a real DELETE,
-- but this file never granted it). See
-- add_contact_opt_outs_delete_grant.sql for the standalone fix to run
-- against the live database. It's also missing DELETE on
-- campaign_recipients, needed once the Curated List feature (see
-- CLAUDE.md / bulk-send-build-plan.md) was redesigned to reuse this
-- same table instead of a new one — see
-- add_campaign_recipients_delete_grant.sql for that standalone fix.
-- This file is kept accurate for a from-scratch database only, where
-- DELETE on both tables below is included from the start.
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

GRANT SELECT, INSERT, UPDATE ON
    campaigns, campaign_recipients, contact_opt_outs, template_variable_mappings
    TO streamlit_campaign_writer;

-- DELETE only where the Python side actually deletes a row: opting a
-- customer back in (remove_opt_out) is a real DELETE, not a soft flag;
-- so is removing someone from a Curated List before it's ever sent
-- (remove_curated_recipient) — the Curated List feature deliberately
-- reuses campaign_recipients/campaigns instead of its own table (one
-- "draft" campaigns row per ZID, identified by template_id = '', with
-- filters_used = '[]' marking it as hand-curated rather than
-- filter-built — see processing/wf_bulk_campaign.py and
-- bulk-send-build-plan.md). Every other table/row here is INSERT/UPDATE
-- only once a real send starts, so DELETE is deliberately withheld from
-- the rest — narrowest grant that actually works.
GRANT DELETE ON contact_opt_outs, campaign_recipients TO streamlit_campaign_writer;

-- BIGSERIAL id columns need explicit sequence USAGE — without this, INSERT
-- (which relies on the column default nextval(...)) fails even with table
-- INSERT privilege granted above.
GRANT USAGE, SELECT ON SEQUENCE
    campaigns_id_seq, campaign_recipients_id_seq, contact_opt_outs_id_seq, template_variable_mappings_id_seq
    TO streamlit_campaign_writer;
