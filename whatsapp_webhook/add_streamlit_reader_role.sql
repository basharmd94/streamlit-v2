-- Read-only Postgres role for Streamlit's SELECT-only access into the
-- whatsapp_webhooks database (core/whatsapp_webhook_db.py -- WhatsFly
-- Messaging's inline conversation history, and Campaign History's
-- delivered/read/failure-reason enrichment). Confirmed by grepping every
-- query in core/whatsapp_webhook_db.py -- SELECT only, on exactly these
-- four tables: webhook_events, messages, contacts, message_status_events.
-- Never templates/account_alerts (not queried by that module) and never
-- campaigns/campaign_recipients/contact_opt_outs/template_variable_mappings
-- (those belong to streamlit_campaign_writer, a completely separate role
-- -- see add_streamlit_campaign_role.sql -- kept apart on purpose so this
-- read-only role can never accidentally write anything).
--
-- STATUS: unlike streamlit_campaign_writer, there was never a tracked
-- script for this role anywhere in this repo -- it must have been created
-- manually at some point before this session, with no record of it. Found
-- while diagnosing a real "config/whatsapp_webhook_db.ini not found" error
-- on the live server.
--
-- BEFORE running this: check whether the role already exists --
--   psql -h <host> -U <admin> -d whatsapp_webhooks -c "\du streamlit_reader"
-- If it already exists, skip the CREATE ROLE line below and run only the
-- GRANT statements (they're safe to re-run). If it doesn't exist, run the
-- whole file, then fill the real password into config/whatsapp_webhook_db.ini
-- (see core/whatsapp_webhook_db.py's own error message for the exact
-- format) -- never commit that file.

CREATE ROLE streamlit_reader LOGIN PASSWORD 'REPLACE_ME_WITH_A_REAL_PASSWORD';

GRANT CONNECT ON DATABASE whatsapp_webhooks TO streamlit_reader;
GRANT USAGE ON SCHEMA public TO streamlit_reader;

GRANT SELECT ON webhook_events, messages, contacts, message_status_events TO streamlit_reader;
