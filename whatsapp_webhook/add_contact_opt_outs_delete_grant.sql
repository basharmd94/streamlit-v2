-- Fixes a real gap: streamlit_campaign_writer was created (see
-- add_streamlit_campaign_role.sql) with only SELECT/INSERT/UPDATE on
-- contact_opt_outs -- but Phase 5's "↩️ Remove Opt-Out" button
-- (processing/wf_bulk_campaign.py::remove_opt_out) does a real DELETE.
-- On the live server this would fail with a permission error the moment
-- someone actually removes an opt-out (never previously noticed --
-- earlier verification ran against a scratch database as its owning
-- role, which bypasses grants entirely). Found while adding the grants
-- for the new curated-list table below.
--
-- Run once, against the live database:
--   psql -h <host> -U <admin> -d whatsapp_webhooks -f add_contact_opt_outs_delete_grant.sql

GRANT DELETE ON contact_opt_outs TO streamlit_campaign_writer;
