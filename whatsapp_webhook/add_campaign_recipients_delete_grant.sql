-- Needed once the Curated List feature (see CLAUDE.md /
-- Whatsfly_Integration_docs/bulk-send-build-plan.md) was redesigned to
-- reuse campaign_recipients/campaigns directly instead of a new table --
-- removing someone from a curated list before it's ever sent
-- (processing/wf_bulk_campaign.py::remove_curated_recipient) is a real
-- DELETE, and the existing streamlit_campaign_writer role was never
-- granted it (a real campaign in progress never deletes a recipient row,
-- so this was never needed until now).
--
-- Run once, against the live database:
--   psql -h <host> -U <admin> -d whatsapp_webhooks -f add_campaign_recipients_delete_grant.sql

GRANT DELETE ON campaign_recipients TO streamlit_campaign_writer;
