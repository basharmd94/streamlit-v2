-- Grants the EXISTING streamlit_campaign_writer role (see
-- add_streamlit_campaign_role.sql, already applied for real against the
-- live database) access to the new curated_list_contacts table -- same
-- role, no new one needed, same overall feature area (Bulk Messaging)
-- and same database. Includes DELETE (unlike template_variable_mappings'
-- own grant script) since removing a curated-list entry is a real
-- DELETE, not an upsert -- see remove_from_curated in
-- processing/wf_curated_list.py.
--
-- Run once, after add_curated_list_table.sql has created the table this
-- grants against:
--   psql -h <host> -U <admin> -d whatsapp_webhooks -f add_curated_list_grants.sql

GRANT SELECT, INSERT, UPDATE, DELETE ON curated_list_contacts TO streamlit_campaign_writer;
GRANT USAGE, SELECT ON SEQUENCE curated_list_contacts_id_seq TO streamlit_campaign_writer;
