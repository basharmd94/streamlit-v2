-- Grants the EXISTING streamlit_campaign_writer role (see
-- add_streamlit_campaign_role.sql, already applied for real against the
-- live database) access to the new template_variable_mappings table too
-- -- same role, no new one needed, since this is the same overall
-- feature area (Bulk Messaging) and the same database.
--
-- Run once, after add_template_variable_mappings_table.sql has created
-- the table this grants against:
--   psql -h <host> -U <admin> -d whatsapp_webhooks -f add_template_mapping_grants.sql

GRANT SELECT, INSERT, UPDATE ON template_variable_mappings TO streamlit_campaign_writer;
GRANT USAGE, SELECT ON SEQUENCE template_variable_mappings_id_seq TO streamlit_campaign_writer;
