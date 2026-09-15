-- Widens template_variable_mappings.source_type's CHECK constraint to
-- also allow 'item_attribute' -- the app started saving mappings with
-- this source_type (Template Mapping's new Item Code/Item Name/Standard
-- Price/Wholesale-Discounted Price/Product List options, see
-- views/marketing.py::_show_wf_template_mapping and
-- processing/wf_template_mapping.py::ITEM_ATTRIBUTES) before this table's
-- own constraint was widened to match -- a real bug, caught live by a
-- CheckViolation on the very first save attempt:
--   psycopg2.errors.CheckViolation: new row for relation
--   "template_variable_mappings" violates check constraint
--   "template_variable_mappings_source_type_check"
--
-- Run this ONCE against a server that already ran
-- add_template_variable_mappings_table.sql (i.e. any real production
-- server today) -- a brand-new setup should use the already-updated
-- CREATE TABLE in that script instead and does not need this file.
--
-- The constraint name below (template_variable_mappings_source_type_check)
-- is Postgres's own auto-generated name for an inline, unnamed column
-- CHECK -- confirmed it matches exactly what the live error above named.
--
-- Pre-9.5 Postgres server, same as every other migration script in this
-- repo -- no `IF EXISTS` on the DROP (errors, not a silent no-op, if run
-- twice -- same convention as add_marketing_leads_area_lead_cost_columns.sql).
--
--   psql -h <host> -U <admin> -d whatsapp_webhooks -f add_item_attribute_source_type.sql

ALTER TABLE template_variable_mappings
    DROP CONSTRAINT template_variable_mappings_source_type_check;

ALTER TABLE template_variable_mappings
    ADD CONSTRAINT template_variable_mappings_source_type_check
    CHECK (source_type IN ('customer_attribute', 'flat_value', 'item_attribute'));
