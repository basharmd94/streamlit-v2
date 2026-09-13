-- Template variable -> customer-attribute (or flat-value) mapping, saved
-- once per WhatsFly template so building a real campaign around an
-- ad-hoc-created template doesn't need a new Python handler each time
-- (see Whatsfly_Integration_docs/bulk-send-build-plan.md, Phase 6 --
-- this supersedes the original "write code together per campaign" plan
-- with a genuinely self-service mapping tool, per explicit follow-up
-- once it became clear templates would be created ad hoc, not one at a
-- time with a build session each time).
--
-- Account-wide, not per-ZID -- one WhatsApp Business number / WhatsFly
-- account serves every ZID (see CLAUDE.md), and templates are a property
-- of that one account, not any single business.
--
-- Run once, against the existing whatsapp_webhooks database (same one
-- the campaign tables live in), e.g.:
--   psql -h <host> -U <admin> -d whatsapp_webhooks -f add_template_variable_mappings_table.sql
-- Then run add_template_mapping_grants.sql to give the existing
-- streamlit_campaign_writer role access to this new table.
--
-- Pre-9.5 Postgres server, same as every other migration script in this
-- repo -- no `IF NOT EXISTS`, no `ON CONFLICT` (saving a mapping is an
-- UPDATE-then-INSERT-if-0-rows from the Python side instead, matching
-- every other upsert in this app). Safe to run once; errors (not a
-- silent no-op) if run twice.

CREATE TABLE template_variable_mappings (
    id              BIGSERIAL PRIMARY KEY,
    template_id     TEXT NOT NULL,      -- WhatsFly's short internal `id` field -- the send-time identity (see the "naming trap" note elsewhere: NOT the longer `template_id` the template-list response also returns). Stable even if template_name were ever reused.
    template_name   TEXT NOT NULL,      -- denormalized, for display only
    variable_name   TEXT NOT NULL,      -- exactly as it appears in the template's own variable_map (e.g. "CUSCODE") -- WhatsFly's own name, chosen when the template was built
    source_type     TEXT NOT NULL CHECK (source_type IN ('customer_attribute', 'flat_value')),
    source_key      TEXT,               -- e.g. "cusid" when source_type = 'customer_attribute' -- NULL for 'flat_value' (its actual value is entered fresh per campaign at send time, never stored here)
    created_by      TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (template_id, variable_name)
);
CREATE INDEX idx_template_variable_mappings_template_id ON template_variable_mappings(template_id);
